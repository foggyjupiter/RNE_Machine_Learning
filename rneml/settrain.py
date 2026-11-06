"""Training and evaluation of the condition-set model (``model_type: set``).

Same losses as the clip model (rneml.train): Huber on N, property MSE, Gaussian NLL for
σ, same-N invariance and same-material ranking on contrast-pair systems, optional
domain-adversarial adaptation with unlabeled real clips. One sample = one system and
one recording setup (``system_id``, ``render``) with its conditions; conditions are
dropped at random while training (``cond_dropout``).

Evaluation reports the full condition set and named subsets (``eval_condition_subsets``,
e.g. 372 rpm only / three speeds / all five): how many conditions are needed to break
the count-material degeneracy. The real first-study clips are sets of one condition
(372 rpm, Γ ≈ 2.07).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from rneml import data as D
from rneml import splits as SP
from rneml.model import grad_reverse
from rneml.setmodel import (GAMMA_REF, SetCountNet, batch_arrays, build_sets, normalize_levels,
                            predict_sets, set_forward)
from rneml.train import (Norm, _device, _gnll, conformal_metrics,
                         metrics, real_split, scalar_columns)

REAL_GAMMA = GAMMA_REF            # first-study recordings: 372 rpm at A = 1.34 cm


def _features(cfg, rows, what, log):
    from rneml.train import feats
    img, sc = feats(cfg, rows, what, log)
    return img, sc[:, scalar_columns(cfg, sc.shape[1])]


def _set_norm(sets, img, sc, gamma, mode, K, rng, n_max=300) -> Norm:
    pick = [sets[i] for i in rng.choice(len(sets), min(n_max, len(sets)), replace=False)]
    I, S, G, M = batch_arrays(pick, img, sc, gamma, K)
    I, S = normalize_levels(I, S, M, mode)
    return Norm(I[M], S[M])


def train_set(cfg, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    dev = _device(cfg.device)
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

    rows = SP.assign(D.read_metadata(cfg.data_dir), cfg.split, cfg.split_seed)
    SP.check_no_leak(rows)
    SP.save(rows, cfg.split, cfg.split_seed, out / "split.json")
    keep = [r for r in rows if r["role"] in ("train", "val", "pairs")]
    img, sc = _features(cfg, keep, Path(cfg.data_dir).name, log)
    gamma = np.array([float(r.get("gamma", REAL_GAMMA)) for r in keep], np.float32)
    y = np.array([r["N"] for r in keep], np.float32) / 100.0
    props = np.stack([D.aux_targets(r) for r in keep])
    mask_p = np.ones_like(props)
    mask_p[y == 0] = 0.0
    sets = build_sets(keep)
    role = [keep[s[0]]["role"] for s in sets]
    tr = [s for s, r in zip(sets, role) if r == "train"]
    va = [s for s, r in zip(sets, role) if r == "val"]
    pr = [s for s, r in zip(sets, role) if r == "pairs"]
    K = max(cfg.max_conditions, max(len(s) for s in sets))
    log(f"sets: train {len(tr)}, val {len(va)}, pair sets {len(pr)}; up to {K} conditions; "
        f"agg={cfg.agg} diff={cfg.diff_path} norm={cfg.norm_mode} cond={cfg.use_condition}")

    by_pair: dict = {}
    for s in pr:
        r0 = keep[s[0]]
        by_pair.setdefault((r0["pair_id"], r0.get("render", 0), r0.get("pair_type", "")), []).append(s)
    pool = [(v[0], v[1], 0 if k[2] == "same_N" else 1) for k, v in by_pair.items() if len(v) == 2]

    # unlabeled real clips for domain adaptation (sets of one condition)
    _, real_unl, _ = real_split(cfg)
    if real_unl and cfg.dann_weight > 0:
        ri, rs = _features(cfg, real_unl, "real", log)
        k0 = len(keep)
        img, sc = np.concatenate([img, ri]), np.concatenate([sc, rs])
        gamma = np.concatenate([gamma, np.full(len(real_unl), REAL_GAMMA, np.float32)])
        real_sets = [[k0 + i] for i in range(len(real_unl))]
    else:
        real_sets = []

    norm = _set_norm(tr, img, sc, gamma, cfg.norm_mode, K, rng)
    in_ch = img.shape[1] if img.ndim == 4 else 1
    model = SetCountNet(sc.shape[1], props.shape[1], cfg.width, cfg.dropout, cfg.use_scalars,
                        cfg.time_pool, cfg.agg, cfg.diff_path, cfg.use_condition,
                        cfg.count_head, cfg.class_step, in_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    nb = math.ceil(len(tr) / cfg.batch)
    total = max(cfg.epochs * nb, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total)
    huber = torch.nn.SmoothL1Loss(beta=0.05)
    bce = torch.nn.BCEWithLogitsLoss()
    first = lambda ss: np.array([s[0] for s in ss], int)

    best, best_ep, hist, step = math.inf, -1, [], 0
    for ep in range(cfg.epochs):
        model.train()
        perm = rng.permutation(len(tr))
        tl, t0 = 0.0, time.time()
        for i in range(0, len(perm), cfg.batch):
            bs = [tr[j] for j in perm[i:i + cfg.batch]]
            pa, pb, kinds = [], [], np.zeros(0, int)
            if cfg.pairs_per_batch and pool:
                sel = rng.choice(len(pool), min(cfg.pairs_per_batch, len(pool)), replace=False)
                pa, pb = [pool[k][0] for k in sel], [pool[k][1] for k in sel]
                kinds = np.array([pool[k][2] for k in sel])
            ru = [real_sets[k] for k in rng.choice(len(real_sets), min(cfg.real_unlabeled_per_batch,
                                                                     len(real_sets)), replace=False)] \
                if real_sets else []
            allsets = bs + pa + pb + ru
            I, S, G, M = batch_arrays(allsets, img, sc, gamma, K, cfg.cond_dropout, rng)
            if cfg.gain_db > 0 and cfg.norm_mode == "absolute":   # same gain for a whole set
                d = rng.uniform(-cfg.gain_db, cfg.gain_db, len(allsets)).astype(np.float32)
                I = (I.astype(np.float32) + d.reshape(-1, *([1] * (I.ndim - 1)))).astype(np.float16)
            o = set_forward(model, I, S, G, M, norm, cfg.norm_mode, dev)
            Ssup = len(bs) + len(pa) + len(pb)
            sup = first(bs + pa + pb)
            yt = torch.from_numpy(y[sup]).to(dev)
            n_hat = o.n[:Ssup]
            if cfg.count_head == "regression":
                loss = huber(n_hat, yt)
                if cfg.nll_weight:
                    loss = loss + cfg.nll_weight * _gnll(n_hat.detach(), yt, o.n_logvar[:Ssup]).mean()
            else:
                loss = model.count.loss(o.count_logits[:Ssup], yt, cfg.label_sigma / 100.0)
            if model.props is not None and cfg.prop_weight:
                pt = torch.from_numpy(props[sup]).to(dev)
                mk = torch.from_numpy(mask_p[sup]).to(dev)
                dn = mk.sum().clamp(min=1.0)
                pm = o.props[:Ssup]
                loss = loss + cfg.prop_weight * (((pm - pt) ** 2) * mk).sum() / dn
                if cfg.nll_weight:
                    loss = loss + cfg.nll_weight * (_gnll(pm.detach(), pt, o.props_logvar[:Ssup]) * mk).sum() / dn
            if len(pa):
                B, P = len(bs), len(pa)
                na, nb_ = n_hat[B:B + P], n_hat[B + P:Ssup]
                kn = torch.from_numpy(kinds).to(dev)
                if (kn == 0).any() and cfg.pair_weight:
                    loss = loss + cfg.pair_weight * torch.mean((na - nb_)[kn == 0] ** 2)
                if (kn == 1).any() and cfg.rank_weight:
                    sgn = torch.sign(yt[B:B + P] - yt[B + P:Ssup])[kn == 1]
                    loss = loss + cfg.rank_weight * torch.mean(torch.relu(cfg.rank_margin - sgn * (na - nb_)[kn == 1]))
            if ru:
                lam = cfg.dann_weight * (2.0 / (1.0 + math.exp(-10 * step / total)) - 1.0)
                dom = torch.cat([torch.zeros(Ssup), torch.ones(len(ru))]).to(dev)
                loss = loss + lam * bce(model.domain(grad_reverse(o.emb, 1.0)).squeeze(1), dom)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tl += float(loss) * len(bs)
        P = predict_sets(model, va, img, sc, gamma, norm, cfg.norm_mode, dev, K)
        mv = metrics(P["n"], y[first(va)] * 100, P["n_std"])
        hist.append({"epoch": ep, "train_loss": tl / max(len(tr), 1), **{f"val_{k}": v for k, v in mv.items()
                                                                         if not isinstance(v, dict)},
                     "sec": time.time() - t0})
        log(f"ep {ep:3d} loss {tl / max(len(tr), 1):.4f} val MAE {mv.get('mae', float('nan')):.2f} "
            f"({time.time() - t0:.0f}s)")
        if mv.get("mae", math.inf) < best:
            best, best_ep = mv["mae"], ep
            torch.save({"model": model.state_dict(), "norm": norm.state(), "n_scalars": sc.shape[1],
                        "n_props": props.shape[1], "K": K, "cfg": asdict(cfg), "in_channels": in_ch},
                       out / "model.pt")
        elif ep - best_ep >= cfg.patience:
            log(f"early stop (best epoch {best_ep})")
            break
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    # σ calibration and conformal quantiles on validation
    ck = torch.load(out / "model.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    P = predict_sets(model, va, img, sc, gamma, norm, cfg.norm_mode, dev, K)
    t = y[first(va)] * 100
    sn = float(np.sqrt(np.mean(((P["n"] - t) / P["n_std"]) ** 2)))
    r = np.sort(np.abs(P["n"] - t) / (P["n_std"] * sn))
    ck["sigma_scale"] = {"n": sn, "conformal": {
        str(a): float(r[min(len(r) - 1, int(np.ceil((len(r) + 1) * (1 - a))) - 1)]) for a in cfg.conformal_alpha}}
    if props.shape[1]:
        zp = ((P["props"] - props[first(va)]) / P["props_std"]) ** 2
        mv_ = mask_p[first(va)].astype(bool)
        ck["sigma_scale"]["props"] = [float(np.sqrt(zp[mv_[:, k], k].mean())) if mv_[:, k].any() else 1.0
                                      for k in range(props.shape[1])]
    torch.save(ck, out / "model.pt")
    return evaluate_set(out, cfg, log)


def load_set_model(run, device="cpu"):
    ck = torch.load(Path(run) / "model.pt", map_location=device, weights_only=False)
    c = ck["cfg"]
    m = SetCountNet(ck["n_scalars"], ck["n_props"], c["width"], c["dropout"], c["use_scalars"],
                    c.get("time_pool", 1), c["agg"], c["diff_path"], c["use_condition"],
                    c.get("count_head", "regression"), c.get("class_step", 5), ck.get("in_channels", 1))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), Norm.from_state(ck["norm"]), ck


def evaluate_set(run, cfg, log=print) -> dict:
    from rneml.evaluate import _by_material, degeneracy, property_metrics

    run = Path(run)
    dev = _device(cfg.device)
    model, norm, ck = load_set_model(run, dev)
    K = ck["K"]
    ss = ck.get("sigma_scale", {})
    conf = ss.get("conformal")
    sp = np.asarray(ss.get("props", 1.0))

    def scaled(P):
        return {**P, "n_std": P["n_std"] * ss.get("n", 1.0), "props_std": P["props_std"] * sp}

    eval_dir = cfg.eval_data_dir or cfg.data_dir
    rows = SP.assign(D.read_metadata(eval_dir), cfg.split if not cfg.eval_data_dir else None, cfg.split_seed)
    res: dict = {"model": "set", "agg": cfg.agg, "diff_path": cfg.diff_path, "norm_mode": cfg.norm_mode,
                 "use_condition": cfg.use_condition, "feature": cfg.feature}
    preds = []
    for split in ("test", "test_ood"):
        rs = SP.select_role(rows, split)
        if not rs:
            continue
        img, sc = _features(cfg, rs, Path(eval_dir).name, log)
        gamma = np.array([float(r.get("gamma", REAL_GAMMA)) for r in rs], np.float32)
        subsets = dict(cfg.eval_condition_subsets or {"all": None})
        for name, conds in subsets.items():
            sets = build_sets(rs, conds)
            if not sets:
                continue
            P = scaled(predict_sets(model, sets, img, sc, gamma, norm, cfg.norm_mode, dev, K))
            first = [rs[s[0]] for s in sets]
            t = np.array([r["N"] for r in first], float)
            m = metrics(P["n"], t, P["n_std"])
            m["conformal"] = conformal_metrics(P["n"], t, P["n_std"], conf)
            m["mean_set_size"] = float(np.mean([len(s) for s in sets]))
            res.setdefault("by_condition_subset", {}).setdefault(name, {})[split] = m
            if conds is None:
                res[split] = m
                if model.props is not None:
                    res[f"properties_{split}"] = property_metrics(P["props"], P["props_std"], first)
                    res[f"degeneracy_{split}"] = degeneracy(P["n"], P["props"], first)
                if split == "test":
                    res["test_by_material"] = _by_material(first, P["n"], t)
                preds += [{"split": split, "run_id": str(r.get("system_id", r["run_id"])),
                           "render": r.get("render"), "N": r["N"], "pred": float(P["n"][k]),
                           "pred_std": float(P["n_std"][k]), "diameter": r["particle_diameter_m"],
                           "density": r["particle_density"], "e_pw": r["restitution_pw"],
                           "e_pp": r["restitution_pp"]} for k, r in enumerate(first)]
    _, _, real_test = real_split(cfg)
    if real_test:
        img, sc = _features(cfg, real_test, "real", log)
        gamma = np.full(len(real_test), REAL_GAMMA, np.float32)
        P = scaled(predict_sets(model, [[i] for i in range(len(real_test))], img, sc, gamma, norm,
                                cfg.norm_mode, dev, K))
        t = np.array([r["N"] for r in real_test], float)
        res["real"] = metrics(P["n"], t, P["n_std"])
        res["real"]["conformal"] = conformal_metrics(P["n"], t, P["n_std"], conf)
        res["real_by_N"] = {int(n): {"mean_pred": float(P["n"][t == n].mean()),
                                     "std_of_pred": float(P["n"][t == n].std())} for n in np.unique(t)}
        preds += [{"split": "real", "run_id": r["run_id"], "N": r["N"], "pred": float(P["n"][k]),
                   "pred_std": float(P["n_std"][k])} for k, r in enumerate(real_test)]
    (run / "metrics.json").write_text(json.dumps(res, indent=1))
    with open(run / "predictions.jsonl", "w") as fh:
        for q in preds:
            fh.write(json.dumps(q) + "\n")
    from rneml.evaluate import _plot
    _plot(run, preds)
    for name, v in res.get("by_condition_subset", {}).items():
        for split, m in v.items():
            log(f"[{name:12s}] {split:9s} N: MAE {m['mae']:.2f}  ±5 {m['within_5']:.2f}  "
                f"(sets {m['n']}, mean size {m['mean_set_size']:.1f})")
    if "real" in res:
        log(f"real (1 condition) N: MAE {res['real']['mae']:.2f}  ±5 {res['real']['within_5']:.2f}")
    return res
