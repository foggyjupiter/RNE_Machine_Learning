"""Training: count + material-property regression with uncertainty, contrast-pair losses,
optional sim-to-real adaptation.

Every head predicts a mean and a log-variance. Loss:

    L = Huber(N̂, N)                                    accuracy of the count (N/100 scale)
      + prop_weight · masked MSE(p̂, p)                 material properties (normalised)
      + nll_weight  · Gaussian NLL(sg(mean), y, σ²)     calibrated uncertainty, all heads
      + pair_weight · mean (N̂_a − N̂_b)²                same_N pairs: invariance to material
      + rank_weight · max(0, m − s·(N̂_a − N̂_b))         same_material pairs: order in N
      + λ(t) · BCE(domain(GRL(embedding)), sim/real)    unlabeled real clips (DANN)

sg = stop-gradient: the variance learns how wrong the mean is without pulling the mean.
A property mask allows rows where only some properties are known (real recordings:
diameter, density and mass of the first-study BB are known, restitution etc. are not).

Augmentation: random time crop (crop_frac of the clip; full clip + crops averaged at
test time), SpecAugment (time/frequency masks) and a random level shift (± gain_db on
the dB image and the level scalars), so the absolute level — confounded with particle
mass and microphone gain — cannot be the only cue.

Real recordings (first study, 15 clips per N): the lowest-numbered clips per N can be
used for adaptation — ``real_train_per_N`` with labels, ``real_unlabeled_per_N``
without labels (domain-adversarial). All other real clips are evaluation only.

After training, σ of every head is rescaled on the validation set so that
mean((error/σ)²) = 1 (variance calibration).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from rneml import data as D
from rneml.features import BASIC_SCALARS, FEATURE_VERSION, build_cache
from rneml.model import CountNet, grad_reverse

LEVEL_SCALARS = [BASIC_SCALARS.index(k) for k in ("db_mean", "db_max", "rms_db")]


@dataclass
class TrainConfig:
    name: str = "blind_mel"
    data_dir: str = "data/v1"
    cache_dir: str = "cache"
    out_dir: str = "outputs"
    real_dir: str | None = None          # first-study clips {N}-{k}.wav
    real_train_per_N: int = 0            # labeled real clips per N used for training
    real_repeat: int = 10                # oversampling of the labeled real clips
    real_unlabeled_per_N: int = 0        # unlabeled real clips per N for domain adaptation
    dann_weight: float = 0.3             # max weight of the domain-adversarial loss
    real_unlabeled_per_batch: int = 8
    train_subset: str = "all"            # "all" (blind) or "bb_like" (known-material baseline)
    feature: str = "mel"                 # "mel" or "scalogram"
    use_modulation: bool = True          # modulation-spectrum features in the scalar branch
    epochs: int = 60
    batch: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    width: int = 32
    dropout: float = 0.3
    use_scalars: bool = True
    prop_weight: float = 0.5
    nll_weight: float = 0.2
    pair_weight: float = 1.0
    rank_weight: float = 0.5
    rank_margin: float = 0.02
    pairs_per_batch: int = 8
    gain_db: float = 3.0
    crop_frac: float = 0.8
    tta_crops: int = 4
    spec_time_masks: int = 2
    spec_freq_masks: int = 2
    calibrate_sigma: bool = True
    patience: int = 12
    seed: int = 0
    workers: int = 4
    threads: int = 0                     # torch CPU threads (0 = default)
    device: str = "auto"
    extra: dict = field(default_factory=dict)


def load_config(path: str | Path | None, **over) -> TrainConfig:
    d = yaml.safe_load(Path(path).read_text()) if path else {}
    if "aux_weight" in d:                # name used by the first version
        d["prop_weight"] = d.pop("aux_weight")
    d.update({k: v for k, v in over.items() if v is not None})
    return TrainConfig(**d)


# ----------------------------------------------------------------------------- features

def features_for(rows: list[dict], cache: Path, workers: int, log=print, kind: str = "mel"):
    """(image [n,F,T] float16, scalars [n,k]) aligned with rows; computes missing ones."""
    cache = Path(cache)
    paths = [r["path"] for r in rows]
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = {p: k for k, p in enumerate(z["paths"].tolist())}
    missing = [p for p in dict.fromkeys(paths) if p not in have]
    if missing:
        log(f"computing {kind} features for {len(missing)} clips → {cache}")
        tmp = build_cache(missing, cache.with_suffix(".new.npz"), workers, log, kind)
        n = np.load(tmp)
        if have:
            o = np.load(cache)
            img = np.concatenate([o["mel"], n["mel"]])
            sc = np.concatenate([o["scalars"], n["scalars"]])
            ps = np.concatenate([o["paths"], n["paths"]])
        else:
            img, sc, ps = n["mel"], n["scalars"], n["paths"]
        np.savez(cache, mel=img, scalars=sc, paths=ps)
        tmp.unlink()
        have = {p: k for k, p in enumerate(ps.tolist())}
    z = np.load(cache)
    idx = np.array([have[p] for p in paths], dtype=np.int64)
    return z["mel"][idx], z["scalars"][idx]


def cache_path(cfg: TrainConfig, what: str) -> Path:
    return Path(cfg.cache_dir) / f"{what}_{cfg.feature}_v{FEATURE_VERSION}.npz"


def scalar_columns(cfg: TrainConfig | dict, n_total: int) -> np.ndarray:
    use_mod = cfg["use_modulation"] if isinstance(cfg, dict) else cfg.use_modulation
    return np.arange(n_total if use_mod else len(BASIC_SCALARS))


class Norm:
    def __init__(self, img: np.ndarray, sc: np.ndarray):
        self.m_mu = float(img.astype(np.float32).mean())
        self.m_sd = float(img.astype(np.float32).std() + 1e-6)
        self.s_mu = sc.mean(0)
        self.s_sd = sc.std(0) + 1e-6

    def __call__(self, img, sc):
        return ((img.astype(np.float32) - self.m_mu) / self.m_sd,
                ((sc - self.s_mu) / self.s_sd).astype(np.float32))

    def state(self):
        return {"m_mu": self.m_mu, "m_sd": self.m_sd, "s_mu": self.s_mu, "s_sd": self.s_sd}

    @classmethod
    def from_state(cls, s):
        o = cls.__new__(cls)
        o.m_mu, o.m_sd = float(s["m_mu"]), float(s["m_sd"])
        o.s_mu, o.s_sd = np.asarray(s["s_mu"]), np.asarray(s["s_sd"])
        return o


# ----------------------------------------------------------------------------- helpers

def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _augment(img: torch.Tensor, sc: torch.Tensor, cfg: TrainConfig, norm: Norm, g):
    B, F, T = img.shape
    if cfg.crop_frac < 1.0:
        L = max(8, int(round(cfg.crop_frac * T)))
        st = torch.randint(0, T - L + 1, (B,), generator=g)
        img = torch.stack([img[b, :, st[b]:st[b] + L] for b in range(B)])
        T = L
    if cfg.gain_db > 0:
        d = (torch.rand(B, generator=g) * 2 - 1) * cfg.gain_db
        img = img + (d / norm.m_sd)[:, None, None].to(img)
        s_sd = torch.as_tensor(norm.s_sd[LEVEL_SCALARS], dtype=sc.dtype)
        sc = sc.clone()
        sc[:, LEVEL_SCALARS] += d[:, None].to(sc) / s_sd.to(sc)
    img = img.clone()
    for _ in range(cfg.spec_time_masks):
        w = torch.randint(0, T // 10 + 1, (B,), generator=g)
        s = torch.randint(0, T, (B,), generator=g)
        for b in range(B):
            img[b, :, s[b]:s[b] + w[b]] = 0.0
    for _ in range(cfg.spec_freq_masks):
        w = torch.randint(0, F // 10 + 1, (B,), generator=g)
        s = torch.randint(0, F, (B,), generator=g)
        for b in range(B):
            img[b, s[b]:s[b] + w[b], :] = 0.0
    return img, sc


def _gnll(mean, y, logvar):
    """Gaussian negative log-likelihood (constant dropped), elementwise."""
    return 0.5 * (logvar + (y - mean) ** 2 * torch.exp(-logvar))


def _views(T: int, crop_frac: float, n_crops: int) -> list[slice]:
    """Full clip + n_crops evenly spaced crops (test-time augmentation)."""
    v = [slice(0, T)]
    if crop_frac < 1.0 and n_crops > 0:
        L = max(8, int(round(crop_frac * T)))
        for s in np.linspace(0, T - L, n_crops).astype(int):
            v.append(slice(int(s), int(s) + L))
    return v


def predict(model, img, sc, norm: Norm, device, batch: int = 64, crop_frac: float = 1.0,
            tta_crops: int = 0, sigma_scale: dict | None = None) -> dict:
    """N mean/std (particles) and normalised property means/stds (averaged over views)."""
    model.eval()
    views = _views(img.shape[2], crop_frac, tta_crops)
    acc = {"n": 0.0, "n_var": 0.0, "props": 0.0, "props_var": 0.0}
    with torch.no_grad():
        for v in views:
            n, nv, p, pv = [], [], [], []
            for i in range(0, len(img), batch):
                m, s = norm(img[i:i + batch, :, v], sc[i:i + batch])
                o = model(torch.from_numpy(m).to(device), torch.from_numpy(s).to(device))
                n.append(o.n.cpu().numpy())
                nv.append(np.exp(o.n_logvar.cpu().numpy()))
                p.append(o.props.cpu().numpy())
                pv.append(np.exp(o.props_logvar.cpu().numpy()))
            cat = (lambda a: np.concatenate(a) if a else np.zeros(0))
            acc["n"] = acc["n"] + cat(n) / len(views)
            acc["n_var"] = acc["n_var"] + cat(nv) / len(views)
            acc["props"] = acc["props"] + cat(p) / len(views)
            acc["props_var"] = acc["props_var"] + cat(pv) / len(views)
    sn = (sigma_scale or {}).get("n", 1.0)
    sp = np.asarray((sigma_scale or {}).get("props", 1.0))
    return {"n": np.asarray(acc["n"]) * 100.0, "n_std": np.sqrt(acc["n_var"]) * 100.0 * sn,
            "props": np.asarray(acc["props"]), "props_std": np.sqrt(acc["props_var"]) * sp}


def metrics(pred: np.ndarray, true: np.ndarray, std: np.ndarray | None = None) -> dict:
    if not len(true):
        return {"n": 0}
    e = pred - true
    out = {"n": int(len(true)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
           "bias": float(e.mean()), "within_5": float((np.abs(e) <= 5).mean()),
           "within_10": float((np.abs(e) <= 10).mean())}
    if std is not None and len(std):
        out.update({"mean_pred_std": float(std.mean()),
                    "in_1sigma": float((np.abs(e) <= std).mean()),      # ideal ≈ 0.68
                    "in_2sigma": float((np.abs(e) <= 2 * std).mean()),  # ideal ≈ 0.95
                    "std_err_corr": float(np.corrcoef(std, np.abs(e))[0, 1]) if len(e) > 2 else None})
    return out


def real_split(cfg: TrainConfig | dict) -> tuple[list[dict], list[dict], list[dict]]:
    """(labeled adaptation clips, unlabeled adaptation clips, evaluation clips)."""
    g = (lambda k: cfg[k]) if isinstance(cfg, dict) else (lambda k: getattr(cfg, k))
    if not g("real_dir"):
        return [], [], []
    rr = D.real_recordings(g("real_dir"))
    nl, nu = g("real_train_per_N"), g("real_unlabeled_per_N")
    lab, unl, test = [], [], []
    for r in rr:
        k = int(r["run_id"].split("-")[1])
        if k <= nl:
            lab.append(r)
        elif k <= nl + nu:
            unl.append(r)
        else:
            test.append(r)
    return lab, unl, test


def calibrate(model, img, sc, y, props, mask, norm, dev, cfg) -> dict:
    """σ scale per head so that mean((error/σ)²) = 1 on the given (validation) rows."""
    P = predict(model, img, sc, norm, dev, crop_frac=cfg.crop_frac, tta_crops=cfg.tta_crops)
    zn = ((P["n"] - y * 100) / P["n_std"]) ** 2
    out = {"n": float(np.sqrt(zn.mean()))}
    if props.shape[1]:
        zp = ((P["props"] - props) / P["props_std"]) ** 2
        m = mask.astype(bool)
        out["props"] = [float(np.sqrt(zp[m[:, k], k].mean())) if m[:, k].any() else 1.0
                        for k in range(props.shape[1])]
    return out


# ----------------------------------------------------------------------------- training

def train(cfg: TrainConfig, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    dev = _device(cfg.device)
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

    rows = D.read_metadata(cfg.data_dir)
    tr = D.select(rows, "train")
    if cfg.train_subset == "bb_like":
        tr = D.bb_like(tr)
    va = D.select(rows, "val")
    pairs = D.select(rows, "pairs")
    real_lab, real_unl, _ = real_split(cfg)
    log(f"train {len(tr)} sim clips (+{len(real_lab)} labeled real ×{cfg.real_repeat}, "
        f"{len(real_unl)} unlabeled real), val {len(va)}, pairs {len(pairs)} "
        f"({cfg.train_subset}, {cfg.feature}{', modulation' if cfg.use_modulation else ''})")

    sim_rows = tr + va + pairs
    img, sc = features_for(sim_rows, cache_path(cfg, Path(cfg.data_dir).name), cfg.workers, log,
                           cfg.feature)
    cols = scalar_columns(cfg, sc.shape[1])
    sc = sc[:, cols]
    y = np.array([r["N"] for r in sim_rows], np.float32) / 100.0
    props = np.stack([D.aux_targets(r) for r in sim_rows])
    mask = np.ones_like(props)
    mask[y == 0] = 0.0                                   # an empty box has no material
    is_real = np.zeros(len(sim_rows), bool)
    ntr, nva = len(tr), len(va)
    train_idx = np.arange(ntr)

    def append_real(rr):
        nonlocal img, sc, y, props, mask, is_real
        ri, rs = features_for(rr, cache_path(cfg, "real"), cfg.workers, log, cfg.feature)
        k0 = len(y)
        img, sc = np.concatenate([img, ri]), np.concatenate([sc, rs[:, cols]])
        yr = np.array([r["N"] for r in rr], np.float32) / 100.0
        y = np.concatenate([y, yr])
        rp, rm = D.real_property_targets(len(rr))
        rm[yr == 0] = 0.0
        props, mask = np.concatenate([props, rp]), np.concatenate([mask, rm])
        is_real = np.concatenate([is_real, np.ones(len(rr), bool)])
        return np.arange(k0, k0 + len(rr))

    if real_lab:
        lab_idx = append_real(real_lab)
        train_idx = np.concatenate([train_idx, np.tile(lab_idx, cfg.real_repeat)])
        (out / "real_train.txt").write_text("\n".join(r["path"] for r in real_lab))
    unl_idx = append_real(real_unl) if real_unl else np.zeros(0, int)
    if len(unl_idx):
        (out / "real_unlabeled.txt").write_text("\n".join(r["path"] for r in real_unl))
    norm = Norm(img[:ntr], sc[:ntr])

    # contrast pairs: index pairs (a, b) within sim_rows, matched by pair_id and render
    by_key: dict = {}
    for k, r in enumerate(sim_rows[ntr + nva:], start=ntr + nva):
        by_key.setdefault((r["pair_id"], int(r["render"])), []).append(k)
    same_n = [tuple(v) for (pid, _), v in by_key.items() if len(v) == 2 and pid.startswith("pN")]
    same_m = [tuple(v) for (pid, _), v in by_key.items() if len(v) == 2 and pid.startswith("pM")]
    pool = [(a, b, 0) for a, b in same_n] + [(a, b, 1) for a, b in same_m]
    log(f"contrast pairs: same_N {len(same_n)}, same_material {len(same_m)}")

    model = CountNet(sc.shape[1], props.shape[1], cfg.width, cfg.dropout, cfg.use_scalars).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    nb = math.ceil(len(train_idx) / cfg.batch)
    total = max(cfg.epochs * nb, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total)
    huber = torch.nn.SmoothL1Loss(beta=0.05)
    bce = torch.nn.BCEWithLogitsLoss()
    use_dann = len(unl_idx) > 0 and cfg.dann_weight > 0

    best, best_ep, hist, step = math.inf, -1, [], 0
    for ep in range(cfg.epochs):
        model.train()
        perm = rng.permutation(train_idx)
        tl, dl, t0 = 0.0, 0.0, time.time()
        for i in range(0, len(perm), cfg.batch):
            idx = perm[i:i + cfg.batch]
            pa = pb = kinds = np.zeros(0, int)
            if cfg.pairs_per_batch and pool:
                sel = rng.choice(len(pool), min(cfg.pairs_per_batch, len(pool)), replace=False)
                pa, pb, kinds = (np.array([pool[k][j] for k in sel]) for j in range(3))
            ru = rng.choice(unl_idx, min(cfg.real_unlabeled_per_batch, len(unl_idx)), replace=False) \
                if use_dann else np.zeros(0, int)
            sup = np.concatenate([idx, pa, pb]).astype(int)
            full = np.concatenate([sup, ru]).astype(int)
            m, s = norm(img[full], sc[full])
            m, s = _augment(torch.from_numpy(m), torch.from_numpy(s), cfg, norm, g)
            o = model(m.to(dev), s.to(dev))
            S = len(sup)
            yt = torch.from_numpy(y[sup]).to(dev)
            n_hat, n_lv = o.n[:S], o.n_logvar[:S]
            loss = huber(n_hat, yt)
            if cfg.nll_weight:
                loss = loss + cfg.nll_weight * _gnll(n_hat.detach(), yt, n_lv).mean()
            if model.props is not None and (cfg.prop_weight or cfg.nll_weight):
                pt = torch.from_numpy(props[sup]).to(dev)
                mk = torch.from_numpy(mask[sup]).to(dev)
                denom = mk.sum().clamp(min=1.0)
                pm, plv = o.props[:S], o.props_logvar[:S]
                loss = loss + cfg.prop_weight * (((pm - pt) ** 2) * mk).sum() / denom
                if cfg.nll_weight:
                    loss = loss + cfg.nll_weight * (_gnll(pm.detach(), pt, plv) * mk).sum() / denom
            if len(pa):
                B, P = len(idx), len(pa)
                na, nb_ = n_hat[B:B + P], n_hat[B + P:S]
                kn = torch.from_numpy(kinds).to(dev)
                if (kn == 0).any() and cfg.pair_weight:
                    loss = loss + cfg.pair_weight * torch.mean((na - nb_)[kn == 0] ** 2)
                if (kn == 1).any() and cfg.rank_weight:
                    sgn = torch.sign(yt[B:B + P] - yt[B + P:S])[kn == 1]
                    loss = loss + cfg.rank_weight * torch.mean(
                        torch.relu(cfg.rank_margin - sgn * (na - nb_)[kn == 1]))
            if use_dann:
                prog = step / total
                lam = cfg.dann_weight * (2.0 / (1.0 + math.exp(-10 * prog)) - 1.0)
                dom = torch.from_numpy(is_real[full].astype(np.float32)).to(dev)
                logit = model.domain(grad_reverse(o.emb, 1.0)).squeeze(1)
                ld = bce(logit, dom)
                loss = loss + lam * ld
                dl += float(ld) * len(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tl += float(loss) * len(idx)
        pv = predict(model, img[ntr:ntr + nva], sc[ntr:ntr + nva], norm, dev)
        mv = metrics(pv["n"], y[ntr:ntr + nva] * 100, pv["n_std"])
        hist.append({"epoch": ep, "train_loss": tl / len(perm), "domain_loss": dl / len(perm),
                     **{f"val_{k}": v for k, v in mv.items()}, "sec": time.time() - t0})
        log(f"ep {ep:3d} loss {tl / len(perm):.4f}"
            + (f" dom {dl / len(perm):.3f}" if use_dann else "")
            + f" val MAE {mv.get('mae', float('nan')):.2f} ±5 {mv.get('within_5', float('nan')):.2f}"
            f" ({time.time() - t0:.0f}s)")
        score = mv.get("mae", math.inf)
        if score < best or nva == 0:
            best, best_ep = score, ep
            torch.save({"model": model.state_dict(), "norm": norm.state(), "n_scalars": sc.shape[1],
                        "n_props": props.shape[1], "prop_names": D.AUX_NAMES, "cfg": asdict(cfg),
                        "feature_version": FEATURE_VERSION}, out / "model.pt")
        elif ep - best_ep >= cfg.patience:
            log(f"early stop (best epoch {best_ep})")
            break
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    if cfg.calibrate_sigma and nva:
        ck = torch.load(out / "model.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        vs = slice(ntr, ntr + nva)
        ck["sigma_scale"] = calibrate(model, img[vs], sc[vs], y[vs], props[vs], mask[vs], norm, dev, cfg)
        torch.save(ck, out / "model.pt")
        log(f"σ calibration factors: N {ck['sigma_scale']['n']:.2f}")
    from rneml.evaluate import evaluate_run
    return evaluate_run(out, cfg, log=log)
