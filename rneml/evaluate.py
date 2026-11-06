"""Evaluation of a trained run or an ensemble of runs (metrics.json, predictions.jsonl, figures).

Count N — MAE, RMSE, bias, within ±5 / ±10 (first-study ±1 / ±2 classes), and the
predicted uncertainty: mean σ, fraction of errors within 1σ / 2σ (ideal 0.68 / 0.95),
correlation of σ with |error|:

  test              same material distribution as training, unseen runs
  test_ood          held-out material region (never trained on)
  test_by_material  N error by diameter / density / wall-restitution tertiles
  same_N_pair_gap   |N̂_a − N̂_b| for equal N and different material (invariance)
  by_N_range        MAE in N bands 0-19 / 20-49 / 50-79 / 80-100
  conformal         normalised split-conformal intervals N̂ ± q·σ (q from validation):
                    coverage vs target (90 %, 95 %) and mean width, also per N band
  *_gain            the same test clips re-scored with the microphone gain × 0.25…4
                    (with device clipping) when gain_test is set
  real              first-study recordings (BB, 0-100) — sim-to-real; real clips used
                    for adaptation (labeled or unlabeled) are never scored

Material properties (diameter, density, mass, restitution pp/pw, friction pp/pw,
modulus) — MAE in physical units, relative MAE, R², calibration, on test / test_ood
(N = 0 rows skipped: an empty box has no material). On the real recordings the
predicted properties are reported next to the known BB values (6 mm, 0.12 g).

Degeneracy diagnostic: correlation of the N error with the diameter / density / mass
error ("many small" vs "few large" shows up as a negative N-diameter correlation) and
the relative error of the layer number F = N d²/A versus that of N.

Ensembles (``rneml ensemble``): member means are averaged; the predictive variance is
the mean member variance plus the variance of the member means.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from rneml import data as D
from rneml.model import CountNet


@lru_cache(maxsize=4096)
def clip_shake_frequency(path: str) -> float:
    """Shaker speed of a recording, estimated from its sound [Hz]."""
    from rneml.features import HOP, SR, frame_energy, mel_db, power_spectrogram, read_audio, shake_frequency
    return shake_frequency(frame_energy(mel_db(power_spectrogram(read_audio(path)))), SR / HOP)


def speed_diagnostic(f_est: np.ndarray, resid: np.ndarray, n_true: np.ndarray) -> dict:
    """Does the count error follow the shaker speed? The first-study recordings drift
    6.04-6.64 Hz (N = 95/100 at ≈ 6.5 Hz), and a model trained at one exact speed may read
    a faster shaker as more particles. Within-N slope (residual and speed demeaned per true
    N, so N itself cannot drive it) in counts per Hz, its correlation, and the speed range."""
    rd, fd = [], []
    for n in np.unique(n_true):
        m = n_true == n
        if m.sum() >= 3:
            rd.append(resid[m] - resid[m].mean())
            fd.append(f_est[m] - f_est[m].mean())
    out = {"f_min": float(f_est.min()), "f_max": float(f_est.max()), "f_median": float(np.median(f_est))}
    if rd:
        rd, fd = np.concatenate(rd), np.concatenate(fd)
        if fd.std() > 0 and rd.std() > 0:
            out.update({"within_N_slope_per_hz": float((fd * rd).sum() / (fd * fd).sum()),
                        "within_N_corr": float(np.corrcoef(fd, rd)[0, 1]), "n": int(len(rd))})
    return out


def load_model(run: str | Path, device="cpu"):
    from rneml.train import Norm

    ck = torch.load(Path(run) / "model.pt", map_location=device, weights_only=False)
    c = ck["cfg"]
    m = CountNet(ck["n_scalars"], ck["n_props"], c["width"], c["dropout"], c["use_scalars"],
                 c.get("time_pool", 1), c.get("count_head", "regression"), c.get("class_step", 5),
                 ck.get("in_channels", 1), c.get("arch", "cnn"), c.get("temporal", "transformer"))
    sd = dict(ck["model"])
    expected = m.state_dict().keys()
    for k in ("weight", "bias"):          # checkpoints from before the CountHead refactor
        # (the CORAL head has its own count.bias — leave keys the model expects alone)
        if f"count.{k}" in sd and f"count.{k}" not in expected and f"count.lin.{k}" in expected:
            sd[f"count.lin.{k}"] = sd.pop(f"count.{k}")
    m.load_state_dict(sd)
    return m.to(device).eval(), Norm.from_state(ck["norm"]), c, ck.get("sigma_scale")


def predict_runs(runs: list[Path], rows: list[dict], what: str, device, log=print,
                 gain: float = 1.0) -> dict:
    """Ensemble prediction of all runs for rows ('what' names the feature cache)."""
    from rneml.train import TrainConfig, feats, predict, prepare

    outs = []
    for run in runs:
        model, norm, c, ss = load_model(run, device)
        cfg = TrainConfig(**c)
        img, sc = feats(cfg, rows, what, log, gain)
        n_extra = len(cfg.material_input)
        if cfg.component_scalars:
            from rneml.components import N_BANDS, TYPES, selected_columns
            n_extra += (N_BANDS * len(TYPES) if cfg.component_scalars == "all"
                        else len(selected_columns(cfg.component_scalars)))
        n_audio = len(norm.s_mu) - n_extra        # older models: fewer audio scalars
        img, sc = prepare(img, sc, cfg, rows, n_audio)
        outs.append(predict(model, img, sc, norm, device, crop_frac=cfg.crop_frac,
                            tta_crops=cfg.tta_crops, sigma_scale=ss))
    if len(outs) == 1:
        return outs[0]
    mean = lambda k: np.mean([o[k] for o in outs], axis=0)
    var = lambda k, s: np.mean([o[s] ** 2 for o in outs], axis=0) + np.var([o[k] for o in outs], axis=0)
    return {"n": mean("n"), "n_std": np.sqrt(var("n", "n_std")),
            "props": mean("props"), "props_std": np.sqrt(var("props", "props_std"))}


def property_metrics(pz: np.ndarray, pz_std: np.ndarray, rs: list[dict]) -> dict:
    keep = np.array([r["N"] > 0 for r in rs])
    out = {}
    for k, n in enumerate(D.AUX_NAMES):
        col, unit, scale = D.PROPERTIES[n][:3]
        truth = np.array([r[col] for r in rs], float)[keep]
        pred = D.to_physical(n, pz[keep, k])
        zt = np.array([D.PROPERTIES[n][3](v) for v in truth])
        ez = np.abs(pz[keep, k] - zt)
        ss = np.sum((truth - truth.mean()) ** 2)
        out[n] = {"unit": unit, "mae": float(np.abs(pred - truth).mean() * scale),
                  "mean_guess_mae": float(np.abs(truth - truth.mean()).mean() * scale),
                  "rel_mae": float(np.mean(np.abs(pred - truth) / np.abs(truth))),
                  "r2": float(1 - np.sum((pred - truth) ** 2) / ss) if ss > 0 else None,
                  "in_1sigma": float((ez <= pz_std[keep, k]).mean()),
                  "in_2sigma": float((ez <= 2 * pz_std[keep, k]).mean())}
    return out


BOX_AREA = 0.05 * 0.05   # container floor area [m²] (5 × 5 cm)


def degeneracy(pN: np.ndarray, pz: np.ndarray, rs: list[dict]) -> dict:
    """Does the model trade N against the material ("many small" vs "few large")?

    corr(N error, property error): negative for diameter means N is overestimated when
    the diameter is underestimated. Layer number F = N d² / A: if the confusion runs
    along F, the relative error of F̂ is smaller than that of N̂.
    """
    keep = np.array([r["N"] > 0 for r in rs])
    if keep.sum() < 5:
        return {}
    N = np.array([r["N"] for r in rs], float)[keep]
    eN = pN[keep] - N
    out = {"corr_N_error_with": {}}
    for name in ("diameter", "density", "mass"):
        k = D.AUX_NAMES.index(name)
        col = D.PROPERTIES[name][0]
        truth = np.array([r[col] for r in rs], float)[keep]
        e = np.log(np.maximum(D.to_physical(name, pz[keep, k]), 1e-12)) - np.log(truth)
        out["corr_N_error_with"][name] = float(np.corrcoef(eN, e)[0, 1])
    k = D.AUX_NAMES.index("diameter")
    d_true = np.array([r["particle_diameter_m"] for r in rs], float)[keep]
    d_pred = D.to_physical("diameter", pz[keep, k])
    F_true = N * d_true ** 2 / BOX_AREA
    F_pred = np.maximum(pN[keep], 0) * d_pred ** 2 / BOX_AREA
    out["rel_mae_N"] = float(np.mean(np.abs(eN) / N))
    out["rel_mae_layers_F"] = float(np.mean(np.abs(F_pred - F_true) / F_true))
    return out


def _by_material(rs, p, t) -> dict:
    out = {}
    for col in ("particle_diameter_m", "particle_density", "restitution_pw"):
        v = np.array([r[col] for r in rs], float)
        q = np.quantile(v, [1 / 3, 2 / 3])
        b = np.digitize(v, q)
        out[col] = [{"bin": int(k), "range": [float(v[b == k].min()), float(v[b == k].max())],
                     "mae": float(np.abs(p[b == k] - t[b == k]).mean()), "n": int((b == k).sum())}
                    for k in range(3) if (b == k).any()]
    return out


def evaluate_runs(runs: list[str | Path], cfg, out: str | Path, log=print) -> dict:
    """Evaluate one run (or an ensemble) with the data/real settings of ``cfg``."""
    from rneml import splits as SP
    from rneml.train import _device, conformal_metrics, metrics, real_split

    runs = [Path(r) for r in runs]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = _device(cfg.device)
    m0, _, _, ss0 = load_model(runs[0], dev)
    has_props = m0.props is not None
    conf = (ss0 or {}).get("conformal")
    eval_dir = cfg.eval_data_dir or cfg.data_dir
    split_spec = cfg.split if not cfg.eval_data_dir else None
    rows = SP.assign(D.read_metadata(eval_dir), split_spec, cfg.split_seed)
    SP.check_no_leak(rows)
    what = Path(eval_dir).name
    res: dict = {"runs": [str(r) for r in runs], "feature": cfg.feature, "eval_data": str(eval_dir),
                 "material_input": list(cfg.material_input),
                 "train_subset": cfg.train_subset, "real_train_per_N": cfg.real_train_per_N,
                 "real_unlabeled_per_N": cfg.real_unlabeled_per_N}
    preds = []
    for split in ("test", "test_ood", "pairs"):
        rs = SP.select_role(rows, split)
        if not rs:
            continue
        P = predict_runs(runs, rs, what, dev, log)
        t = np.array([r["N"] for r in rs], float)
        res[split] = metrics(P["n"], t, P["n_std"])
        res[split]["conformal"] = conformal_metrics(P["n"], t, P["n_std"], conf)
        if split in ("test", "test_ood"):
            for g in cfg.gain_test:
                Pg = predict_runs(runs, rs, what, dev, log, gain=float(g))
                res.setdefault(f"{split}_gain", {})[str(g)] = metrics(Pg["n"], t, Pg["n_std"])
        for k, r in enumerate(rs):
            q = {"split": split, "run_id": r["run_id"], "render": r.get("render"), "N": r["N"],
                 "pred": float(P["n"][k]), "pred_std": float(P["n_std"][k]),
                 "diameter": r["particle_diameter_m"], "density": r["particle_density"],
                 "e_pw": r["restitution_pw"], "e_pp": r["restitution_pp"],
                 "pair_id": r.get("pair_id", ""), "pair_type": r.get("pair_type", "")}
            if has_props:
                q["props_pred"] = {n: float(D.to_physical(n, P["props"][k, j]))
                                   for j, n in enumerate(D.AUX_NAMES)}
            preds.append(q)
        if split in ("test", "test_ood") and has_props:
            res[f"properties_{split}"] = property_metrics(P["props"], P["props_std"], rs)
            res[f"degeneracy_{split}"] = degeneracy(P["n"], P["props"], rs)
        if split == "test":
            res["test_by_material"] = _by_material(rs, P["n"], t)
        if split == "pairs":
            same_n: dict = {}
            for k, r in enumerate(rs):
                if r.get("pair_type") == "same_N":
                    same_n.setdefault((r["pair_id"], r["render"]), []).append(P["n"][k])
            d = [abs(v[0] - v[1]) for v in same_n.values() if len(v) == 2]
            res["same_N_pair_gap"] = {"n": len(d), "mean_abs_diff": float(np.mean(d)) if d else None}
    _, _, real_test = real_split(cfg)
    if real_test:
        P = predict_runs(runs, real_test, "real", dev, log)
        t = np.array([r["N"] for r in real_test], float)
        res["real"] = metrics(P["n"], t, P["n_std"])
        res["real"]["conformal"] = conformal_metrics(P["n"], t, P["n_std"], conf)
        res["real_by_N"] = {int(n): {"mean_pred": float(P["n"][t == n].mean()),
                                     "std_of_pred": float(P["n"][t == n].std())} for n in np.unique(t)}
        if has_props:
            m = t > 0
            res["real_properties"] = {}
            for j, n in enumerate(D.AUX_NAMES):
                v = D.to_physical(n, P["props"][m, j]) * D.PROPERTIES[n][2]
                res["real_properties"][n] = {
                    "unit": D.PROPERTIES[n][1], "pred_median": float(np.median(v)),
                    "pred_iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                    "true": (D.REAL_BB[n] * D.PROPERTIES[n][2]) if n in D.REAL_BB else None}
        f_est = np.array([clip_shake_frequency(r["path"]) for r in real_test])
        res["real_speed"] = speed_diagnostic(f_est, P["n"] - t, t)
        for k, r in enumerate(real_test):
            preds.append({"split": "real", "run_id": r["run_id"], "N": r["N"],
                          "pred": float(P["n"][k]), "pred_std": float(P["n_std"][k]),
                          "f_shake_est": float(f_est[k])})
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    with open(out / "predictions.jsonl", "w") as fh:
        for q in preds:
            fh.write(json.dumps(q) + "\n")
    _plot(out, preds)
    for k in ("test", "test_ood", "real"):
        if k in res and res[k].get("n"):
            r = res[k]
            c90 = r.get("conformal", {}).get("0.1", {})
            log(f"{k:9s} N: MAE {r['mae']:.2f}  ±5 {r['within_5']:.2f}  ±10 {r['within_10']:.2f}  "
                f"bias {r['bias']:+.2f}  σ̄ {r['mean_pred_std']:.1f} (1σ {r['in_1sigma']:.2f})"
                + (f"  90% interval: cover {c90['coverage']:.2f} width {c90['mean_width']:.1f}" if c90 else "")
                + f"  n={r['n']}")
    for k in ("test_gain", "test_ood_gain"):
        if k in res:
            log(k + ": " + ", ".join(f"×{g}: MAE {v['mae']:.2f}" for g, v in res[k].items()))
    if res.get("degeneracy_test"):
        dg = res["degeneracy_test"]
        log("degeneracy (test): corr(N err, diameter err) "
            f"{dg['corr_N_error_with']['diameter']:+.2f}, rel. error N {dg['rel_mae_N']:.2f} vs "
            f"layers F {dg['rel_mae_layers_F']:.2f}")
    for k in ("properties_test", "properties_test_ood"):
        if k in res:
            log(k + ": " + ", ".join(f"{n} {v['mae']:.3g} {v['unit']} (R² {v['r2']:.2f})"
                                    for n, v in res[k].items() if v["r2"] is not None))
    return res


def evaluate_run(run: str | Path, cfg, log=print) -> dict:
    return evaluate_runs([run], cfg, run, log)


def _plot(run: Path, preds: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = [s for s in ("test", "test_ood", "real") if any(q["split"] == s for q in preds)]
    if not splits:
        return
    fig, ax = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 4), squeeze=False)
    for a, s in zip(ax[0], splits):
        q = [x for x in preds if x["split"] == s]
        t = np.array([x["N"] for x in q])
        p = np.array([x["pred"] for x in q])
        sd = np.array([x["pred_std"] for x in q])
        a.errorbar(t, p, yerr=sd, fmt="o", ms=2, alpha=0.4, elinewidth=0.5)
        a.plot([0, 100], [0, 100], "k--", lw=1)
        a.set_title(f"{s}: MAE {np.abs(p - t).mean():.1f}")
        a.set_xlabel("true N")
        a.set_ylabel("predicted N (± σ)")
    fig.tight_layout()
    fig.savefig(run / "pred_vs_true.png", dpi=110)
    plt.close(fig)
    # real recordings: mean prediction per true N (first-study style calibration curve)
    q = [x for x in preds if x["split"] == "real"]
    if q:
        t = np.array([x["N"] for x in q])
        p = np.array([x["pred"] for x in q])
        u = np.unique(t)
        fig, a = plt.subplots(figsize=(4.6, 4.2))
        a.errorbar(u, [p[t == n].mean() for n in u], yerr=[p[t == n].std() for n in u], fmt="o-", ms=3)
        a.plot([0, 100], [0, 100], "k--", lw=1)
        a.set_xlabel("true N (real BB recordings)")
        a.set_ylabel("predicted N (mean ± sd over clips)")
        a.set_title(f"real: MAE {np.abs(p - t).mean():.1f}")
        fig.tight_layout()
        fig.savefig(run / "real_by_N.png", dpi=110)
        plt.close(fig)
    # how the count error depends on the (unknown) material
    q = [x for x in preds if x["split"] in ("test", "test_ood") and "diameter" in x]
    if q:
        err = np.array([x["pred"] - x["N"] for x in q])
        cols = [("diameter", 1e3, "diameter [mm]"), ("density", 1.0, "density [kg/m³]"),
                ("e_pw", 1.0, "e (particle-wall)")]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        for a, (k, sc, lab) in zip(ax, cols):
            v = np.array([x[k] for x in q]) * sc
            a.scatter(v, err, s=5, alpha=0.4, c=[x["N"] for x in q], cmap="viridis")
            a.axhline(0, color="k", lw=0.8)
            a.set_xlabel(lab)
            a.set_ylabel("N error (pred − true)")
        fig.tight_layout()
        fig.savefig(run / "error_by_material.png", dpi=110)
        plt.close(fig)
    q = [x for x in preds if x["split"] == "test" and "props_pred" in x and x["N"] > 0]
    if q:
        keys = [("diameter", "diameter", 1e3, "mm"), ("density", "density", 1.0, "kg/m³"),
                ("e_pw", "e_pw", 1.0, "-"), ("e_pp", "e_pp", 1.0, "-")]
        fig, ax = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.8))
        for a, (pk, tk, sc, unit) in zip(ax, keys):
            t = np.array([x[tk] for x in q]) * sc
            p = np.array([x["props_pred"][pk] for x in q]) * sc
            a.scatter(t, p, s=5, alpha=0.4)
            lo, hi = t.min(), t.max()
            a.plot([lo, hi], [lo, hi], "k--", lw=1)
            a.set_title(f"{pk} [{unit}]")
            a.set_xlabel("true")
            a.set_ylabel("predicted")
        fig.tight_layout()
        fig.savefig(run / "properties_test.png", dpi=110)
        plt.close(fig)
