"""Evaluation of a trained run.

Material properties (diameter, density, mass, restitution, friction, modulus): MAE in
physical units and R² on test / test_ood; on the real recordings the predicted
properties are reported next to the known BB values (6 mm, 0.12 g).

Reports, for the count N:
  * test (same material distribution as training, unseen runs)
  * test_ood (held-out material region: never seen in training)
  * by material bin (mass and wall restitution tertiles) — how flat is the error?
  * same_N pairs: |N̂_a − N̂_b| for identical N and different material (invariance)
  * real first-study recordings (BB, 0-100) — sim-to-real transfer
and the auxiliary material predictions (correlation with the truth).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rneml import data as D
from rneml.model import CountNet


def load_model(run: str | Path, device="cpu"):
    from rneml.train import Norm

    ck = torch.load(Path(run) / "model.pt", map_location=device, weights_only=False)
    c = ck["cfg"]
    m = CountNet(ck["n_scalars"], ck["n_aux"], c["width"], c["dropout"], c["use_scalars"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), Norm.from_state(ck["norm"]), c


def _aux_pred(model, mel, sc, norm, device, batch=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(mel), batch):
            m, s = norm(mel[i:i + batch], sc[i:i + batch])
            _, a, _ = model(torch.from_numpy(m).to(device), torch.from_numpy(s).to(device))
            out.append(a.cpu().numpy())
    return np.concatenate(out)


def evaluate_run(run: str | Path, cfg, log=print) -> dict:
    from rneml.train import _device, features_for, metrics, predict

    run = Path(run)
    dev = _device(cfg.device)
    model, norm, _ = load_model(run, dev)
    rows = D.read_metadata(cfg.data_dir)
    cache = Path(cfg.cache_dir) / (Path(cfg.data_dir).name + "_features.npz")
    res: dict = {}
    preds = []
    for split in ("test", "test_ood", "pairs"):
        rs = D.select(rows, split)
        if not rs:
            continue
        mel, sc = features_for(rs, cache, cfg.workers, log)
        p = predict(model, mel, sc, norm, dev)
        t = np.array([r["N"] for r in rs], float)
        res[split] = metrics(p, t)
        for r, pi in zip(rs, p):
            preds.append({"split": split, "run_id": r["run_id"], "render": r.get("render"),
                          "N": r["N"], "pred": float(pi), "mass": r["particle_mass_kg"],
                          "diameter": r["particle_diameter_m"], "density": r["particle_density"],
                          "e_pw": r["restitution_pw"], "e_pp": r["restitution_pp"],
                          "pair_id": r.get("pair_id", ""), "pair_type": r.get("pair_type", "")})
        if split in ("test", "test_ood") and model.aux is not None:
            a = _aux_pred(model, mel, sc, norm, dev)
            res[f"properties_{split}"] = property_metrics(a, rs)
            if split == "test":
                res["test_by_material"] = _by_material(rs, p, t)
        if split == "pairs":
            same_n = {}
            for r, pi in zip(rs, p):
                if r.get("pair_type") == "same_N":
                    same_n.setdefault((r["pair_id"], r["render"]), []).append(pi)
            d = [abs(v[0] - v[1]) for v in same_n.values() if len(v) == 2]
            res["same_N_pair_gap"] = {"n": len(d), "mean_abs_diff": float(np.mean(d)) if d else None}
    if cfg.real_dir:
        rr = D.real_recordings(cfg.real_dir)
        if rr:
            mel, sc = features_for(rr, Path(cfg.cache_dir) / "real_features.npz", cfg.workers, log)
            p = predict(model, mel, sc, norm, dev)
            t = np.array([r["N"] for r in rr], float)
            res["real"] = metrics(p, t)
            res["real_by_N"] = {int(n): float(p[t == n].mean()) for n in np.unique(t)}
            if model.aux is not None:
                a = _aux_pred(model, mel, sc, norm, dev)
                m = t > 0  # an empty box has no material
                res["real_properties"] = {
                    n: {"pred_median": float(np.median(D.to_physical(n, a[m, k])) * D.PROPERTIES[n][2]),
                        "pred_iqr": [float(np.percentile(D.to_physical(n, a[m, k]), q) * D.PROPERTIES[n][2])
                                     for q in (25, 75)],
                        "unit": D.PROPERTIES[n][1],
                        "true": (D.REAL_BB[n] * D.PROPERTIES[n][2]) if n in D.REAL_BB else None}
                    for k, n in enumerate(D.AUX_NAMES)}
            for r, pi in zip(rr, p):
                preds.append({"split": "real", "run_id": r["run_id"], "N": r["N"], "pred": float(pi)})
    (run / "metrics.json").write_text(json.dumps(res, indent=1))
    with open(run / "predictions.jsonl", "w") as fh:
        for q in preds:
            fh.write(json.dumps(q) + "\n")
    _plot(run, preds)
    for k in ("test", "test_ood", "real"):
        if k in res:
            log(f"{k:9s} N: MAE {res[k]['mae']:.2f}  ±5 {res[k]['within_5']:.2f}  ±10 {res[k]['within_10']:.2f}"
                f"  bias {res[k]['bias']:+.2f}  (n={res[k]['n']})")
    for k in ("properties_test", "properties_test_ood"):
        if k in res:
            log(k + ": " + ", ".join(f"{n} MAE {v['mae']:.3g} {v['unit']} (R² {v['r2']:.2f})"
                                    for n, v in res[k].items() if v["r2"] is not None))
    return res


def property_metrics(a: np.ndarray, rs: list[dict]) -> dict:
    """Per-property error in physical units and R² (rows with N = 0 are skipped)."""
    keep = np.array([r["N"] > 0 for r in rs])
    out = {}
    for k, n in enumerate(D.AUX_NAMES):
        col, unit, scale = D.PROPERTIES[n][:3]
        truth = np.array([r[col] for r in rs])[keep]
        pred = D.to_physical(n, a[keep, k])
        err = (pred - truth) * scale
        ss = np.sum((truth - truth.mean()) ** 2)
        out[n] = {"unit": unit, "mae": float(np.abs(err).mean()),
                  "rel_mae": float(np.mean(np.abs(pred - truth) / np.abs(truth))),
                  "r2": float(1 - np.sum((pred - truth) ** 2) / ss) if ss > 0 else None}
    return out


def _by_material(rs, p, t) -> dict:
    out = {}
    for col in ("particle_diameter_m", "particle_density", "restitution_pw"):
        v = np.array([r[col] for r in rs])
        q = np.quantile(v, [1 / 3, 2 / 3])
        b = np.digitize(v, q)
        out[col] = [{"bin": int(k), "range": [float(v[b == k].min()), float(v[b == k].max())],
                     "mae": float(np.abs(p[b == k] - t[b == k]).mean())} for k in range(3) if (b == k).any()]
    return out


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
        a.scatter(t, p, s=6, alpha=0.5)
        a.plot([0, 100], [0, 100], "k--", lw=1)
        a.set_title(f"{s}: MAE {np.abs(p - t).mean():.1f}")
        a.set_xlabel("true N")
        a.set_ylabel("predicted N")
    fig.tight_layout()
    fig.savefig(run / "pred_vs_true.png", dpi=110)
    plt.close(fig)
