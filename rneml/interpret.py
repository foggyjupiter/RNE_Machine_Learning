"""What did the model learn? (clip model; uses the embedding before the heads)

* ``embedding_map``  PCA of the test embeddings, coloured by N and by diameter / density /
  wall restitution side by side. Ordered along N and mixed along the material = a
  material-invariant count representation.
* ``probe``  linear (ridge, CV grouped by run) read-out from the frozen embedding of
  N, every material property, the first-study order parameters (ψ3 kurtosis, ψ2
  entropy, period-folded peakiness ~ ψ1) and the simulator's ground-truth phase
  indicators (collectivity, phase concentration, free flight, sustained contact,
  bed height). High R² for the ground-truth phase means the network found, on its own,
  the observables a physicist would design.
* ``band_masking``  replace one group of mel bands by its mean and measure how much
  the N error and each property error grow — do count and material live in different
  frequency bands?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rneml import data as D
from rneml.features import BASIC_SCALARS

GT_KEYS = ["gt_collectivity", "gt_phase_concentration", "gt_phase_concentration_impulse",
           "gt_free_flight", "gt_sustained_fraction", "gt_height_mean", "gt_height_std",
           "gt_impact_rate_per_particle"]
ORDER_KEYS = {"psi3_kurtosis": "kurtosis", "psi2_entropy": "entropy", "psi1_fold_peak": "fold_peak"}


def _load(run, device="cpu"):
    from rneml.evaluate import load_model
    from rneml.train import TrainConfig

    model, norm, c, ss = load_model(run, device)
    return model, norm, TrainConfig(**c), ss


def _embed(model, img, sc, norm, device, batch=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(img), batch):
            m, s = norm(img[i:i + batch], sc[i:i + batch])
            out.append(model(torch.from_numpy(m).to(device), torch.from_numpy(s).to(device)).emb.cpu().numpy())
    return np.concatenate(out)


def _ridge_cv(X, y, groups, alphas=(0.1, 1, 10, 100), k=5, seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ug = np.unique(groups)
    fold = dict(zip(np.random.default_rng(seed).permutation(ug), np.arange(len(ug)) % k))
    f = np.array([fold[g] for g in groups])
    best = -np.inf
    for a in alphas:
        p = np.zeros_like(y)
        for j in range(k):
            m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X[f != j], y[f != j])
            p[f == j] = m.predict(X[f == j])
        r2 = 1 - np.mean((p - y) ** 2) / np.var(y)
        best = max(best, r2)
    return float(best)


def interpret(run: str | Path, data_dir: str | None = None, split: str = "test", log=print) -> dict:
    from rneml import splits as SP
    from rneml.train import feats, prepare

    run = Path(run)
    dev = "cpu"
    model, norm, cfg, _ = _load(run, dev)
    ddir = data_dir or cfg.eval_data_dir or cfg.data_dir
    rows = SP.assign(D.read_metadata(ddir), cfg.split if not cfg.eval_data_dir else None, cfg.split_seed)
    rs = [r for r in SP.select_role(rows, split)]
    if split == "test":
        rs += SP.select_role(rows, "test_ood")
    rs = [r for r in rs if r["N"] > 0]
    img_raw, sc_raw = feats(cfg, rs, Path(ddir).name, log)
    img, sc = prepare(img_raw, sc_raw, cfg)
    sc = sc[:, :len(norm.s_mu)]
    Z = _embed(model, img, sc, norm, dev)
    groups = np.array([str(r.get("system_id") or r["run_id"]) for r in rs])
    res: dict = {"n_clips": len(rs), "probe_r2": {}}

    # linear probes
    targets = {"N": np.array([r["N"] for r in rs], float)}
    for name in D.AUX_NAMES:
        targets[name] = np.array([D.aux_targets(r)[D.AUX_NAMES.index(name)] for r in rs])
    for k, col in ORDER_KEYS.items():
        targets[k] = sc_raw[:, BASIC_SCALARS.index(col)].astype(float)
    for k in GT_KEYS:
        if k in rs[0] and rs[0][k] is not None:
            targets[k] = np.array([float(r[k]) for r in rs])
    for k, y in targets.items():
        if np.isfinite(y).all() and np.var(y) > 0:
            res["probe_r2"][k] = _ridge_cv(Z, y, groups)
    log("linear probe R² from the embedding:")
    for k, v in res["probe_r2"].items():
        log(f"  {k:32s} {v:5.2f}")

    # band masking (mel/scalogram rows grouped into 8 bands)
    from rneml.train import predict
    base = predict(model, img, sc, norm, dev)
    tN = targets["N"]
    tz = np.stack([D.aux_targets(r) for r in rs])
    def errs(P):
        e = {"N": float(np.abs(P["n"] - tN).mean())}
        if P["props"].shape[1]:
            for j, n in enumerate(D.AUX_NAMES):
                e[n] = float(np.abs(P["props"][:, j] - tz[:, j]).mean())
        return e
    e0 = errs(base)
    F = img.shape[-2]
    edges = np.linspace(0, F, 9).astype(int)
    res["band_masking"] = {"bands": [[int(a), int(b)] for a, b in zip(edges[:-1], edges[1:])],
                           "baseline_error": e0, "error_increase": []}
    for a, b in zip(edges[:-1], edges[1:]):
        im = img.astype(np.float32).copy()
        im[..., a:b, :] = im[..., a:b, :].mean(axis=(0, -1), keepdims=True)
        e = errs(predict(model, im.astype(np.float16), sc, norm, dev))
        res["band_masking"]["error_increase"].append({k: e[k] - e0[k] for k in e})
    _plots(run, Z, rs, res)
    (run / "interpret.json").write_text(json.dumps(res, indent=1))
    log(f"wrote {run / 'interpret.json'}, embedding_map.png, band_masking.png")
    return res


def _plots(run: Path, Z, rs, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Zc @ Vt[:2].T
    cols = [("N", np.array([r["N"] for r in rs]), "N"),
            ("diameter", np.array([r["particle_diameter_m"] for r in rs]) * 1e3, "diameter [mm]"),
            ("density", np.array([r["particle_density"] for r in rs]), "density [kg/m³]"),
            ("e_pw", np.array([r["restitution_pw"] for r in rs]), "e (particle-wall)")]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for a, (_, v, lab) in zip(ax, cols):
        s = a.scatter(P[:, 0], P[:, 1], c=v, s=6, cmap="viridis")
        fig.colorbar(s, ax=a, label=lab)
        a.set_xticks([])
        a.set_yticks([])
    ax[0].set_title("embedding PCA, coloured by N")
    fig.tight_layout()
    fig.savefig(run / "embedding_map.png", dpi=110)
    plt.close(fig)
    bm = res["band_masking"]
    keys = ["N"] + [k for k in ("diameter", "density", "e_pw", "e_pp") if k in bm["baseline_error"]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(bm["bands"]))
    for i, k in enumerate(keys):
        inc = np.array([d[k] for d in bm["error_increase"]]) / max(bm["baseline_error"][k], 1e-9)
        ax.bar(x + i * 0.15, inc, 0.15, label=k)
    ax.set_xticks(x + 0.3)
    ax.set_xticklabels([f"{a}-{b}" for a, b in bm["bands"]], rotation=30)
    ax.set_xlabel("masked band (image rows, low → high frequency)")
    ax.set_ylabel("relative error increase")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "band_masking.png", dpi=110)
    plt.close(fig)
