"""Non-deep baseline: ridge regression on the hand-crafted features.

Uses the scalar features only (12 physics features + 192 modulation-spectrum values),
standardised, one ridge model per target (N and each material property), α chosen on
the validation split. Written in the same metrics.json format as the CNN runs, so
``rneml report`` puts both in one table. No uncertainty: σ is the validation RMSE.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rneml import data as D


def _fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = (X - mu) / sd
    b = y.mean(0)
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y - b))
    return lambda Z: ((Z - mu) / sd) @ w + b


def run_ridge(cfg, log=print) -> dict:
    from rneml.evaluate import _by_material, degeneracy, property_metrics
    from rneml.train import cache_path, features_for, metrics, prepare, real_split

    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    rows = D.read_metadata(cfg.data_dir)
    what = Path(cfg.data_dir).name
    tr, va = D.select(rows, "train"), D.select(rows, "val")

    def feats(rs, w=what):
        img, sc = features_for(rs, cache_path(cfg, w), cfg.workers, log, cfg.feature)
        return prepare(img, sc, cfg)[1].astype(np.float64)

    Xtr, Xva = feats(tr), feats(va)
    Y = lambda rs: np.column_stack([np.array([r["N"] for r in rs], float)] +
                                   [np.array([D.aux_targets(r)[k] for r in rs]) for k in range(len(D.AUX_NAMES))])
    Ytr, Yva = Y(tr), Y(va)
    keep_tr = Ytr[:, 0] > 0
    best = {}
    for j in range(Ytr.shape[1]):
        m = np.ones(len(Ytr), bool) if j == 0 else keep_tr        # no material at N = 0
        mv = np.ones(len(Yva), bool) if j == 0 else Yva[:, 0] > 0
        cands = []
        for a in (0.1, 1, 10, 100, 1000):
            f = _fit(Xtr[m], Ytr[m, j], a)
            cands.append((float(np.abs(f(Xva[mv]) - Yva[mv, j]).mean()), a, f,
                          float(np.sqrt(np.mean((f(Xva[mv]) - Yva[mv, j]) ** 2)))))
        best[j] = min(cands, key=lambda c: c[0])

    def predict(X):
        n = best[0][2](X)
        pz = np.column_stack([best[j][2](X) for j in range(1, Ytr.shape[1])])
        return {"n": n, "n_std": np.full(len(X), best[0][3]), "props": pz,
                "props_std": np.tile([best[j][3] for j in range(1, Ytr.shape[1])], (len(X), 1))}

    eval_dir = cfg.eval_data_dir or cfg.data_dir
    erows = D.read_metadata(eval_dir)
    res = {"model": "ridge on scalar features", "feature": "scalars",
           "train_subset": cfg.train_subset, "alphas": {str(j): best[j][1] for j in best}}
    for split in ("test", "test_ood"):
        rs = D.select(erows, split)
        if not rs:
            continue
        P = predict(feats(rs, Path(eval_dir).name))
        t = np.array([r["N"] for r in rs], float)
        res[split] = metrics(P["n"], t, P["n_std"])
        res[f"properties_{split}"] = property_metrics(P["props"], P["props_std"], rs)
        res[f"degeneracy_{split}"] = degeneracy(P["n"], P["props"], rs)
        if split == "test":
            res["test_by_material"] = _by_material(rs, P["n"], t)
    _, _, real_test = real_split(cfg)
    if real_test:
        P = predict(feats(real_test, "real"))
        t = np.array([r["N"] for r in real_test], float)
        res["real"] = metrics(P["n"], t, P["n_std"])
        res["real_by_N"] = {int(n): {"mean_pred": float(P["n"][t == n].mean()),
                                     "std_of_pred": float(P["n"][t == n].std())} for n in np.unique(t)}
    (out / "config.yaml").write_text(json.dumps(asdict(cfg), indent=1))
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    for k in ("test", "test_ood", "real"):
        if k in res:
            log(f"{k:9s} N: MAE {res[k]['mae']:.2f}  ±5 {res[k]['within_5']:.2f}  (ridge)")
    return res
