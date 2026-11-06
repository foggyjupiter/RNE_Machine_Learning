"""Non-deep baselines on the hand-crafted scalar features (same splits, same metrics).

| name            | inputs                                          | question it answers |
|---|---|---|
| ``rms``         | clip RMS level only (cubic)                     | how far does loudness alone go? (and how badly does it break when the material changes) |
| ``onset``       | detected impact-onset rate only (cubic)         | the most intuitive physical method: count the clicks |
| ``order_params``| ψ3 kurtosis, ψ2 entropy, period-folded peakiness | first-study order parameters as scalars (vs the learned field) |
| ``ridge_all``   | all 204 scalars (physics + modulation), ridge   | linear use of everything hand-crafted |
| ``gbm_physics`` | 12 physics scalars, gradient boosting           | non-linear use of the first-study-style features |
| ``gbm_all``     | all 204 scalars, gradient boosting              | strongest hand-crafted baseline |

Each baseline predicts N and every material property (N = 0 rows carry no material).
Hyper-parameters are chosen on the validation split only. σ = validation RMSE (no
per-clip uncertainty); conformal intervals use validation residual quantiles.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rneml import data as D
from rneml.features import BASIC_SCALARS

IDX = {k: i for i, k in enumerate(BASIC_SCALARS)}
BASELINES = ("rms", "onset", "order_params", "ridge_all", "gbm_physics", "gbm_all")


def _design(kind: str, sc: np.ndarray) -> np.ndarray:
    if kind == "rms":
        x = sc[:, [IDX["rms_db"]]]
        return np.hstack([x, x ** 2, x ** 3])
    if kind == "onset":
        x = np.log1p(np.maximum(sc[:, [IDX["onset_rate"]]], 0))
        return np.hstack([x, x ** 2, x ** 3])
    if kind == "order_params":
        x = sc[:, [IDX["kurtosis"], IDX["entropy"], IDX["fold_peak"]]].copy()
        x[:, 0] = np.sign(x[:, 0]) * np.log1p(np.abs(x[:, 0]))
        sq = np.stack([x[:, i] * x[:, j] for i in range(3) for j in range(i, 3)], 1)
        return np.hstack([x, sq])
    if kind == "gbm_physics":
        return sc[:, :len(BASIC_SCALARS)]
    return sc                                      # ridge_all, gbm_all


def _model(kind: str, hp):
    if kind.startswith("gbm"):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=hp, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=0)
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=hp))


def run_baseline(cfg, kind: str, log=print) -> dict:
    from rneml import splits as SP
    from rneml.evaluate import _by_material, degeneracy, property_metrics
    from rneml.train import cache_path, conformal_metrics, features_for, metrics, real_split

    if kind not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}")
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    rows = SP.assign(D.read_metadata(cfg.data_dir), cfg.split, cfg.split_seed)
    SP.check_no_leak(rows)
    SP.save(rows, cfg.split, cfg.split_seed, out / "split.json")
    tr, va = SP.select_role(rows, "train"), SP.select_role(rows, "val")
    if cfg.train_subset == "bb_like":
        tr = D.bb_like(tr)

    def X(rs, what):
        _, sc = features_for(rs, cache_path(cfg, what), cfg.workers, log, cfg.feature)
        return _design(kind, sc.astype(np.float64))

    what = Path(cfg.data_dir).name
    Xtr, Xva = X(tr, what), X(va, what)
    Y = lambda rs: np.column_stack([np.array([r["N"] for r in rs], float)] +
                                   [np.array([D.aux_targets(r)[k] for r in rs]) for k in range(len(D.AUX_NAMES))])
    Ytr, Yva = Y(tr), Y(va)
    grid = (100, 300, 600) if kind.startswith("gbm") else (0.1, 1, 10, 100, 1000)
    fitted = {}
    for j in range(Ytr.shape[1]):
        m = np.ones(len(Ytr), bool) if j == 0 else Ytr[:, 0] > 0
        mv = np.ones(len(Yva), bool) if j == 0 else Yva[:, 0] > 0
        if j > 0 and np.ptp(Ytr[m, j]) == 0:       # constant material (fixed-material data)
            c = float(Ytr[m, j][0])
            fitted[j] = (None, c, 1.0)
            continue
        best = None
        for hp in grid:
            f = _model(kind, hp).fit(Xtr[m], Ytr[m, j])
            e = f.predict(Xva[mv]) - Yva[mv, j]
            if best is None or np.abs(e).mean() < best[0]:
                best = (float(np.abs(e).mean()), f, float(np.sqrt(np.mean(e ** 2))), e)
        fitted[j] = (best[1], None, best[2])
        if j == 0:
            r = np.sort(np.abs(best[3]) / best[2])
            conf = {str(a): float(r[min(len(r) - 1, int(np.ceil((len(r) + 1) * (1 - a))) - 1)])
                    for a in cfg.conformal_alpha}

    def predict(Xe):
        cols = []
        for j in range(Ytr.shape[1]):
            f, c, _ = fitted[j]
            cols.append(np.full(len(Xe), c) if f is None else f.predict(Xe))
        P = np.column_stack(cols)
        return {"n": P[:, 0], "n_std": np.full(len(Xe), fitted[0][2]), "props": P[:, 1:],
                "props_std": np.tile([fitted[j][2] for j in range(1, Ytr.shape[1])], (len(Xe), 1))}

    eval_dir = cfg.eval_data_dir or cfg.data_dir
    erows = SP.assign(D.read_metadata(eval_dir), cfg.split if not cfg.eval_data_dir else None,
                      cfg.split_seed)
    res = {"model": f"baseline:{kind}", "feature": "scalars", "train_subset": cfg.train_subset,
           "split": cfg.split, "hyper": {str(j): (None if fitted[j][0] is None else str(fitted[j][0])[:60])
                                         for j in fitted}}
    preds = []
    for split in ("test", "test_ood"):
        rs = SP.select_role(erows, split)
        if not rs:
            continue
        P = predict(X(rs, Path(eval_dir).name))
        t = np.array([r["N"] for r in rs], float)
        res[split] = metrics(P["n"], t, P["n_std"])
        res[split]["conformal"] = conformal_metrics(P["n"], t, P["n_std"], conf)
        res[f"properties_{split}"] = property_metrics(P["props"], P["props_std"], rs)
        res[f"degeneracy_{split}"] = degeneracy(P["n"], P["props"], rs)
        if split == "test":
            res["test_by_material"] = _by_material(rs, P["n"], t)
        preds += [{"split": split, "run_id": r["run_id"], "render": r.get("render"), "N": r["N"],
                   "pred": float(P["n"][k]), "pred_std": float(P["n_std"][k])} for k, r in enumerate(rs)]
    _, _, real_test = real_split(cfg)
    if real_test:
        P = predict(X(real_test, "real"))
        t = np.array([r["N"] for r in real_test], float)
        res["real"] = metrics(P["n"], t, P["n_std"])
        res["real"]["conformal"] = conformal_metrics(P["n"], t, P["n_std"], conf)
        res["real_by_N"] = {int(n): {"mean_pred": float(P["n"][t == n].mean()),
                                     "std_of_pred": float(P["n"][t == n].std())} for n in np.unique(t)}
        preds += [{"split": "real", "run_id": r["run_id"], "N": r["N"], "pred": float(P["n"][k]),
                   "pred_std": float(P["n_std"][k])} for k, r in enumerate(real_test)]
    (out / "config.yaml").write_text(json.dumps(asdict(cfg), indent=1, default=str))
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    with open(out / "predictions.jsonl", "w") as fh:
        for q in preds:
            fh.write(json.dumps(q) + "\n")
    for k in ("test", "test_ood", "real"):
        if k in res:
            log(f"[{kind}] {k:9s} N: MAE {res[k]['mae']:.2f}  ±5 {res[k]['within_5']:.2f}")
    return res


def run_ridge(cfg, log=print) -> dict:   # kept for the earlier CLI name
    return run_baseline(cfg, "ridge_all", log)
