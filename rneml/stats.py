"""Repeated runs and paired comparisons.

* :func:`aggregate` — mean ± std over seeds of the key metrics of several runs.
* :func:`paired_compare` — two runs evaluated on the same clips: Wilcoxon signed-rank
  test on per-clip absolute errors and a bootstrap CI (resampling dynamics runs, so
  the 4 correlated clips of a run stay together) of the MAE difference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KEYS = [("test", "mae"), ("test", "within_5"), ("test_ood", "mae"), ("test_ood", "within_5"),
        ("real", "mae"), ("real", "within_5"), ("test", "in_1sigma")]
PROP_KEYS = ["diameter", "density", "e_pw", "e_pp"]


def aggregate(runs: list[str | Path]) -> dict:
    ms = [json.loads((Path(r) / "metrics.json").read_text()) for r in runs
          if (Path(r) / "metrics.json").exists()]
    out = {"n_runs": len(ms), "runs": [str(r) for r in runs]}
    for s, k in KEYS:
        v = [m[s][k] for m in ms if s in m and k in m[s]]
        if v:
            out[f"{s}.{k}"] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    for p in PROP_KEYS:
        v = [m["properties_test"][p]["r2"] for m in ms if "properties_test" in m]
        if v:
            out[f"r2.{p}"] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    return out


def _errors(run: str | Path, split: str) -> dict:
    out = {}
    for line in open(Path(run) / "predictions.jsonl"):
        q = json.loads(line)
        if q["split"] == split:
            out[(q["run_id"], q.get("render"))] = abs(q["pred"] - q["N"])
    return out


def paired_compare(run_a: str | Path, run_b: str | Path, split: str = "test",
                   n_boot: int = 2000, seed: int = 0) -> dict:
    from scipy.stats import wilcoxon

    ea, eb = _errors(run_a, split), _errors(run_b, split)
    keys = sorted(set(ea) & set(eb))
    if len(keys) < 5:
        return {"n": len(keys), "note": "too few common clips"}
    a = np.array([ea[k] for k in keys])
    b = np.array([eb[k] for k in keys])
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k[0], []).append(i)
    g = list(groups.values())
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))])
        diffs.append(a[idx].mean() - b[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
    return {"split": split, "n_clips": len(keys), "n_runs": len(g), "mae_a": float(a.mean()),
            "mae_b": float(b.mean()), "mae_diff_a_minus_b": float(a.mean() - b.mean()),
            "ci95_run_bootstrap": [float(lo), float(hi)], "wilcoxon_p": p}
