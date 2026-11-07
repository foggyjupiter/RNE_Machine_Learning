"""Ensemble of finished runs by averaging their saved predictions (no data or model needed).

    python scripts/ensemble_predictions.py NAME RUN [RUN ...] [--out outputs/ensembles.json]

Runs must share the split (same data and split seed; seeds of the model may differ). The
ensemble mean is the average of the per-run means; its uncertainty combines the average
predicted variance with the spread between runs. Prints MAE / bias / within ±5 for test,
test_ood and real, and real bias per N range.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BINS = [(0, 0), (5, 25), (30, 50), (55, 75), (80, 100)]


def load(run):
    out = {}
    for line in open(Path(run) / "predictions.jsonl"):
        r = json.loads(line)
        out[(r["split"], r["run_id"], r.get("render"))] = r
    return out


def ensemble(runs):
    preds = [load(r) for r in runs]
    keys = set(preds[0]).intersection(*preds[1:])
    rows = defaultdict(list)
    for k in sorted(keys, key=str):
        mu = np.array([p[k]["pred"] for p in preds])
        sd = np.array([p[k].get("pred_std", 0.0) for p in preds])
        rows[k[0]].append((preds[0][k]["N"], mu.mean(), float(np.sqrt((sd ** 2).mean() + mu.var()))))
    res = {}
    for split, rs in rows.items():
        N = np.array([r[0] for r in rs], float)
        P = np.array([r[1] for r in rs])
        if split == "real":
            P = np.clip(P, 0, None)
        S = np.array([r[2] for r in rs])
        e = P - N
        res[split] = {"n": len(rs), "mae": float(np.abs(e).mean()), "bias": float(e.mean()),
                      "within_5": float((np.abs(e) <= 5).mean()), "in_2sigma": float((np.abs(e) <= 2 * S).mean())}
        if split == "real":
            res[split]["bias_by_N"] = {f"{a}-{b}": float(e[(N >= a) & (N <= b)].mean()) for a, b in BINS}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="outputs/ensembles.json")
    a = ap.parse_args()
    res = ensemble(a.runs)
    res["members"] = a.runs
    p = Path(a.out)
    allres = json.loads(p.read_text()) if p.exists() else {}
    allres[a.name] = res
    p.write_text(json.dumps(allres, indent=1))
    f = lambda s: f"{res[s]['mae']:.1f}" if s in res else "–"
    print(f"{a.name}: test {f('test')} | ood {f('test_ood')} | real {f('real')} "
          f"(bias {res['real']['bias']:+.1f}, ±5 {res['real']['within_5']:.0%}, 2σ {res['real']['in_2sigma']:.0%}) "
          f"| real bias by N {', '.join(f'{k} {v:+.1f}' for k, v in res['real']['bias_by_N'].items())}")
