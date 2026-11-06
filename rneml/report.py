"""Comparison table of several evaluated runs (markdown), from their metrics.json."""

from __future__ import annotations

import json
from pathlib import Path


def _f(v, fmt="{:.2f}"):
    return "–" if v is None else fmt.format(v)


def compare(runs: list[str | Path]) -> str:
    rows = []
    for r in runs:
        p = Path(r) / "metrics.json"
        if p.exists():
            rows.append((Path(r).name, json.loads(p.read_text())))
    head = ("| run | feature | train | real adapt | test MAE | test ±5 | OOD MAE | OOD ±5 | "
            "real MAE | real ±5 | 1σ cover | pair gap | diameter R² | density R² | e_pw R² |")
    lines = [head, "|" + "|".join(["---"] * (head.count("|") - 1)) + "|"]
    for name, m in rows:
        g = lambda s, k: m.get(s, {}).get(k)
        pr = m.get("properties_test", {})
        adapt = []
        if m.get("real_train_per_N"):
            adapt.append(f"{m['real_train_per_N']} lab")
        if m.get("real_unlabeled_per_N"):
            adapt.append(f"{m['real_unlabeled_per_N']} unl")
        lines.append("| " + " | ".join([
            name, str(m.get("feature", "")), str(m.get("train_subset", "")), "+".join(adapt) or "–",
            _f(g("test", "mae")), _f(g("test", "within_5")), _f(g("test_ood", "mae")),
            _f(g("test_ood", "within_5")), _f(g("real", "mae")), _f(g("real", "within_5")),
            _f(g("test", "in_1sigma")), _f(m.get("same_N_pair_gap", {}).get("mean_abs_diff")),
            _f(pr.get("diameter", {}).get("r2")), _f(pr.get("density", {}).get("r2")),
            _f(pr.get("e_pw", {}).get("r2"))]) + " |")
    return "\n".join(lines)
