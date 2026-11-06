"""Data splits — always by dynamics run (``run_id``; ``system_id`` in multi-condition
data), never by clip.

The four clips of one dynamics run share the trajectory; if they were split across
train and test, a model could recognise the trajectory instead of counting. Every
split here assigns whole runs, and :func:`check_no_leak` verifies it.

Split kinds (``split`` in the training config):

  default              the dataset's own ``split`` column (train/val/test/test_ood/pairs)
  material_extrap      train on ``axis`` within [lo, hi], test outside
  material_band        hold out ``axis`` within [lo, hi] (a hole in the middle), test there
  count_extrap         train on N ≤ ``n_train_max``, test on N ≥ ``n_test_min``
  count_sparse         train only on N within ±``halfwidth`` of multiples of ``step``,
                       test on the N in between (continuity of the count estimate)

For all but ``default``, validation = a random 10 % of the training runs (seeded), and
contrast pairs are kept only if both members are in the training region.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROLES = ("train", "val", "test", "test_ood", "pairs")


def _run_of(r: dict) -> str:
    """The leakage unit: the dynamics run, or the whole system in multi-condition data
    (all conditions of one system share N, material and recording setup)."""
    return str(r.get("system_id") or r["run_id"])


def assign(rows: list[dict], spec: dict | None, seed: int = 0) -> list[dict]:
    """Return rows with a 'role' key. ``spec`` = {"kind": ..., params}; None = default."""
    spec = spec or {"kind": "default"}
    kind = spec.get("kind", "default")
    out = []
    if kind == "default":
        for r in rows:
            out.append({**r, "role": r["split"]})
        return out

    def in_train(r) -> bool | None:
        """True = training region, False = test region, None = unused (gap)."""
        if kind == "material_extrap":
            v = r[spec["axis"]]
            return spec["lo"] <= v <= spec["hi"]
        if kind == "material_band":
            v = r[spec["axis"]]
            return not (spec["lo"] <= v <= spec["hi"])
        if kind == "count_extrap":
            if r["N"] <= spec.get("n_train_max", 70):
                return True
            return False if r["N"] >= spec.get("n_test_min", 80) else None
        if kind == "count_sparse":
            step, hw = spec.get("step", 10), spec.get("halfwidth", 1)
            d = abs(r["N"] - step * round(r["N"] / step))
            if d <= hw:
                return True
            return False if d >= hw + 2 else None
        raise ValueError(f"unknown split kind {kind!r}")

    rng = np.random.default_rng(seed)
    # validation = 10 % of the runs in the *training region* (at least one)
    runs = sorted({_run_of(r) for r in rows if r["split"] != "pairs" and in_train(r) is True})
    val_runs = set(rng.choice(runs, max(1, len(runs) // 10), replace=False)) if runs else set()
    pair_ok: dict = {}
    for r in rows:
        if r["split"] == "pairs":
            pair_ok.setdefault(r["pair_id"], []).append(in_train(r) is True)
    for r in rows:
        t = in_train(r)
        if r["split"] == "pairs":
            if all(pair_ok[r["pair_id"]]):
                out.append({**r, "role": "pairs"})
            continue
        if t is None:
            continue
        role = ("val" if _run_of(r) in val_runs else "train") if t else "test"
        out.append({**r, "role": role})
    return out


def check_no_leak(rows: list[dict]) -> None:
    """Raise if any dynamics run appears in more than one role."""
    seen: dict = {}
    for r in rows:
        seen.setdefault(_run_of(r), set()).add(r["role"])
    bad = {k: v for k, v in seen.items() if len(v) > 1}
    if bad:
        raise AssertionError(f"{len(bad)} runs appear in several roles, e.g. {next(iter(bad.items()))}")


def save(rows: list[dict], spec: dict | None, seed: int, path: str | Path) -> str:
    """Write {run_id: role} plus the spec and seed; returns the split hash."""
    m = {}
    for r in rows:
        m[_run_of(r)] = r["role"]
    blob = json.dumps({"spec": spec, "seed": seed, "roles": dict(sorted(m.items()))}, sort_keys=True)
    h = hashlib.sha1(blob.encode()).hexdigest()[:12]
    Path(path).write_text(json.dumps({"hash": h, **json.loads(blob)}, indent=1))
    return h


def select_role(rows: list[dict], role: str) -> list[dict]:
    return [r for r in rows if r["role"] == role]
