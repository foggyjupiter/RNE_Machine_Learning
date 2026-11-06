"""Sim-to-real gap of the hand-crafted features, by N.

For each N of the real recordings, real clips are compared with simulated clips of
similar N (±tol) and, optionally, similar material (``material`` filter, e.g. the
first-study BB). Reported per feature: z = (real mean − sim mean) / sim std. Large |z|
points at what the simulator gets wrong (e.g. too many distinct impacts, too periodic).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rneml import data as D
from rneml.features import BASIC_SCALARS

BB_LIKE = {"particle_diameter_m": (5.0e-3, 7.0e-3), "particle_density": (0.0, 2000.0)}


def gap(cfg, n_tol: int = 8, material: dict | None = None, top: int = 5, log=print) -> dict:
    from rneml.train import feats, real_split

    rows = D.read_metadata(cfg.data_dir)
    if material:
        for k, (lo, hi) in material.items():
            rows = [r for r in rows if lo <= r[k] <= hi]
    _, _, real = real_split(cfg)
    if not real:
        real = D.real_recordings(cfg.real_dir)
    _, S = feats(cfg, rows, Path(cfg.data_dir).name, log)
    _, R = feats(cfg, real, "real", log)
    S, R = S[:, :len(BASIC_SCALARS)], R[:, :len(BASIC_SCALARS)]
    sN = np.array([r["N"] for r in rows])
    rN = np.array([r["N"] for r in real])
    out = {}
    for n in np.unique(rN):
        sel = np.abs(sN - n) <= n_tol
        if sel.sum() < 4:
            continue
        mu, sd = S[sel].mean(0), S[sel].std(0) + 1e-9
        z = (R[rN == n].mean(0) - mu) / sd
        out[int(n)] = {"n_sim": int(sel.sum()), "z": {k: float(z[i]) for i, k in enumerate(BASIC_SCALARS)}}
        worst = np.argsort(-np.abs(z))[:top]
        log(f"N={n:3d} (sim {sel.sum():3d}): " + ", ".join(f"{BASIC_SCALARS[i]} {z[i]:+.1f}" for i in worst))
    allz = np.array([[v["z"][k] for k in BASIC_SCALARS] for v in out.values()])
    if len(allz):
        out["mean_abs_z"] = {k: float(np.abs(allz[:, i]).mean()) for i, k in enumerate(BASIC_SCALARS)}
        log("mean |z| over N: " + ", ".join(f"{k} {v:.1f}" for k, v in
                                           sorted(out["mean_abs_z"].items(), key=lambda kv: -kv[1])[:6]))
    return out
