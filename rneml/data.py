"""Datasets: simulated clips from the Hugging Face repo and real first-study recordings.

Simulated data: ``FoggyJupiter/rne-granular-acoustics`` (generator:
foggyjupiter/2-2-rne-simulation). ``metadata.csv`` has one row per clip with the label
``N``, every material parameter, acoustic conditions and the ``split``:

  train / val / test   80/10/10 by dynamics run (clips of one run never cross splits)
  test_ood             held-out material region (heavy, low wall restitution)
  pairs                contrast pairs (pair_type same_N / same_material, pair_id)

Real data: a folder of ``{N}-{k}.wav`` clips (first-study layout, BB 6 mm 0.12 g, 0-100).

Besides N, the material properties in :data:`PROPERTIES` (diameter, density, mass,
restitution, friction, modulus) are training targets and model outputs.
"""

from __future__ import annotations

import csv
import glob
import math
import re
from pathlib import Path

import numpy as np

REPO_ID = "FoggyJupiter/rne-granular-acoustics"

# Material properties estimated alongside N.
# name: (metadata column, unit label, unit scale, to_normalised, from_normalised)
# Normalised targets are ≈ zero-mean, unit-spread over the sweep ranges.
PROPERTIES = {
    "diameter":    ("particle_diameter_m", "mm", 1e3,
                    lambda v: (v - 6e-3) / 1.15e-3, lambda z: 6e-3 + 1.15e-3 * z),
    "density":     ("particle_density", "kg/m3", 1.0,
                    lambda v: math.log(v / 2000.0) / 0.4, lambda z: 2000.0 * math.exp(0.4 * z)),
    "mass":        ("particle_mass_kg", "g", 1e3,
                    lambda v: math.log(v / 2.5e-4) / 0.8, lambda z: 2.5e-4 * math.exp(0.8 * z)),
    "e_pp":        ("restitution_pp", "-", 1.0,
                    lambda v: (v - 0.735) / 0.107, lambda z: 0.735 + 0.107 * z),
    "e_pw":        ("restitution_pw", "-", 1.0,
                    lambda v: (v - 0.735) / 0.107, lambda z: 0.735 + 0.107 * z),
    "mu_pp":       ("friction_pp", "-", 1.0,
                    lambda v: (v - 0.3) / 0.115, lambda z: 0.3 + 0.115 * z),
    "mu_pw":       ("friction_pw", "-", 1.0,
                    lambda v: (v - 0.3) / 0.115, lambda z: 0.3 + 0.115 * z),
    "E":           ("particle_E_acoustic_pa", "GPa", 1e-9,
                    lambda v: math.log(v / 2.7e9) / 0.35, lambda z: 2.7e9 * math.exp(0.35 * z)),
}
AUX_NAMES = list(PROPERTIES)
# first-study BB (for reporting predictions on the real recordings)
REAL_BB = {"diameter": 6e-3, "mass": 1.2e-4, "density": 1.2e-4 / (math.pi / 6 * 6e-3 ** 3)}
# not measured for the first-study BB: the values the sim-vs-real comparison fitted best
# (docs/RESEARCH_NOTES.md §4 of the simulator). Used only when an experiment GIVES these
# properties to the model and is scored on the real recordings (reported as estimates).
REAL_BB_ESTIMATED = {"e_pp": 0.95, "e_pw": 0.70}


DATASETS = ("v1", "v3", "bb_fixed", "multi_rpm", "v1_r2", "v1_sig", "v1_clk", "v3_nosig", "multi_rpm_nosig", "lowe",
            # 실제 상자(바닥 7×7 cm, 높이 5 cm)로 다시 만든 데이터 (2026-11-10 상자 치수 정정)
            "box7/v3_nosig", "box7/multi_rpm_nosig", "box7/bb_fixed", "box7/lowe")    # v1 = repo root


def fetch(local_dir: str | Path, repo_id: str = REPO_ID, audio: bool = True,
          sets: list[str] | None = None) -> Path:
    """Download (or update) the dataset; returns the local folder (v1 at its root, the
    others in sub-folders). ``sets``: only these of DATASETS (default all)."""
    from huggingface_hub import snapshot_download

    sets = list(sets or DATASETS)
    bad = [s for s in sets if s not in DATASETS]
    if bad:
        raise ValueError(f"unknown data sets {bad}; choose from {DATASETS}")
    pats = ["README.md"]
    for sub in sets:
        pre = "" if sub == "v1" else f"{sub}/"
        pats += [f"{pre}metadata.csv", f"{pre}manifest.jsonl", f"{pre}sweep_spec.yaml"]
        if audio:
            pats.append(f"{pre}audio/**")
    return Path(snapshot_download(repo_id, repo_type="dataset", local_dir=str(local_dir),
                                  allow_patterns=pats))


def read_metadata(root: str | Path) -> list[dict]:
    root = Path(root)
    rows = []
    with open(root / "metadata.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k, v in list(r.items()):
                try:
                    r[k] = float(v) if v not in ("", "True", "False") else (v == "True" if v else None)
                except ValueError:
                    pass
            r["N"] = int(r["N"])
            r["path"] = str(root / r["file_name"])
            rows.append(r)
    return [r for r in rows if Path(r["path"]).exists()]


def aux_targets(row: dict) -> np.ndarray:
    """Normalised material targets in AUX_NAMES order."""
    return np.array([PROPERTIES[n][3](row[PROPERTIES[n][0]]) for n in AUX_NAMES], np.float32)


def real_property_targets(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(normalised targets, mask) for n first-study clips: only the BB's diameter, density
    and mass are known; restitution, friction and modulus are masked out."""
    t = np.zeros((n, len(AUX_NAMES)), np.float32)
    m = np.zeros_like(t)
    for k, name in enumerate(AUX_NAMES):
        if name in REAL_BB:
            t[:, k] = PROPERTIES[name][3](REAL_BB[name])
            m[:, k] = 1.0
    return t, m


def material_inputs(rows: list[dict], names: list[str]) -> np.ndarray:
    """Normalised values of the given material properties, to be fed to the model as
    inputs (the 'material known' experiments). Real recordings: the first-study BB's
    diameter, density and mass are known; restitution gets the sim2real estimate
    (REAL_BB_ESTIMATED); friction and modulus are unknown and set to the training mean (0)."""
    out = np.zeros((len(rows), len(names)), np.float32)
    for i, r in enumerate(rows):
        for j, n in enumerate(names):
            col = PROPERTIES[n][0]
            if r.get(col) is not None:
                out[i, j] = PROPERTIES[n][3](r[col])
            elif r.get("split") == "real" and n in REAL_BB:
                out[i, j] = PROPERTIES[n][3](REAL_BB[n])
            elif r.get("split") == "real" and n in REAL_BB_ESTIMATED:
                out[i, j] = PROPERTIES[n][3](REAL_BB_ESTIMATED[n])
    return out


def to_physical(name: str, z: np.ndarray) -> np.ndarray:
    """Normalised prediction → SI value (z clipped to ±10 so log-scaled ones cannot overflow)."""
    return np.vectorize(PROPERTIES[name][4])(np.clip(z, -10.0, 10.0))


def real_recordings(folder: str | Path) -> list[dict]:
    """First-study clips named {N}-{k}.wav → rows with N and path."""
    rows = []
    for p in sorted(glob.glob(str(Path(folder) / "*.wav"))):
        m = re.match(r"(\d+)-(\d+)\.wav$", Path(p).name)
        if m:
            rows.append({"N": int(m.group(1)), "path": p, "split": "real", "run_id": Path(p).stem})
    return rows


def select(rows: list[dict], split: str | list[str], where: dict | None = None) -> list[dict]:
    """Rows of the given split(s), optionally filtered by {column: (low, high)}."""
    splits = {split} if isinstance(split, str) else set(split)
    out = [r for r in rows if r["split"] in splits]
    for k, (lo, hi) in (where or {}).items():
        out = [r for r in out if lo <= r[k] <= hi]
    return out


def bb_like(rows: list[dict]) -> list[dict]:  # small subset; prefer the bb_fixed dataset
    """'Known-material' subset resembling the first study (6 mm, light, e ≈ 0.8): the
    baseline that assumes a fixed material, to compare with blind training."""
    return [r for r in rows if abs(r["particle_diameter_m"] - 6e-3) <= 0.6e-3
            and r["particle_density"] <= 1600 and abs(r["restitution_pw"] - 0.8) <= 0.1
            and abs(r["restitution_pp"] - 0.8) <= 0.1]
