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


def fetch(local_dir: str | Path, repo_id: str = REPO_ID, audio: bool = True) -> Path:
    """Download (or update) the dataset; returns the local folder."""
    from huggingface_hub import snapshot_download

    pats = ["metadata.csv", "manifest.jsonl", "sweep_spec.yaml", "README.md"]
    if audio:
        pats.append("audio/**")
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


def to_physical(name: str, z: np.ndarray) -> np.ndarray:
    """Normalised prediction → SI value."""
    return np.vectorize(PROPERTIES[name][4])(z)


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


def bb_like(rows: list[dict]) -> list[dict]:
    """'Known-material' subset resembling the first study (6 mm, light, e ≈ 0.8): the
    baseline that assumes a fixed material, to compare with blind training."""
    return [r for r in rows if abs(r["particle_diameter_m"] - 6e-3) <= 0.6e-3
            and r["particle_density"] <= 1600 and abs(r["restitution_pw"] - 0.8) <= 0.1
            and abs(r["restitution_pp"] - 0.8) <= 0.1]
