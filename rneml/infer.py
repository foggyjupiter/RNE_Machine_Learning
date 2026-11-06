"""Apply trained run(s) to new recordings: N ± σ and material properties ± σ.

Any wav works (resampled to 48 kHz, centre 5 s). The recording should be made like the
training data: the 5×5×7 cm box shaken at 372 rpm, phone microphone at ~2.7 m.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rneml import data as D


def predict_files(runs: list[str | Path], files: list[str], device: str = "cpu") -> list[dict]:
    from rneml.evaluate import load_model
    from rneml.features import extract
    from rneml.train import TrainConfig, predict, prepare

    outs = []
    for run in runs:
        model, norm, c, ss = load_model(run, device)
        cfg = TrainConfig(**c)
        feats = [extract(f, cfg.feature) for f in files]
        img = np.stack([a for a, _ in feats]).astype(np.float16)
        sc = np.stack([b for _, b in feats])
        img, sc = prepare(img, sc, cfg)
        outs.append(predict(model, img, sc, norm, device, crop_frac=cfg.crop_frac,
                            tta_crops=cfg.tta_crops, sigma_scale=ss))
    n = np.mean([o["n"] for o in outs], 0)
    n_sd = np.sqrt(np.mean([o["n_std"] ** 2 for o in outs], 0) + np.var([o["n"] for o in outs], 0))
    pz = np.mean([o["props"] for o in outs], 0)
    pz_sd = np.sqrt(np.mean([o["props_std"] ** 2 for o in outs], 0) + np.var([o["props"] for o in outs], 0))
    res = []
    for i, f in enumerate(files):
        r = {"file": f, "N": float(n[i]), "N_std": float(n_sd[i]), "properties": {}}
        for k, name in enumerate(D.AUX_NAMES[: pz.shape[1]]):
            col, unit, scale = D.PROPERTIES[name][:3]
            mid = float(D.to_physical(name, pz[i, k]))
            lo = float(D.to_physical(name, pz[i, k] - pz_sd[i, k]))
            hi = float(D.to_physical(name, pz[i, k] + pz_sd[i, k]))
            r["properties"][name] = {"value": mid * scale, "range_1sigma": [lo * scale, hi * scale],
                                     "unit": unit}
        res.append(r)
    return res


def format_result(r: dict) -> str:
    lines = [f"{Path(r['file']).name}: N = {r['N']:.1f} ± {r['N_std']:.1f}"]
    for name, v in r["properties"].items():
        lines.append(f"    {name:9s} {v['value']:.3g} {v['unit']}  "
                     f"(1σ: {v['range_1sigma'][0]:.3g} – {v['range_1sigma'][1]:.3g})")
    return "\n".join(lines)
