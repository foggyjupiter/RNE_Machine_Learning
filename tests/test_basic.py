import math

import numpy as np
import torch

from rneml import data as D
from rneml.features import (BASIC_SCALARS, HOP, MOD_EDGES, N_MELS, SCALAR_NAMES, SCALO_BANDS, SR,
                            _morlet_bank, log_mel, modulation_features, scalar_features, scalogram)
from rneml.model import CountNet


def test_property_transforms_roundtrip():
    row = {"particle_diameter_m": 7e-3, "particle_density": 3100.0, "particle_mass_kg": 3e-4,
           "restitution_pp": 0.6, "restitution_pw": 0.9, "friction_pp": 0.12,
           "friction_pw": 0.45, "particle_E_acoustic_pa": 4.2e9}
    z = D.aux_targets(row)
    for k, n in enumerate(D.AUX_NAMES):
        assert math.isclose(float(D.to_physical(n, z[k])), row[D.PROPERTIES[n][0]], rel_tol=1e-5)


def test_features_shapes_and_level():
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal(SR * 5)).astype(np.float32)
    S = log_mel(x)
    assert S.shape[0] == N_MELS and abs(S.shape[1] - SR * 5 // HOP) <= 2
    s = scalar_features(x, S)
    assert s.shape == (len(BASIC_SCALARS),) and np.isfinite(s).all()
    assert len(SCALAR_NAMES) == len(BASIC_SCALARS) + modulation_features(S).size
    # fixed dB reference: 2x amplitude → +6 dB
    assert np.mean(log_mel(2 * x) - S) == np.float32(np.mean(log_mel(2 * x) - S))
    assert abs(np.mean(log_mel(2 * x) - S) - 6.02) < 0.05


def test_model_forward():
    m = CountNet(len(SCALAR_NAMES), len(D.AUX_NAMES))
    for F in (N_MELS, SCALO_BANDS):
        o = m(torch.randn(3, F, 500), torch.randn(3, len(SCALAR_NAMES)))
        assert o.n.shape == o.n_logvar.shape == (3,)
        assert o.props.shape == o.props_logvar.shape == (3, len(D.AUX_NAMES))
        assert o.emb.shape == (3, 128)


def test_scalogram_tone_peaks_in_right_band():
    t = np.arange(SR * 5) / SR
    x = (0.1 * np.sin(2 * np.pi * 9400 * t)).astype(np.float32)
    S = scalogram(x)
    fc, _ = _morlet_bank(len(x))
    assert S.shape == (SCALO_BANDS, SR * 5 // HOP)
    assert abs(fc[np.argmax(S.mean(1))] - 9400) / 9400 < 0.06


def test_real_property_mask():
    t, m = D.real_property_targets(4)
    known = [n for n in D.AUX_NAMES if n in D.REAL_BB]
    assert m.sum() == 4 * len(known)
    k = D.AUX_NAMES.index("diameter")
    assert abs(float(D.to_physical("diameter", t[0, k])) - 6e-3) < 1e-9


def test_gaussian_nll_prefers_honest_variance():
    from rneml.train import _gnll
    y, mean = torch.zeros(1000), torch.randn(1000) * 0.1     # errors with σ = 0.1
    l = [float(_gnll(mean, y, torch.full((1000,), 2 * np.log(s))).mean()) for s in (0.03, 0.1, 0.3)]
    assert l[1] < l[0] and l[1] < l[2]


def _clicks(times, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(SR * 5, np.float32)
    for c in times:
        i = int(c * SR)
        x[i:i + 200] += rng.standard_normal(200).astype(np.float32) * 0.3
    return x


def test_modulation_features_see_shaking_period_and_ignore_level_and_shift():
    per = _clicks(np.arange(0.01, 5, 1 / 6.2))
    rnd = _clicks(np.sort(np.random.default_rng(1).uniform(0, 5, 31)))
    fp = modulation_features(log_mel(per)).reshape(16, -1)
    fr = modulation_features(log_mel(rnd)).reshape(16, -1)
    centres = np.sqrt(MOD_EDGES[:-1] * MOD_EDGES[1:])
    k = int(np.argmin(np.abs(centres - 6.2)))
    assert fp[8, k] > fr[8, k] + 0.2                  # periodic clicks → peak near 6.2 Hz
    assert np.allclose(modulation_features(log_mel(3 * per)), modulation_features(log_mel(per)),
                       atol=1e-3)                     # level-free
    shifted = np.roll(per, SR // 20)                  # phase-free
    assert np.abs(modulation_features(log_mel(shifted)) - modulation_features(log_mel(per))).mean() < 0.05


def test_gradient_reversal():
    from rneml.model import grad_reverse
    x = torch.ones(3, requires_grad=True)
    (grad_reverse(x, 0.5) * 2).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -1.0))


def test_tta_views_and_predict_shapes():
    from rneml.train import Norm, _views, predict
    v = _views(500, 0.8, 4)
    assert v[0] == slice(0, 500) and len(v) == 5 and all(s.stop - s.start == 400 for s in v[1:])
    m = CountNet(len(SCALAR_NAMES), len(D.AUX_NAMES)).eval()
    img = np.random.default_rng(0).standard_normal((5, N_MELS, 500)).astype(np.float16)
    sc = np.random.default_rng(1).standard_normal((5, len(SCALAR_NAMES))).astype(np.float32)
    P = predict(m, img, sc, Norm(img, sc), "cpu", crop_frac=0.8, tta_crops=4,
                sigma_scale={"n": 2.0, "props": [1.0] * len(D.AUX_NAMES)})
    P1 = predict(m, img, sc, Norm(img, sc), "cpu", crop_frac=0.8, tta_crops=4)
    assert P["n"].shape == (5,) and P["props"].shape == (5, len(D.AUX_NAMES))
    assert np.allclose(P["n_std"], 2 * P1["n_std"])


def test_level_free_removes_gain():
    from rneml.train import level_free
    rng = np.random.default_rng(0)
    x = (0.05 * rng.standard_normal(SR * 5)).astype(np.float32)
    S1, S2 = log_mel(x), log_mel(4 * x)
    s = np.zeros((2, len(SCALAR_NAMES)), np.float32)
    img, sc = level_free(np.stack([S1, S2]).astype(np.float16), s)
    d = np.abs(img[0].astype(np.float32) - img[1].astype(np.float32))
    assert np.percentile(d, 99) < 0.1     # only cells at the -100 dB floor can differ


def test_predict_files_end_to_end(tmp_path):
    from dataclasses import asdict

    from scipy.io import wavfile

    from rneml.infer import format_result, predict_files
    from rneml.train import Norm, TrainConfig

    rng = np.random.default_rng(0)
    w = tmp_path / "rec.wav"
    wavfile.write(w, SR, (0.1 * rng.standard_normal(SR * 6) * 32767).astype(np.int16))
    run = tmp_path / "run"
    run.mkdir()
    m = CountNet(len(SCALAR_NAMES), len(D.AUX_NAMES))
    norm = Norm(rng.normal(-40, 15, (4, N_MELS, 10)).astype(np.float16),
                rng.standard_normal((4, len(SCALAR_NAMES))).astype(np.float32))
    torch.save({"model": m.state_dict(), "norm": norm.state(), "n_scalars": len(SCALAR_NAMES),
                "n_props": len(D.AUX_NAMES), "cfg": asdict(TrainConfig()),
                "sigma_scale": {"n": 1.0, "props": [1.0] * len(D.AUX_NAMES)}}, run / "model.pt")
    r = predict_files([run, run], [str(w)])[0]
    assert np.isfinite(r["N"]) and r["N_std"] > 0
    assert set(r["properties"]) == set(D.AUX_NAMES)
    assert "diameter" in format_result(r)


def test_degeneracy_detects_n_diameter_trade():
    from rneml.evaluate import degeneracy
    rng = np.random.default_rng(0)
    rows, pN, pz = [], [], []
    k = D.AUX_NAMES.index("diameter")
    for _ in range(200):
        N, d = int(rng.integers(10, 100)), rng.uniform(4e-3, 8e-3)
        rows.append({"N": N, "particle_diameter_m": d, "particle_density": 1500.0,
                     "particle_mass_kg": 1500.0 * np.pi / 6 * d ** 3})
        f = np.exp(rng.normal(0, 0.15))          # keep N d² fixed: N × f, d / sqrt(f)
        pN.append(N * f)
        z = np.zeros(len(D.AUX_NAMES))
        z[k] = D.PROPERTIES["diameter"][3](d / np.sqrt(f))
        z[D.AUX_NAMES.index("density")] = D.PROPERTIES["density"][3](1500.0)
        z[D.AUX_NAMES.index("mass")] = D.PROPERTIES["mass"][3](1500.0 * np.pi / 6 * (d / np.sqrt(f)) ** 3)
        pz.append(z)
    out = degeneracy(np.array(pN), np.array(pz), rows)
    assert out["corr_N_error_with"]["diameter"] < -0.8
    assert out["rel_mae_layers_F"] < 0.2 * out["rel_mae_N"]


def _fake_rows(n_runs=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_runs):
        N = int(rng.integers(0, 101))
        mat = {"restitution_pw": rng.uniform(0.55, 0.92), "particle_diameter_m": rng.uniform(4e-3, 8e-3)}
        split = rng.choice(["train", "val", "test", "test_ood"], p=[0.7, 0.1, 0.1, 0.1])
        for k in range(4):
            rows.append({"run_id": f"r{i:03d}", "render": k, "N": N, "split": str(split),
                         "pair_id": "", **mat})
    for j in range(5):   # one same_N pair per j
        N = int(rng.integers(0, 101))
        for side in "ab":
            for k in range(4):
                rows.append({"run_id": f"pN{j}{side}", "render": k, "N": N, "split": "pairs",
                             "pair_id": f"pN{j}", "restitution_pw": rng.uniform(0.55, 0.92),
                             "particle_diameter_m": 6e-3})
    return rows


def test_splits_never_leak_and_respect_ranges():
    from rneml import splits as SP
    rows = _fake_rows()
    specs = [None,
             {"kind": "material_extrap", "axis": "restitution_pw", "lo": 0.55, "hi": 0.75},
             {"kind": "material_band", "axis": "particle_diameter_m", "lo": 5.5e-3, "hi": 6.5e-3},
             {"kind": "count_extrap", "n_train_max": 70, "n_test_min": 80},
             {"kind": "count_sparse", "step": 10, "halfwidth": 1}]
    for spec in specs:
        a = SP.assign(rows, spec, seed=0)
        SP.check_no_leak(a)
        tr, te = SP.select_role(a, "train") + SP.select_role(a, "val"), SP.select_role(a, "test")
        if spec and spec["kind"] == "material_extrap":
            assert all(0.55 <= r["restitution_pw"] <= 0.75 for r in tr)
            assert all(not (0.55 <= r["restitution_pw"] <= 0.75) for r in te)
        if spec and spec["kind"] == "count_extrap":
            assert max(r["N"] for r in tr) <= 70 and min(r["N"] for r in te) >= 80
        if spec and spec["kind"] == "count_sparse":
            assert all(abs(r["N"] - 10 * round(r["N"] / 10)) <= 1 for r in tr)
            assert all(abs(r["N"] - 10 * round(r["N"] / 10)) >= 3 for r in te)
    bad = SP.assign(rows, None)
    bad[0] = {**bad[0], "role": "test" if bad[0]["role"] != "test" else "train"}
    import pytest
    with pytest.raises(AssertionError):
        SP.check_no_leak(bad)


def test_conformal_interval_coverage():
    from rneml.train import conformal_metrics
    rng = np.random.default_rng(0)
    true = rng.uniform(0, 100, 5000)
    std = rng.uniform(2, 8, 5000)
    pred = true + rng.standard_normal(5000) * std
    q90 = 1.6449                                    # Gaussian 90 % two-sided
    c = conformal_metrics(pred, true, std, {"0.1": q90})
    assert abs(c["0.1"]["coverage"] - 0.90) < 0.02


def test_paired_compare_detects_better_run(tmp_path):
    import json as _json

    from rneml.stats import paired_compare
    rng = np.random.default_rng(0)
    for name, noise in (("a", 2.0), ("b", 6.0)):
        d = tmp_path / name
        d.mkdir()
        with open(d / "predictions.jsonl", "w") as fh:
            for i in range(60):
                for k in range(4):
                    N = (i * 7) % 101
                    fh.write(_json.dumps({"split": "test", "run_id": f"r{i}", "render": k, "N": N,
                                          "pred": N + rng.standard_normal() * noise}) + "\n")
    c = paired_compare(tmp_path / "a", tmp_path / "b")
    assert c["mae_diff_a_minus_b"] < 0 and c["ci95_run_bootstrap"][1] < 0 and c["wilcoxon_p"] < 1e-3
