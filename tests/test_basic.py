import math

import numpy as np
import torch

from rneml import data as D
from rneml.features import (HOP, N_MELS, SCALAR_NAMES, SCALO_BANDS, SR, _morlet_bank, log_mel,
                            scalar_features, scalogram)
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
    assert s.shape == (len(SCALAR_NAMES),) and np.isfinite(s).all()
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
