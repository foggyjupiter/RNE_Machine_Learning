import math

import numpy as np
import torch

from rneml import data as D
from rneml.features import HOP, N_MELS, SCALAR_NAMES, SR, log_mel, scalar_features
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
    n, a, e = m(torch.randn(3, N_MELS, 501), torch.randn(3, len(SCALAR_NAMES)))
    assert n.shape == (3,) and a.shape == (3, len(D.AUX_NAMES)) and e.shape == (3, 128)
