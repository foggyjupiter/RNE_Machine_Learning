"""CountNet: CNN over a time-frequency image + scalar physics features.

Input: a time-frequency image (log-mel or wavelet scalogram, see features.py) and the
scalar physics features. **No material property is ever an input** — diameter,
density, restitution, friction and modulus are only targets.

Conv blocks (3×3 conv, BN, ReLU, max-pool; channels w-2w-4w-4w) → mean over
frequency → statistics pooling over time (mean, std, max; as in the first study) →
concatenated with the normalised scalar features → shared MLP → heads, each giving a
mean and a log-variance (heteroscedastic regression):

  count   N / 100
  props   material properties (diameter, density, mass, restitution pp/pw,
          friction pp/pw, modulus), normalised

Optional domain-adversarial head: a sim-vs-real classifier on the embedding behind a
gradient-reversal layer pushes the embedding to look the same for simulated and real
recordings (unlabeled real clips suffice).

The variance lets the model say "not sure" where different (N, material) sound alike
(e.g. many small vs few large particles), instead of silently averaging.

Predicting the material jointly is the physical route to a material-agnostic count:
the sound depends on combinations such as (1 − e²)·N, so a model that can name the
material can also undo its effect on the count.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

LOGVAR_MIN, LOGVAR_MAX = -12.0, 4.0


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Identity forward, gradient × (−lam) backward (domain-adversarial training)."""
    return _GradReverse.apply(x, lam)


class Output(NamedTuple):
    n: torch.Tensor          # [B]      N / 100
    n_logvar: torch.Tensor   # [B]
    props: torch.Tensor      # [B, P]   normalised properties (empty if P = 0)
    props_logvar: torch.Tensor
    emb: torch.Tensor        # [B, 128]


def block(cin: int, cout: int, pool: tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(pool))


class CountNet(nn.Module):
    def __init__(self, n_scalars: int, n_props: int, width: int = 32, dropout: float = 0.3,
                 use_scalars: bool = True):
        super().__init__()
        w = width
        self.conv = nn.Sequential(block(1, w, (2, 2)), block(w, 2 * w, (2, 2)),
                                  block(2 * w, 4 * w, (2, 2)), block(4 * w, 4 * w, (2, 1)))
        self.use_scalars = use_scalars
        self.n_props = n_props
        d = 3 * 4 * w + (n_scalars if use_scalars else 0)
        self.trunk = nn.Sequential(nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                   nn.Linear(256, 128), nn.ReLU(inplace=True))
        self.count = nn.Linear(128, 2)
        self.props = nn.Linear(128, 2 * n_props) if n_props else None
        # sim-vs-real discriminator for unsupervised domain adaptation (used only when
        # training with unlabeled real clips; trained through a gradient-reversal layer)
        self.domain = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def embed(self, image: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        h = self.conv(image.unsqueeze(1))         # [B, C, F', T']
        h = h.mean(dim=2)                          # over frequency
        z = torch.cat([h.mean(-1), h.std(-1), h.amax(-1)], dim=1)
        if self.use_scalars:
            z = torch.cat([z, scalars], dim=1)
        return self.trunk(z)

    def forward(self, image, scalars) -> Output:
        e = self.embed(image, scalars)
        c = self.count(e)
        n, n_lv = c[:, 0], c[:, 1].clamp(LOGVAR_MIN, LOGVAR_MAX)
        if self.props is not None:
            p = self.props(e)
            pm, plv = p[:, :self.n_props], p[:, self.n_props:].clamp(LOGVAR_MIN, LOGVAR_MAX)
        else:
            pm = plv = e.new_zeros(e.shape[0], 0)
        return Output(n, n_lv, pm, plv, e)
