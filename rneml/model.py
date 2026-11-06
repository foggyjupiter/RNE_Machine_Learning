"""CountNet: CNN over the log-mel spectrogram + scalar physics features.

Conv blocks (3×3 conv, BN, ReLU, max-pool; channels 32-64-128-128) → mean over
frequency → statistics pooling over time (mean, std, max; as in the first study) →
concatenated with the normalised scalar features → shared MLP → heads:

  count   N / 100 (regression; an ordinal quantity, so no classification grid)
  aux     material properties (diameter, density, mass, restitution pp/pw,
          friction pp/pw, modulus), normalised — outputs in their own right

Predicting the material jointly is the physical route to a material-agnostic count:
the sound depends on combinations such as (1 − e²)·N, so a model that can name the
material can also undo its effect on the count.
"""

from __future__ import annotations

import torch
from torch import nn


def block(cin: int, cout: int, pool: tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(pool))


class CountNet(nn.Module):
    def __init__(self, n_scalars: int, n_aux: int, width: int = 32, dropout: float = 0.3,
                 use_scalars: bool = True):
        super().__init__()
        w = width
        self.conv = nn.Sequential(block(1, w, (2, 2)), block(w, 2 * w, (2, 2)),
                                  block(2 * w, 4 * w, (2, 2)), block(4 * w, 4 * w, (2, 1)))
        self.use_scalars = use_scalars
        d = 3 * 4 * w + (n_scalars if use_scalars else 0)
        self.trunk = nn.Sequential(nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                   nn.Linear(256, 128), nn.ReLU(inplace=True))
        self.count = nn.Linear(128, 1)
        self.aux = nn.Linear(128, n_aux) if n_aux else None

    def embed(self, mel: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        h = self.conv(mel.unsqueeze(1))          # [B, C, F', T']
        h = h.mean(dim=2)                         # over frequency
        z = torch.cat([h.mean(-1), h.std(-1), h.amax(-1)], dim=1)
        if self.use_scalars:
            z = torch.cat([z, scalars], dim=1)
        return self.trunk(z)

    def forward(self, mel, scalars):
        e = self.embed(mel, scalars)
        n = self.count(e).squeeze(1)
        a = self.aux(e) if self.aux is not None else None
        return n, a, e
