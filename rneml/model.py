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
    n: torch.Tensor          # [B]      N / 100 (for class heads: expected value)
    n_logvar: torch.Tensor   # [B]      (for class heads: log variance of the distribution)
    props: torch.Tensor      # [B, P]   normalised properties (empty if P = 0)
    props_logvar: torch.Tensor
    emb: torch.Tensor        # [B, 128]
    count_logits: torch.Tensor | None = None   # class / CORAL heads


N_MAX = 100


class CountHead(nn.Module):
    """Count output designs (ML spec 4.3), all returning N/100 and a log-variance:

    regression      mean and log-variance directly (Huber + Gaussian NLL)
    classification  softmax over class centres 0, step, …, 100; expected value decoding
    soft_label      same head, trained on Gaussian-smoothed labels (far misses cost more)
    coral           K−1 ordinal thresholds "N > c_k?" with a shared weight (monotone)
    """

    def __init__(self, d: int, kind: str = "regression", step: int = 5):
        super().__init__()
        self.kind = kind
        self.centres = torch.arange(0, N_MAX + 1, step, dtype=torch.float32)
        k = len(self.centres)
        if kind == "regression":
            self.lin = nn.Linear(d, 2)
        elif kind in ("classification", "soft_label"):
            self.lin = nn.Linear(d, k)
        elif kind == "coral":
            self.lin = nn.Linear(d, 1, bias=False)
            self.bias = nn.Parameter(torch.linspace(2.0, -2.0, k - 1))
        else:
            raise ValueError("count_head must be regression, classification, soft_label or coral")

    def forward(self, e):
        if self.kind == "regression":
            c = self.lin(e)
            return c[:, 0], c[:, 1].clamp(LOGVAR_MIN, LOGVAR_MAX), None
        cen = self.centres.to(e.device) / 100.0
        if self.kind == "coral":
            logits = self.lin(e) + self.bias                       # [B, K-1]
            p_gt = torch.sigmoid(logits)                           # P(N > c_k)
            step = cen[1] - cen[0]
            mean = p_gt.sum(1) * step
            var = (p_gt * (1 - p_gt)).sum(1) * step ** 2 + (step ** 2) / 12
            return mean, torch.log(var).clamp(LOGVAR_MIN, LOGVAR_MAX), logits
        logits = self.lin(e)
        p = torch.softmax(logits, 1)
        mean = (p * cen).sum(1)
        var = (p * (cen - mean[:, None]) ** 2).sum(1) + ((cen[1] - cen[0]) ** 2) / 12
        return mean, torch.log(var).clamp(LOGVAR_MIN, LOGVAR_MAX), logits

    def loss(self, logits, y01, label_sigma: float):
        """Classification-type losses; y01 = N / 100."""
        cen = self.centres.to(y01.device) / 100.0
        if self.kind == "classification":
            tgt = torch.argmin(torch.abs(y01[:, None] - cen[None, :]), 1)
            return nn.functional.cross_entropy(logits, tgt)
        if self.kind == "soft_label":
            q = torch.exp(-0.5 * ((cen[None, :] - y01[:, None]) / label_sigma) ** 2)
            q = q / q.sum(1, keepdim=True)
            return -(q * torch.log_softmax(logits, 1)).sum(1).mean()
        tgt = (y01[:, None] > cen[None, :-1] + 1e-6).float()      # coral
        return nn.functional.binary_cross_entropy_with_logits(logits, tgt)


def block(cin: int, cout: int, pool: tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(pool))


class CountNet(nn.Module):
    def __init__(self, n_scalars: int, n_props: int, width: int = 32, dropout: float = 0.3,
                 use_scalars: bool = True, time_pool: int = 1, count_head: str = "regression",
                 class_step: int = 5, in_channels: int = 1):
        super().__init__()
        w = width
        self.time_pool = time_pool   # average-pool input frames (2 → 20 ms frames, ~3× faster)
        self.conv = nn.Sequential(block(in_channels, w, (2, 2)), block(w, 2 * w, (2, 2)),
                                  block(2 * w, 4 * w, (2, 2)), block(4 * w, 4 * w, (2, 1)))
        self.use_scalars = use_scalars
        self.n_props = n_props
        d = 3 * 4 * w + (n_scalars if use_scalars else 0)
        self.trunk = nn.Sequential(nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                   nn.Linear(256, 128), nn.ReLU(inplace=True))
        self.count = CountHead(128, count_head, class_step)
        self.props = nn.Linear(128, 2 * n_props) if n_props else None
        # sim-vs-real discriminator for unsupervised domain adaptation (used only when
        # training with unlabeled real clips; trained through a gradient-reversal layer)
        self.domain = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def embed(self, image: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        x = image.unsqueeze(1) if image.dim() == 3 else image     # [B, C, F, T]
        if self.time_pool > 1:
            x = nn.functional.avg_pool2d(x, (1, self.time_pool))
        h = self.conv(x)                           # [B, C, F', T']
        h = h.mean(dim=2)                          # over frequency
        z = torch.cat([h.mean(-1), h.std(-1), h.amax(-1)], dim=1)
        if self.use_scalars:
            z = torch.cat([z, scalars], dim=1)
        return self.trunk(z)

    def forward(self, image, scalars) -> Output:
        e = self.embed(image, scalars)
        n, n_lv, logits = self.count(e)
        if self.props is not None:
            p = self.props(e)
            pm, plv = p[:, :self.n_props], p[:, self.n_props:].clamp(LOGVAR_MIN, LOGVAR_MAX)
        else:
            pm = plv = e.new_zeros(e.shape[0], 0)
        return Output(n, n_lv, pm, plv, e, logits)
