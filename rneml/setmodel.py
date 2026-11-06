"""Condition-set model: one system observed under several shaking conditions.

Why: from one condition, the sound mixes count and material (e.g. "many small" vs "few
large" particles, or low vs high restitution). Material and microphone gain scale the
level of every condition alike (a dB offset); the count changes how the response varies
with Γ — where the bed fluidises, as in the first study's voltage sweeps. A set of
conditions therefore carries information a single clip lacks.

Structure
  clip encoder      the CountNet CNN + scalar branch, shared by every condition
  condition value   log Γ → Fourier features → MLP, added to the clip embedding
  aggregation       'set' (self-attention + masked mean), 'ordered' (Γ-sorted 1-D conv),
                    or 'mean'
  difference path   Δz between Γ-adjacent conditions, pooled ('none' | 'diff' | 'both')
  heads             N and material properties, each mean + log-variance

Level normalisation (``norm_mode``)
  absolute   fixed dB reference (level kept)
  per_clip   each clip's mean dB removed, level scalars zeroed (level discarded)
  per_set    the set's common mean dB removed from every clip (and from the level
             scalars): a gain or mass factor common to all conditions vanishes, the
             level differences between conditions remain

Sets are padded to K conditions with a mask; padded entries never influence the
output (tested). A single clip is a set of size 1 — the same model scores the
single-condition real recordings.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from rneml.features import BASIC_SCALARS
from rneml.model import LOGVAR_MAX, LOGVAR_MIN, CountHead, CountNet, Output, grad_reverse

LEVEL_SCALARS = [BASIC_SCALARS.index(k) for k in ("db_mean", "db_max", "rms_db")]
GAMMA_REF = 2.07


def normalize_levels(img: np.ndarray, sc: np.ndarray, mask: np.ndarray, mode: str):
    """img [B,K,(C,)F,T] (dB), sc [B,K,S], mask [B,K] bool → normalised copies (float32)."""
    img = img.astype(np.float32)
    sc = sc.astype(np.float32).copy()
    if mode == "absolute":
        return img, sc
    ax = tuple(range(2, img.ndim))
    expand = (slice(None), slice(None)) + (None,) * len(ax)
    clip_mean = img.mean(axis=ax)                                       # [B,K]
    if mode == "per_clip":
        img = img - clip_mean[expand]
        sc[:, :, LEVEL_SCALARS] = 0.0
        return img, sc
    if mode == "per_set":
        m = mask.astype(np.float32)
        off = (clip_mean * m).sum(1) / np.maximum(m.sum(1), 1)          # [B]
        img = img - off.reshape(-1, *([1] * (img.ndim - 1)))
        lv = sc[:, :, LEVEL_SCALARS]
        lv_off = (lv * m[:, :, None]).sum(1) / np.maximum(m.sum(1), 1)[:, None]
        sc[:, :, LEVEL_SCALARS] = lv - lv_off[:, None, :]
        return img, sc
    raise ValueError("norm_mode must be absolute, per_clip or per_set")


class ConditionEmbedding(nn.Module):
    def __init__(self, d: int = 128, n_freq: int = 4):
        super().__init__()
        self.n_freq = n_freq
        self.mlp = nn.Sequential(nn.Linear(1 + 2 * n_freq, 64), nn.ReLU(inplace=True), nn.Linear(64, d))

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:          # [B,K] → [B,K,d]
        x = (torch.log(gamma.clamp(min=1e-3)) - math.log(GAMMA_REF)) / 0.5
        f = [x]
        for k in range(self.n_freq):
            f += [torch.sin((2 ** k) * math.pi * x), torch.cos((2 ** k) * math.pi * x)]
        return self.mlp(torch.stack(f, -1))


class SetCountNet(nn.Module):
    def __init__(self, n_scalars: int, n_props: int, width: int = 32, dropout: float = 0.3,
                 use_scalars: bool = True, time_pool: int = 1, agg: str = "set",
                 diff_path: str = "both", use_condition: bool = True,
                 count_head: str = "regression", class_step: int = 5, in_channels: int = 1,
                 arch: str = "cnn", temporal: str = "transformer"):
        super().__init__()
        self.enc = CountNet(n_scalars, 0, width, dropout, use_scalars, time_pool,
                            in_channels=in_channels, arch=arch, temporal=temporal)
        d = 128
        self.agg, self.diff_path, self.use_condition = agg, diff_path, use_condition
        self.n_props = n_props
        self.cond = ConditionEmbedding(d)
        if agg == "set":
            layer = nn.TransformerEncoderLayer(d, 4, 256, dropout=0.1, batch_first=True)
            self.mix = nn.TransformerEncoder(layer, 2)
        elif agg == "ordered":
            self.mix = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv1d(d, d, 3, padding=1))
        elif agg != "mean":
            raise ValueError("agg must be set, ordered or mean")
        self.diff = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True))
        n_in = {"none": d, "diff": d, "both": 2 * d}[diff_path]
        self.fuse = nn.Sequential(nn.Linear(n_in, d), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.count = CountHead(d, count_head, class_step)
        self.props = nn.Linear(d, 2 * n_props) if n_props else None
        self.domain = nn.Sequential(nn.Linear(d, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    @staticmethod
    def _masked_mean(z, mask):
        m = mask.unsqueeze(-1).to(z.dtype)
        return (z * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, img, sc, gamma, mask) -> Output:
        """img [B,K,F,T], sc [B,K,S], gamma [B,K], mask [B,K] (bool, conditions sorted by Γ)."""
        B, K = mask.shape
        z = self.enc.embed(img.reshape(B * K, *img.shape[2:]), sc.reshape(B * K, -1)).view(B, K, -1)
        z = z * mask.unsqueeze(-1).to(z.dtype)                     # padded entries → 0
        parts = []
        if self.diff_path in ("none", "both"):
            h = z + (self.cond(gamma) * mask.unsqueeze(-1).to(z.dtype) if self.use_condition else 0)
            if self.agg == "set":
                h = self.mix(h, src_key_padding_mask=~mask)
            elif self.agg == "ordered":
                h = self.mix(h.transpose(1, 2)).transpose(1, 2)
            parts.append(self._masked_mean(h, mask))
        if self.diff_path in ("diff", "both"):
            dm = mask[:, 1:] & mask[:, :-1]
            dz = self.diff(z[:, 1:] - z[:, :-1]) if K > 1 else z.new_zeros(B, 0, z.shape[-1])
            parts.append(self._masked_mean(dz, dm) if K > 1 else z.new_zeros(B, z.shape[-1]))
        e = self.fuse(torch.cat(parts, -1))
        n, n_lv, logits = self.count(e)
        if self.props is not None:
            p = self.props(e)
            pm, plv = p[:, :self.n_props], p[:, self.n_props:].clamp(LOGVAR_MIN, LOGVAR_MAX)
        else:
            pm = plv = e.new_zeros(B, 0)
        return Output(n, n_lv, pm, plv, e, logits)


def condition_gamma(row: dict, default: float) -> float:
    """Γ the set model is told: the *nominal* one when the dataset has it (the speed the
    experimenter set; the simulated clips carry a few per cent of speed jitter, like the
    real shaker), else the simulated one, else ``default``."""
    for k in ("gamma_nominal", "gamma"):
        v = row.get(k)
        if v not in (None, ""):
            return float(v)
    return float(default)


def build_sets(rows: list[dict], conditions: list[int] | None = None) -> list[list[int]]:
    """Indices of rows grouped by (system_id, render), sorted by Γ. ``conditions``
    keeps only these condition indices (evaluation with fewer conditions)."""
    groups: dict = {}
    for i, r in enumerate(rows):
        if conditions is not None and int(r.get("condition_index", 0)) not in conditions:
            continue
        key = (str(r.get("system_id", r["run_id"])), r.get("render", 0))
        groups.setdefault(key, []).append(i)
    out = []
    for key in sorted(groups):
        out.append(sorted(groups[key], key=lambda i: condition_gamma(rows[i], GAMMA_REF)))
    return out


def batch_arrays(sets, img, sc, gamma, K: int, drop_p: float = 0.0, rng=None):
    """Pad sets to K → (img [B,K,F,T] f16, sc [B,K,S], gamma [B,K], mask [B,K]).
    ``drop_p`` randomly removes conditions (at least one kept)."""
    B = len(sets)
    I = np.zeros((B, K, *img.shape[1:]), np.float16)
    S = np.zeros((B, K, sc.shape[1]), np.float32)
    G = np.full((B, K), GAMMA_REF, np.float32)
    M = np.zeros((B, K), bool)
    for b, s in enumerate(sets):
        idx = list(s)
        if drop_p > 0 and len(idx) > 1:
            keep = [i for i in idx if rng.random() >= drop_p] or [idx[int(rng.integers(len(idx)))]]
            idx = keep
        idx = idx[:K]
        I[b, :len(idx)] = img[idx]
        S[b, :len(idx)] = sc[idx]
        G[b, :len(idx)] = gamma[idx]
        M[b, :len(idx)] = True
    return I, S, G, M


def set_forward(model, I, S, G, M, norm, mode, device):
    """Normalise levels, standardise, run the model (numpy in, Output out)."""
    I, S = normalize_levels(I, S, M, mode)
    I = (I - norm.m_mu) / norm.m_sd
    S = (S - norm.s_mu) / norm.s_sd
    I[~M] = 0.0
    S[~M] = 0.0
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    return model(t(I), t(S), t(G), t(M, torch.bool))


def predict_sets(model, sets, img, sc, gamma, norm, mode, device, K, batch: int = 16) -> dict:
    model.eval()
    n, nv, p, pv = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(sets), batch):
            o = set_forward(model, *batch_arrays(sets[i:i + batch], img, sc, gamma, K), norm, mode, device)
            n.append(o.n.cpu().numpy() * 100)
            nv.append(np.exp(o.n_logvar.cpu().numpy()))
            p.append(o.props.cpu().numpy())
            pv.append(np.exp(o.props_logvar.cpu().numpy()))
    cat = lambda a: np.concatenate(a) if a else np.zeros(0)
    return {"n": cat(n), "n_std": np.sqrt(cat(nv)) * 100, "props": cat(p), "props_std": np.sqrt(cat(pv))}
