"""Training: count regression + auxiliary material heads + contrast-pair losses.

Loss = Huber(N̂/100, N/100)
     + aux_weight  · MSE(material̂, material)            (sim labels only)
     + pair_weight · mean (N̂_a − N̂_b)²                   same_N pairs (invariance)
     + rank_weight · max(0, m − s·(N̂_a − N̂_b))           same_material pairs (order)

Augmentation: SpecAugment (time/frequency masks) and a random level shift (± gain_db
on the dB spectrogram and the level scalars) so the absolute level — confounded with
particle mass and microphone gain — cannot be the only cue.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from rneml import data as D
from rneml.features import SCALAR_NAMES, build_cache
from rneml.model import CountNet

LEVEL_SCALARS = [SCALAR_NAMES.index(k) for k in ("db_mean", "db_max", "rms_db")]


@dataclass
class TrainConfig:
    name: str = "blind_v1"
    data_dir: str = "data/v1"
    cache_dir: str = "cache"
    out_dir: str = "outputs"
    real_dir: str | None = None          # first-study clips {N}-{k}.wav for real evaluation
    train_subset: str = "all"            # "all" (blind) or "bb_like" (known-material baseline)
    epochs: int = 60
    batch: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    width: int = 32
    dropout: float = 0.3
    use_scalars: bool = True
    aux_weight: float = 0.5
    pair_weight: float = 1.0
    rank_weight: float = 0.5
    rank_margin: float = 0.02
    pairs_per_batch: int = 8
    gain_db: float = 3.0
    spec_time_masks: int = 2
    spec_freq_masks: int = 2
    patience: int = 12
    seed: int = 0
    workers: int = 4
    device: str = "auto"
    extra: dict = field(default_factory=dict)


def load_config(path: str | Path | None, **over) -> TrainConfig:
    d = yaml.safe_load(Path(path).read_text()) if path else {}
    d.update({k: v for k, v in over.items() if v is not None})
    return TrainConfig(**d)


# ----------------------------------------------------------------------------- features

def features_for(rows: list[dict], cache: Path, workers: int, log=print):
    """(mel [n,F,T] float16, scalars [n,k]) aligned with rows; computes missing ones."""
    paths = [r["path"] for r in rows]
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = {p: k for k, p in enumerate(z["paths"].tolist())}
    missing = [p for p in paths if p not in have]
    if missing:
        log(f"computing features for {len(missing)} clips → {cache}")
        tmp = build_cache(missing, cache.with_suffix(".new.npz"), workers, log)
        n = np.load(tmp)
        if have:
            o = np.load(cache)
            mel = np.concatenate([o["mel"], n["mel"]])
            sc = np.concatenate([o["scalars"], n["scalars"]])
            ps = np.concatenate([o["paths"], n["paths"]])
        else:
            mel, sc, ps = n["mel"], n["scalars"], n["paths"]
        np.savez(cache, mel=mel, scalars=sc, paths=ps)
        tmp.unlink()
        have = {p: k for k, p in enumerate(ps.tolist())}
    z = np.load(cache)
    idx = np.array([have[p] for p in paths], dtype=np.int64)
    return z["mel"][idx], z["scalars"][idx]


class Norm:
    def __init__(self, mel: np.ndarray, sc: np.ndarray):
        self.m_mu = float(mel.astype(np.float32).mean())
        self.m_sd = float(mel.astype(np.float32).std() + 1e-6)
        self.s_mu = sc.mean(0)
        self.s_sd = sc.std(0) + 1e-6

    def __call__(self, mel, sc):
        return ((mel.astype(np.float32) - self.m_mu) / self.m_sd,
                ((sc - self.s_mu) / self.s_sd).astype(np.float32))

    def state(self):
        return {"m_mu": self.m_mu, "m_sd": self.m_sd, "s_mu": self.s_mu, "s_sd": self.s_sd}

    @classmethod
    def from_state(cls, s):
        o = cls.__new__(cls)
        o.m_mu, o.m_sd = float(s["m_mu"]), float(s["m_sd"])
        o.s_mu, o.s_sd = np.asarray(s["s_mu"]), np.asarray(s["s_sd"])
        return o


# ----------------------------------------------------------------------------- training

def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _augment(mel: torch.Tensor, sc: torch.Tensor, cfg: TrainConfig, norm: Norm, g):
    B, F, T = mel.shape
    if cfg.gain_db > 0:
        d = (torch.rand(B, generator=g) * 2 - 1) * cfg.gain_db
        mel = mel + (d / norm.m_sd)[:, None, None].to(mel)
        s_sd = torch.as_tensor(norm.s_sd[LEVEL_SCALARS], dtype=sc.dtype)
        sc = sc.clone()
        sc[:, LEVEL_SCALARS] += d[:, None].to(sc) / s_sd.to(sc)
    mel = mel.clone()
    for _ in range(cfg.spec_time_masks):
        w = torch.randint(0, T // 10 + 1, (B,), generator=g)
        s = torch.randint(0, T, (B,), generator=g)
        for b in range(B):
            mel[b, :, s[b]:s[b] + w[b]] = 0.0
    for _ in range(cfg.spec_freq_masks):
        w = torch.randint(0, F // 10 + 1, (B,), generator=g)
        s = torch.randint(0, F, (B,), generator=g)
        for b in range(B):
            mel[b, s[b]:s[b] + w[b], :] = 0.0
    return mel, sc


def predict(model, mel, sc, norm: Norm, device, batch: int = 64) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(mel), batch):
            m, s = norm(mel[i:i + batch], sc[i:i + batch])
            n, _, _ = model(torch.from_numpy(m).to(device), torch.from_numpy(s).to(device))
            out.append(n.cpu().numpy() * 100.0)
    return np.concatenate(out) if out else np.zeros(0)


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    e = pred - true
    return {"n": int(len(true)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()), "within_5": float((np.abs(e) <= 5).mean()),
            "within_10": float((np.abs(e) <= 10).mean())} if len(true) else {"n": 0}


def train(cfg: TrainConfig, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    dev = _device(cfg.device)
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

    rows = D.read_metadata(cfg.data_dir)
    cache = Path(cfg.cache_dir) / (Path(cfg.data_dir).name + "_features.npz")
    tr = D.select(rows, "train")
    if cfg.train_subset == "bb_like":
        tr = D.bb_like(tr)
    va = D.select(rows, "val")
    pairs = D.select(rows, "pairs")
    log(f"train {len(tr)} clips, val {len(va)}, pairs {len(pairs)} ({cfg.train_subset})")

    all_rows = tr + va + pairs
    mel, sc = features_for(all_rows, cache, cfg.workers, log)
    ntr, nva = len(tr), len(va)
    norm = Norm(mel[:ntr], sc[:ntr])
    y = np.array([r["N"] for r in all_rows], np.float32) / 100.0
    aux = np.stack([D.aux_targets(r) for r in all_rows])

    # contrast pairs: index pairs (a, b) within all_rows, matched by pair_id and render
    by_key: dict = {}
    for k, r in enumerate(all_rows[ntr + nva:], start=ntr + nva):
        by_key.setdefault((r["pair_id"], int(r["render"])), []).append(k)
    same_n = [tuple(v) for (pid, _), v in by_key.items() if len(v) == 2 and pid.startswith("pN")]
    same_m = [tuple(v) for (pid, _), v in by_key.items() if len(v) == 2 and pid.startswith("pM")]
    log(f"contrast pairs: same_N {len(same_n)}, same_material {len(same_m)}")

    model = CountNet(sc.shape[1], aux.shape[1], cfg.width, cfg.dropout, cfg.use_scalars).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(ntr / cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(steps, 1))
    huber = torch.nn.SmoothL1Loss(beta=0.05)

    def batch_tensors(idx):
        m, s = norm(mel[idx], sc[idx])
        return torch.from_numpy(m), torch.from_numpy(s)

    best, best_ep, hist = math.inf, -1, []
    for ep in range(cfg.epochs):
        model.train()
        perm = rng.permutation(ntr)
        tl, t0 = 0.0, time.time()
        for i in range(0, ntr, cfg.batch):
            idx = perm[i:i + cfg.batch]
            pa = pb = np.zeros(0, int)
            kinds = np.zeros(0, int)
            if cfg.pairs_per_batch and (same_n or same_m):
                pool = [(a, b, 0) for a, b in same_n] + [(a, b, 1) for a, b in same_m]
                sel = rng.choice(len(pool), min(cfg.pairs_per_batch, len(pool)), replace=False)
                pa = np.array([pool[k][0] for k in sel])
                pb = np.array([pool[k][1] for k in sel])
                kinds = np.array([pool[k][2] for k in sel])
            full = np.concatenate([idx, pa, pb])
            m, s = batch_tensors(full)
            m, s = _augment(m, s, cfg, norm, g)
            n_hat, a_hat, _ = model(m.to(dev), s.to(dev))
            yt = torch.from_numpy(y[full]).to(dev)
            loss = huber(n_hat, yt)
            if a_hat is not None and cfg.aux_weight:
                loss = loss + cfg.aux_weight * torch.mean((a_hat - torch.from_numpy(aux[full]).to(dev)) ** 2)
            if len(pa):
                B, P = len(idx), len(pa)
                na, nb = n_hat[B:B + P], n_hat[B + P:]
                kn = torch.from_numpy(kinds).to(dev)
                if (kn == 0).any() and cfg.pair_weight:
                    loss = loss + cfg.pair_weight * torch.mean((na - nb)[kn == 0] ** 2)
                if (kn == 1).any() and cfg.rank_weight:
                    sgn = torch.sign(yt[B:B + P] - yt[B + P:])[kn == 1]
                    loss = loss + cfg.rank_weight * torch.mean(
                        torch.relu(cfg.rank_margin - sgn * (na - nb)[kn == 1]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss) * len(idx)
        pv = predict(model, mel[ntr:ntr + nva], sc[ntr:ntr + nva], norm, dev)
        mv = metrics(pv, y[ntr:ntr + nva] * 100)
        hist.append({"epoch": ep, "train_loss": tl / ntr, **{f"val_{k}": v for k, v in mv.items()},
                     "sec": time.time() - t0})
        log(f"ep {ep:3d} loss {tl / ntr:.4f} val MAE {mv['mae']:.2f} ±5 {mv['within_5']:.2f} "
            f"({time.time() - t0:.0f}s)")
        if mv["mae"] < best:
            best, best_ep = mv["mae"], ep
            torch.save({"model": model.state_dict(), "norm": norm.state(),
                        "n_scalars": sc.shape[1], "n_aux": aux.shape[1], "cfg": asdict(cfg)},
                       out / "model.pt")
        elif ep - best_ep >= cfg.patience:
            log(f"early stop (best epoch {best_ep})")
            break
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    from rneml.evaluate import evaluate_run
    return evaluate_run(out, cfg, log=log)
