"""Training: count + material-property regression with uncertainty, contrast-pair losses,
optional sim-to-real adaptation.

Every head predicts a mean and a log-variance. Loss:

    L = Huber(N̂, N)                                    accuracy of the count (N/100 scale)
      + prop_weight · masked MSE(p̂, p)                 material properties (normalised)
      + nll_weight  · Gaussian NLL(sg(mean), y, σ²)     calibrated uncertainty, all heads
      + pair_weight · mean (N̂_a − N̂_b)²                same_N pairs: invariance to material
      + rank_weight · max(0, m − s·(N̂_a − N̂_b))         same_material pairs: order in N
      + λ(t) · BCE(domain(GRL(embedding)), sim/real)    unlabeled real clips (DANN)

sg = stop-gradient: the variance learns how wrong the mean is without pulling the mean.
A property mask allows rows where only some properties are known (real recordings:
diameter, density and mass of the first-study BB are known, restitution etc. are not).

Augmentation: random time crop (crop_frac of the clip; full clip + crops averaged at
test time), SpecAugment (time/frequency masks) and a random level shift (± gain_db on
the dB image and the level scalars), so the absolute level — confounded with particle
mass and microphone gain — cannot be the only cue.

Real recordings (first study, 15 clips per N): the lowest-numbered clips per N can be
used for adaptation — ``real_train_per_N`` with labels, ``real_unlabeled_per_N``
without labels (domain-adversarial). All other real clips are evaluation only.

After training, σ of every head is rescaled on the validation set so that
mean((error/σ)²) = 1 (variance calibration).
"""

from __future__ import annotations

import json
import math
import time
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from rneml import data as D
from rneml.features import BASIC_SCALARS, FEATURE_VERSION, build_cache
from rneml.model import CountNet, grad_reverse

LEVEL_SCALARS = [BASIC_SCALARS.index(k) for k in ("db_mean", "db_max", "rms_db")]


@dataclass
class TrainConfig:
    name: str = "blind_mel"
    data_dir: str = "data/v1"            # training / validation data
    eval_data_dir: str | None = None     # test data (default: data_dir)
    cache_dir: str = "cache"
    out_dir: str = "outputs"
    real_dir: str | None = None          # first-study clips {N}-{k}.wav
    real_train_per_N: int = 0            # labeled real clips per N used for training
    real_repeat: int = 10                # oversampling of the labeled real clips
    real_unlabeled_per_N: int = 0        # unlabeled real clips per N for domain adaptation
    real_exclude_per_N: int = 0          # real clips per N used elsewhere (e.g. component selection): never scored
    real_fold: int = 0                   # cross-validation of the real adaptation clips: block of 5 clips per N
                                         # (fold 0 = clips 1-5, 1 = 6-10, 2 = 11-15) used for adaptation
    dann_weight: float = 0.3             # max weight of the domain-adversarial loss
    real_unlabeled_per_batch: int = 8
    train_subset: str = "all"            # "all" (blind) or "bb_like" (known-material baseline)
    split: dict | None = None            # see rneml.splits (None = the dataset's split column)
    split_seed: int = 0                  # same split for every experiment
    conformal_alpha: list = field(default_factory=lambda: [0.1, 0.05])
    gain_test: list = field(default_factory=list)   # e.g. [0.25, 0.5, 2, 4] at evaluation
    # condition-set model (multi-condition data; see rneml.setmodel)
    model_type: str = "clip"             # "clip" (one clip) or "set" (a system's conditions)
    agg: str = "set"                     # set | ordered | mean
    diff_path: str = "both"              # none | diff | both
    norm_mode: str = "absolute"          # absolute | per_clip | per_set
    use_condition: bool = True           # inject Γ of each clip
    cond_dropout: float = 0.3            # drop conditions at random while training
    max_conditions: int = 5
    eval_condition_subsets: dict = field(default_factory=lambda: {
        "372rpm_only": [1], "three": [0, 2, 4], "all": None})
    feature: str = "mel"                 # mel | scalogram | mel22k | '+'-joined mel/perc/harm channels
    denoise: bool = False                # spectral subtraction of the motor noise (empty-box clips)
    noise_ref: str | None = None         # folder with empty-box clips 0-*.wav (default: real_dir)
    band_hz: list | None = None          # keep mel bands within [lo, hi] Hz
    use_modulation: bool = True          # modulation-spectrum features in the scalar branch
    scalar_set: str = "all"              # all | basic (12 physics) | first_study (5: dB mean/std/max, flux, active) | none
    material_input: list = field(default_factory=list)  # properties GIVEN to the model (e.g. [diameter, density, mass]); [] = blind
    component_scalars: str | None = None  # append component features: path of components.json (selected) or 'all'
    count_head: str = "regression"       # regression | classification | soft_label | coral
    class_step: int = 5                  # class / threshold spacing for the non-regression heads
    label_sigma: float = 5.0             # soft_label width [particles]
    level_free: bool = False             # remove the absolute level (per-clip mean dB, level scalars)
    epochs: int = 60
    batch: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    arch: str = "cnn"                    # cnn (light) | resnet (full size; GPU)
    temporal: str = "transformer"        # resnet only: transformer | none
    amp: bool = True                     # mixed precision on CUDA
    width: int = 32
    time_pool: int = 1                   # 2 = 20 ms frames (CPU-friendly, ~3× faster)
    dropout: float = 0.3
    use_scalars: bool = True
    prop_weight: float = 0.5
    n_only: bool = False                 # label N only: no material-property heads at all
    nll_weight: float = 0.2
    pair_weight: float = 1.0
    rank_weight: float = 0.5
    rank_margin: float = 0.02
    pairs_per_batch: int = 8
    gain_db: float = 3.0
    crop_frac: float = 0.8
    tta_crops: int = 4
    spec_time_masks: int = 2
    spec_freq_masks: int = 2
    calibrate_sigma: bool = True
    patience: int = 12
    seed: int = 0
    workers: int = 4
    threads: int = 0                     # torch CPU threads (0 = default)
    device: str = "auto"
    extra: dict = field(default_factory=dict)


def load_config(path: str | Path | None, **over) -> TrainConfig:
    d = yaml.safe_load(Path(path).read_text()) if path else {}
    if "aux_weight" in d:                # name used by the first version
        d["prop_weight"] = d.pop("aux_weight")
    d.update({k: v for k, v in over.items() if v is not None})
    return TrainConfig(**d)


# ----------------------------------------------------------------------------- features

def features_for(rows: list[dict], cache: Path, workers: int, log=print, kind: str = "mel",
                 gain: float = 1.0, noise_psd_file: str | None = None, band: tuple | None = None):
    """(image [n,F,T] float16, scalars [n,k]) aligned with rows; computes missing ones."""
    cache = Path(cache)
    paths = [r["path"] for r in rows]
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = {p: k for k, p in enumerate(z["paths"].tolist())}
    missing = [p for p in dict.fromkeys(paths) if p not in have]
    if missing:
        log(f"computing {kind} features for {len(missing)} clips → {cache}")
        tmp = build_cache(missing, cache.with_suffix(".new.npz"), workers, log, kind, gain,
                          noise_psd_file, band)
        n = np.load(tmp)
        if have:
            o = np.load(cache)
            img = np.concatenate([o["mel"], n["mel"]])
            sc = np.concatenate([o["scalars"], n["scalars"]])
            ps = np.concatenate([o["paths"], n["paths"]])
        else:
            img, sc, ps = n["mel"], n["scalars"], n["paths"]
        np.savez(cache, mel=img, scalars=sc, paths=ps)
        tmp.unlink()
        have = {p: k for k, p in enumerate(ps.tolist())}
    z = np.load(cache)
    idx = np.array([have[p] for p in paths], dtype=np.int64)
    # features cached before the NaN guard (silent bands) are cleaned here as well
    return z["mel"][idx], np.nan_to_num(z["scalars"][idx], nan=0.0, posinf=0.0, neginf=0.0)


def cache_path(cfg: TrainConfig, what: str, gain: float = 1.0) -> Path:
    g = "" if gain == 1.0 else f"_gain{gain:g}"
    dn = "_dn" if getattr(cfg, "denoise", False) else ""
    band = getattr(cfg, "band_hz", None)
    b = f"_b{int(band[0])}-{int(band[1])}" if band else ""
    return Path(cfg.cache_dir) / f"{what}_{cfg.feature}{dn}{b}_v{FEATURE_VERSION}{g}.npz"


def noise_psd_path(cfg: TrainConfig, log=print) -> str | None:
    """Mean noise power spectrum of the empty-box clips (cached), for spectral subtraction."""
    if not getattr(cfg, "denoise", False):
        return None
    import glob

    from rneml.features import noise_psd_from
    ref = cfg.noise_ref or cfg.real_dir
    if not ref:
        raise ValueError("denoise needs noise_ref (or real_dir) with empty-box clips 0-*.wav")
    out = Path(cfg.cache_dir) / "noise_psd.npy"
    if not out.exists():
        files = sorted(glob.glob(str(Path(ref) / "0-*.wav")))
        if not files:
            raise FileNotFoundError(f"no empty-box clips 0-*.wav in {ref}")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, noise_psd_from(files))
        log(f"noise spectrum from {len(files)} empty-box clips → {out}")
    return str(out)


def feats(cfg: TrainConfig, rows: list[dict], what: str, log=print, gain: float = 1.0):
    """Cached features of rows for this config (feature kind, denoising, band, gain)."""
    return features_for(rows, cache_path(cfg, what, gain), cfg.workers, log, cfg.feature, gain,
                        noise_psd_path(cfg, log), tuple(cfg.band_hz) if cfg.band_hz else None)


def level_free(img: np.ndarray, sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop the absolute level: per-clip mean dB removed from the image, level scalars zeroed.
    For deployment with an unknown microphone gain / distance."""
    img = img.astype(np.float32)
    img = img - img.mean(axis=tuple(range(1, img.ndim)), keepdims=True)
    sc = sc.copy()
    sc[:, LEVEL_SCALARS] = 0.0
    return img.astype(np.float16), sc


def prepare(img: np.ndarray, sc: np.ndarray, cfg: TrainConfig | dict, rows: list[dict] | None = None,
            n_scalars: int | None = None):
    """Column selection, optional level removal, and — for the 'material known'
    experiments — the given material properties appended to the scalar branch.
    ``n_scalars``: number of audio scalars the model was trained with (older models)."""
    get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    sc = sc[:, scalar_columns(cfg, sc.shape[1])]
    if n_scalars is not None:
        sc = sc[:, :n_scalars]
    if get("level_free"):
        img, sc = level_free(img, sc)
    comp = get("component_scalars")
    if comp:
        if rows is None:
            raise ValueError("component_scalars needs the metadata rows")
        from rneml.components import scalar_block
        sc = np.concatenate([sc, scalar_block(rows, comp, get("cache_dir", "cache"), get("workers", 1))], axis=1)
    names = get("material_input") or []
    if names:
        if rows is None:
            raise ValueError("material_input needs the metadata rows")
        sc = np.concatenate([sc, D.material_inputs(rows, names)], axis=1)
    return img, sc


def scalar_columns(cfg: TrainConfig | dict, n_total: int) -> np.ndarray:
    get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    sset = get("scalar_set", "all")
    if sset == "none":                               # only component / material scalars
        return np.arange(0)
    if sset == "first_study":
        return np.arange(5)                          # db_mean, db_std, db_max, flux, active_ratio
    if sset == "basic" or not get("use_modulation", True):
        return np.arange(len(BASIC_SCALARS))
    return np.arange(n_total)


class Norm:
    def __init__(self, img: np.ndarray, sc: np.ndarray):
        self.m_mu = float(img.astype(np.float32).mean())
        self.m_sd = float(img.astype(np.float32).std() + 1e-6)
        self.s_mu = sc.mean(0)
        self.s_sd = sc.std(0) + 1e-6

    def __call__(self, img, sc):
        return ((img.astype(np.float32) - self.m_mu) / self.m_sd,
                ((sc - self.s_mu) / self.s_sd).astype(np.float32))

    def state(self):
        return {"m_mu": self.m_mu, "m_sd": self.m_sd, "s_mu": self.s_mu, "s_sd": self.s_sd}

    @classmethod
    def from_state(cls, s):
        o = cls.__new__(cls)
        o.m_mu, o.m_sd = float(s["m_mu"]), float(s["m_sd"])
        o.s_mu, o.s_sd = np.asarray(s["s_mu"]), np.asarray(s["s_sd"])
        return o


# ----------------------------------------------------------------------------- helpers

def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _augment(img: torch.Tensor, sc: torch.Tensor, cfg: TrainConfig, norm: Norm, g):
    B = img.shape[0]
    F, T = img.shape[-2:]
    if cfg.crop_frac < 1.0:
        L = max(8, int(round(cfg.crop_frac * T)))
        st = torch.randint(0, T - L + 1, (B,), generator=g)
        img = torch.stack([img[b, ..., st[b]:st[b] + L] for b in range(B)])
        T = L
    if cfg.gain_db > 0 and not cfg.level_free:
        d = (torch.rand(B, generator=g) * 2 - 1) * cfg.gain_db
        img = img + (d / norm.m_sd).view(-1, *([1] * (img.dim() - 1))).to(img)
        s_sd = torch.as_tensor(norm.s_sd[LEVEL_SCALARS], dtype=sc.dtype)
        sc = sc.clone()
        sc[:, LEVEL_SCALARS] += d[:, None].to(sc) / s_sd.to(sc)
    img = img.clone()
    for _ in range(cfg.spec_time_masks):
        w = torch.randint(0, T // 10 + 1, (B,), generator=g)
        s = torch.randint(0, T, (B,), generator=g)
        for b in range(B):
            img[b, ..., s[b]:s[b] + w[b]] = 0.0
    for _ in range(cfg.spec_freq_masks):
        w = torch.randint(0, F // 10 + 1, (B,), generator=g)
        s = torch.randint(0, F, (B,), generator=g)
        for b in range(B):
            img[b, ..., s[b]:s[b] + w[b], :] = 0.0
    return img, sc


def _gnll(mean, y, logvar):
    """Gaussian negative log-likelihood (constant dropped), elementwise."""
    return 0.5 * (logvar + (y - mean) ** 2 * torch.exp(-logvar))


def _views(T: int, crop_frac: float, n_crops: int) -> list[slice]:
    """Full clip + n_crops evenly spaced crops (test-time augmentation)."""
    v = [slice(0, T)]
    if crop_frac < 1.0 and n_crops > 0:
        L = max(8, int(round(crop_frac * T)))
        for s in np.linspace(0, T - L, n_crops).astype(int):
            v.append(slice(int(s), int(s) + L))
    return v


def predict(model, img, sc, norm: Norm, device, batch: int = 64, crop_frac: float = 1.0,
            tta_crops: int = 0, sigma_scale: dict | None = None) -> dict:
    """N mean/std (particles) and normalised property means/stds (averaged over views)."""
    model.eval()
    views = _views(img.shape[-1], crop_frac, tta_crops)
    acc = {"n": 0.0, "n_var": 0.0, "props": 0.0, "props_var": 0.0}
    with torch.no_grad():
        for v in views:
            n, nv, p, pv = [], [], [], []
            for i in range(0, len(img), batch):
                m, s = norm(img[i:i + batch, ..., v], sc[i:i + batch])
                o = model(torch.from_numpy(m).to(device), torch.from_numpy(s).to(device))
                n.append(o.n.cpu().numpy())
                nv.append(np.exp(o.n_logvar.cpu().numpy()))
                p.append(o.props.cpu().numpy())
                pv.append(np.exp(o.props_logvar.cpu().numpy()))
            cat = (lambda a: np.concatenate(a) if a else np.zeros(0))
            acc["n"] = acc["n"] + cat(n) / len(views)
            acc["n_var"] = acc["n_var"] + cat(nv) / len(views)
            acc["props"] = acc["props"] + cat(p) / len(views)
            acc["props_var"] = acc["props_var"] + cat(pv) / len(views)
    sn = (sigma_scale or {}).get("n", 1.0)
    sp = np.asarray((sigma_scale or {}).get("props", 1.0))
    return {"n": np.asarray(acc["n"]) * 100.0, "n_std": np.sqrt(acc["n_var"]) * 100.0 * sn,
            "props": np.asarray(acc["props"]), "props_std": np.sqrt(acc["props_var"]) * sp}


def metrics(pred: np.ndarray, true: np.ndarray, std: np.ndarray | None = None) -> dict:
    if not len(true):
        return {"n": 0}
    e = pred - true
    out = {"n": int(len(true)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
           "bias": float(e.mean()), "within_5": float((np.abs(e) <= 5).mean()),
           "within_10": float((np.abs(e) <= 10).mean())}
    rngs = {"0-19": (0, 19), "20-49": (20, 49), "50-79": (50, 79), "80-100": (80, 100)}
    out["by_N_range"] = {k: float(np.abs(e[(true >= lo) & (true <= hi)]).mean())
                         for k, (lo, hi) in rngs.items() if ((true >= lo) & (true <= hi)).any()}
    if std is not None and len(std):
        out.update({"mean_pred_std": float(std.mean()),
                    "in_1sigma": float((np.abs(e) <= std).mean()),      # ideal ≈ 0.68
                    "in_2sigma": float((np.abs(e) <= 2 * std).mean()),  # ideal ≈ 0.95
                    "std_err_corr": float(np.corrcoef(std, np.abs(e))[0, 1]) if len(e) > 2 else None})
    return out


def conformal_metrics(pred, true, std, conformal: dict | None) -> dict:
    """Coverage and width of normalised split-conformal intervals pred ± q·σ."""
    if not conformal or not len(true):
        return {}
    out = {}
    e = np.abs(pred - true)
    for a, q in conformal.items():
        w = q * std
        ok = e <= w
        out[a] = {"target": 1 - float(a), "coverage": float(ok.mean()), "mean_width": float(2 * w.mean())}
        for k, (lo, hi) in {"0-19": (0, 19), "20-49": (20, 49), "50-79": (50, 79), "80-100": (80, 100)}.items():
            m = (true >= lo) & (true <= hi)
            if m.any():
                out[a][f"width_{k}"] = float(2 * w[m].mean())
    return out


def real_split(cfg: TrainConfig | dict) -> tuple[list[dict], list[dict], list[dict]]:
    """(labeled adaptation clips, unlabeled adaptation clips, evaluation clips)."""
    g = (lambda k: cfg[k]) if isinstance(cfg, dict) else (lambda k: getattr(cfg, k))
    if not g("real_dir"):
        return [], [], []
    rr = D.real_recordings(g("real_dir"))
    nl, nu = g("real_train_per_N"), g("real_unlabeled_per_N")
    opt = (lambda k, d: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d: getattr(cfg, k, d))
    nx = opt("real_exclude_per_N", 0)
    lo = 5 * int(opt("real_fold", 0))      # adaptation block: clips lo+1 .. lo+5 of every N
    lab, unl, test = [], [], []
    for r in rr:
        k = int(r["run_id"].split("-")[1])
        j = k - lo
        if 1 <= j <= nl:
            lab.append(r)
        elif nl < j <= nl + nu:
            unl.append(r)
        elif (lo == 0 and k > max(nx, nl + nu)) or (lo > 0 and k > nx and not 1 <= j <= 5):
            test.append(r)
    return lab, unl, test


def calibrate(model, img, sc, y, props, mask, norm, dev, cfg) -> dict:
    """σ scale per head so that mean((error/σ)²) = 1 on the given (validation) rows."""
    P = predict(model, img, sc, norm, dev, crop_frac=cfg.crop_frac, tta_crops=cfg.tta_crops)
    zn = ((P["n"] - y * 100) / P["n_std"]) ** 2
    out = {"n": float(np.sqrt(zn.mean()))}
    # normalised split conformal: q = finite-sample quantile of |error| / σ_calibrated
    r = np.sort(np.abs(P["n"] - y * 100) / (P["n_std"] * out["n"]))
    out["conformal"] = {}
    for a in cfg.conformal_alpha:
        k = min(len(r) - 1, int(np.ceil((len(r) + 1) * (1 - a))) - 1)
        out["conformal"][str(a)] = float(r[k]) if len(r) else float("nan")
    if props.shape[1]:
        zp = ((P["props"] - props) / P["props_std"]) ** 2
        m = mask.astype(bool)
        out["props"] = [float(np.sqrt(zp[m[:, k], k].mean())) if m[:, k].any() else 1.0
                        for k in range(props.shape[1])]
    return out


# ----------------------------------------------------------------------------- training

def train(cfg: TrainConfig, log=print) -> dict:
    if cfg.model_type == "set":
        from rneml.settrain import train_set
        return train_set(cfg, log)
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    dev = _device(cfg.device)
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        commit = ""
    (out / "provenance.json").write_text(json.dumps({"git_commit": commit, "seed": cfg.seed}))

    from rneml import splits as SP

    rows = SP.assign(D.read_metadata(cfg.data_dir), cfg.split, cfg.split_seed)
    SP.check_no_leak(rows)
    split_hash = SP.save(rows, cfg.split, cfg.split_seed, out / "split.json")
    tr = SP.select_role(rows, "train")
    if cfg.train_subset == "bb_like":
        tr = D.bb_like(tr)
    va = SP.select_role(rows, "val")
    pairs = SP.select_role(rows, "pairs")
    log(f"split {cfg.split or 'default'} (hash {split_hash})")
    real_lab, real_unl, _ = real_split(cfg)
    log(f"train {len(tr)} sim clips (+{len(real_lab)} labeled real ×{cfg.real_repeat}, "
        f"{len(real_unl)} unlabeled real), val {len(va)}, pairs {len(pairs)} "
        f"({cfg.train_subset}, {cfg.feature}{', modulation' if cfg.use_modulation else ''})")

    sim_rows = tr + va + pairs
    img, sc = feats(cfg, sim_rows, Path(cfg.data_dir).name, log)
    img, sc = prepare(img, sc, cfg, sim_rows)
    y = np.array([r["N"] for r in sim_rows], np.float32) / 100.0
    props = np.stack([D.aux_targets(r) for r in sim_rows])
    if cfg.n_only:                                       # no property targets, no property head
        props = props[:, :0]
    mask = np.ones_like(props)
    mask[y == 0] = 0.0                                   # an empty box has no material
    is_real = np.zeros(len(sim_rows), bool)
    ntr, nva = len(tr), len(va)
    train_idx = np.arange(ntr)

    def append_real(rr):
        nonlocal img, sc, y, props, mask, is_real
        ri, rs = feats(cfg, rr, "real", log)
        ri, rs = prepare(ri, rs, cfg, rr)
        k0 = len(y)
        img, sc = np.concatenate([img, ri]), np.concatenate([sc, rs])
        yr = np.array([r["N"] for r in rr], np.float32) / 100.0
        y = np.concatenate([y, yr])
        rp, rm = D.real_property_targets(len(rr))
        if cfg.n_only:
            rp, rm = rp[:, :0], rm[:, :0]
        rm[yr == 0] = 0.0
        props, mask = np.concatenate([props, rp]), np.concatenate([mask, rm])
        is_real = np.concatenate([is_real, np.ones(len(rr), bool)])
        return np.arange(k0, k0 + len(rr))

    if real_lab:
        lab_idx = append_real(real_lab)
        train_idx = np.concatenate([train_idx, np.tile(lab_idx, cfg.real_repeat)])
        (out / "real_train.txt").write_text("\n".join(r["path"] for r in real_lab))
    unl_idx = append_real(real_unl) if real_unl else np.zeros(0, int)
    if len(unl_idx):
        (out / "real_unlabeled.txt").write_text("\n".join(r["path"] for r in real_unl))
    norm = Norm(img[:ntr], sc[:ntr])

    # contrast pairs: index pairs (a, b) within sim_rows, matched by pair_id and render
    by_key: dict = {}
    for k, r in enumerate(sim_rows[ntr + nva:], start=ntr + nva):
        by_key.setdefault((r["pair_id"], int(r["render"])), []).append(k)
    same_n = [tuple(v) for (pid, _), v in by_key.items() if len(v) == 2 and pid.startswith("pN")]
    same_m = [tuple(v) for (pid, _), v in by_key.items() if len(v) == 2 and pid.startswith("pM")]
    pool = [(a, b, 0) for a, b in same_n] + [(a, b, 1) for a, b in same_m]
    log(f"contrast pairs: same_N {len(same_n)}, same_material {len(same_m)}")

    in_ch = img.shape[1] if img.ndim == 4 else 1
    model = CountNet(sc.shape[1], props.shape[1], cfg.width, cfg.dropout, cfg.use_scalars,
                     cfg.time_pool, cfg.count_head, cfg.class_step, in_ch, cfg.arch, cfg.temporal).to(dev)
    use_amp = cfg.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    nb = math.ceil(len(train_idx) / cfg.batch)
    total = max(cfg.epochs * nb, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total)
    huber = torch.nn.SmoothL1Loss(beta=0.05)
    bce = torch.nn.BCEWithLogitsLoss()
    use_dann = len(unl_idx) > 0 and cfg.dann_weight > 0

    best, best_ep, hist, step = math.inf, -1, [], 0
    for ep in range(cfg.epochs):
        model.train()
        perm = rng.permutation(train_idx)
        tl, dl, t0 = 0.0, 0.0, time.time()
        for i in range(0, len(perm), cfg.batch):
            idx = perm[i:i + cfg.batch]
            pa = pb = kinds = np.zeros(0, int)
            if cfg.pairs_per_batch and pool:
                sel = rng.choice(len(pool), min(cfg.pairs_per_batch, len(pool)), replace=False)
                pa, pb, kinds = (np.array([pool[k][j] for k in sel]) for j in range(3))
            ru = rng.choice(unl_idx, min(cfg.real_unlabeled_per_batch, len(unl_idx)), replace=False) \
                if use_dann else np.zeros(0, int)
            sup = np.concatenate([idx, pa, pb]).astype(int)
            full = np.concatenate([sup, ru]).astype(int)
            m, s = norm(img[full], sc[full])
            m, s = _augment(torch.from_numpy(m), torch.from_numpy(s), cfg, norm, g)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                o = model(m.to(dev), s.to(dev))
            o = o._replace(**{k: (v.float() if torch.is_tensor(v) else v) for k, v in o._asdict().items()})
            S = len(sup)
            yt = torch.from_numpy(y[sup]).to(dev)
            n_hat, n_lv = o.n[:S], o.n_logvar[:S]
            if cfg.count_head == "regression":
                loss = huber(n_hat, yt)
                if cfg.nll_weight:
                    loss = loss + cfg.nll_weight * _gnll(n_hat.detach(), yt, n_lv).mean()
            else:
                loss = model.count.loss(o.count_logits[:S], yt, cfg.label_sigma / 100.0)
            if model.props is not None and (cfg.prop_weight or cfg.nll_weight):
                pt = torch.from_numpy(props[sup]).to(dev)
                mk = torch.from_numpy(mask[sup]).to(dev)
                denom = mk.sum().clamp(min=1.0)
                pm, plv = o.props[:S], o.props_logvar[:S]
                loss = loss + cfg.prop_weight * (((pm - pt) ** 2) * mk).sum() / denom
                if cfg.nll_weight:
                    loss = loss + cfg.nll_weight * (_gnll(pm.detach(), pt, plv) * mk).sum() / denom
            if len(pa):
                B, P = len(idx), len(pa)
                na, nb_ = n_hat[B:B + P], n_hat[B + P:S]
                kn = torch.from_numpy(kinds).to(dev)
                if (kn == 0).any() and cfg.pair_weight:
                    loss = loss + cfg.pair_weight * torch.mean((na - nb_)[kn == 0] ** 2)
                if (kn == 1).any() and cfg.rank_weight:
                    sgn = torch.sign(yt[B:B + P] - yt[B + P:S])[kn == 1]
                    loss = loss + cfg.rank_weight * torch.mean(
                        torch.relu(cfg.rank_margin - sgn * (na - nb_)[kn == 1]))
            if use_dann:
                prog = step / total
                lam = cfg.dann_weight * (2.0 / (1.0 + math.exp(-10 * prog)) - 1.0)
                dom = torch.from_numpy(is_real[full].astype(np.float32)).to(dev)
                logit = model.domain(grad_reverse(o.emb, 1.0)).squeeze(1).float()
                ld = bce(logit, dom)
                loss = loss + lam * ld
                dl += float(ld) * len(idx)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            tl += float(loss) * len(idx)
        pv = predict(model, img[ntr:ntr + nva], sc[ntr:ntr + nva], norm, dev)
        mv = metrics(pv["n"], y[ntr:ntr + nva] * 100, pv["n_std"])
        hist.append({"epoch": ep, "train_loss": tl / len(perm), "domain_loss": dl / len(perm),
                     **{f"val_{k}": v for k, v in mv.items()}, "sec": time.time() - t0})
        log(f"ep {ep:3d} loss {tl / len(perm):.4f}"
            + (f" dom {dl / len(perm):.3f}" if use_dann else "")
            + f" val MAE {mv.get('mae', float('nan')):.2f} ±5 {mv.get('within_5', float('nan')):.2f}"
            f" ({time.time() - t0:.0f}s)")
        score = mv.get("mae", math.inf)
        if score < best or nva == 0:
            best, best_ep = score, ep
            torch.save({"model": model.state_dict(), "norm": norm.state(), "n_scalars": sc.shape[1],
                        "n_props": props.shape[1], "prop_names": D.AUX_NAMES, "cfg": asdict(cfg),
                        "in_channels": in_ch, "n_audio_scalars": sc.shape[1] - len(cfg.material_input),
                        "feature_version": FEATURE_VERSION}, out / "model.pt")
        elif ep - best_ep >= cfg.patience:
            log(f"early stop (best epoch {best_ep})")
            break
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    if cfg.calibrate_sigma and nva:
        ck = torch.load(out / "model.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        vs = slice(ntr, ntr + nva)
        ck["sigma_scale"] = calibrate(model, img[vs], sc[vs], y[vs], props[vs], mask[vs], norm, dev, cfg)
        torch.save(ck, out / "model.pt")
        log(f"σ calibration factors: N {ck['sigma_scale']['n']:.2f}")
    from rneml.evaluate import evaluate_run
    return evaluate_run(out, cfg, log=log)
