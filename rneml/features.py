"""Audio → model inputs.

Two time-frequency images are available (``feature`` in the training config):

* ``mel``: log-mel spectrogram at the native 48 kHz, mel bands up to 24 kHz.
* ``scalogram``: Morlet wavelet scalogram (constant Q, 64 log-spaced bands 100 Hz-22 kHz),
  energy averaged into the same 10 ms frames. The first study found the wavelet
  scalogram best for this signal (short repeated transients) — there at 8 kHz, here
  over the full band.

Details of the mel image: Most impact
  energy lies at 8-24 kHz (box modes ≈ 9-10 kHz), which the first study discarded by
  resampling to 22.05 kHz.
* dB with a *fixed* reference (not per-clip max), so the absolute level is kept, as in
  the first study. Level is informative of N but confounded with particle mass and
  microphone gain; the training-time gain augmentation (train.py) controls how much
  the model may rely on it.
* Scalar physics features (first-study fusion idea, extended): level statistics,
  spectral flux, active-frame ratio, waveform kurtosis (order parameter ψ3),
  spectral entropy (ψ2), period-folded energy peakiness at the shaking frequency,
  spectral centroid, high-band onset rate.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal, stats
from scipy.io import wavfile

SR = 48000
N_FFT = 2048
HOP = 480            # 10 ms
N_MELS = 128
F_MIN, F_MAX = 50.0, 24000.0
DB_FLOOR = -100.0
CLIP_S = 5.0
F_SHAKE = 6.2        # first-study working speed (372 rpm)

BASIC_SCALARS = ["db_mean", "db_std", "db_max", "flux", "active_ratio", "kurtosis",
                 "entropy", "fold_peak", "centroid_khz", "onset_rate", "rms_db", "clip_frac"]

# Modulation spectrum: how the energy of each band fluctuates in time. The shaking
# frequency (6.2 Hz) and its harmonics, and the texture of collision rates, show up as
# modulation peaks. Relative (level-free) and shift-invariant, so it needs no phase
# reference and is unaffected by microphone gain or particle mass scaling the level.
MOD_GROUPS = 16                                  # mel bands grouped 8 by 8
MOD_EDGES = np.geomspace(0.8, 50.0, 13)          # 12 modulation bins [Hz] (frame rate 100 Hz)
MOD_NAMES = [f"mod_g{g:02d}_b{b:02d}" for g in range(MOD_GROUPS) for b in range(len(MOD_EDGES) - 1)]
SCALAR_NAMES = BASIC_SCALARS + MOD_NAMES
FEATURE_VERSION = 2                              # bump when the cached features change


def read_audio(path: str | Path, sr: int = SR, duration: float = CLIP_S) -> np.ndarray:
    """Mono float in [-1, 1], resampled to sr, centre-cropped / zero-padded to duration."""
    fs, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max + 1)
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != sr:
        g = np.gcd(int(fs), int(sr))
        x = signal.resample_poly(x, sr // g, fs // g).astype(np.float32)
    n = int(duration * sr)
    if len(x) >= n:
        s = (len(x) - n) // 2
        return x[s:s + n]
    return np.pad(x, (0, n - len(x)))


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


@lru_cache(maxsize=4)
def mel_filterbank(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = F_MIN, fmax: float = F_MAX) -> np.ndarray:
    """Triangular, area-normalised (Slaney style) mel filterbank [n_mels, n_fft//2+1]."""
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(min(fmax, sr / 2)), n_mels + 2))
    fb = np.zeros((n_mels, f.size), np.float32)
    for m in range(n_mels):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        up = (f - lo) / (c - lo)
        down = (hi - f) / (hi - c)
        fb[m] = np.maximum(0.0, np.minimum(up, down)) * (2.0 / (hi - lo))
    return fb


def power_spectrogram(x: np.ndarray, sr: int = SR) -> np.ndarray:
    _, _, Z = signal.stft(x, fs=sr, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP,
                          boundary="even", padded=True)
    return (np.abs(Z) ** 2).astype(np.float32)


def log_mel(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """[N_MELS, frames] in dB re 1 (fixed reference), floored at DB_FLOOR."""
    P = power_spectrogram(x, sr)
    M = mel_filterbank(sr) @ P
    return np.maximum(10 * np.log10(M + 1e-12), DB_FLOOR).astype(np.float32)


SCALO_BANDS = 64
SCALO_FMIN, SCALO_FMAX = 100.0, 22000.0
SCALO_Q = 6.0          # centre frequency / bandwidth (Gaussian, σ_f = fc / (2 Q))


@lru_cache(maxsize=4)
def _morlet_bank(n: int, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """Frequency-domain Gaussian (Morlet) filters on the positive half, unit peak gain."""
    fc = np.geomspace(SCALO_FMIN, min(SCALO_FMAX, 0.46 * sr), SCALO_BANDS)
    f = np.fft.rfftfreq(n, 1 / sr)
    sig = fc / (2 * SCALO_Q)
    return fc, np.exp(-0.5 * ((f[None, :] - fc[:, None]) / sig[:, None]) ** 2).astype(np.float32)


def scalogram(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """[SCALO_BANDS, frames] wavelet energy in dB re 1, frames of HOP samples."""
    n = len(x)
    X = np.fft.rfft(x)
    _, bank = _morlet_bank(n, sr)
    frames = n // HOP
    out = np.empty((SCALO_BANDS, frames), np.float32)
    full = np.zeros(n, np.complex128)
    for b in range(SCALO_BANDS):
        full[:X.size] = 2 * X * bank[b]        # analytic signal: positive frequencies only
        full[X.size:] = 0
        env2 = np.abs(np.fft.ifft(full)) ** 2
        out[b] = env2[:frames * HOP].reshape(frames, HOP).mean(1)
    return np.maximum(10 * np.log10(out + 1e-12), DB_FLOOR).astype(np.float32)


def scalar_features(x: np.ndarray, S_db: np.ndarray, sr: int = SR,
                    f_shake: float = F_SHAKE) -> np.ndarray:
    frame_db = S_db.mean(axis=0)
    flux = np.mean(np.abs(np.diff(S_db, axis=1)))
    active = np.mean(frame_db > frame_db.mean())
    P = np.abs(np.fft.rfft(x)) ** 2
    p = P / max(P.sum(), 1e-20)
    nz = p[p > 0]
    entropy = 1.0 - (-(nz * np.log(nz)).sum() / np.log(len(p)))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    centroid = (f * p).sum() / 1000.0
    # period-folded energy peakiness (phase-free)
    ph = (f_shake * np.arange(len(x)) / sr) % 1.0
    b = np.minimum((ph * 64).astype(int), 63)
    e = np.bincount(b, weights=x.astype(np.float64) ** 2, minlength=64)
    c = np.bincount(b, minlength=64)
    prof = e / np.maximum(c, 1)
    fold_peak = prof.max() / max(prof.mean(), 1e-20)
    # high-band (> 12 kHz) onset rate: broadband clicks vs the 9-10 kHz ring
    bb, aa = signal.butter(4, 12000 / (sr / 2), "high")
    y = signal.lfilter(bb, aa, x)
    env = np.sqrt(np.maximum(signal.lfilter(*signal.butter(2, 1000 / (sr / 2)), y * y), 0))
    lag = int(0.001 * sr)
    r = 20 * np.log10((env[lag:] + 1e-5) / (env[:-lag] + 1e-5))
    on = np.flatnonzero((r[1:] > 15) & (r[:-1] <= 15))
    on = on[env[on + lag] > 3 * np.median(env)] if on.size else on
    n_on = 0
    last = -10 ** 9
    for o in on:
        if o - last > 0.003 * sr:
            n_on += 1
            last = o
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2))
    return np.array([frame_db.mean(), S_db.std(), S_db.max(), flux, active,
                     stats.kurtosis(x), entropy, fold_peak, centroid, n_on / (len(x) / sr),
                     20 * np.log10(rms + 1e-9), np.mean(np.abs(x) >= 0.99)], np.float32)


def modulation_features(S_db: np.ndarray, frame_rate: float = SR / HOP) -> np.ndarray:
    """[MOD_GROUPS × 12] log modulation-spectrum magnitudes of the band envelopes."""
    P = 10 ** (S_db / 10)
    g = P[: (P.shape[0] // MOD_GROUPS) * MOD_GROUPS].reshape(MOD_GROUPS, -1, P.shape[1]).sum(1)
    g = g / (g.mean(1, keepdims=True) + 1e-20) - 1.0          # relative fluctuation
    w = np.hanning(g.shape[1])
    M = np.abs(np.fft.rfft(g * w, axis=1)) / w.sum()
    f = np.fft.rfftfreq(g.shape[1], 1 / frame_rate)
    out = np.empty((MOD_GROUPS, len(MOD_EDGES) - 1), np.float32)
    for b in range(len(MOD_EDGES) - 1):
        m = (f >= MOD_EDGES[b]) & (f < MOD_EDGES[b + 1])
        out[:, b] = M[:, m].max(1) if m.any() else 0.0   # peaks (shaking harmonics) kept
    return np.log10(out + 1e-6).ravel()


def extract(path: str | Path, kind: str = "mel") -> tuple[np.ndarray, np.ndarray]:
    """(time-frequency image in dB, scalar features). Scalars always use the log-mel."""
    x = read_audio(path)
    S = log_mel(x)
    img = S if kind == "mel" else scalogram(x)
    return img, np.concatenate([scalar_features(x, S), modulation_features(S)])


def build_cache(paths: list[str], out: str | Path, workers: int = 4, log=print,
                kind: str = "mel") -> Path:
    """Compute features for all paths into one .npz (image as float16)."""
    from concurrent.futures import ProcessPoolExecutor

    if kind not in ("mel", "scalogram"):
        raise ValueError("feature must be 'mel' or 'scalogram'")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mels, scal = [None] * len(paths), [None] * len(paths)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (S, s) in enumerate(ex.map(extract, paths, [kind] * len(paths), chunksize=8)):
            mels[k], scal[k] = S.astype(np.float16), s
            if (k + 1) % 500 == 0:
                log(f"features {k + 1}/{len(paths)}")
    np.savez(out, mel=np.stack(mels), scalars=np.stack(scal), paths=np.array(paths))
    return out
