"""Audio → model inputs.

Preprocessing that focuses the model on particular components (training config):

* ``denoise: true`` — spectral subtraction of the stationary motor/shaker noise, as in the
  first study: the mean power spectrum of empty-box recordings (``noise_ref``, default the
  N = 0 clips of ``real_dir``) is subtracted with a floor, P' = max(P − N̄, 0.05 P). Applied
  identically to simulated and real clips.
* multi-channel images, ``feature: "mel+perc+harm"`` — harmonic/percussive separation
  (median filtering of the power spectrogram along time vs frequency, soft masks):
  ``perc`` holds the impact clicks (vertical lines: when and how often particles hit),
  ``harm`` the sustained ring of the box modes and room and the motor hum (horizontal
  lines: the material/box timbre and the background). Separating them lets the model
  weigh "how many impacts" and "what rings" independently — and the click-to-ring
  balance is where simulation and recordings differ most.
* ``band_hz: [lo, hi]`` — keep only mel bands with centres in this range.

Images (``feature`` in the training config):

* ``mel``: log-mel spectrogram at the native 48 kHz, mel bands up to 24 kHz.
* ``mel22k``: the first-study pipeline's view — resampled to 22.05 kHz, mel up to 11 kHz
  (ablation: what the 11-24 kHz band adds). Scalar features still use the full band.
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
  spectral entropy (ψ2), period-folded energy peakiness at the clip's own (estimated)
  shaking frequency,
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

# The shaker's speed is estimated from each clip (not assumed): the first-study recordings
# drift between 6.04 and 6.64 Hz at a nominal 6.2 Hz (N = 95/100 were recorded at ≈ 6.5 Hz),
# and folding or binning at a fixed 6.2 Hz smears exactly those clips. Search range covers
# the first study and the multi-rpm dataset (300-650 rpm).
SHAKE_RANGE_HZ = (4.5, 11.5)
# Modulation spectrum: how the energy of each band fluctuates in time, in bins placed
# relative to the clip's own shaking frequency f0 (so speed drift does not move energy
# between bins): peaks at the harmonics k·f0 (collective motion locked to the shaker),
# at f0/2 (period doubling), the floor between harmonics (unlocked collisions), and the
# absolute slow / texture bands. Relative (level-free) and shift-invariant, so it needs no
# phase reference and is unaffected by microphone gain or particle mass scaling the level.
MOD_GROUPS = 16                                  # mel bands grouped 8 by 8
MOD_BINS = ["h1", "h2", "h3", "h4", "sub", "ih1", "ih2", "ih3", "below", "texture", "slow", "broad"]
MOD_NAMES = [f"mod_g{g:02d}_{b}" for g in range(MOD_GROUPS) for b in MOD_BINS]
# Spectral kurtosis per band group: the first study's waveform kurtosis ψ3 resolved in
# frequency (which bands are impulsive), from the band power envelopes.
SK_GROUPS = 16
SK_NAMES = [f"sk_g{g:02d}" for g in range(SK_GROUPS)]
SCALAR_NAMES = BASIC_SCALARS + MOD_NAMES + SK_NAMES
FEATURE_VERSION = 4                              # bump when the cached features change
MULTI_CHANNELS = ("mel", "perc", "harm")
HPSS_KERNEL = 17                                 # 170 ms along time / ≈ 400 Hz along frequency


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


def mel_db(P: np.ndarray, sr: int = SR) -> np.ndarray:
    """Power spectrogram → [N_MELS, frames] dB re 1 (fixed reference), floored."""
    M = mel_filterbank(sr) @ P
    return np.maximum(10 * np.log10(M + 1e-12), DB_FLOOR).astype(np.float32)


def log_mel(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """[N_MELS, frames] in dB re 1 (fixed reference), floored at DB_FLOOR."""
    return mel_db(power_spectrogram(x, sr), sr)


def denoise_power(P: np.ndarray, noise_psd: np.ndarray, over: float = 1.0,
                  floor: float = 0.05) -> np.ndarray:
    """Spectral subtraction of a stationary noise power spectrum (per STFT bin)."""
    return np.maximum(P - over * noise_psd[:, None], floor * P).astype(np.float32)


def noise_psd_from(files: list[str]) -> np.ndarray:
    """Mean STFT power spectrum of noise-only recordings (empty box)."""
    acc = None
    for f in files:
        P = power_spectrogram(read_audio(f))
        m = P.mean(1)
        acc = m if acc is None else acc + m
    return (acc / max(len(files), 1)).astype(np.float32)


def hpss(P: np.ndarray, k: int = HPSS_KERNEL) -> tuple[np.ndarray, np.ndarray]:
    """(harmonic, percussive) power by median filtering along time / frequency, soft masks."""
    from scipy.ndimage import median_filter
    H = median_filter(P, size=(1, k))
    Q = median_filter(P, size=(k, 1))
    Mp = Q ** 2 / (Q ** 2 + H ** 2 + 1e-30)
    return ((1 - Mp) * P).astype(np.float32), (Mp * P).astype(np.float32)


def spectral_kurtosis(S_db: np.ndarray) -> np.ndarray:
    """Excess kurtosis over time of the band-group power envelopes [SK_GROUPS]."""
    P = 10 ** (S_db / 10)
    g = P[: (P.shape[0] // SK_GROUPS) * SK_GROUPS].reshape(SK_GROUPS, -1, P.shape[1]).sum(1)
    k = stats.kurtosis(g, axis=1)
    return np.sign(k) * np.log1p(np.abs(k))          # compress heavy tails


def band_rows(lo: float, hi: float) -> np.ndarray:
    """Indices of mel rows whose centre frequency lies in [lo, hi] Hz."""
    pts = _mel_to_hz(np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2))[1:-1]
    return np.flatnonzero((pts >= lo) & (pts <= hi))


def log_mel_22k(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """First-study style image: resampled to 22.05 kHz (mel up to 11 kHz), same 10 ms frames
    and 128 bands. For the ablation 'does the 11-24 kHz band help?'."""
    y = signal.resample_poly(x, 147, 320).astype(np.float32)        # 48000 → 22050
    sr2, nfft, hop = 22050, 1024, 220                                 # ≈ 46 ms window, 10 ms hop
    _, _, Z = signal.stft(y, fs=sr2, window="hann", nperseg=nfft, noverlap=nfft - hop,
                          boundary="even", padded=True)
    fb = mel_filterbank(sr2, nfft, N_MELS, F_MIN, sr2 / 2)
    M = fb @ (np.abs(Z) ** 2).astype(np.float32)
    S = np.maximum(10 * np.log10(M + 1e-12), DB_FLOOR).astype(np.float32)
    return S[:, : int(len(x) / sr / 0.01) + 1]


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


def shake_frequency(env: np.ndarray, rate: float, lo: float = SHAKE_RANGE_HZ[0],
                    hi: float = SHAKE_RANGE_HZ[1], n_harm: int = 3) -> float:
    """Shaking frequency [Hz] from the periodicity of an energy envelope sampled at ``rate``.

    Harmonic sum of the zero-padded envelope spectrum over [lo, hi]; if the winner's half
    also carries a clear peak (a harmonic picked as fundamental, or period doubling), the
    half is returned.
    """
    e = np.asarray(env, np.float64)
    if e.size < 8 or not np.isfinite(e).all() or e.std() == 0:
        return float(F_SHAKE)
    e = (e - e.mean()) * np.hanning(e.size)
    n = 1 << int(np.ceil(np.log2(e.size * 32)))
    S = np.abs(np.fft.rfft(e, n))
    f = np.fft.rfftfreq(n, 1 / rate)
    cand = f[(f >= lo) & (f <= hi)]
    score = sum(np.interp(h * cand, f, S, right=0.0) for h in range(1, n_harm + 1))
    f0 = float(cand[int(np.argmax(score))])
    m = (f > 0.485 * f0) & (f < 0.515 * f0) & (f >= lo)
    if m.any() and S[m].max() > 0.5 * np.interp(f0, f, S):
        f0 = float(f[m][int(np.argmax(S[m]))])
    return f0


def frame_energy(S_db: np.ndarray) -> np.ndarray:
    return (10 ** (S_db.astype(np.float64) / 10)).sum(0)


def scalar_features(x: np.ndarray, S_db: np.ndarray, sr: int = SR,
                    f_shake: float | None = None) -> np.ndarray:
    """BASIC_SCALARS. f_shake None → the clip's own shaking frequency (estimated)."""
    if f_shake is None:
        f_shake = shake_frequency(frame_energy(S_db), sr / HOP)
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


def modulation_features(S_db: np.ndarray, frame_rate: float = SR / HOP,
                        f_shake: float | None = None) -> np.ndarray:
    """[MOD_GROUPS × len(MOD_BINS)] log modulation-spectrum magnitudes of the band
    envelopes, binned relative to the shaking frequency (estimated if None)."""
    P = 10 ** (S_db.astype(np.float64) / 10)
    if f_shake is None:
        f_shake = shake_frequency(P.sum(0), frame_rate)
    f0 = float(f_shake)
    g = P[: (P.shape[0] // MOD_GROUPS) * MOD_GROUPS].reshape(MOD_GROUPS, -1, P.shape[1]).sum(1)
    g = g / (g.mean(1, keepdims=True) + 1e-20) - 1.0          # relative fluctuation
    w = np.hanning(g.shape[1])
    n = 1 << int(np.ceil(np.log2(g.shape[1] * 4)))            # 4× zero padding for the peaks
    M = np.abs(np.fft.rfft(g * w, n, axis=1)) / w.sum()
    f = np.fft.rfftfreq(n, 1 / frame_rate)
    nyq = frame_rate / 2
    half_w = max(0.4, 0.06 * f0)                              # ≥ Hann main-lobe half width (5 s)

    def peak(fc):
        m = (f >= fc - half_w) & (f <= fc + half_w)
        return M[:, m].max(1) if (m.any() and fc + half_w < nyq) else np.zeros(MOD_GROUPS)

    def floor(lo, hi):
        m = (f >= lo) & (f < min(hi, nyq))
        return np.median(M[:, m], 1) if m.sum() > 1 else np.zeros(MOD_GROUPS)

    cols = [peak(k * f0) for k in (1, 2, 3, 4)] + [peak(0.5 * f0)]
    cols += [floor((k + 0.25) * f0, (k + 0.75) * f0) for k in (1, 2, 3)]
    cols += [floor(0.15 * f0, 0.4 * f0), floor(4.5 * f0, nyq), floor(0.3, 1.5), floor(1.0, nyq)]
    out = np.stack(cols, 1).astype(np.float32)
    return np.log10(out + 1e-6).ravel()


def extract(path: str | Path, kind: str = "mel", gain: float = 1.0,
            noise_psd_file: str | None = None, band: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(time-frequency image(s) in dB, scalar features).

    ``kind``: 'mel' | 'mel22k' | 'scalogram' | '+'-joined channels of 'mel', 'perc', 'harm'
    (→ image [C, F, T]). ``gain`` ≠ 1 simulates another microphone gain on the file (with
    device clipping at ±1). ``noise_psd_file``: spectral subtraction before everything
    mel-based. ``band``: keep mel rows in [lo, hi] Hz. Scalars use the (denoised) log-mel
    and the raw waveform.
    """
    x = read_audio(path)
    if gain != 1.0:
        x = np.clip(gain * x, -1.0, 1.0).astype(np.float32)
    P = power_spectrogram(x)
    if noise_psd_file:
        P = denoise_power(P, np.load(noise_psd_file))
    S = mel_db(P)
    if "+" in kind or kind in ("perc", "harm"):
        chans = kind.split("+")
        bad = [c for c in chans if c not in MULTI_CHANNELS]
        if bad:
            raise ValueError(f"unknown channels {bad}; use {MULTI_CHANNELS}")
        Ph, Pp = hpss(P) if ("perc" in chans or "harm" in chans) else (None, None)
        img = np.stack([{"mel": S, "perc": mel_db(Pp) if Pp is not None else None,
                         "harm": mel_db(Ph) if Ph is not None else None}[c] for c in chans])
        if len(chans) == 1:
            img = img[0]
    else:
        img = {"mel": lambda: S, "scalogram": lambda: scalogram(x), "mel22k": lambda: log_mel_22k(x)}[kind]()
    if band is not None and kind not in ("scalogram", "mel22k"):
        rows = band_rows(*band)
        img = img[..., rows, :]
    f0 = shake_frequency(frame_energy(S), SR / HOP)
    sc = np.concatenate([scalar_features(x, S, f_shake=f0), modulation_features(S, f_shake=f0),
                         spectral_kurtosis(S)])
    return img, sc.astype(np.float32)


def build_cache(paths: list[str], out: str | Path, workers: int = 4, log=print,
                kind: str = "mel", gain: float = 1.0, noise_psd_file: str | None = None,
                band: tuple | None = None) -> Path:
    """Compute features for all paths into one .npz (image as float16)."""
    from concurrent.futures import ProcessPoolExecutor

    if not (kind in ("mel", "scalogram", "mel22k") or all(c in MULTI_CHANNELS for c in kind.split("+"))):
        raise ValueError("feature must be mel, scalogram, mel22k or '+'-joined mel/perc/harm")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mels, scal = [None] * len(paths), [None] * len(paths)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        n = len(paths)
        for k, (S, s) in enumerate(ex.map(extract, paths, [kind] * n, [gain] * n,
                                          [noise_psd_file] * n, [band] * n, chunksize=8)):
            mels[k], scal[k] = S.astype(np.float16), s
            if (k + 1) % 500 == 0:
                log(f"features {k + 1}/{len(paths)}")
    np.savez(out, mel=np.stack(mels), scalars=np.stack(scal), paths=np.array(paths))
    return out
