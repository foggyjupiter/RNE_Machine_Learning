"""Which components of the sound tell the count apart? (no deep learning)

The sound is decomposed along two axes:

  frequency band   16 groups of 8 mel bands (≈ 50 Hz … 24 kHz)
  component type   per band:
    level          mean dB                         how loud
    variability    std of dB over time             how much it fluctuates
    impulsiveness  kurtosis of the band envelope   how spiky (ψ3 resolved in frequency)
    onset_rate     rises > 9 dB within 20 ms /s    distinct impacts heard in the band
    mod_shake      modulation at k·f_shake, k=1..4 collective motion locked to the shaker
    mod_texture    modulation 20-50 Hz             texture of the collision rate
    mod_slow       modulation 0.3-2 Hz             slow drifts
    fold_peak      max/mean of the period-folded   energy concentrated at one phase (ψ1,
                   envelope                        phase-free)
    perc_share     percussive / total energy       share of impact clicks (HPSS)
    perc_level     percussive level dB             clicks only
    harm_level     harmonic level dB               ring / hum only

For every component type (all bands), every band (all types) and every single feature,
four scores are computed with ridge regression of log(1+N) (R²; cross-validation
grouped by dynamics run for simulation, by recording for real clips):

  sim_blind     simulated clips, material unknown
  sim_known     same with the true material appended (gain = material dependence)
  real_cv       the first-study recordings alone (one material: what carries N in reality)
  transfer      fit on simulation, score on the real recordings (does the relation carry
                over? a component informative in simulation but not here is a
                simulation-only cue)

A component is useful for blind counting when sim_blind, real_cv and transfer are all
high; ``select`` keeps such features, and the selection is verified against using all
of them (simulation test / held-out material / real).

With the simulator's contrast pairs a causal measure is added (``pair_specificity``):
how much a component changes per 10 particles at fixed material, divided by how much
it changes between materials at fixed N. Impossible to measure in experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal, stats

from rneml import data as D
from rneml.features import HOP, N_MELS, SR, hpss, mel_db, power_spectrogram, read_audio

N_BANDS = 16
TYPES = ["level", "variability", "impulsiveness", "onset_rate", "mod_shake", "mod_texture",
         "mod_slow", "fold_peak", "perc_share", "perc_level", "harm_level"]
F_SHAKE = 6.2
FR = SR / HOP                                     # 100 frames per second


def band_centres() -> np.ndarray:
    from rneml.features import F_MAX, F_MIN, _hz_to_mel, _mel_to_hz
    pts = _mel_to_hz(np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2))[1:-1]
    return pts.reshape(N_BANDS, -1).mean(1)


def _group(M_lin: np.ndarray) -> np.ndarray:
    return M_lin.reshape(N_BANDS, -1, M_lin.shape[1]).sum(1)          # [16, T]


def component_features(path: str, f_shake: float = F_SHAKE) -> np.ndarray:
    """[N_BANDS, len(TYPES)] for one clip."""
    x = read_audio(path)
    P = power_spectrogram(x)
    from rneml.features import mel_filterbank
    fb = mel_filterbank()
    E = _group(fb @ P)                                                 # band energy envelopes
    H, Q = hpss(P)
    Ep, Eh = _group(fb @ Q), _group(fb @ H)
    dB = 10 * np.log10(E + 1e-12)
    out = np.zeros((N_BANDS, len(TYPES)), np.float32)
    T = E.shape[1]
    w = np.hanning(T)
    f = np.fft.rfftfreq(T, 1 / FR)
    ph = (f_shake * np.arange(T) / FR) % 1.0
    b = np.minimum((ph * 32).astype(int), 31)
    lag = int(0.02 * FR) or 1
    for g in range(N_BANDS):
        e = E[g]
        rel = e / (e.mean() + 1e-20) - 1
        M = np.abs(np.fft.rfft(rel * w)) / w.sum()
        shake = sum(M[np.argmin(np.abs(f - k * f_shake))] for k in range(1, 5))
        prof = np.bincount(b, weights=e, minlength=32) / np.maximum(np.bincount(b, minlength=32), 1)
        rise = dB[g, lag:] - dB[g, :-lag]
        on = np.flatnonzero((rise[1:] > 9) & (rise[:-1] <= 9))
        out[g] = [dB[g].mean(), dB[g].std(), np.log1p(max(stats.kurtosis(e), -0.99) + 1),
                  len(on) / (T / FR), np.log10(shake + 1e-6),
                  np.log10(M[(f >= 20) & (f <= 50)].mean() + 1e-6),
                  np.log10(M[(f >= 0.3) & (f <= 2)].mean() + 1e-6),
                  prof.max() / (prof.mean() + 1e-20),
                  Ep[g].sum() / (E[g].sum() + 1e-20),
                  10 * np.log10(Ep[g].mean() + 1e-12), 10 * np.log10(Eh[g].mean() + 1e-12)]
    return out


def compute(paths: list[str], cache: Path, workers: int = 1, log=print) -> np.ndarray:
    """Component features of paths [n, N_BANDS, len(TYPES)], cached by path (one cache file
    can hold simulated and real clips; legacy per-set caches next to it are merged)."""
    from concurrent.futures import ProcessPoolExecutor
    cache = Path(cache)
    have = {}
    for c in [cache, *sorted(cache.parent.glob("components_*.npz"))]:
        if c.exists():
            z = np.load(c, allow_pickle=False)
            have.update(zip(z["paths"].tolist(), z["X"]))
    todo = [p for p in dict.fromkeys(paths) if p not in have]
    if todo:
        log(f"component features for {len(todo)} clips ({workers} worker)")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, (p, v) in enumerate(zip(todo, ex.map(component_features, todo, chunksize=4))):
                have[p] = v
                if (k + 1) % 200 == 0:
                    log(f"  {k + 1}/{len(todo)}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, paths=np.array(list(have)), X=np.stack(list(have.values())))
    return np.stack([have[p] for p in paths])


# ---------------------------------------------------------------- scoring

def _ridge_cv_r2(X, y, groups, k=5, alpha=1.0, seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ug = np.unique(groups)
    fold = dict(zip(np.random.default_rng(seed).permutation(ug), np.arange(len(ug)) % k))
    f = np.array([fold[g] for g in groups])
    p = np.zeros_like(y)
    for j in range(k):
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[f != j], y[f != j])
        p[f == j] = m.predict(X[f == j])
    return float(1 - np.mean((p - y) ** 2) / np.var(y))


def _fit_score(Xa, ya, Xb, yb, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xa, ya)
    p = m.predict(Xb)
    return float(1 - np.mean((p - yb) ** 2) / np.var(yb)), p


def scores(Xs, ys, gs, Ms, Xr, yr, gr, Xs_ood=None, ys_ood=None) -> dict:
    """Four scores for one feature subset (columns already selected)."""
    out = {"sim_blind": _ridge_cv_r2(Xs, ys, gs),
           "sim_known": _ridge_cv_r2(np.hstack([Xs, Ms]), ys, gs),
           "real_cv": _ridge_cv_r2(Xr, yr, gr) if len(Xr) else None,
           "transfer": _fit_score(Xs, ys, Xr, yr)[0] if len(Xr) else None}
    if Xs_ood is not None and len(Xs_ood):
        out["ood"] = _fit_score(Xs, ys, Xs_ood, ys_ood)[0]
    return out


def analyse(sim_rows, Xs_all, real_rows, Xr_all, log=print) -> dict:
    """Scores per component type, per band, and per (band, type)."""
    y_s = np.log1p([r["N"] for r in sim_rows])
    g_s = np.array([str(r.get("system_id") or r["run_id"]) for r in sim_rows])
    M_s = D.material_inputs(sim_rows, ["diameter", "density", "e_pp", "e_pw", "mu_pp", "mu_pw", "E"])
    ood = np.array([r["split"] == "test_ood" for r in sim_rows])
    tr = ~ood
    y_r = np.log1p([r["N"] for r in real_rows])
    g_r = np.array([r["run_id"].split("-")[1] for r in real_rows])        # recording index
    res = {"by_type": {}, "by_band": {}, "by_cell": {}, "band_centres_hz": band_centres().tolist()}

    def sc(cols_fn):
        Xs = cols_fn(Xs_all)
        Xr = cols_fn(Xr_all)
        return scores(Xs[tr], y_s[tr], g_s[tr], M_s[tr], Xr, y_r, g_r, Xs[ood], y_s[ood])

    for t, name in enumerate(TYPES):
        res["by_type"][name] = sc(lambda X, t=t: X[:, :, t])
    for g in range(N_BANDS):
        res["by_band"][str(g)] = sc(lambda X, g=g: X[:, g, :])
    for g in range(N_BANDS):
        for t, name in enumerate(TYPES):
            res["by_cell"][f"{g}:{name}"] = sc(lambda X, g=g, t=t: X[:, g, t][:, None])
    res["all"] = sc(lambda X: X.reshape(len(X), -1))
    log("component type        sim_blind  sim_known  ood   real_cv  transfer")
    for k, v in res["by_type"].items():
        log(f"  {k:18s} {v['sim_blind']:8.2f} {v['sim_known']:9.2f} {v.get('ood', float('nan')):6.2f} "
            f"{v['real_cv']:8.2f} {v['transfer']:9.2f}")
    v = res["all"]
    log(f"  {'ALL':18s} {v['sim_blind']:8.2f} {v['sim_known']:9.2f} {v.get('ood', float('nan')):6.2f} "
        f"{v['real_cv']:8.2f} {v['transfer']:9.2f}")
    return res


def pair_specificity(sim_rows, Xs_all) -> dict | None:
    """Count specificity per component from the simulator's contrast pairs (causal: only one
    factor changes inside a pair).

      signal  = median |x_a − x_b| / |N_a − N_b| × 10   same material, different N  (per 10 particles)
      nuisance = median |x_a − x_b|                      same N, different material
      specificity = signal / nuisance

    High specificity: the component moves with the count and hardly with the material —
    what a blind count needs. Pairs are matched on the same render index (same recording
    setup draw)."""
    by = {}
    for i, r in enumerate(sim_rows):
        if r.get("pair_type"):
            by.setdefault((r["pair_id"], r.get("render"), r["pair_type"]), []).append(i)
    sig, nui = [], []
    for (pid, rend, kind), ii in by.items():
        if len(ii) != 2:
            continue
        a, b = ii
        d = np.abs(Xs_all[a] - Xs_all[b])                               # [bands, types]
        if kind == "same_material":
            dn = abs(sim_rows[a]["N"] - sim_rows[b]["N"])
            if dn >= 5:
                sig.append(d / dn * 10)
        elif kind == "same_N":
            nui.append(d)
    if len(sig) < 5 or len(nui) < 5:
        return None
    S, Nn = np.median(sig, 0), np.median(nui, 0)
    spec = S / (Nn + 1e-9)
    return {"n_pairs_signal": len(sig), "n_pairs_nuisance": len(nui),
            "by_cell": {f"{g}:{n}": {"signal_per10": float(S[g, t]), "nuisance": float(Nn[g, t]),
                                     "specificity": float(spec[g, t])}
                        for g in range(N_BANDS) for t, n in enumerate(TYPES)},
            "by_type": {n: float(np.median(spec[:, t])) for t, n in enumerate(TYPES)},
            "by_band": {str(g): float(np.median(spec[g])) for g in range(N_BANDS)}}


def select(res: dict, min_sim=0.2, min_real=0.2, min_transfer=0.0, top: int | None = 40) -> list[str]:
    """Cells that carry N in simulation (blind) and in the recordings, and transfer."""
    cells = []
    for k, v in res["by_cell"].items():
        if v["sim_blind"] >= min_sim and (v["real_cv"] or -1) >= min_real and (v["transfer"] or -9) >= min_transfer:
            cells.append((min(v["sim_blind"], v["real_cv"]) + 0.5 * max(v["transfer"], 0), k))
    cells.sort(reverse=True)
    return [k for _, k in cells[:top]] if top else [k for _, k in cells]


def verify(sim_rows, Xs_all, real_rows, Xr_all, chosen: list[str], log=print) -> dict:
    """Selected components vs all components vs level only, with a ridge and a GBM:
    MAE on simulation test (default split), held-out material, and real recordings."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    idx = {f"{g}:{n}": (g, t) for g in range(N_BANDS) for t, n in enumerate(TYPES)}
    sel = [idx[c] for c in chosen]
    subsets = {
        "all_components": lambda X: X.reshape(len(X), -1),
        "selected": lambda X: np.stack([X[:, g, t] for g, t in sel], 1),
        "level_only": lambda X: X[:, :, TYPES.index("level")],
        "no_level": lambda X: X[:, :, [i for i, n in enumerate(TYPES) if "level" not in n]].reshape(len(X), -1),
    }
    split = np.array([r["split"] for r in sim_rows])
    N_s = np.array([r["N"] for r in sim_rows], float)
    N_r = np.array([r["N"] for r in real_rows], float)
    tr = np.isin(split, ["train", "val", "pairs"])
    out = {}
    for name, fn in subsets.items():
        Xs, Xr = fn(Xs_all), fn(Xr_all)
        row = {}
        for mname, model in (("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
                             ("gbm", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                   max_leaf_nodes=15, random_state=0))):
            model.fit(Xs[tr], np.log1p(N_s[tr]))
            pr = lambda X: np.expm1(model.predict(X))
            for s in ("test", "test_ood"):
                m = split == s
                if m.any():
                    row[f"{mname}_{s}_mae"] = float(np.abs(pr(Xs[m]) - N_s[m]).mean())
            row[f"{mname}_real_mae"] = float(np.abs(pr(Xr) - N_r).mean())
        out[name] = row
    log("subset            " + "  ".join(f"{k:>16s}" for k in next(iter(out.values()))))
    for name, row in out.items():
        log(f"  {name:16s}" + "  ".join(f"{v:16.2f}" for v in row.values()))
    return out


def plot(res: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("sim_blind", "시뮬레이션: 물질 모를 때"), ("real_cv", "실측 녹음 안에서"), ("transfer", "시뮬레이션→실측")]
    fc = np.array(res["band_centres_hz"]) / 1000
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, (k, title) in zip(ax, keys):
        Mx = np.array([[res["by_cell"][f"{g}:{n}"][k] or np.nan for n in TYPES] for g in range(N_BANDS)])
        im = a.imshow(np.clip(Mx, -0.5, 1), aspect="auto", origin="lower", cmap="viridis", vmin=-0.2, vmax=0.9)
        a.set_xticks(range(len(TYPES)))
        a.set_xticklabels(TYPES, rotation=60, ha="right", fontsize=8)
        a.set_yticks(range(N_BANDS))
        a.set_yticklabels([f"{c:.1f}" for c in fc], fontsize=7)
        a.set_ylabel("band centre [kHz]")
        a.set_title(f"R² of log(1+N): {k}")
        fig.colorbar(im, ax=a)
    fig.tight_layout()
    fig.savefig(out / "component_relevance.png", dpi=110)
    plt.close(fig)


def run(data_dir: str, real_dir: str, out_dir: str = "outputs/components", cache_dir: str = "cache",
        workers: int = 1, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sim_rows = [r for r in D.read_metadata(data_dir) if r["N"] > 0]
    real_rows = [r for r in D.real_recordings(real_dir) if r["N"] > 0]
    Xs = compute([r["path"] for r in sim_rows], Path(cache_dir) / "components.npz", workers, log)
    Xr = compute([r["path"] for r in real_rows], Path(cache_dir) / "components.npz", workers, log)
    res = analyse(sim_rows, Xs, real_rows, Xr, log)
    ps = pair_specificity(sim_rows, Xs)
    if ps:
        res["pair_specificity"] = ps
        log("count specificity (contrast pairs; change per 10 particles / change from material):")
        for k, v in sorted(ps["by_type"].items(), key=lambda kv: -kv[1]):
            log(f"  {k:18s} {v:6.2f}")
    else:
        log("contrast pairs not (yet) in the data: specificity skipped")
    chosen = select(res)
    log(f"selected {len(chosen)} components: {chosen[:12]} …")
    res["selected"] = chosen
    res["verification"] = verify(sim_rows, Xs, real_rows, Xr, chosen, log)
    res["n_sim_clips"], res["n_real_clips"] = len(sim_rows), len(real_rows)
    (out / "components.json").write_text(json.dumps(res, indent=1))
    plot(res, out)
    return res


def selected_columns(path: str | Path) -> list[tuple[int, int]]:
    """(band, type) indices of the components chosen by a previous ``run``."""
    res = json.loads(Path(path).read_text())
    idx = {f"{g}:{n}": (g, t) for g in range(N_BANDS) for t, n in enumerate(TYPES)}
    return [idx[c] for c in res["selected"]]


def scalar_block(rows: list[dict], selection: str, cache_dir: str, workers: int = 1, log=print) -> np.ndarray:
    """Component features to append to the model's scalar branch: ``selection`` = path of a
    components.json (its selected cells) or 'all'."""
    X = compute([r["path"] for r in rows], Path(cache_dir) / "components.npz", workers, log)
    if selection == "all":
        return X.reshape(len(X), -1).astype(np.float32)
    cols = selected_columns(selection)
    return np.stack([X[:, g, t] for g, t in cols], 1).astype(np.float32)
