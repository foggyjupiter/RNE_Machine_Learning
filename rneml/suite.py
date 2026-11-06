"""Experiment suite for the report: resumable, priority-ordered, one results table.

Each experiment = a base config + overrides (+ the data it needs). Finished experiments
(outputs/<name>/metrics.json present) are skipped, so the suite can be stopped and
restarted at any time. ``report()`` collects every finished experiment into
outputs/REPORT.md, grouped by the question it answers.

Data sets: ``v1`` (single condition, material varied), ``v1_r2`` (the same v1 dynamics
re-rendered with a per-impact radiation spread), ``bb_fixed`` (first-study BB only),
``multi_rpm`` (v2, each system at five shaker speeds).
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

ALL_PROPS = ["diameter", "density", "mass", "e_pp", "e_pw", "mu_pp", "mu_pw", "E"]

# size presets (``--size``): light CNN for a CPU, full ResNet for a GPU
SIZES = {
    "cpu": {"arch": "cnn", "width": 16, "time_pool": 2, "epochs": 30, "patience": 8},
    "default": {},
    "full": {"arch": "resnet", "width": 32, "time_pool": 1, "temporal": "transformer",
             "epochs": 80, "patience": 15},
}
FULL = SIZES["full"]

# (priority, name, group, data, base config, overrides)
EXPERIMENTS = [
    # A. material knowledge — the central question
    (1, "A1_blind", "A", "v1", "configs/blind.yaml", {}),
    (1, "A2_known_size_weight", "A", "v1", "configs/blind.yaml",
     {"material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}),
    (1, "A3_known_all", "A", "v1", "configs/blind.yaml", {"material_input": ALL_PROPS, "prop_weight": 0.0}),
    (2, "A4_first_study_repro", "A", "bb_fixed", "configs/first_study_repro.yaml", {}),
    (2, "A5_bb_trained", "A", "bb_fixed", "configs/bb_baseline.yaml", {}),
    # F. non-deep baselines (fast)
    *[(1, f"F_{k}", "F", "v1", "configs/blind.yaml", {"baseline": k})
      for k in ("rms", "onset", "order_params", "ridge_all", "gbm_physics", "gbm_all")],
    # C. training components
    (2, "C1_dann", "C", "v1", "configs/blind.yaml", {"real_unlabeled_per_N": 5}),
    (3, "C2_no_property_heads", "C", "v1", "configs/blind.yaml", {"prop_weight": 0.0}),
    (3, "C3_no_pair_losses", "C", "v1", "configs/blind.yaml", {"pair_weight": 0.0, "rank_weight": 0.0}),
    (4, "C4_level_free", "C", "v1", "configs/blind.yaml", {"level_free": True}),
    (4, "C5_no_gain_augmentation", "C", "v1", "configs/blind.yaml", {"gain_db": 0.0}),
    # B. input representation / preprocessing
    (2, "B1_hpss_channels", "B", "v1", "configs/blind.yaml", {"feature": "mel+perc+harm"}),
    (2, "B2_denoise", "B", "v1", "configs/blind.yaml", {"denoise": True}),
    (3, "B3_mel22k", "B", "v1", "configs/blind.yaml", {"feature": "mel22k"}),
    (3, "B4_scalogram", "B", "v1", "configs/blind.yaml", {"feature": "scalogram"}),
    (4, "B5_band_6k_24k", "B", "v1", "configs/blind.yaml", {"band_hz": [6000, 24000]}),
    (4, "B6_basic_scalars", "B", "v1", "configs/blind.yaml", {"scalar_set": "basic"}),
    (4, "B7_no_scalars", "B", "v1", "configs/blind.yaml", {"use_scalars": False}),
    # selection used real clips 1-5 per N → those are never scored (real_exclude_per_N)
    (2, "B9_selected_components", "B", "v1", "configs/blind.yaml",
     {"component_scalars": "outputs/components/components.json", "real_exclude_per_N": 5}),
    # transfer-safe input: only the >8 kHz image and the selected components (no shaker-locked
    # modulation, period folding or onset counts — simulation-only cues in the analysis)
    (2, "B11_transfer_safe", "B", "v1", "configs/blind.yaml",
     {"band_hz": [8000, 24000], "scalar_set": "none",
      "component_scalars": "outputs/components/components.json", "real_exclude_per_N": 5}),
    (3, "B12_transfer_safe_dann", "B", "v1", "configs/blind.yaml",
     {"band_hz": [8000, 24000], "scalar_set": "none", "real_unlabeled_per_N": 5,
      "component_scalars": "outputs/components/components.json", "real_exclude_per_N": 5}),
    (3, "B10_all_components", "B", "v1", "configs/blind.yaml",
     {"component_scalars": "all", "real_exclude_per_N": 5}),
    (3, "B8_hpss_denoise_dann", "B", "v1", "configs/blind.yaml",
     {"feature": "mel+perc+harm", "denoise": True, "real_unlabeled_per_N": 5}),
    # E. generalisation splits
    (3, "E1_extrap_restitution", "E", "v1", "configs/blind.yaml",
     {"split": {"kind": "material_extrap", "axis": "restitution_pw", "lo": 0.55, "hi": 0.75}}),
    (3, "E2_hole_diameter", "E", "v1", "configs/blind.yaml",
     {"split": {"kind": "material_band", "axis": "particle_diameter_m", "lo": 5.5e-3, "hi": 6.5e-3}}),
    (4, "E3_extrap_count", "E", "v1", "configs/blind.yaml",
     {"split": {"kind": "count_extrap", "n_train_max": 70, "n_test_min": 80}}),
    # D. count output design on a sparse count grid
    *[(4, f"D_{h}", "D", "v1", "configs/blind.yaml",
       {"count_head": h, "split": {"kind": "count_sparse", "step": 10, "halfwidth": 1}})
      for h in ("regression", "classification", "soft_label", "coral")],
    # G. few labelled real recordings
    *[(5, f"G_real_labeled_{k}", "G", "v1", "configs/blind.yaml", {"real_train_per_N": k})
      for k in (1, 2, 5)],
    # H. multi-condition (v2)
    (2, "H1_set_per_set", "H", "multi_rpm", "configs/set_multi_rpm.yaml", {}),
    (3, "H2_set_absolute", "H", "multi_rpm", "configs/set_multi_rpm.yaml", {"norm_mode": "absolute"}),
    (3, "H3_set_per_clip", "H", "multi_rpm", "configs/set_multi_rpm.yaml", {"norm_mode": "per_clip"}),
    (3, "H4_set_diff_only", "H", "multi_rpm", "configs/set_multi_rpm.yaml", {"diff_path": "diff"}),
    (3, "H5_set_no_diff", "H", "multi_rpm", "configs/set_multi_rpm.yaml", {"diff_path": "none"}),
    (4, "H6_set_ordered", "H", "multi_rpm", "configs/set_multi_rpm.yaml", {"agg": "ordered"}),
    (4, "H7_set_no_condition_value", "H", "multi_rpm", "configs/set_multi_rpm.yaml", {"use_condition": False}),
    (3, "H8_set_known_size_weight", "H", "multi_rpm", "configs/set_multi_rpm.yaml",
     {"material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}),
    # Z. full-size models (ResNet + SE + temporal transformer) of the key comparisons —
    # meant for a GPU (on a 4-core CPU one run takes many hours)
    *[(6, f"Z{i}_full_{n}", "Z", d, b, {**FULL, **o}) for i, (n, d, b, o) in enumerate([
        ("blind", "v1", "configs/blind.yaml", {}),
        ("known_size_weight", "v1", "configs/blind.yaml",
         {"material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}),
        ("known_all", "v1", "configs/blind.yaml", {"material_input": ALL_PROPS, "prop_weight": 0.0}),
        ("hpss_denoise_dann", "v1", "configs/blind.yaml",
         {"feature": "mel+perc+harm", "denoise": True, "real_unlabeled_per_N": 5}),
        ("set_per_set", "multi_rpm", "configs/set_multi_rpm.yaml", {}),
        ("set_known_size_weight", "multi_rpm", "configs/set_multi_rpm.yaml",
         {"material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}),
    ], start=1)],
    # R. acoustic domain randomisation: the same v1 dynamics re-rendered (gransim rerender)
    # with an impact-to-impact radiation spread σ ~ U(0, 1.5), level-neutral — the recorded
    # high-band envelope distribution matches σ ≈ 1.2, the original v1 audio has σ = 0.
    # Compare with A1 / A2 / B11 / B8 (same dynamics, same splits) on the real recordings.
    (2, "R1_r2_blind", "R", "v1_r2", "configs/blind.yaml", {}),
    (2, "R2_r2_known_size_weight", "R", "v1_r2", "configs/blind.yaml",
     {"material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}),
    (3, "R3_r2_transfer_safe", "R", "v1_r2", "configs/blind.yaml",
     {"band_hz": [8000, 24000], "scalar_set": "none",
      "component_scalars": "outputs/components/components.json", "real_exclude_per_N": 5}),
    (3, "R4_r2_hpss_denoise_dann", "R", "v1_r2", "configs/blind.yaml",
     {"feature": "mel+perc+harm", "denoise": True, "real_unlabeled_per_N": 5}),
    # S. repeated seeds of the key comparisons
    *[(5, f"S_A1_blind_s{s}", "S", "v1", "configs/blind.yaml", {"seed": s}) for s in (1, 2)],
    *[(5, f"S_A2_known_size_weight_s{s}", "S", "v1", "configs/blind.yaml",
       {"seed": s, "material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}) for s in (1, 2)],
]

GROUP_TITLES = {
    "A": "A. 물질을 알려 주면 얼마나 좋아지나 (블라인드 vs 크기·무게 앎 vs 전부 앎 vs 물질 고정 학습)",
    "F": "F. 딥러닝 없는 기준선 (같은 분할·같은 지표)",
    "B": "B. 입력 표현과 전처리",
    "C": "C. 학습 구성 요소 제거",
    "E": "E. 일반화 분할 (물성 외삽·구멍, 개수 외삽)",
    "D": "D. 개수 출력 방식 (학습에 없던 개수: count_sparse 분할)",
    "G": "G. 실측 녹음 몇 개에 정답을 붙이면",
    "H": "H. 다중 조건 (5개 rpm) 모델",
    "Z": "Z. 큰 모델 (ResNet + SE + 시간축 트랜스포머)",
    "R": "R. 음향 도메인 무작위화 (같은 동역학, 충돌마다 방사 효율이 다른 소리로 다시 합성한 v1)",
    "S": "S. 시드 반복",
}


def run_suite(paths: dict, real_dir: str | None, out_dir: str = "outputs", max_priority: int = 5,
              only: list[str] | None = None, common: dict | None = None,
              budget_hours: float | None = None, log=print) -> None:
    from rneml.baselines import run_baseline
    from rneml.train import load_config, train

    t0 = time.time()
    todo = sorted((e for e in EXPERIMENTS if e[0] <= max_priority and (not only or e[1] in only)),
                  key=lambda e: e[0])
    for prio, name, group, data, base, over in todo:
        if budget_hours and time.time() - t0 > budget_hours * 3600:
            log("time budget reached; run again to continue")
            return
        out = Path(out_dir) / name
        if (out / "metrics.json").exists():
            continue
        if not paths.get(data) or not Path(paths[data], "metadata.csv").exists():
            log(f"skip {name}: data '{data}' not available")
            continue
        over = dict(over)
        kind = over.pop("baseline", None)
        params = {**(common or {}), **over}            # the experiment's own settings win
        if data == "bb_fixed":
            params.setdefault("eval_data_dir", paths.get("v1"))
        cfg = load_config(base, name=name, out_dir=out_dir, data_dir=paths[data], real_dir=real_dir,
                          **params)
        log(f"=== {name} (priority {prio}, group {group})")
        try:
            run_baseline(cfg, kind, log) if kind else train(cfg, log)
        except Exception as e:                       # keep the suite going
            out.mkdir(parents=True, exist_ok=True)
            (out / "FAILED.txt").write_text(traceback.format_exc())
            log(f"FAILED {name}: {e!r}")


def _f(v, fmt="{:.2f}"):
    return "–" if v is None else fmt.format(v)


def report(out_dir: str = "outputs") -> str:
    """outputs/REPORT.md: one table per group of every finished experiment."""
    rows = {}
    for prio, name, group, data, base, over in EXPERIMENTS:
        p = Path(out_dir) / name / "metrics.json"
        if p.exists():
            rows.setdefault(group, []).append((name, data, over, json.loads(p.read_text())))
    lines = ["# 실험 결과 요약", "",
             "MAE는 개수 오차(개). test = 학습과 같은 물성 분포의 새 실행, OOD = 학습에 없던 물성 영역, "
             "real = 1차 연구 실제 녹음(BB, 0~100개; 적응에 쓴 클립 제외). ±5 = 오차 5개 이내 비율. "
             "R²는 물성 추정. corr(N,d) = N 오차와 지름 오차의 상관(음수면 '작은 입자 여러 개 vs 큰 입자 몇 개' 혼동).", ""]
    for g in GROUP_TITLES:
        if g not in rows:
            continue
        lines += [f"## {GROUP_TITLES[g]}", ""]
        if g == "H":
            lines += ["| 실험 | 설정 | 372 rpm만 MAE | 3개 속도 MAE | 5개 전부 MAE | OOD(5개) MAE | real MAE |",
                      "|---|---|---|---|---|---|---|"]
            for name, data, over, m in rows[g]:
                bc = m.get("by_condition_subset", {})
                cell = lambda k, s="test": _f(bc.get(k, {}).get(s, {}).get("mae"))
                lines.append(f"| {name} | `{json.dumps(over, ensure_ascii=False)}` | {cell('372rpm_only')} | "
                             f"{cell('three')} | {cell('all')} | {cell('all', 'test_ood')} | "
                             f"{_f(m.get('real', {}).get('mae'))} |")
        else:
            lines += ["| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | "
                      "R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for name, data, over, m in rows[g]:
                pr = m.get("properties_test", {})
                dg = m.get("degeneracy_test", {}).get("corr_N_error_with", {})
                conf = m.get("test", {}).get("conformal", {}).get("0.1", {})
                lines.append("| " + " | ".join([
                    name, data, f"`{json.dumps(over, ensure_ascii=False)}`" if over else "–",
                    _f(m.get("test", {}).get("mae")), _f(m.get("test", {}).get("within_5")),
                    _f(m.get("test_ood", {}).get("mae")), _f(m.get("real", {}).get("mae")),
                    _f(m.get("real", {}).get("within_5")),
                    _f(pr.get("diameter", {}).get("r2")), _f(pr.get("density", {}).get("r2")),
                    _f(pr.get("e_pw", {}).get("r2")), _f(dg.get("diameter")),
                    _f(conf.get("coverage"))]) + " |")
        lines.append("")
    text = "\n".join(lines)
    Path(out_dir, "REPORT.md").write_text(text)
    return text
