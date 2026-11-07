# 실험 결과 요약

MAE는 개수 오차(개). test = 학습과 같은 물성 분포의 새 실행, OOD = 학습에 없던 물성 영역, real = 1차 연구 실제 녹음(BB, 0~100개)의 개수별 6~15번 클립 210개 (1~5번은 성분 선택·실측 충돌음 라이브러리·적응에 쓰는 개발용이라 모든 실험에서 제외). ±5 = 오차 5개 이내 비율. R²는 물성 추정. corr(N,d) = N 오차와 지름 오차의 상관(음수면 '작은 입자 여러 개 vs 큰 입자 몇 개' 혼동). real 속도 기울기 = 같은 N 안에서 실측 가진 속도가 1 Hz 빠를 때 예측이 몇 개 늘어나는지 (실측 속도는 6.04~6.64 Hz로 흔들림; 0에 가까울수록 속도에 휘둘리지 않음).

## A. 물질을 알려 주면 얼마나 좋아지나 (블라인드 vs 크기·무게 앎 vs 전부 앎 vs 물질 고정 학습)

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1_blind | v1 | – | 8.89 | 0.41 | 9.12 | 25.42 | 0.15 | 0.62 | 0.36 | 0.49 | -0.44 | 0.90 | -12 |
| A2_known_size_weight | v1 | `{"material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}` | 6.84 | 0.51 | 7.46 | 17.17 | 0.17 | -0.36 | -0.11 | -0.09 | 0.06 | 0.89 | -0 |
| A3_known_all | v1 | `{"material_input": ["diameter", "density", "mass", "e_pp", "e_pw", "mu_pp", "mu_pw", "E"], "prop_weight": 0.0}` | 4.75 | 0.65 | 4.84 | 20.61 | 0.15 | -0.20 | -0.10 | -0.35 | -0.05 | 0.89 | 10 |
| A4_first_study_repro | bb_fixed | – | 14.94 | 0.24 | 14.67 | 29.43 | 0.14 | -0.54 | -0.97 | -4.40 | -0.47 | 0.57 | 4 |
| A5_bb_trained | bb_fixed | – | 15.84 | 0.23 | 19.70 | 18.47 | 0.13 | -2.58 | -0.16 | -0.16 | -0.68 | 0.59 | 4 |

## F. 딥러닝 없는 기준선 (같은 분할·같은 지표)

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F_rms | v1 | `{"baseline": "rms"}` | 20.06 | 0.15 | 21.54 | 18.52 | 0.17 | 0.18 | 0.11 | 0.02 | -0.16 | 0.93 | – |
| F_onset | v1 | `{"baseline": "onset"}` | 19.59 | 0.09 | 19.22 | 20.65 | 0.10 | 0.16 | 0.05 | 0.05 | -0.19 | 0.95 | – |
| F_order_params | v1 | `{"baseline": "order_params"}` | 14.54 | 0.24 | 16.16 | 21.36 | 0.10 | 0.36 | 0.21 | 0.26 | -0.30 | 0.93 | – |
| F_ridge_all | v1 | `{"baseline": "ridge_all"}` | 10.31 | 0.32 | 11.16 | 25.35 | 0.12 | 0.61 | 0.31 | 0.42 | -0.37 | 0.92 | – |
| F_gbm_physics | v1 | `{"baseline": "gbm_physics"}` | 10.82 | 0.30 | 11.29 | 27.26 | 0.12 | 0.55 | 0.27 | 0.34 | -0.40 | 0.95 | – |
| F_gbm_all | v1 | `{"baseline": "gbm_all"}` | 9.71 | 0.33 | 9.66 | 24.91 | 0.18 | 0.58 | 0.30 | 0.49 | -0.42 | 0.91 | – |

## B. 입력 표현과 전처리

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_hpss_channels | v1 | `{"feature": "mel+perc+harm"}` | 9.36 | 0.40 | 8.73 | 24.86 | 0.14 | 0.61 | 0.33 | 0.45 | -0.45 | 0.89 | -15 |
| B2_denoise | v1 | `{"denoise": true}` | 8.95 | 0.40 | 8.99 | 25.66 | 0.14 | 0.62 | 0.36 | 0.51 | -0.46 | 0.90 | -17 |
| B3_mel22k | v1 | `{"feature": "mel22k"}` | 9.15 | 0.39 | 9.08 | 23.84 | 0.16 | 0.61 | 0.36 | 0.47 | -0.46 | 0.90 | -8 |
| B4_scalogram | v1 | `{"feature": "scalogram"}` | 8.81 | 0.39 | 8.90 | 23.17 | 0.13 | 0.64 | 0.37 | 0.51 | -0.43 | 0.90 | -11 |
| B5_band_6k_24k | v1 | `{"band_hz": [6000, 24000]}` | 9.74 | 0.37 | 9.90 | 23.75 | 0.14 | 0.59 | 0.29 | 0.43 | -0.36 | 0.90 | -9 |
| B6_basic_scalars | v1 | `{"scalar_set": "basic"}` | 9.64 | 0.38 | 9.52 | 30.67 | 0.16 | 0.61 | 0.36 | 0.56 | -0.45 | 0.90 | 3 |
| B7_no_scalars | v1 | `{"use_scalars": false}` | 9.32 | 0.42 | 10.11 | 26.84 | 0.16 | 0.60 | 0.37 | 0.57 | -0.42 | 0.91 | 3 |
| B9_selected_components | v1 | `{"component_scalars": "outputs/components/components.json", "real_exclude_per_N": 5}` | 9.08 | 0.40 | 9.01 | 24.77 | 0.17 | 0.63 | 0.34 | 0.49 | -0.46 | 0.92 | -13 |
| B11_transfer_safe | v1 | `{"band_hz": [8000, 24000], "scalar_set": "none", "component_scalars": "outputs/components/components.json", "real_exclude_per_N": 5}` | 9.31 | 0.38 | 9.27 | 32.07 | 0.14 | 0.64 | 0.33 | 0.58 | -0.43 | 0.92 | -1 |
| B12_transfer_safe_dann | v1 | `{"band_hz": [8000, 24000], "scalar_set": "none", "real_unlabeled_per_N": 5, "component_scalars": "outputs/components/components.json", "real_exclude_per_N": 5}` | 9.52 | 0.41 | 10.18 | 44.96 | 0.05 | 0.60 | 0.31 | 0.48 | -0.44 | 0.91 | 7 |
| B10_all_components | v1 | `{"component_scalars": "all", "real_exclude_per_N": 5}` | 8.77 | 0.42 | 8.44 | 20.89 | 0.17 | 0.61 | 0.36 | 0.53 | -0.48 | 0.91 | -13 |
| B8_hpss_denoise_dann | v1 | `{"feature": "mel+perc+harm", "denoise": true, "real_unlabeled_per_N": 5}` | 9.47 | 0.40 | 9.36 | 17.86 | 0.16 | 0.58 | 0.28 | 0.46 | -0.43 | 0.90 | -8 |

## C. 학습 구성 요소 제거

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1_dann | v1 | `{"real_unlabeled_per_N": 5}` | 10.07 | 0.35 | 10.25 | 14.15 | 0.18 | 0.59 | 0.32 | 0.46 | -0.46 | 0.88 | -10 |
| C2_no_property_heads | v1 | `{"prop_weight": 0.0}` | 9.22 | 0.40 | 8.94 | 24.85 | 0.12 | -1.99 | -0.49 | -0.02 | -0.25 | 0.89 | -11 |
| C3_no_pair_losses | v1 | `{"pair_weight": 0.0, "rank_weight": 0.0}` | 8.87 | 0.41 | 8.41 | 26.02 | 0.17 | 0.64 | 0.35 | 0.51 | -0.45 | 0.89 | -5 |
| C4_level_free | v1 | `{"level_free": true}` | 9.09 | 0.37 | 9.18 | 25.22 | 0.14 | 0.64 | 0.37 | 0.49 | -0.45 | 0.89 | -15 |
| C5_no_gain_augmentation | v1 | `{"gain_db": 0.0}` | 9.06 | 0.41 | 9.13 | 24.05 | 0.13 | 0.63 | 0.37 | 0.51 | -0.47 | 0.91 | -9 |

## E. 일반화 분할 (물성 외삽·구멍, 개수 외삽)

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1_extrap_restitution | v1 | `{"split": {"kind": "material_extrap", "axis": "restitution_pw", "lo": 0.55, "hi": 0.75}}` | 12.36 | 0.34 | – | 19.16 | 0.17 | 0.58 | 0.26 | -10.53 | -0.46 | 0.80 | -8 |
| E2_hole_diameter | v1 | `{"split": {"kind": "material_band", "axis": "particle_diameter_m", "lo": 0.0055, "hi": 0.0065}}` | 10.12 | 0.35 | – | 25.38 | 0.11 | -5.22 | 0.29 | 0.36 | -0.59 | 0.91 | -13 |
| E3_extrap_count | v1 | `{"split": {"kind": "count_extrap", "n_train_max": 70, "n_test_min": 80}}` | 28.79 | 0.00 | – | 20.75 | 0.13 | 0.41 | 0.12 | 0.13 | -0.25 | 0.00 | -6 |

## D. 개수 출력 방식 (학습에 없던 개수: count_sparse 분할)

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D_regression | v1 | `{"count_head": "regression", "split": {"kind": "count_sparse", "step": 10, "halfwidth": 1}}` | 10.12 | 0.38 | – | 27.35 | 0.16 | 0.57 | 0.29 | 0.33 | -0.52 | 0.86 | -4 |
| D_classification | v1 | `{"count_head": "classification", "split": {"kind": "count_sparse", "step": 10, "halfwidth": 1}}` | 10.02 | 0.38 | – | 29.06 | 0.16 | 0.55 | 0.24 | 0.33 | -0.51 | 0.84 | 4 |
| D_soft_label | v1 | `{"count_head": "soft_label", "split": {"kind": "count_sparse", "step": 10, "halfwidth": 1}}` | 9.83 | 0.40 | – | 30.89 | 0.13 | 0.56 | 0.23 | 0.33 | -0.52 | 0.86 | 4 |
| D_coral | v1 | `{"count_head": "coral", "split": {"kind": "count_sparse", "step": 10, "halfwidth": 1}}` | 9.94 | 0.37 | – | 30.35 | 0.13 | 0.57 | 0.27 | 0.34 | -0.54 | 0.88 | -2 |

## G. 실측 녹음 몇 개에 정답을 붙이면

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G_real_labeled_1 | v1 | `{"real_train_per_N": 1}` | 9.52 | 0.39 | 9.18 | 9.74 | 0.36 | 0.62 | 0.36 | 0.48 | -0.46 | 0.89 | -11 |
| G_real_labeled_2 | v1 | `{"real_train_per_N": 2}` | 9.20 | 0.40 | 9.10 | 7.76 | 0.47 | 0.62 | 0.36 | 0.50 | -0.46 | 0.89 | -2 |
| G_real_labeled_5 | v1 | `{"real_train_per_N": 5}` | 9.32 | 0.40 | 9.10 | 5.88 | 0.53 | 0.61 | 0.33 | 0.46 | -0.46 | 0.88 | -3 |

## H. 다중 조건 (5개 rpm) 모델

| 실험 | 설정 | 372 rpm만 MAE | 3개 속도 MAE | 5개 전부 MAE | OOD(5개) MAE | real MAE |
|---|---|---|---|---|---|---|
| H1_set_per_set | `{}` | 10.79 | 9.25 | 8.79 | 9.53 | 21.21 |
| H2_set_absolute | `{"norm_mode": "absolute"}` | 10.40 | 9.51 | 8.63 | 9.56 | 30.67 |
| H3_set_per_clip | `{"norm_mode": "per_clip"}` | 10.34 | 9.34 | 8.61 | 9.47 | 24.19 |
| H4_set_diff_only | `{"diff_path": "diff"}` | 36.53 | 16.01 | 16.73 | 14.31 | 34.84 |
| H5_set_no_diff | `{"diff_path": "none"}` | 10.63 | 9.11 | 8.61 | 10.00 | 27.27 |
| H6_set_ordered | `{"agg": "ordered"}` | 9.84 | 9.69 | 8.98 | 10.10 | 27.21 |
| H7_set_no_condition_value | `{"use_condition": false}` | 10.10 | 10.43 | 9.20 | 10.03 | 26.46 |
| H8_set_known_size_weight | `{"material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}` | 8.40 | 7.24 | 6.87 | 7.71 | 24.53 |
| H9_nosig_set_per_set | `{}` | 7.95 | 7.62 | 6.59 | 7.31 | 11.18 |
| H10_nosig_set_known_size_weight | `{"material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}` | 6.16 | 6.55 | 5.73 | 5.68 | 15.52 |
| H11_nosig_set_dann | `{"real_unlabeled_per_N": 5}` | 8.75 | 7.60 | 6.89 | 6.87 | 9.14 |

## R. 실측 보정의 효과 (R0 = 보정 전 v1, R1~R4 = 보정 전 동역학 + 보정한 소리, A = 동역학·소리 모두 보정한 v3)

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_v1old_blind | v1_old | – | 7.46 | 0.44 | 8.85 | 11.50 | 0.31 | 0.72 | 0.42 | 0.42 | -0.48 | 0.91 | -6 |
| R0_v1old_known_size_weight | v1_old | `{"material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}` | 4.73 | 0.62 | 5.49 | 14.40 | 0.20 | -0.34 | -0.20 | -0.48 | -0.04 | 0.94 | -7 |
| R1_r2_blind | v1_r2 | – | 8.23 | 0.44 | 9.70 | 27.53 | 0.14 | 0.68 | 0.36 | 0.41 | -0.52 | 0.94 | -9 |
| R2_r2_known_size_weight | v1_r2 | `{"material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}` | 6.13 | 0.54 | 7.27 | 18.54 | 0.14 | -0.02 | -0.38 | -0.65 | -0.00 | 0.95 | -4 |
| R3_r2_transfer_safe | v1_r2 | `{"band_hz": [8000, 24000], "scalar_set": "none", "component_scalars": "outputs/components/components.json", "real_exclude_per_N": 5}` | 9.71 | 0.35 | 11.27 | 29.22 | 0.09 | 0.63 | 0.30 | 0.22 | -0.49 | 0.93 | -9 |
| R4_r2_hpss_denoise_dann | v1_r2 | `{"feature": "mel+perc+harm", "denoise": true, "real_unlabeled_per_N": 5}` | 9.28 | 0.37 | 10.19 | 21.65 | 0.17 | 0.60 | 0.32 | 0.39 | -0.46 | 0.94 | -5 |

## W. 보정한 소리가 실측에서 나빠진 원인 (충돌별 세기 차이만 / 실제 충돌음만, 동역학은 보정 전 v1)

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W1_sigma_only_blind | v1_sig | – | 7.81 | 0.46 | 9.94 | 25.94 | 0.16 | 0.72 | 0.43 | 0.41 | -0.52 | 0.93 | -8 |
| W2_clicks_only_blind | v1_clk | – | 6.17 | 0.58 | 7.87 | 13.78 | 0.15 | 0.72 | 0.43 | 0.49 | -0.57 | 0.93 | -7 |

## Q. 실측 적응 녹음의 교차검증 (적응에 쓰는 녹음 번호 5개 묶음을 바꿔 가며; 1~5번은 평가하지 않음)

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1_cv_real_labeled_5_f1 | v3_nosig | `{"real_train_per_N": 5, "real_fold": 1}` | 6.78 | 0.51 | 7.23 | 6.08 | 0.53 | 0.65 | 0.38 | 0.65 | -0.41 | 0.88 | -4 |
| Q2_cv_real_labeled_5_f2 | v3_nosig | `{"real_train_per_N": 5, "real_fold": 2}` | 6.64 | 0.50 | 6.84 | 6.49 | 0.47 | 0.67 | 0.40 | 0.65 | -0.40 | 0.87 | 4 |
| Q3_cv_dann_f1 | v3_nosig | `{"real_unlabeled_per_N": 5, "real_fold": 1}` | 7.61 | 0.41 | 8.47 | 12.49 | 0.20 | 0.53 | 0.23 | 0.39 | -0.43 | 0.92 | -1 |
| Q4_cv_dann_f2 | v3_nosig | `{"real_unlabeled_per_N": 5, "real_fold": 2}` | 7.06 | 0.50 | 7.23 | 11.40 | 0.18 | 0.65 | 0.30 | 0.58 | -0.39 | 0.89 | -6 |
| Q5_cv_dann_known_size_weight_f1 | v3_nosig | `{"real_unlabeled_per_N": 5, "material_input": ["diameter", "density", "mass"], "prop_weight": 0.0, "real_fold": 1}` | 5.97 | 0.57 | 7.04 | 11.13 | 0.25 | -0.03 | 0.00 | -0.56 | -0.07 | 0.88 | 2 |
| Q6_cv_dann_known_size_weight_f2 | v3_nosig | `{"real_unlabeled_per_N": 5, "material_input": ["diameter", "density", "mass"], "prop_weight": 0.0, "real_fold": 2}` | 6.17 | 0.54 | 7.01 | 11.00 | 0.17 | -0.07 | -0.04 | -0.29 | -0.17 | 0.92 | -2 |

## U. 단일 속도 + 다중 속도 데이터를 합쳐 학습한 단일 클립 모델 (세기 퍼짐 없음)

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U1_merged_blind | merged_nosig | – | 7.52 | 0.48 | 8.59 | 16.82 | 0.19 | 0.71 | 0.41 | 0.52 | -0.45 | 0.91 | -1 |
| U2_merged_dann | merged_nosig | `{"real_unlabeled_per_N": 5}` | 7.77 | 0.46 | 9.13 | 9.46 | 0.38 | 0.64 | 0.34 | 0.47 | -0.42 | 0.88 | -5 |
| U3_merged_dann_known_size_weight | merged_nosig | `{"real_unlabeled_per_N": 5, "material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}` | 6.54 | 0.53 | 7.67 | 8.52 | 0.40 | -7.01 | -1.43 | -1.28 | -0.09 | 0.90 | -3 |

## V. 충돌별 세기 퍼짐을 뺀 주 데이터 (W에서 찾은 원인 제거: v3 동역학·실제 충돌음 50 %·녹음 조건 그대로, 세기 퍼짐 0)

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V1_nosig_blind | v3_nosig | – | 6.82 | 0.52 | 6.91 | 13.70 | 0.19 | 0.65 | 0.34 | 0.59 | -0.44 | 0.89 | -8 |
| V2_nosig_known_size_weight | v3_nosig | `{"material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}` | 5.07 | 0.63 | 5.44 | 11.52 | 0.20 | -0.54 | -0.45 | -0.68 | 0.03 | 0.88 | -1 |
| V3_nosig_known_all | v3_nosig | `{"material_input": ["diameter", "density", "mass", "e_pp", "e_pw", "mu_pp", "mu_pw", "E"], "prop_weight": 0.0}` | 4.01 | 0.71 | 4.39 | 13.86 | 0.14 | -0.27 | -0.45 | -0.46 | -0.07 | 0.89 | 1 |
| V4_nosig_dann | v3_nosig | `{"real_unlabeled_per_N": 5}` | 6.87 | 0.51 | 7.16 | 9.96 | 0.39 | 0.63 | 0.30 | 0.60 | -0.42 | 0.88 | -2 |
| V5_nosig_hpss_denoise_dann | v3_nosig | `{"feature": "mel+perc+harm", "denoise": true, "real_unlabeled_per_N": 5}` | 7.05 | 0.47 | 7.28 | 12.28 | 0.20 | 0.63 | 0.30 | 0.59 | -0.42 | 0.88 | -7 |
| V6_nosig_real_labeled_2 | v3_nosig | `{"real_train_per_N": 2}` | 6.90 | 0.50 | 6.91 | 7.43 | 0.44 | 0.67 | 0.38 | 0.64 | -0.42 | 0.83 | -2 |
| V7_nosig_real_labeled_5 | v3_nosig | `{"real_train_per_N": 5}` | 7.22 | 0.48 | 7.31 | 6.20 | 0.53 | 0.64 | 0.39 | 0.51 | -0.40 | 0.88 | -3 |
| V8_nosig_dann_known_size_weight | v3_nosig | `{"real_unlabeled_per_N": 5, "material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}` | 5.49 | 0.62 | 6.13 | 7.32 | 0.48 | -0.43 | -0.80 | -0.96 | 0.03 | 0.86 | -0 |
| V9_nosig_blind_s1 | v3_nosig | `{"seed": 1}` | 6.80 | 0.53 | 7.17 | 13.45 | 0.22 | 0.66 | 0.38 | 0.59 | -0.42 | 0.85 | -8 |
| V10_nosig_dann_s1 | v3_nosig | `{"real_unlabeled_per_N": 5, "seed": 1}` | 7.26 | 0.48 | 7.27 | 10.24 | 0.27 | 0.62 | 0.31 | 0.58 | -0.43 | 0.85 | -1 |
| V11_nosig_real_labeled_1 | v3_nosig | `{"real_train_per_N": 1}` | 7.11 | 0.49 | 7.07 | 9.17 | 0.39 | 0.65 | 0.36 | 0.58 | -0.42 | 0.86 | -7 |

## K. 어떤 물성을 알려 주면 도움이 되나 (주 데이터, 실측에서 지름·밀도·질량은 실제 값, 반발계수는 추정값, 마찰·탄성률은 모름)

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K1_diameter | v1 | `{"material_input": ["diameter"], "prop_weight": 0.0}` | 7.45 | 0.45 | 8.12 | 18.73 | 0.14 | -0.16 | -0.22 | 0.13 | 0.07 | 0.89 | -6 |
| K2_mass | v1 | `{"material_input": ["mass"], "prop_weight": 0.0}` | 7.85 | 0.44 | 7.76 | 21.06 | 0.17 | -0.01 | -0.07 | -0.00 | -0.22 | 0.91 | -10 |
| K3_density | v1 | `{"material_input": ["density"], "prop_weight": 0.0}` | 9.14 | 0.38 | 9.49 | 25.45 | 0.17 | -0.02 | -0.09 | 0.06 | -0.15 | 0.86 | -11 |
| K4_diameter_mass | v1 | `{"material_input": ["diameter", "mass"], "prop_weight": 0.0}` | 7.50 | 0.46 | 7.76 | 19.51 | 0.14 | 0.07 | -0.56 | -0.23 | -0.03 | 0.90 | -5 |
| K5_restitution | v1 | `{"material_input": ["e_pp", "e_pw"], "prop_weight": 0.0}` | 8.84 | 0.40 | 8.95 | 30.39 | 0.15 | -0.18 | -0.75 | -0.22 | -0.25 | 0.88 | -12 |
| K6_friction | v1 | `{"material_input": ["mu_pp", "mu_pw"], "prop_weight": 0.0}` | 10.00 | 0.34 | 9.96 | 24.48 | 0.17 | -0.03 | -0.18 | -0.22 | -0.23 | 0.88 | -6 |
| K7_modulus | v1 | `{"material_input": ["E"], "prop_weight": 0.0}` | 8.82 | 0.41 | 8.80 | 26.25 | 0.14 | -0.03 | -0.04 | -0.01 | -0.27 | 0.89 | -8 |
| K8_size_weight_restitution | v1 | `{"material_input": ["diameter", "density", "mass", "e_pp", "e_pw"], "prop_weight": 0.0}` | 4.78 | 0.63 | 4.93 | 23.37 | 0.18 | -0.63 | -0.02 | -0.06 | -0.05 | 0.90 | 7 |

## S. 시드 반복

| 실험 | 데이터 | 설정 | test MAE | test ±5 | OOD MAE | real MAE | real ±5 | R² 지름 | R² 밀도 | R² e_pw | corr(N,d) | 90% 구간 포함률 | real 속도 기울기 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S_A1_blind_s1 | v1 | `{"seed": 1}` | 8.99 | 0.41 | 8.84 | 24.08 | 0.16 | 0.64 | 0.36 | 0.51 | -0.44 | 0.90 | -13 |
| S_A1_blind_s2 | v1 | `{"seed": 2}` | 9.19 | 0.37 | 8.82 | 26.52 | 0.13 | 0.62 | 0.37 | 0.51 | -0.44 | 0.89 | -16 |
| S_A2_known_size_weight_s1 | v1 | `{"seed": 1, "material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}` | 7.07 | 0.50 | 7.56 | 17.42 | 0.17 | -0.06 | -0.29 | 0.05 | 0.03 | 0.87 | -6 |
| S_A2_known_size_weight_s2 | v1 | `{"seed": 2, "material_input": ["diameter", "density", "mass"], "prop_weight": 0.0}` | 6.95 | 0.50 | 7.20 | 17.06 | 0.16 | -1.04 | -0.21 | -0.20 | 0.04 | 0.91 | -1 |
