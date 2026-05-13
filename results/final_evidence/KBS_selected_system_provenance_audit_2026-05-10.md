# KBS selected-system provenance audit (2026-05-10)

This audit maps the manuscript `Ours (selected)` cells to local experiment artifacts when a rounded metric match is available.

The manuscript tables report values rounded to three decimals. `status_json_exact` means a completed `experiment_runner/parallel_run_*/status/*.json` record rounds to the displayed MSE and MAE on the same dataset and horizon. `summary_log_exact` means the same rounded match is recovered from a `summary.json` entry and its linked terminal log. `terminal_log_exact` means the match is recovered directly from a terminal log whose final result block records the dataset, horizon, model type, seed, MSE, and MAE. When multiple rounded matches exist, the CSV reports the first deterministic match and the candidate count. This file is an audit aid, not a substitute for a final public run manifest.

## Summary

- Parsed manuscript cells: $28$ non-average `Ours (selected)` settings.
- Rounded exact matches: $28$.
- Status-JSON exact matches: $16$.
- Summary-log exact matches: $10$.
- Direct terminal-log exact matches: $2$.
- Nearest-only matches: $0$.
- Missing local FRWKV-family status candidates: $0$.
- CSV: `ts_forecasting_framework/docs/KBS_selected_system_provenance_audit_2026-05-10.csv`.

## Audit Table

| Dataset | Horizon | Displayed MSE/MAE | Marker | Trace status | Candidate count | Exact run ID | Seed | Model type | Exact metric source | Source file |
| --- | ---: | --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| ETTm1 | 96 | 0.308 / 0.339 |  | status_json_exact | 14 | parallel_run_20260419_113220 | 2026 | frwkv_crossbranchphasegate_adaptive | 0.307998 / 0.339446 | experiment_runner/parallel_run_20260419_113220/status/kbs_release_ettm1_pl96_local_p48_r4_a0025_bm25_seed2026.json |
| ETTm1 | 192 | 0.357 / 0.370 |  | status_json_exact | 2 | parallel_run_20260416_213910 | 2026 | frwkv_crossbranchphasegate_adaptive_linearproj | 0.357307 / 0.369640 | experiment_runner/parallel_run_20260416_213910/status/kbs_ettm1_pl192_confirm_linear_base_seed2026.json |
| ETTm1 | 336 | 0.389 / 0.391 |  | status_json_exact | 31 | parallel_run_20260415_233903 | 2025 | frwkv_crossbranchphasegate_adaptive | 0.389460 / 0.390789 | experiment_runner/parallel_run_20260415_233903/status/ettm1_pl336_pp48_r4_a0.05_b-3.0_es4_el3.json |
| ETTm1 | 720 | 0.452 / 0.427 |  | status_json_exact | 88 | parallel_run_20260416_115318 | 2025 | frwkv_crossbranchphasegate_adaptive | 0.452379 / 0.426739 | experiment_runner/parallel_run_20260416_115318/status/ettm1_pl720_pp144_r8_a0.10_b-2.0_es8_el2.json |
| ETTm2 | 96 | 0.171 / 0.249 |  | status_json_exact | 23 | parallel_run_20260503_131828 | 2024 | frwkv_crossbranchgate | 0.170719 / 0.249356 | experiment_runner/parallel_run_20260503_131828/status/kbs_fullabl16_crossbranchgate_ettm2_pl96_seed2024.json |
| ETTm2 | 192 | 0.233 / 0.291 |  | summary_log_exact | 1 | parallel_run_20260408_222932 | 2025 | frwkv_crossbranchphasegate_adaptive | 0.233051 / 0.290963 | experiment_runner/parallel_run_20260408_222932/logs/overnight_ETTm2_pl192_pp24_r2_a0.02_b-3.0.log |
| ETTm2 | 336 | 0.292 / 0.328 |  | status_json_exact | 4 | parallel_run_20260416_224740 | 2025 | frwkv_crossbranchphasegate_adaptive | 0.291799 / 0.328495 | experiment_runner/parallel_run_20260416_224740/status/kbs_adaptive_ms_ettm2_pl336_seed2025_es16.json |
| ETTm2 | 720 | 0.390 / 0.388 |  | status_json_exact | 14 | parallel_run_20260503_131828 | 2026 | frwkv_crossbranchgate | 0.389563 / 0.387606 | experiment_runner/parallel_run_20260503_131828/status/kbs_fullabl16_crossbranchgate_ettm2_pl720_seed2026.json |
| ETTh1 | 96 | 0.371 / 0.388 |  | summary_log_exact | 2 | parallel_run_20260408_004029 | 2025 | frwkv_crossbranchphasegate_adaptive | 0.370847 / 0.388467 | experiment_runner/parallel_run_20260408_004029/logs/frwkv_crossbranchphasegate_adaptive_etth1_pl96_pp12_seed2025.log |
| ETTh1 | 192 | 0.424 / 0.419 |  | status_json_exact | 6 | parallel_run_20260416_224740 | 2025 | frwkv_crossbranchphasegate_adaptive | 0.423557 / 0.419051 | experiment_runner/parallel_run_20260416_224740/status/kbs_r1_etth1_pl192_patch24_l1_es16.json |
| ETTh1 | 336 | 0.462 / 0.441 |  | status_json_exact | 4 | parallel_run_20260416_224740 | 2025 | frwkv_crossbranchphasegate_adaptive | 0.462054 / 0.440882 | experiment_runner/parallel_run_20260416_224740/status/kbs_adaptive_ms_etth1_pl336_seed2025_es16.json |
| ETTh1 | 720 | 0.463 / 0.460 | ddagger | status_json_exact | 4 | parallel_run_20260419_201650 | 2034 | frwkv_crossbranchphasegate_adaptive | 0.462557 / 0.460319 | experiment_runner/parallel_run_20260419_201650/status/kbs_mae_etth1_pl720_apg_es4_p24_r4_a006_bm30_seed2034.json |
| ETTh2 | 96 | 0.278 / 0.327 |  | status_json_exact | 9 | parallel_run_20260419_022815 | 2034 | frwkv_crossbranchphasegate_adaptive | 0.277616 / 0.327223 | experiment_runner/parallel_run_20260419_022815/status/kbs_search_etth2_pl96_p12_r2_a001_bm45_es16_seed2034.json |
| ETTh2 | 192 | 0.358 / 0.379 |  | status_json_exact | 14 | parallel_run_20260416_224740 | 2028 | frwkv_crossbranchphasegate_adaptive | 0.358200 / 0.379347 | experiment_runner/parallel_run_20260416_224740/status/kbs_ablation_frwkv_crossbranchphasegate_adaptive_etth2_pl192_seed2028_es16.json |
| ETTh2 | 336 | 0.396 / 0.410 |  | summary_log_exact | 2 | parallel_run_20260409_133922 | 2025 | frwkv_crossbranchphasegate_adaptive | 0.395515 / 0.410494 | experiment_runner/parallel_run_20260409_133922/logs/phasegate_hard_etth2_pl336_pp24_r1_a0.05_b-2.5_seed2025.log |
| ETTh2 | 720 | 0.409 / 0.430 |  | summary_log_exact | 1 | parallel_run_20260409_133922 | 2024 | frwkv_crossbranchphasegate_adaptive | 0.409381 / 0.429917 | experiment_runner/parallel_run_20260409_133922/logs/phasegate_hard_etth2_pl720_pp48_r2_a0.1_b-2.0_seed2024.log |
| Weather | 96 | 0.156 / 0.194 |  | summary_log_exact | 1 | parallel_run_20260408_093213 | 2025 | frwkv_crossbranchphasegate_adaptive | 0.156426 / 0.193769 | experiment_runner/parallel_run_20260408_093213/logs/frwkv_phasegate_transfer_weather_pl96_pp12_seed2025.log |
| Weather | 192 | 0.205 / 0.238 |  | terminal_log_exact | 1 | parallel_run_20260411_181159 | 2028 | frwkv_crossbranchphasegate_adaptive | 0.205043 / 0.238439 | experiment_runner/parallel_run_20260411_181159/logs/kbs_adaptive_ms_weather_pl192_seed2028.log |
| Weather | 336 | 0.264 / 0.282 |  | terminal_log_exact | 1 | parallel_run_20260411_181159 | 2024 | frwkv_crossbranchphasegate_adaptive | 0.263671 / 0.282206 | experiment_runner/parallel_run_20260411_181159/logs/kbs_adaptive_ms_weather_pl336_seed2024.log |
| Weather | 720 | 0.342 / 0.335 | S | status_json_exact | 15 | parallel_run_20260418_161524 | 2033 | frwkv_crossbranchphasegate_adaptive | 0.341614 / 0.335317 | experiment_runner/parallel_run_20260418_161524/status/kbs_encattn_weather_pl720_linattn_seed2033.json |
| ILI | 24 | 1.432 / 0.721 |  | summary_log_exact | 1 | parallel_run_20260409_170109 | 2032 | frwkv_crossbranchphasegate_adaptive | 1.431946 / 0.721287 | experiment_runner/parallel_run_20260409_170109/logs/phasegate_hard_stage2_ili_pl24_pp6_r2_a0.02_b-3.0_seed2032.log |
| ILI | 36 | 1.392 / 0.714 |  | status_json_exact | 4 | parallel_run_20260416_224740 | 2025 | frwkv_crossbranchphasegate_adaptive | 1.391716 / 0.714074 | experiment_runner/parallel_run_20260416_224740/status/kbs_ablation_frwkv_crossbranchphasegate_adaptive_ili_pl36_seed2025_es16.json |
| ILI | 48 | 1.467 / 0.730 |  | summary_log_exact | 1 | parallel_run_20260409_133922 | 2025 | frwkv_crossbranchphasegate_adaptive | 1.467154 / 0.729867 | experiment_runner/parallel_run_20260409_133922/logs/phasegate_hard_ili_pl48_pp12_r4_a0.05_b-2.5_seed2025.log |
| ILI | 60 | 1.628 / 0.773 |  | summary_log_exact | 1 | parallel_run_20260409_170109 | 2038 | frwkv_crossbranchphasegate_adaptive | 1.628406 / 0.772938 | experiment_runner/parallel_run_20260409_170109/logs/phasegate_hard_stage2_ili_pl60_pp12_r4_a0.05_b-2.5_seed2038.log |
| Exchange | 96 | 0.081 / 0.198 |  | status_json_exact | 3 | parallel_run_20260503_131828 | 2026 | frwkv_crossbranchgate | 0.080592 / 0.197849 | experiment_runner/parallel_run_20260503_131828/status/kbs_fullabl16_crossbranchgate_exchange_pl96_seed2026.json |
| Exchange | 192 | 0.172 / 0.296 |  | summary_log_exact | 1 | parallel_run_20260411_220655 | 2027 | frwkv_crossbranchphasegate_adaptive | 0.172078 / 0.295598 | experiment_runner/parallel_run_20260411_220655/logs/kbs_ablation_frwkv_crossbranchphasegate_adaptive_exchange_pl192_seed2027.log |
| Exchange | 336 | 0.315 / 0.405 |  | status_json_exact | 4 | parallel_run_20260418_232012 | 2033 | frwkv_crossbranchphasegate_adaptive | 0.315031 / 0.404994 | experiment_runner/parallel_run_20260418_232012/status/kbs_release_anchor_exchange336_a_l1l2_r4_a02_b30_seed2033.json |
| Exchange | 720 | 0.840 / 0.690 | dagger | summary_log_exact | 1 | parallel_run_20260409_170109 | 2033 | frwkv_crossbranchphasegate_adaptive | 0.840333 / 0.689541 | experiment_runner/parallel_run_20260409_170109/logs/phasegate_hard_stage2_exchange_rate_pl720_pp24_r1_a0.02_b-3.0_seed2033.log |

## Remaining Risk

All parsed non-average `Ours (selected)` cells now have rounded exact local provenance through completed status JSON records, summary-linked terminal logs, or direct terminal logs. This resolves the internal audit gap identified by the earlier status-only scan. A final public release should still include the audit CSV or an equivalent run manifest so reviewers can inspect the run identifiers, seeds, model types, metric values, and source files.
