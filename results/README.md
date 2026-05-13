# Results Folder

This folder contains the paper-oriented result artifacts included in the release.

## Current Manuscript Evidence

Use `final_evidence/` as the current source of truth for the manuscript-facing evidence chain:

- `KBS_selected_system_provenance_audit_2026-05-10.csv`
  Provenance audit for the 28 selected-system cells in the main result table.
- `KBS_selected_system_provenance_audit_2026-05-10.md`
  Human-readable summary of the selected-system audit.
- `KBS_table4_selected_single_run_provenance_2026-05-11.csv`
  Selected single-run provenance for the table comparing `Ours(selected)` with FRWKV-family variants.
- `KBS_full_family_matched16_final_analysis_2026-05-09.json`
  Machine-readable matched 16-seed validation and claim-strength summary.
- `KBS_full_family_matched16_final_analysis_2026-05-09.csv`
  Tabular matched 16-seed family-ablation results.
- `KBS_full_family_matched16_final_raw_rows_2026-05-09.csv`
  Row-level matched 16-seed evidence with one row per dataset-horizon-model-seed result.
- `KBS_full_family_matched16_final_analysis_2026-05-09.md`
  Human-readable matched 16-seed analysis report.
- `KBS_main_model_runtime_profile_results_2026-05-11.csv`
  Runtime/profile evidence for the 28 current main-model runs.
- `KBS_main_model_runtime_profile_results_2026-05-11.md`
  Human-readable runtime/profile summary.
- `source_snapshots/`
  Copied status/log snapshots for the main-table, selected single-run table, and runtime-profile rows that directly reference runner output files.
- `source_snapshot_manifest_2026-05-13.csv`
  Snapshot path, byte size, and SHA-256 manifest for the copied source files.
- `RELEASE_INTEGRITY_AUDIT_2026-05-13.md`
  Short audit note summarizing what the release evidence supports and where rerun variability remains.

Run the release integrity checker from the repository root:

```bash
python scripts/verify_release_repro.py
```

## Historical Artifacts

- `paper_results.jsonl`
  Curated April-era experiment records extracted from the larger working log. Kept for auditability and older table-builder compatibility.
- `etth1_seq96_pred96_all_results.csv`
  Baseline comparison file used in earlier table drafts.

These historical artifacts are useful for tracing the development process, but the current manuscript should be audited against `final_evidence/`.
