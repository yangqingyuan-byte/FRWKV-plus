# Release Integrity Audit

Date: 2026-05-13

Scope: `adaptive_phasegate_kbs_release`

## Overall Verdict

PASS with explicit reproducibility boundaries.

The release now contains the current manuscript source and PDF, the packaged recipe config, row-level final evidence, and copied source snapshots for the runner files directly referenced by the final provenance tables. The package supports audit of the reported values without relying only on prose summaries.

## Checks

### Ground Truth and Metrics

PASS. The forecasting metrics in the release evidence are standard test MSE and MAE fields from experiment runner outputs or row-level aggregation files. No evidence file normalizes scores by model outputs.

### Result Existence and Traceability

PASS. `scripts/verify_release_repro.py` checks:

- 28 exact selected-system main-table traces.
- 76 selected single-run provenance rows.
- 2240 matched 16-seed raw rows.
- 28 completed runtime/profile rows.
- 112 copied source snapshots with SHA-256 checksums.
- 28 packaged current main-table recipes.

### Scope

PASS with boundary. The matched family ablation covers seven datasets, 28 dataset-horizon settings, five FRWKV-family variants, and 16 seeds per setting. The evidence supports selective periodic-position correction and largest MSE winner coverage within the FRWKV family. It does not support a claim that FRWKV+ is uniformly best on every dataset, horizon, metric, or average-rank criterion.

### Release-Code Alignment

PASS. The release training entrypoint supports `--config_name`, the packaged `kbs_ours_recipes.json`, the current periodic-position argument names, legacy phase argument aliases, and the FRWKV-family variants referenced by the provenance files.

## Residual Limitations

- Fresh reruns can differ slightly across PyTorch, CUDA, driver, GPU, and host-load conditions.
- Some selected main-table values come from selected provenance records rather than a single universal fixed recipe. The selection is explicitly represented in the provenance CSV and source snapshots.
- Runtime/profile values document the profiled campaign and should not be interpreted as hardware-independent speed guarantees.

## Required Release Check

Run this command from the release root before publication:

```bash
python scripts/verify_release_repro.py
```
