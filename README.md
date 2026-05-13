# FRWKV+ KBS Release

<p align="right">
  <a href="README.zh-CN.md">
    <img alt="简体中文" src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue?style=for-the-badge">
  </a>
</p>

Open-source release package for the manuscript:

**FRWKV+: Adaptive Periodic-Position Branch Interaction for Frequency-Space Linear Time Series Forecasting**

This repository packages the code, benchmark data, recipe configuration, final evidence manifests, and manuscript assets for the KBS paper line. Its main purpose is to make the paper numbers traceable: every reported final-result cell should be linked to concrete local evidence, and current paper recipes should be executable from the release package.

## What Is Included

- `src/adaptive_phasegate_kbs/`
  Core training code, data loaders, utilities, and the FRWKV-family models used in the paper.
- `src/adaptive_phasegate_kbs/configs/kbs_ours_recipes.json`
  Current paper recipe config. The main-table group is `paper_current.main_table_all`.
- `src/adaptive_phasegate_kbs/dataset/`
  Bundled benchmark CSV files used in the experiments.
- `results/final_evidence/`
  Final manuscript evidence files: selected-run provenance, selected single-run table provenance, matched 16-seed family ablation, runtime profile evidence, raw matched 16-seed rows, and source snapshots for directly referenced status/log files.
- `scripts/verify_release_repro.py`
  Integrity checker for the final evidence files and packaged recipe config.
- `paper/`
  LaTeX source, compiled PDF, figures, bibliography, and submission-side text assets.
- `scripts/` and `jobs/`
  Experiment runners and historical job generators retained for reruns and auditability.
- `docs/`
  Human-readable summaries. Some April-era table drafts are kept as historical summaries; the current manuscript evidence source is `results/final_evidence/`.

## Validated Scope

The release focuses on the KBS paper line and the FRWKV-family variants needed to inspect the reported results:

- `frwkv`
- `frwkv_crossbranchgate`
- `frwkv_crossbranchphasegate`
- `frwkv_crossbranchphasegate_fullcontextdelta`
- `frwkv_crossbranchperiodicpositiongate_adaptive`
- `frwkv_crossbranchphasegate_adaptive`
- selected adaptive embedding/projection variants used by provenance records

The current paper’s `Ours` recipe is the adaptive periodic-position branch-interaction model, exposed as `frwkv_crossbranchperiodicpositiongate_adaptive`.

## Environment Setup

PyTorch is installed separately so users can choose the correct CUDA or CPU build for their hardware.

### Conda

```bash
conda create -n apgkbs python=3.10 pip -y
conda activate apgkbs
pip install -r requirements-cu128.txt
pip install -e .[notify]
```

For CPU-only execution:

```bash
conda create -n apgkbs-cpu python=3.10 pip -y
conda activate apgkbs-cpu
pip install -r requirements-cpu.txt
pip install -e .
```

### uv

```bash
python -m pip install --user uv
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements-cu128.txt
uv pip install -e .[notify]
```

## Required Integrity Check

From the release root:

```bash
python scripts/verify_release_repro.py
```

This checks that:

- the selected-system provenance contains 28 exact traced main-table cells;
- the selected single-run table contains 76 traced rows;
- the matched 16-seed FRWKV-family ablation contains 2240 deduplicated rows with no missing seed groups;
- the runtime profile contains 28 completed runs;
- the raw matched 16-seed row file contains 2240 unique dataset-horizon-model-seed rows;
- the 112 copied source snapshots match their checksum manifest;
- `kbs_ours_recipes.json` contains the 28 current main-table recipes and the ETTh2-96 recipe used by the manuscript.

Passing this script means the release package contains the evidence needed to audit the reported paper values. It is not a claim that a fresh rerun will produce bit-identical metrics across every PyTorch, CUDA, driver, and GPU combination.

## Reproducing a Paper Recipe

The preferred entrypoint is `--config_name`, which loads settings from the packaged recipe file:

```bash
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
python -m adaptive_phasegate_kbs.train \
  --config_name paper_current.adaptive.etth2_pl96 \
  --seed 2034 \
  --model_tag reproduce_etth2_pl96
```

For a quick syntax/smoke check without launching a full experiment:

```bash
python -m adaptive_phasegate_kbs.train --help
python -m adaptive_phasegate_kbs.train --config_name paper_current.adaptive.etth2_pl96 --help
```

The recipe config is stored at:

- `src/adaptive_phasegate_kbs/configs/kbs_ours_recipes.json`

## Final Evidence Files

Current manuscript-facing evidence is under `results/final_evidence/`:

- `KBS_selected_system_provenance_audit_2026-05-10.csv`
  Main-table selected-system provenance for 28 dataset-horizon cells.
- `KBS_table4_selected_single_run_provenance_2026-05-11.csv`
  Selected single-run provenance for the table comparing `Ours(selected)` and FRWKV-family variants.
- `KBS_full_family_matched16_final_analysis_2026-05-09.json`
  Machine-readable matched 16-seed family-ablation summary, including validation status, winner counts, average ranks, paired comparisons, and the claim-strength label.
- `KBS_full_family_matched16_final_analysis_2026-05-09.csv`
  Table-form matched 16-seed family-ablation results.
- `KBS_full_family_matched16_final_raw_rows_2026-05-09.csv`
  Row-level matched 16-seed evidence with one row per dataset-horizon-model-seed result used in the final family-ablation analysis.
- `KBS_main_model_runtime_profile_results_2026-05-11.csv`
  Runtime/profile evidence for 28 current main-model jobs.
- `source_snapshots/` and `source_snapshot_manifest_2026-05-13.csv`
  Copied status/log snapshots for the main-table, selected single-run table, and runtime-profile records that directly reference runner output files.
- `RELEASE_INTEGRITY_AUDIT_2026-05-13.md`
  Audit summary describing what the release evidence supports and the remaining reproducibility boundaries.

The older `results/paper_results.jsonl` and `docs/KBS_phasegate_tables_draft.md` are retained for historical auditability. They should not be treated as the final source of truth for the current manuscript.

## Manuscript Assets

- Source: `paper/adaptive_phasegate_kbs.tex`
- Compiled PDF: `paper/adaptive_phasegate_kbs.pdf`
- Figures: `paper/figs/`
- Bibliography: `paper/cas-refs.bib`

The release copy of the paper is synchronized with the current manuscript title, author metadata, funding statement, generative-AI declaration, and final architecture figures.

## Notes on Reproducibility

- The bundled CSV benchmark files are the dataset files used by the release training code.
- The final evidence files preserve run IDs, seeds, model types, config names or model tags, metric values, and source-file references wherever available.
- Fresh reruns can differ slightly because PyTorch kernels, CUDA libraries, GPU models, and host load affect numerical paths and early stopping. The package therefore separates two responsibilities: evidence integrity through `scripts/verify_release_repro.py`, and rerun execution through the same recipe configuration used by the paper.
- Throughput numbers are hardware dependent. The runtime profile evidence is included to document the profiled campaign rather than to prescribe expected speed on other machines.
