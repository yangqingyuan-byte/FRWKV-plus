# Data Availability Statement

The datasets used in this study are standard public benchmarks for long-term time series forecasting, including ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange, and ILI. The processed dataset files used in our experiments are organized in the project repository under:

- `src/adaptive_phasegate_kbs/dataset`

The code used to run the forecasting experiments, aggregate results, and generate the manuscript tables is available in the same repository, primarily under:

- `src/adaptive_phasegate_kbs`
- `scripts`
- `paper`

The manuscript-facing evidence files used to audit the reported numbers are organized under:

- `results/final_evidence`

This directory includes row-level matched 16-seed evidence and copied source snapshots for directly referenced runner status or log outputs.

The release integrity checker is:

- `scripts/verify_release_repro.py`

Before submission, the authors should replace these local paths with the final public repository URL and, if required by the journal workflow, provide a persistent DOI or repository archive link.
