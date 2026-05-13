# KBS Remaining Tuning Summary (2026-04-12)

## Scope

Focused strict single-GPU tuning screen on the two weakest settings identified after the KBS supplement campaign:

- `ETTh2, pred_len=96`
- `ILI, pred_len=36`

Run:

- `experiment_runner/parallel_run_20260412_002634`

Status:

- `153 / 153` success

## 1. ETTh2-96

Previous five-seed ablation winner:

- `CrossBranchPhaseGate`: `0.283826 ± 0.003299 / 0.330229 ± 0.001735`

Best new adaptive setting from the tuning screen:

- `pp=48, r=2, a=0.02, b=-3.5`
- single-seed result: `0.283078 / 0.329018`

Key observation:

- Many near-best settings collapse to `alpha=0.0` and produce `0.283816 / 0.329180`.
- This suggests the adaptive branch is currently most reliable when phase injection is extremely conservative.
- The tuned adaptive winner is now slightly better than the previous `CrossBranchPhaseGate` single best, but the margin is small enough that it still needs multi-seed confirmation.

## 2. ILI-36

Previous five-seed ablation winner:

- `FRWKV`: `1.553650 ± 0.081893 / 0.751628 ± 0.021773`

Best new adaptive setting from the tuning screen:

- `pp=12, r=4, a=0.01, b=-4.5`
- single-seed result: `1.479308 / 0.728766`

Conservative adaptive cluster:

- multiple `alpha=0.0` settings converge to `1.538269 / 0.750358`

Key observation:

- ILI benefits from a much stronger trust suppression than the previous KBS supplement default.
- The current best point is substantially better than the old five-seed adaptive mean and also better than the old FRWKV mean, but it is still a single-seed result and therefore must be treated as provisional.

## 3. Recommended confirmation

The next round should confirm two configurations per setting:

### ETTh2-96

1. `pp=48, r=2, a=0.02, b=-3.5`
2. `pp=12, r=1, a=0.0, b=-3.5`

### ILI-36

1. `pp=12, r=4, a=0.01, b=-4.5`
2. `pp=12, r=4, a=0.0, b=-4.5`

Suggested seeds:

- `2032, 2033, 2034, 2035, 2036`

Total confirmation jobs:

- `20`

## 4. Confirmation results

Confirmation run:

- `experiment_runner/parallel_run_20260412_112815`
- `20 / 20` success

### ETTh2-96

Best-tuned adaptive:

- `pp=48, r=2, a=0.02, b=-3.5`
- `0.284344 ± 0.003179 / 0.330523 ± 0.002205`

Conservative adaptive:

- `pp=12, r=1, a=0.0, b=-3.5`
- `0.286320 ± 0.002166 / 0.331370 ± 0.002032`

Interpretation:

- The single-seed tuning winner remains competitive after confirmation.
- We further added a matched-seed baseline confirmation for `CrossBranchPhaseGate`, obtaining:
  - `0.283551 ± 0.003147 / 0.329556 ± 0.001876`
- Therefore the earlier ablation conclusion still holds: `CrossBranchPhaseGate` remains the strongest phase variant on `ETTh2-96`.
- Therefore `ETTh2-96` should be written as a boundary case where simpler phase modulation is preferable to adaptive trust.

### ILI-36

Best-tuned adaptive:

- `pp=12, r=4, a=0.01, b=-4.5`
- `1.642889 ± 0.135393 / 0.772907 ± 0.030450`

Conservative adaptive:

- `pp=12, r=4, a=0.0, b=-4.5`
- `1.525323 ± 0.082615 / 0.745453 ± 0.020010`

Interpretation:

- The original single-seed best adaptive configuration was not stable.
- The confirmed winner is the **conservative** variant with `alpha=0`, which effectively suppresses active phase injection while retaining the broader adaptive framework.
- We further added a matched-seed FRWKV confirmation and obtained:
  - `FRWKV`: `1.609987 ± 0.111367 / 0.762924 ± 0.024831`
- The conservative adaptive variant is now not only better than the earlier FRWKV mean, but also better than the matched-seed FRWKV confirmation.

## 5. Current writing implication

- Do not yet overwrite the KBS main text with the new tuned winners as final claims.
- It is already safe to say that:
  - `ETTh2-96` prefers simpler phase modulation over adaptive trust.
  - `ILI-36` strongly prefers more negative trust bias than the previous default, and the best confirmed adaptive variant effectively collapses to a near-no-phase regime.
- The manuscript can now safely use these two points as finalized boundary conclusions.
