# Legacy KBS PhaseGate Tables Draft

This file is generated from `results/paper_results.jsonl`, an April-era curated log retained for historical auditability.
For the current manuscript evidence chain, use `results/final_evidence/` and run `python scripts/verify_release_repro.py`.

## Strong-Regime Multi-Seed Completion

| Dataset | Pred | Seeds | Adaptive PhaseGate MSE | Adaptive PhaseGate MAE |
| --- | --- | --- | --- | --- |
| etth1 | 336 | 5 | 0.471136 ± 0.002923 | 0.443831 ± 0.002236 |
| etth1 | 720 | 5 | 0.476644 ± 0.005223 | 0.469053 ± 0.002123 |
| ettm1 | 96 | 5 | 0.312329 ± 0.002187 | 0.343462 ± 0.001872 |
| ettm1 | 192 | 5 | 0.360489 ± 0.001691 | 0.370743 ± 0.001445 |
| ettm1 | 336 | 5 | 0.393606 ± 0.002559 | 0.394765 ± 0.002133 |
| ettm1 | 720 | 5 | 0.456542 ± 0.002926 | 0.430062 ± 0.001244 |
| ettm2 | 96 | 5 | 0.172697 ± 0.001695 | 0.250203 ± 0.001051 |
| ettm2 | 192 | 5 | 0.237061 ± 0.002130 | 0.293408 ± 0.001356 |
| ettm2 | 336 | 5 | 0.297668 ± 0.002829 | 0.332609 ± 0.001770 |
| ettm2 | 720 | 5 | 0.392623 ± 0.002545 | 0.389023 ± 0.001307 |
| weather | 96 | 5 | 0.158733 ± 0.001559 | 0.194827 ± 0.001381 |
| weather | 192 | 5 | 0.206799 ± 0.001316 | 0.240720 ± 0.001436 |
| weather | 336 | 5 | 0.265021 ± 0.001167 | 0.284103 ± 0.001111 |
| weather | 720 | 5 | 0.343600 ± 0.001578 | 0.336639 ± 0.001179 |

## Five-Seed Ablation

| Setting | FRWKV | CrossBranchGate | CrossBranchPhaseGate | Adaptive PhaseGate |
| --- | --- | --- | --- | --- |
| etth2-96 | 0.285056 ± 0.004891 | 0.284967 ± 0.001752 | 0.283826 ± 0.003299 | 0.285394 ± 0.002059 |
| etth2-192 | 0.365279 ± 0.004271 | 0.363300 ± 0.003047 | 0.364038 ± 0.001586 | 0.360196 ± 0.001676 |
| exchange-192 | 0.182735 ± 0.003054 | 0.181880 ± 0.004490 | 0.182792 ± 0.001572 | 0.181739 ± 0.005309 |
| ili-36 | 1.553650 ± 0.081893 | 1.561185 ± 0.078786 | 1.555548 ± 0.040734 | 1.589565 ± 0.104826 |

## Focused Tuning Best Single-Seed Candidates

| Setting | phase\_period\_len | routers | alpha | trust bias | MSE | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| etth2-96 | 48 | 2 | 0.02 | -3.5 | 0.283078 | 0.329018 |
| ili-36 | 12 | 4 | 0.01 | -4.5 | 1.479308 | 0.728766 |

## Confirmation Runs

| Setting | Label | Seeds | MSE | MAE |
| --- | --- | --- | --- | --- |
| etth2-96 | best | 5 | 0.284344 ± 0.003179 | 0.330523 ± 0.002205 |
| etth2-96 | conservative | 5 | 0.286320 ± 0.002166 | 0.331370 ± 0.002032 |
| ili-36 | best | 5 | 1.642889 ± 0.135393 | 0.772907 ± 0.030450 |
| ili-36 | conservative | 5 | 1.525323 ± 0.082615 | 0.745453 ± 0.020010 |

## Matched Baseline Confirmation

| Setting | Model | Seeds | MSE | MAE |
| --- | --- | --- | --- | --- |
| etth2-96 | crossbranchphasegate | 5 | 0.283551 ± 0.003147 | 0.329556 ± 0.001876 |
| ili-36 | frwkv | 5 | 1.609987 ± 0.111367 | 0.762924 ± 0.024831 |
