# Adaptive PhaseGate 七数据集总表

当前主线模型：
- `FRWKV-CrossBranchPhaseGateAdaptive`

当前已覆盖数据集：
- `ETTh1`
- `ETTh2`
- `ETTm1`
- `ETTm2`
- `Weather`
- `Exchange`
- `ILI`

说明：
- `ETTh1 / ETTh2` 的 `96/192` 使用 tuned multiseed 结果
- `ETTh2 / Exchange / ILI` 已进一步补充高 seeds 复核结果
- `ETTh1-336/720`、`ETTm1`、`ETTm2`、`Weather` 已进一步补充 `5 seeds` 复核结果
- `ILI` 的 protocol 为 `seq_len=36`, `pred_len in {24,36,48,60}`
- `stage2` 高 seeds 复核来自：
  - `experiment_runner/parallel_run_20260409_170109`
  - 共 `288` 个任务，`284` 成功，`4` 个因共享服务器下的外部显存占用触发 OOM
- `strong-regime` 补实验来自：
  - `experiment_runner/parallel_run_20260411_181159`
  - `experiment_runner/parallel_run_20260411_220655`
  - 共 `150` 个任务，`150/150` 成功，其中 `70` 个为 strong-regime adaptive multi-seed completion，`80` 个为结构消融

## 1. ETT / Weather 主表

| Dataset | Pred | MSE | MAE | Notes |
| --- | --- | --- | --- | --- |
| ETTh1 | 96 | `0.372933 ± 0.002223` | `0.389600 ± 0.001283` | tuned multiseed |
| ETTh1 | 192 | `0.426700 ± 0.000748` | `0.421100 ± 0.000082` | tuned multiseed |
| ETTh1 | 336 | `0.471136 ± 0.002923` | `0.443831 ± 0.002236` | 5-seed supplement |
| ETTh1 | 720 | `0.476644 ± 0.005223` | `0.469053 ± 0.002123` | 5-seed supplement |
| ETTh2 | 96 | `0.284183 ± 0.002139` | `0.330405 ± 0.001570` | combined 15 seeds |
| ETTh2 | 192 | `0.362149 ± 0.002176` | `0.381240 ± 0.001497` | combined 15 seeds |
| ETTh2 | 336 | `0.403581 ± 0.005778` | `0.412894 ± 0.002319` | combined 15 seeds |
| ETTh2 | 720 | `0.418079 ± 0.005672` | `0.433015 ± 0.002529` | combined 15 seeds |
| ETTm1 | 96 | `0.312329 ± 0.002187` | `0.343462 ± 0.001872` | 5-seed supplement |
| ETTm1 | 192 | `0.360489 ± 0.001691` | `0.370743 ± 0.001445` | 5-seed supplement |
| ETTm1 | 336 | `0.393606 ± 0.002559` | `0.394765 ± 0.002133` | 5-seed supplement |
| ETTm1 | 720 | `0.456542 ± 0.002926` | `0.430062 ± 0.001244` | 5-seed supplement |
| ETTm2 | 96 | `0.172697 ± 0.001695` | `0.250203 ± 0.001051` | 5-seed supplement |
| ETTm2 | 192 | `0.237061 ± 0.002130` | `0.293408 ± 0.001356` | 5-seed supplement |
| ETTm2 | 336 | `0.297668 ± 0.002829` | `0.332609 ± 0.001770` | 5-seed supplement |
| ETTm2 | 720 | `0.392623 ± 0.002545` | `0.389023 ± 0.001307` | 5-seed supplement |
| Weather | 96 | `0.158733 ± 0.001559` | `0.194827 ± 0.001381` | 5-seed supplement |
| Weather | 192 | `0.206799 ± 0.001316` | `0.240720 ± 0.001436` | 5-seed supplement |
| Weather | 336 | `0.265021 ± 0.001167` | `0.284103 ± 0.001111` | 5-seed supplement |
| Weather | 720 | `0.343600 ± 0.001578` | `0.336639 ± 0.001179` | 5-seed supplement |

## 2. Exchange / ILI

| Dataset | Pred | MSE | MAE | Notes |
| --- | --- | --- | --- | --- |
| Exchange | 96 | `0.085117 ± 0.002682` | `0.203587 ± 0.003779` | combined 15 seeds |
| Exchange | 192 | `0.182962 ± 0.006497` | `0.304242 ± 0.005802` | combined 15 seeds |
| Exchange | 336 | `0.349942 ± 0.008508` | `0.427296 ± 0.005935` | combined 15 seeds |
| Exchange | 720 | `0.962210 ± 0.120839` | `0.735182 ± 0.036636` | combined 15 seeds |
| ILI | 24 | `1.550766 ± 0.088796` | `0.746606 ± 0.024304` | combined 14 seeds |
| ILI | 36 | `1.628859 ± 0.102518` | `0.765066 ± 0.028000` | combined 15 seeds |
| ILI | 48 | `1.776425 ± 0.140742` | `0.799356 ± 0.031408` | combined 15 seeds |
| ILI | 60 | `1.684156 ± 0.035190` | `0.792787 ± 0.009331` | combined 15 seeds |

## 3. 当前判断

### 表现较强的数据集
- `ETTh1`
- `ETTm1`
- `ETTm2`
- `Weather`

### 表现中等但可接受
- `ETTh2`
- `ILI`

### 当前最弱项
- `Exchange`
- `Exchange` 的所有 horizon 仍然明显弱于其余强周期数据集
- `ILI-36` 在最新的结构消融里对 adaptive phase modulation 不够友好

## 4. 最值得继续改进的空间

1. **dataset-aware phase hyperparameters**
   - `phase_period_len` 在：
     - `ETTh1` 更偏 `12/24`
     - `ETTh2 / ETTm1 / ETTm2` 更偏 `24/48/96`
     - `Weather` 更偏 `12/24`
   - 这说明 phase scale 需要适配数据域

2. **弱周期 / 健康序列专项修正**
   - `Exchange` 仍然是最弱项
   - `ILI` 虽然已经通过高 seeds 复核，但整体方差仍然偏大
   - 最新的 `5-seed` 结构消融还表明：
     - `ETTh2-96` 上 plain `CrossBranchPhaseGate` 比 adaptive 更稳
     - `ILI-36` 上 `FRWKV` 基线反而最好
   - 如果继续补实验，最值得做的是：
     - 更保守的相位注入强度
     - 更短的 phase period
     - 针对不同 horizon 的 phase 策略

3. **共享 GPU 场景下的稳定复核**
   - `stage2` 的 `4` 个失败任务全部是 OOM，而不是模型逻辑错误
   - 这说明当前方法在共享服务器环境下需要更保守的显存策略
   - 如果后续还补实验，应优先：
     - 降 `batch_size`
     - 降 `num_workers`
     - 或等待空闲显存后自动重跑

## 5. 一句话总结

`Adaptive PhaseGate` 在七个数据集里已经证明对强周期结构数据集有效，但最新 multi-seed completion 说明它在 `ETTh1 / ETTm1 / ETTm2 / Weather` 上更适合被描述为“整体有竞争力且通常优于 CrossBranchGate”，而不是在每个 horizon 都显著占优；在 `ETTh2` 上 `192/336/720` 经过高 seeds 复核后较稳，但 `ETTh2-96` 仍值得继续调 phase；`Exchange` 依旧是最明显的弱项，`ILI` 尤其是 `36` 步长则提示 adaptive phase trust 需要更保守。 
