# FRWKV+ KBS Release

<p align="right">
  <a href="README.md">
    <img alt="English" src="https://img.shields.io/badge/Language-English-blue?style=for-the-badge">
  </a>
</p>

本仓库是以下论文的开源发布包：

**FRWKV+: Adaptive Periodic-Position Branch Interaction for Frequency-Space Linear Time Series Forecasting**

本仓库打包了 KBS 论文线所需的代码、基准数据、实验 recipe 配置、最终证据清单和论文资产。它的核心目标是让论文中的数值可追溯：每一个最终报告的结果单元都应能连接到本地的具体证据，并且当前论文 recipe 可以从 release 包中直接启动。

## 包含内容

- `src/adaptive_phasegate_kbs/`
  论文使用的核心训练代码、数据加载器、工具函数，以及 FRWKV-family 模型。
- `src/adaptive_phasegate_kbs/configs/kbs_ours_recipes.json`
  当前论文 recipe 配置。主表 recipe 组为 `paper_current.main_table_all`。
- `src/adaptive_phasegate_kbs/dataset/`
  实验中使用的基准 CSV 数据文件。
- `results/final_evidence/`
  最终稿件证据文件，包括 selected-run provenance、selected single-run table provenance、matched 16-seed family ablation、runtime profile evidence、raw matched 16-seed rows，以及直接引用的状态文件和日志文件快照。
- `scripts/verify_release_repro.py`
  用于检查最终证据文件和打包 recipe 配置完整性的脚本。
- `paper/`
  LaTeX 源文件、已编译 PDF、图、参考文献和投稿侧文本资产。
- `scripts/` 和 `jobs/`
  为复跑和审计保留的实验 runner 与历史 job 生成脚本。
- `docs/`
  便于人工阅读的总结文档。部分 4 月份的表格草稿作为历史摘要保留；当前稿件的证据来源以 `results/final_evidence/` 为准。

## 已验证范围

本 release 聚焦 KBS 论文线，以及检查报告结果所需的 FRWKV-family 变体：

- `frwkv`
- `frwkv_crossbranchgate`
- `frwkv_crossbranchphasegate`
- `frwkv_crossbranchphasegate_fullcontextdelta`
- `frwkv_crossbranchperiodicpositiongate_adaptive`
- `frwkv_crossbranchphasegate_adaptive`
- provenance 记录中使用的若干 adaptive embedding/projection 变体

当前论文中的 `Ours` recipe 是 adaptive periodic-position branch-interaction model，对应暴露的模型类型为 `frwkv_crossbranchperiodicpositiongate_adaptive`。

## 环境配置

PyTorch 需要单独安装，便于用户根据自己的硬件选择 CUDA 或 CPU 版本。

### Conda

```bash
conda create -n apgkbs python=3.10 pip -y
conda activate apgkbs
pip install -r requirements-cu128.txt
pip install -e .[notify]
```

仅 CPU 执行环境：

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

## 必需的完整性检查

在 release 根目录执行：

```bash
python scripts/verify_release_repro.py
```

该脚本会检查：

- selected-system provenance 是否包含 28 个可精确追踪的主表单元；
- selected single-run table 是否包含 76 条可追踪记录；
- matched 16-seed FRWKV-family ablation 是否包含 2240 条去重记录，并且没有缺失 seed 组；
- runtime profile 是否包含 28 个已完成运行；
- raw matched 16-seed row file 是否包含 2240 条唯一的 dataset-horizon-model-seed 结果；
- 复制出来的 112 个 source snapshot 是否和 checksum manifest 一致；
- `kbs_ours_recipes.json` 是否包含 28 个当前主表 recipe，以及稿件使用的 ETTh2-96 recipe。

该脚本通过后，表示 release 包含审计论文报告值所需的证据。fresh rerun 的指标仍会受到 PyTorch、CUDA、driver 和 GPU 组合影响，因此不同环境下的数值可能存在轻微差异。

## 复现一个论文 Recipe

推荐入口是 `--config_name`，它会从打包的 recipe 文件加载设置：

```bash
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
python -m adaptive_phasegate_kbs.train \
  --config_name paper_current.adaptive.etth2_pl96 \
  --seed 2034 \
  --model_tag reproduce_etth2_pl96
```

如果只想做语法或 smoke check，而不启动完整实验：

```bash
python -m adaptive_phasegate_kbs.train --help
python -m adaptive_phasegate_kbs.train --config_name paper_current.adaptive.etth2_pl96 --help
```

recipe 配置文件位于：

- `src/adaptive_phasegate_kbs/configs/kbs_ours_recipes.json`

## 最终证据文件

当前面向稿件的证据位于 `results/final_evidence/`：

- `KBS_selected_system_provenance_audit_2026-05-10.csv`
  28 个 dataset-horizon 单元的主表 selected-system provenance。
- `KBS_table4_selected_single_run_provenance_2026-05-11.csv`
  比较 `Ours(selected)` 与 FRWKV-family 变体的表格所用 selected single-run provenance。
- `KBS_full_family_matched16_final_analysis_2026-05-09.json`
  机器可读的 matched 16-seed family-ablation 汇总，包含验证状态、winner count、average rank、paired comparisons 和 claim-strength label。
- `KBS_full_family_matched16_final_analysis_2026-05-09.csv`
  表格形式的 matched 16-seed family-ablation 结果。
- `KBS_full_family_matched16_final_raw_rows_2026-05-09.csv`
  行级 matched 16-seed 证据。最终 family-ablation 分析中使用的每个 dataset-horizon-model-seed 结果对应一行。
- `KBS_main_model_runtime_profile_results_2026-05-11.csv`
  28 个当前主模型 job 的 runtime/profile 证据。
- `source_snapshots/` 和 `source_snapshot_manifest_2026-05-13.csv`
  主表、selected single-run table 和 runtime-profile 记录直接引用的 runner 输出文件快照，以及对应清单。
- `RELEASE_INTEGRITY_AUDIT_2026-05-13.md`
  审计摘要，说明 release 证据支持的内容和仍然存在的复现边界。

较旧的 `results/paper_results.jsonl` 和 `docs/KBS_phasegate_tables_draft.md` 保留下来用于历史审计。当前稿件的最终事实来源请以 `results/final_evidence/` 为准。

## 论文资产

- 源文件：`paper/adaptive_phasegate_kbs.tex`
- 已编译 PDF：`paper/adaptive_phasegate_kbs.pdf`
- 图：`paper/figs/`
- 参考文献：`paper/cas-refs.bib`

release 中的论文副本已经与当前稿件标题、作者信息、funding statement、generative-AI declaration 和最终架构图同步。

## 可复现性说明

- 打包的 CSV 基准文件就是 release 训练代码使用的数据文件。
- 最终证据文件在可获得的范围内保留了 run ID、seed、model type、config name 或 model tag、metric values，以及 source-file references。
- 重新运行可能因为 PyTorch kernels、CUDA libraries、GPU models 和 host load 等因素产生轻微差异。因此，本 release 将两件事分开处理：通过 `scripts/verify_release_repro.py` 检查证据完整性，通过与论文相同的 recipe 配置执行复跑。
- Throughput 数值依赖硬件。runtime profile 证据用于记录当时的 profiling campaign，而不是规定其他机器上的预期速度。
