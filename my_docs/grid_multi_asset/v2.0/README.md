# grid_multi_asset v2.0

多标的自适应网格 **v2**：在 v1 基础上增加大盘 regime（MA120 / MA250）与三档总仓位比例（BULL / NEUTRAL / BEAR），以及单标的权重 water-filling 上限。

## 文档索引

| 文件 | 内容 |
|------|------|
| [01-design.md](./01-design.md) | 设计说明与测试清单 |
| [02-plan.md](./02-plan.md) | 任务分解与实施顺序 |
| [03-optimization-summary.md](./03-optimization-summary.md) | **Walk-Front + Holdout 调参结论（已完成）** |

## v2 Walk-Front 最优（Trial 29）

结果文件：`strategies/grid_multi_asset_v2/optimization/results/best_params_20260510_011341.json`  
代码已写入：`backtest.py`、`template.py`（及生成的 `optimization/optimized_strategy.py`）。

| 摘要 | 数值 |
|------|------|
| WF 最终综合得分 | **-0.3457** |
| Holdout 综合得分（未参与优化） | **0.9019** |
| Holdout 年化 / 夏普 / 最大回撤 | **+28.94%** / **1.43** / **-12.35%** |

## 对照 v1 Holdout（同区间 2025-01～2026-03）

v1（Trial 53）在同段 Holdout 上收益更高（约年化 +60%），但回撤与市场 Beta 更大；v2 **回撤与 Beta 更小**，更接近「risk-off 结构」。详见 [03-optimization-summary.md](./03-optimization-summary.md) §六。

## 运行（conda 环境 `SimTrade`）

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab

# Holdout（默认已与优化脚本对齐）
/root/miniconda3/envs/SimTrade/bin/python src/simtradelab/backtest/run_backtest.py

# 继续或重跑 WF 优化（断点续传）
/root/miniconda3/envs/SimTrade/bin/python strategies/grid_multi_asset_v2/optimization/optimize_params.py

# v2 单元测试
/root/miniconda3/envs/SimTrade/bin/python -m pytest tests/unit/test_grid_multi_asset_v2.py -q
```

## 后续工作（简述）

- **JoinQuant：** 更新对照表状态见 `strategies_jq/grid_multi_asset/README.md`，`v2/strategy.py` 仍待从 `template.py` 映射粘贴。
- **研究：** 熊市专用区间截面、模拟盘、`run_backtest.py` 全样本长周期冒烟。
