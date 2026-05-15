# 核心仓 + 网格混合 — 回测与实现纪要（v1.x）

**日期：** 2026-05-15  
**策略代码：** `strategies/core_grid_hybrid_v1/backtest.py`  
**默认模式：** `regime_mode='trend_sizing'`（见 `01-design.md` §9）

---

## 1. 回测口径（与仓库入口对齐）

| 项 | 值 |
|----|-----|
| 入口 | `src/simtradelab/backtest/run_backtest.py`，`strategy_name='core_grid_hybrid_v1'` |
| 区间（常用全长） | 2019-01-01～2026-04-20 |
| 标的 | `510300.SS` |
| 本金 | 50 万（`BacktestConfig.initial_capital`） |

运行示例（需在 `SimTradeLab` 下设置 `PYTHONPATH=src` 或已安装包）：

```bash
cd SimTradeLab
PYTHONPATH=src python src/simtradelab/backtest/run_backtest.py
```

---

## 2. 默认 trend_sizing 的典型结果（备忘）

以下为单次本地回测量级，**非业绩承诺**；随数据与依赖版本可能变化。

| 指标 | 量级（约） |
|------|------------|
| 年化收益 | ~3%～4% |
| 最大回撤 | ~-26%～-36% |

**说明：** 单标的宽基 ETF 上同时硬卡「高年化 + 极低回撤」通常不现实；若需控回撤，需扩展资产或规则（见历史讨论纪要，本文不展开）。

---

## 3. 代码与文档关系

| 组件 | 说明 |
|------|------|
| `trend_sizing` | 当前默认执行路径：`order_target_value` + 慢均线 z + 峰值回撤限仓 |
| `pick_grid_action_close` 等 | 保留供单测与设计文档 §2 网格规格对照 |
| 聚宽 | `strategies_jq/core_grid_hybrid/v1/strategy.py` |

---

## 4. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-15 | 首版纪要：默认 trend_sizing、回测口径与结果量级 |

---

*本文档仅供研究与工程记录，不构成投资建议。*
