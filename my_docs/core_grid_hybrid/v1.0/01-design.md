# 核心仓 + 网格混合策略 v1.0 — 设计文档

**版本：** v1.0（设计已定稿：**网格基准 = 方案 A 固定锚定价**）  
**日期：** 2026-05-14  
**已定决策：** 回测与实现以 **A** 为唯一口径；B/C 仅作 v1.x 扩展备选，不得与 v1.0 混用。  
**规格真源（规划）：** `strategies/core_grid_hybrid_v1/`（`backtest.py` + `stats/`）  
**理念来源：** [`core_grid_hybrid_strategy /README.MD`](../../core_grid_hybrid_strategy%20/README.MD)（`my_docs` 内理念说明）  
**目录范式参考：** [`my_docs/grid_multi_asset/`](../grid_multi_asset/README.md) 与 [`strategies/grid_multi_asset_v5/`](../../../strategies/grid_multi_asset_v5/)

---

## 1. 目标与非目标

### 1.1 目标（v1.0）

1. **单标的、单账户、仅现货多头** 下，实现「**战略死仓 + 网格活仓 + 备用现金**」三层分工；与理念文档 §2～§5 数值语义可对齐（参数化，不限死 20 万 / 10 元示例）。
2. **SimTradeLab 可回测**：通过 `BacktestConfig(strategy_name='core_grid_hybrid_v1', ...)` 与现有 [`runner`](../../../src/simtradelab/backtest/runner.py) 链路接入；**CN 市场默认 T+1**（[`BacktestConfig.t_plus_1`](../../../src/simtradelab/backtest/config.py)）。
3. **可观测性**：日志或导出中至少能还原：各日持仓（总股数、活仓可用卖量、现金）、网格状态（**固定 `ref`**、当前触及的买卖台阶索引、是否防御模式、是否因卖光活仓暂停网格）、**整体收益率**（与理念 §5 公式一致）。
4. **v1.0 验收**：给定标的与一段历史，回测可完整跑通且无未定义行为（跳空、涨跌停可按 §5 最小口径处理）。

### 1.2 非目标（v1.0 明确不做）

| 类别 | 说明 |
|------|------|
| 多标的资金分配、Universe、换仓 régime | 属 `grid_multi_asset` 族；本策略 v1.0 **单标的**，不引入锚定池/水填充。 |
| 与 `grid_multi_asset` **相同的三重门禁（I/II/III）** | v1.0 **不强制** 000300 相对收益门槛；可在 `03-optimization-summary` 中 **可选** 披露对比，作为研究报告而非 Hard Gate。 |
| 聚宽双轨强制同步 | 同 v5 约定：**不强制** `strategies_jq`；需要时另开 v1.x。 |
| 全自动基本面选股 | v1.0 **标的由配置传入**（代码 + 初始日期）；文档保留「好公司+低估」为外在假设。 |

---

## 2. v1.0 真源：方案 A（固定锚定价）

### 2.1 已定案行为

- **`ref`：** 等于 **INITIAL_BUILD** 完成后、首笔批次建仓的 **成交均价（VWAP）**，**全程不变**。
- **买卖台阶（相对 `ref` 的几何网格，与理念「每档约 ±3%」一致）：**
  - 第 \(k\) 档 **卖出** 参考触发价：\(P^{sell}_k = ref \times (1 + grid\_step)^k\)，\(k = 1, 2, \ldots\) 直至可选上限 `max_grid_level`（若未配置则由现金/活仓耗尽自然停止）。
  - 第 \(k\) 档 **买入** 参考触发价：\(P^{buy}_k = ref \times (1 - grid\_step)^k\)（**非防御**；与卖侧对称的几何下跌台阶）。
- **当日判定：** 使用回测频率对应的价格序列（默认 **日线收** 或引擎提供的等价价）与上述阈值比较；**已成交过的档位是否允许重复触发** 由 `02-plan` 二选一锁死（推荐：**每档每方向仅「回补」式配对**，即卖出成交后同档回落再买、买入成交后同档反弹再卖——与理念文档纪律一致；实现上等价于维护「当前未完成配对」而非无限重复同一 \(k\)）。
- **防御模式（DEFENSIVE_DOWN）：** 仅 **买单侧** 改用 `defensive_buy_step` 重算 \(P^{buy}_k\)（仍从同一 `ref` 出发：**\(ref \times (1 - defensive\_buy\_step)^k\)**）；**卖单侧** 仍用原 `grid_step` 与同一 `ref`。若与已进入防御前已挂单逻辑冲突，以 **`02-plan`** 中顺序说明为准。

### 2.2 B / C 方案（非 v1.0）

| 方案 | 说明 |
|------|------|
| **B. 成交滚动基准** | v1.1+ 再评估；不纳入 v1.0 代码路径。 |
| **C. 离散档位表（占用标记）** | 若未来需要更精细的跳空多档撮合，可升维至 C；v1.0 用 A + **`02-plan` 约定跳空口径** 即可。 |

---

## 3. 策略状态机与资金记账

### 3.1 状态（概念）

1. **INIT**：回测开始，无仓。  
2. **INITIAL_BUILD**：在配置规定的「建仓日/价」（或首日市价）买入 **目标总股数的一半（向下取整到 100 股）**；另一半保留现金。将该笔总股数拆分为：**死仓股数**、**活仓股数**（比例默认 50:50，可配置）。记录 **建仓 VWAP → `ref`**。  
3. **GRID_ACTIVE**：按 §2.1 **固定 `ref`** 维护几何档位；卖单 **仅能动用活仓库存**（及当日 T+1 可卖量）；买单受现金与 `max_buy` 约束。  
4. **DEFENSIVE_DOWN**：自 ENTER 网格起 **标的收市价相对 `ref` 回撤** 或相对 **建仓后峰值价** 回撤达阈值（默认 **约 -15%**，与理念对齐）→ 进入防御：**买单** 改用 `defensive_buy_step`（§2.1）；可选 **`grid_lot` 缩小**；**卖单侧规则不变**（仍 `grid_step` + 固定 `ref`）。  
5. **GRID_SUSPENDED_UP**：**活仓可卖数量为 0**且规则禁止「卖死仓」→ **停止网格**；仅持有死仓 + 现金，直至 §5 整体止盈。  
6. **EXIT_ALL**：整体收益率 ≥ 阈值 → **清仓该标的全部头寸**，策略本轮结束（v1.0 单标的回测可等价于结束回测或保留现金 idle）。

### 3.2 整体收益率（与理念 §5 对齐）

定义在 **单标的维度**：

\[
\text{ret} = \frac{MV + \text{实现盈亏累计} - \text{累计净投入}}{\text{累计净投入}}
\]

其中「实现盈亏累计」含网格已实现；**净投入** 与买卖资金流一致（初始资金可视为首次净投入，后续是否加仓 v1.0 **默认无加仓**，仅首笔半仓建仓 + 网格）。

**v1.0 调参阈值默认：** `take_profit_ret = 0.20`。

---

## 4. 参数化（建议最小集合）

| 参数 | 含义 | 示例 / 默认 |
|------|------|-------------|
| `symbol` | 标的代码 | 配置必选 |
| `initial_capital` / `max_risk_budget` | 与 BacktestConfig 一致或可覆盖 | 对齐项目常用 50 万或单票 20 万 |
| `initial_position_ratio` | 首轮建仓占用预算比例 | 0.5 |
| `core_ratio` | 死仓占「已买入股数」比例 | 0.5 |
| `grid_step` | 基础步长 | 0.03 |
| `grid_lot` | 每格股数（100 整数倍） | 1000（可缩） |
| `defensive_trigger_drawdown` | 进入防御 | 0.15 |
| `defensive_buy_step` | 防御买步长 | 0.05 或 0.08 |
| `take_profit_ret` | 整体止盈 | 0.20 |
| `grid_reference_mode` | v1.0 固定为 **`FIXED_REF`（方案 A）** | 不设枚举开关；扩展移至 v1.1+ |
| `max_grid_level` | 单侧最大档位数（可选） | `None` 表示不限制（由资金/仓存耗尽约束） |

---

## 5. 执行与约束（最小可行口径）

| 项 | v1.0 口径 |
|----|------------|
| **T+1** | 使用框架默认 CN=True；卖出数量 ≤ 昨收可卖 + 规则内释放（与引擎一致）。 |
| **100 股** | 所有订单数量向下取整到 100。 |
| **手续费** | 继承回测/券商 profile；设计不另造费率模型。 |
| **涨跌停** | 若框架已有无法成交处理则沿用；否则 v1.0 记 **当日不成交** 并在总结中披露简化假设。 |
| **跳空** | **方案 A 下：** 若单根 K 线开盘价/收盘价 **一次性越过** 本应多档依次成交的区间，在 `02-plan` **二选一并单测**：**(i)** 该 bar **最多成交一笔**（保守）；(ii) 同一 bar 内 **按时间顺序** 仅能成交 **不超过 `max_trades_per_bar`** 笔（通常 1）。默认倾向 **(i)** 以降低争议。 |
| **分红送转** | v1.0 可 **忽略** 或 **框架若支持则继承**；须在 `03-optimization-summary` 写明。 |

---

## 6. 仓库目录结构（参考 grid_multi_asset）

与 [`my_docs/core_grid_hybrid/README.md`](../README.md) 一致，摘要如下：

- **文档：** `my_docs/core_grid_hybrid/v1.0/{01-design.md,02-plan.md,03-optimization-summary.md}`  
- **代码：** `strategies/core_grid_hybrid_v1/backtest.py`（及可选 `optimization/`、`stats/`）  
- **入口：** `run_backtest.py` 中 `strategy_name='core_grid_hybrid_v1'`（实现阶段再加）

---

## 7. 测试与交付

| 项 | 要求 |
|----|------|
| **单测** | `tests/unit/test_core_grid_hybrid_v1.py`（或项目约定路径）：档位触发、T+1 可卖量、活仓卖光暂停、整体收益率止盈、防御模式步长切换。 |
| **对比基线** | 同区间 **买入持有（半仓一次性买入与建仓日对齐）** 或全仓 B&H，便于报告相对超额（非门禁）。 |
| **文档闭环** | 设计（本文）→ 计划（`02-plan.md`）→ 调参总结（`03-optimization-summary.md`） |

---

## 8. 已知风险（规格层）

1. **单标的 + 高信念** 放大 idiosyncratic 风险；v1.0 规格不包含行业分散。  
2. **固定锚定价** 在长期单边行情中可能长期不靠档成交，依赖 **整体止盈** 与 **活仓卖光暂停**；若 `ref` 离现价过远，**网格贡献可能长期为 0**（属 A 方案预期行为，在总结中需披露）。  
3. 与 `grid_multi_asset_v5` **不可数值对标**：逻辑与持仓结构不同，仅可并列展示作为「另一条产品线」。

---

## 9. 附录 A：SimTradeLab 当前默认实现（trend_sizing，2026-05）

与 §2～§3 描述的 **v1.0 固定 ref 网格主路径** 并列存在；**`strategies/core_grid_hybrid_v1/backtest.py` 默认 `regime_mode='trend_sizing'`，`handle_data` 仅执行本节逻辑**，不再执行 INITIAL_BUILD + GRID_ACTIVE 日序（网格相关能力以 **纯函数** + **`tests/unit/test_core_grid_hybrid_v1.py`** 保留）。

### 9.1 因子与公式

1. **慢均线 MA**：使用在写入**当日**收盘**之前**的日终收盘序列（避免前视），窗口 `ma_slow_days`；历史不足时用 `min(ma_slow_days, len)`，但至少 **`ma_min_days`** 根后才交易。
2. **\(z = (C - MA) / MA\)**。若 `z < ts_z_cut` 则目标权重 `w = 0`，否则 `w = clip(ts_w_mid + z * ts_w_slope, ts_w_floor, ts_w_cap)`。
3. **峰值回撤**：维护收盘序列上的最高价 `ts_peak_close`；若 `(peak - C)/peak > ts_peak_dd_cut`，则 `w = min(w, ts_w_peak_stress)`。
4. **再平衡**：目标市值 `pv * w`（`pv` = 组合净值）；若 `|w - 当前股票市值/pv| < ts_rebalance_band` 则跳过下单。
5. **整体止盈**：若 `context.portfolio.returns >= take_profit_ret`（默认 **9.99**，近似不触发）则清仓并标记 `done`。

### 9.2 参数一览（以 `initialize` 为准）

| 参数 | 默认（约） | 含义 |
|------|------------|------|
| `symbol` | 510300.SS | 单标的 |
| `ma_slow_days` | 120 | 慢均线窗口 |
| `ma_min_days` | 20 | 最短历史长度 |
| `ts_z_cut` | -0.078 | 低于此 z 则空仓 |
| `ts_w_floor` / `ts_w_cap` / `ts_w_mid` / `ts_w_slope` | 0.52 / 0.99 / 0.68 / 3.5 | 权重映射 |
| `ts_rebalance_band` | 0.026 | 再平衡死区 |
| `ts_peak_dd_cut` | 0.10 | 峰值回撤阈值 |
| `ts_w_peak_stress` | 0.32 | 峰值压力下权重上限 |

### 9.3 聚宽对照

`510300.SS` → `510300.XSHG`；`strategies_jq/core_grid_hybrid/v1/strategy.py`，说明见 `strategies_jq/core_grid_hybrid/README.md`。

---

## 10. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-14 | 初稿：目标/非目标、三方案、状态机、目录与测试 |
| 2026-05-14 | **定稿：选用方案 A**；§2 改为 A 的数学定义与 B/C 降级；§4～§5、§8 同步 |
| 2026-05-15 | **§9 附录 A**：补充当前仓库默认 **trend_sizing** 与 v1.0 网格设计文档的关系；聚宽路径 |

---

*本文档为策略与工程设计说明，不构成投资建议。*
