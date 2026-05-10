# 多标的自适应网格策略 v3.0 — 设计规格

**日期：** 2026-05-10  
**状态：** 已确认（对话头脑风暴定稿）  
**前置版本：** v2.0（`strategies/grid_multi_asset_v2/`），v1.0 文档与口径见 `my_docs/grid_multi_asset/v1.0/`  

---

## 1. 目标与非目标

### 1.1 目标

在 **仅持有现货多头（股票与 ETF）** 的前提下，相对 v2 再引入一层 **机制级** 变化：

- **M1（熊市网格行为）**：在大盘处于下行 regime 时，限制网格的 **净加仓** 或 **最大层数**，避免在持续下跌中被动把弹药打满。
- **M2（防御型 Universe）**：在同一 regime 下，将选股候选池 **收窄** 为预定义的 **低风险宽基/红利等 ETF 子集**（仍全部为多头标的），与 v2 的三档总仓比例、单标 water-filling **正交**，可联合进入 **Walk-Forward + Optuna** 再优化。

验收上需与 v2 **同一时间切分、同一 Holdout** 并列披露：**Holdout**、**2019 年起至约定截止日的全长**、以及 **2021～2022 熊市专项** 三张表，便于与 v2 Trial 29 对照。

### 1.2 非目标（硬约束）

以下 **不得** 纳入 v3.0 设计及回测假设：

- 股指期货、期权等杠杆或做空工具  
- 反向 ETF、做空型或合成空头产品  
- 融券卖空、任何形式的 **Beta 对冲腿** 或「用现金模拟对冲」的虚拟账户  
- 不要求 v3.0 必须与 JoinQuant 同步上线；**SimTradeLab 为规格与实现的首选真源**，JQ 双轨可作后续映射项  

---

## 2. 与 v2 的关系

- **基线：** 从 v2 策略 **fork** 为新目录（建议名：`strategies/grid_multi_asset_v3/`，最终以仓库约定为准）。
- **保留：** `_detect_regime`（沪深300 相对 MA120/MA250 → BULL / NEUTRAL / BEAR）、`context.invested_ratio` 与三档 `BULL_RATIO` / `NEUTRAL_RATIO` / `BEAR_RATIO`、单标权重 water-filling、原有 Universe 打分与网格步长/layer 计算框架。
- **新增：** 仅 **M1**（执行层语义）、**M2**（选股池来源分流），不改变「无衍生品、全多头」的资产类别假设。

---

## 3. 执行顺序（`handle_data`）

与 v2 设计一致，并写明 M1/M2 触点：

```
若 context.day_counter == 1 或 context.day_counter % REBALANCE_FREQ == 0：
    _detect_regime(context)     # 更新 regime、invested_ratio（仅在此处刷新 MA 与状态）
    _refresh_pool(context)      # 【M2】按 regime 选择「全量合并池」或「防御 ETF 子池」并完成换股逻辑

每日：
    _execute_grid(context)      # 【M1】在非 BULL（或可配置 regime）下应用网格加仓限制
```

**约定：**

- `regime` 与 `invested_ratio` **仅在换股日**与 v2 相同方式更新；非换股日使用缓存。
- **M2 只在 `_refresh_pool` 生效**，避免在同一交易日内切换 Universe 语义导致回放不一致。
- **M1 在 `_execute_grid` 生效**，在已应用的 `invested_ratio` 与资金 cap 之内，对已持仓标的限制「较前一日更重」的名义仓位或等价的 layer 上限。

---

## 4. M2：防御型 Universe

### 4.1 行为

- **NEUTRAL / BULL：** 候选池定义与 **v2 完全一致**（指数成分股 + 固定 ETF 列表等，以 v2 `backtest.py` / `template` 为准）。
- **BEAR：** 候选池切换为 **`DEFENSIVE_ETF_POOL`** —— 仅从项目内 **单一常量表**（或可生成列表）选取宽基、红利等低风险 ETF；**不包含**期货、融券、反向产品。

### 4.2 优化参数建议（离散，进入 WF）

| 符号名（建议） | 含义 |
|----------------|------|
| `BEAR_UNIVERSE_MODE` | `SAME`：BEAR 时仍用合并全池（关 M2 对照） / `ETF_DEFENSIVE`：BEAR 仅用防御 ETF 池 |

可选第三档 `ETF_PLUS_LOW_VOL_STOCK` 若后续需要扩大搜索空间，须在实现计划中 **单独评估** 与现有基本面过滤的交互及 trial 数量；**v3.0 规格以两档为默认最小集**，第三档记入「附录·可选扩展」而非本版必达。

---

## 5. M1：熊市网格行为

### 5.1 行为（与 v2 grid 相容）

在 **BEAR**（默认；是否扩展到 NEUTRAL 由离散参数 **`M1_APPLY_REGIMES`** 或等价布尔组定义，建议在实现计划中收口为：**默认仅 BEAR**，避免过早膨胀）下：

| 符号名（建议） | 含义 |
|----------------|------|
| `BEAR_GRID_MODE` | `NORMAL`：与 v2 相同 / `NO_NET_ADD`：相对 **前一交易日持仓**不允许净加仓（可平仓或减仓）/ `CAP_LAYER`：BEAR 下有效最大层数为 `min(GRID_MAX_LAYER, BEAR_GRID_MAX_LAYER_CAP)` |
| `BEAR_GRID_MAX_LAYER_CAP` | 非负小整数，仅当 `BEAR_GRID_MODE=CAP_LAYER` 时参与；建议 WF 候选如 `{0, 1, 2}` |

**NO_NET_ADD 的参照：** 必须使用回测可用的 **无前视** 信息（例如 **前一收盘后**的名义仓位或已实现持仓市值），在具体实现计划中 **单一固定**定义，禁止混用「当日盘中最高价」等易引入歧义的参照。

### 5.2 约束

- `BEAR_GRID_MAX_LAYER_CAP` 在语义上应与 `GRID_MAX_LAYER` 协调（实现上取 `min` 即可）。
- M1 不改变 v2 的 **目标总资金** `min(总资产 × invested_ratio, TARGET_CAPITAL)`，仅约束 **单路径上的加仓方向**。

---

## 6. Walk-Forward 与 Optuna

- **时间切分、Holdout 区间、WF 窗口** 与 v2 `03-optimization-summary` **保持一致**，便于直接对比。
- **参数空间：** v2 已有维度 + §4.2 + §5.1 离散维度；在 `optimize_params.py` 中增加校验：无效组合（如步长上下界）继续拒绝；`CAP_LAYER` 与层数关系见 §5.2。
- **早停、剪枝、journal 断点续传** 与 v2 工程模式对齐，具体数值可在实现计划中引用 v2 当前 `patience` 等并评估是否因维度增加而调整。
- **报告义务：** 除最优 trial 的 Holdout 与全长外，必须输出 **2021-01～2022-12（或文档与代码中锁定的熊市窗）** 专项指标表。

---

## 7. 测试与验收

- **单元测试：** 基于 v2 测试结构新增：`_refresh_pool` 在 `BEAR` + `ETF_DEFENSIVE` 下候选仅为防御池子集；`_execute_grid` 在 `NO_NET_ADD` 下对合成价格路径 **持仓价值不高于前一日**（在固定 cap 与无外部入金假设下）。
- **回归：** v3 不得破坏 v2 目录内代码行为；v3 为独立包路径。
- **文档：** 在 `my_docs/grid_multi_asset/v3.0/` 下按既有规范补充 `01-design.md`（本规格可摘要转发）、`02-plan.md`、`03-optimization-summary.md`（优化完成后撰写）。

---

## 8. 附录：可选的后继项（不纳入 v3.0 必达）

- `BEAR_UNIVERSE_MODE` 第三档与股票滤波的联合 WF  
- JoinQuant `v3/strategy.py` 与模板对齐  
- 将 WF 次级目标改为显式加权「全长最大回撤」等（需在优化脚本层改目标函数，单列版本）  

---

## 规格自检（填写时已完成）

| 检查项 | 结果 |
|--------|------|
| 占位/TBD | 无未决占位；可选第三 universe mode 标明为附录 |
| 内部一致性 | M2 仅在换仓日；M1 在每日网格；均无衍生品 |
| 范围 | 单版本机制 + WF；对冲与做空已排除 |
| 歧义 | NO_NET_ADD 要求实现计划固定「前一日参照」定义 |
