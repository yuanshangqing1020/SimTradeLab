# 核心仓 + 网格混合策略 v1.0 — 实施计划

> **现状（2026-05）：** SimTradeLab `strategies/core_grid_hybrid_v1/backtest.py` 的 **`handle_data` 默认仅执行 `trend_sizing`**（见 `01-design.md` §9）。下文 Task 以 **v1.0 固定 ref 网格** 为原始交付清单；网格主路径已暂从日线 `handle_data` 移除，相关行为由 **`pick_grid_*` 纯函数 + 单元测试** 覆盖。若恢复网格主路径，可另设 `regime_mode` 并在计划中增补回归任务。

> **For agentic workers:** 按下方 Task 顺序实施；步骤使用 `- [ ]` 勾选跟踪。规格真源见同目录 [`01-design.md`](./01-design.md)。

**Goal：** 新建 `strategies/core_grid_hybrid_v1/backtest.py`（及配套目录），在 SimTradeLab 回测链路内实现 **单标的**、**固定锚定价 `ref`（方案 A）**、**死仓 / 活仓虚拟分账**、**几何网格 + 单档回补配对**、**防御买步长**、**活仓卖光暂停网格**、**整体收益率止盈**；默认 **日线收盘撮合**，**每根 K 线最多一笔网格成交**。

**Architecture：** 策略为 **Ptrade 风格** `initialize` / `handle_data`（与 [`strategies/simple/backtest.py`](../../../strategies/simple/backtest.py) 同族）。在 `context` 上持久化：`ref`、死仓/活仓股数、`round_active`、`last_pair_side`、`last_pair_k`、防御标志、峰值价、`rolling_max_close` 等。买卖单只操作 **单一 `symbol`**，股数 **100 股整数倍**，卖出量 **不超过** 框架给出的可卖数量与 **活仓虚拟库存** 的较小值。

**Tech Stack：** Python 3.x、`BacktestRunner` / `BacktestConfig`；CN 默认 **T+1**（`BacktestConfig` 默认行为）。

---

## 规格覆盖自检（撰稿后）

| 设计 § | 本计划落点 |
|--------|------------|
| §2.1 固定 `ref` + 几何台阶 | §3 价格公式、§4 配对规则 |
| §3.1 状态机 | Task 2、§5 日序 |
| §4 参数 | Task 1 常量表 |
| §5 执行约束 | §6 跳空、§7 T+1/涨停、§8 价格口径 |
| §7 测试 | Task 5、`tests/unit/test_core_grid_hybrid_v1.py` |

---

## 本计划在 `01-design` 上锁死的补充口径（v1.0 唯一真源）

以下若与旧版口头讨论冲突，**以本文为准**。

### 1. 触发价：日线收盘

- **v1.0**：所有网格阈值与防御回撤 **仅用当日 `close`**（通过 `get_history` + 当日 BAR 或等价 API 取到与框架一致的收盘价）。
- **不**做盘中 OHLC 路径推演；**不**在同一 BAR 内先模拟最高再最低。

### 2. 几何台阶（与非防御一致）

- \(P^{sell}_k = ref \times (1 + grid\_step)^k\)  
- \(P^{buy}_k = ref \times (1 - grid\_step)^k\)（**正常模式**）  
- **防御模式**：\(P^{buy}_k = ref \times (1 - defensive\_buy\_step)^k\)；**卖侧**仍用 \(P^{sell}_k\)（`grid_step` + 同一 `ref`）。

### 3. 单档回补配对（锁死）

- 任意时刻，**网格层若存在「未完成配对」**，则 **只处理该配对方向**（不再开新档位的反向单）。
- **无在途配对**（`round_active == False`）时：
  - 若收盘价 **同时** 满足多个卖档或买档，**当日只成交一笔**：优先规则见 **§6「多条件同时满足」**。
- **在途配对**（`round_active == True`）：
  - 若 `last_pair_side == 'sell'` 且配对档为 `k`：**仅当** `close <= P^{buy}_k`（防御模式下用防御公式算出的 \(P^{buy}_k\)）时允许 **买入** 一笔 `grid_lot`（受现金与 100 股约束）；成交后清 `round_active`。
  - 若 `last_pair_side == 'buy'` 且配对档为 `k`：**仅当** `close >= P^{sell}_k` 时允许 **卖出** 一笔（受 T+1 与活仓约束）；成交后清 `round_active`。
- **初始建仓后**：视为无在途配对；**首个**可触发动作：若 `close >= P^{sell}_1` **且** `close <= P^{buy}_1`（仅理论边界）—— **§6** 规定优先级。

### 4. 进入防御（锁死）

- **v1.0 默认只采用「峰值回撤」**：自 **INITIAL_BUILD 完成日收盘价** 起维护 `rolling_peak_close`（历史收盘最高价）。当  
  \[
  (rolling\_peak\_close - close) / rolling\_peak\_close \ge defensive\_trigger\_drawdown
  \]  
  则 **进入** `DEFENSIVE_DOWN`（**不可逆**直至回测结束；v1.0 不设「跌多了再切回正常买步长」）。
- **不**与「相对 `ref` 跌幅」做 OR（避免与峰值回撤重复触发）；若未来要加 `ref` 维，走 v1.1 变更本计划。

### 5. GRID_SUSPENDED_UP（锁死）

- 当 **活仓虚拟库存 `grid_shares == 0`** 且 **不允许卖死仓**，则进入 **GRID_SUSPENDED_UP**：**不再下任何网格买/卖单**（与理念「不追高补活仓」一致），仅 **`EXIT_ALL`** 止盈逻辑仍每日检查。

### 6. 每根 K 线、跳空与多条件（锁死 = 设计 §5 选项 **(i)**）

- **`max_trades_per_bar = 1`**：**每个 `handle_data` 交易日最多一笔网格成交**（买或卖）。
- **跳空越过若干档**：仍以 **§3 配对 + §6 一笔** 处理：当日 **最多** 执行 **一步** 合格撮合（可能「浪费」中间档，属保守口径）。
- **多档同时满足（且无在途配对）**：  
  - **优先卖还是买**：以 **离 `ref` 更近的台阶** 为准——即选 **满足条件的 `k` 最小者**（卖：`close >= P^{sell}_k` 的最小 `k`；买：`close <= P^{buy}_k` 的最小 `k`）；若 **卖、买两侧** 同日均可行，**优先卖**（回笼现金与风控更符合「先兑现上方」直觉）。*（若回测发现不合理，仅在 v1.1 修订本段。）*

### 7. INITIAL_BUILD（锁死）

- **建仓日**：默认可配置为 **回测 `start_date` 首日**（或 `context` 常量 `build_date`）；该日 **在网格逻辑之前** 完成成交。
- **预算**：`deployable = initial_capital * initial_position_ratio`，按 **当日收盘价**（或引擎默认市价口径）计算目标股数  
  `target_shares = floor(deployable / close / 100) * 100`，**一次性买入** `target_shares`。  
- **分账**：`core_shares = floor(target_shares * core_ratio / 100) * 100`，`grid_shares = target_shares - core_shares`（余数归 **活仓**，保证和为 `target_shares`）。
- **`ref`**：该笔买入成交均价 = 成交价（单笔当日唯一价则 `ref = close`）；记入 `context`。
- **剩余现金**留在账户（`initial_capital - 成本 - 费用`）。

### 8. 整体收益率 `EXIT_ALL`（锁死）

- 每日在网格之前或之后（**顺序固定**：建议 **先** `EXIT_ALL` 检查）计算：  
  `ret = (MV + realized_pnl - cumulative_net_deposit) / cumulative_net_deposit`  
  其中 **v1.0**：`cumulative_net_deposit = initial_capital`（无后续入金）；`realized_pnl` 用组合或自行维护的已实现盈亏字段（与框架一致即可）。  
- 若 `ret >= take_profit_ret`：**市价清仓该标的全部持仓**（死仓+活仓合一物理仓），策略进入 **DONE**，后续 `handle_data` 仅记录或 no-op。

### 9. 除权分红

- **v1.0：忽略**；在 `03-optimization-summary.md` 声明「未纠权延续市价回测」。

---

## 文件清单

| 操作 | 路径 | 说明 |
|------|------|------|
| 创建 | `strategies/core_grid_hybrid_v1/backtest.py` | 主策略 |
| 创建 | `strategies/core_grid_hybrid_v1/stats/.gitkeep` | 日志目录占位 |
| 创建 | `tests/unit/test_core_grid_hybrid_v1.py` | 几何价、配对、防御、峰值回撤、单 bar 一单 |
| 修改 | `src/simtradelab/backtest/run_backtest.py` | 增加注释行 `# strategy_name = 'core_grid_hybrid_v1'`（**不**改当前默认策略） |
| 可选 | `my_docs/core_grid_hybrid/v1.0/03-optimization-summary.md` | 首跑后补报告（本阶段可仅占位） |

**勿修改** `grid_multi_asset_v*` 目录行为。

---

## Task 0：纯函数抽层（建议，便于单测）

在 `backtest.py` 顶部或同目录 `grid_math.py`（若希望文件更短）实现无副作用函数：

- `grid_sell_prices(ref, grid_step, max_k=None)` → `list[float]`
- `grid_buy_prices(ref, step, max_k, defensive=False)`：若 `defensive` 则 `step=defensive_buy_step`
- `should_enter_defensive(rolling_peak, close, threshold)` → `bool`
- `pick_grid_action_close(...)` → 返回 `None | ('buy', k) | ('sell', k)` **在单 bar 闭包价下** 符合 §3～§6 的 **至多一个** 动作

单测 **只测** 上述纯函数 + 少量状态机转移，**不**强依赖全量回测。

- [ ] Step 0：实现纯函数签名与 docstring（与 §2～§6 一字面对齐）。
- [ ] Step 1：`tests/unit/test_core_grid_hybrid_v1.py` 覆盖边界（见 Task 5）。

---

## Task 1：`strategies/core_grid_hybrid_v1/backtest.py` 脚手架

- [ ] **Step 1：** 创建目录与 `stats/.gitkeep`、`backtest.py` 文件头注释（策略名、`01-design` / `02-plan` 链接）。
- [ ] **Step 2：** 在 `initialize` 中设置 `context.symbol`（单标的常量，如 `'600519.SS'` 可配置）、`set_benchmark` 可选与标的一致或 000300。
- [ ] **Step 3：** 写入 **默认值**（与 `01-design` §4 表一致）：`initial_position_ratio=0.5`、`core_ratio=0.5`、`grid_step=0.03`、`grid_lot=1000`、`defensive_trigger_drawdown=0.15`、`defensive_buy_step=0.05`、`take_profit_ret=0.20`、`max_grid_level=None`。
- [ ] **Step 4：** `initialize` 末尾初始化状态：`round_active=False`、`last_pair_side=None`、`last_pair_k=None`、`defensive=False`、`built=False`、`done=False`、`rolling_peak_close=None`、`grid_shares=0`、`core_shares=0`、`ref=None`。

---

## Task 2：INITIAL_BUILD 与 `handle_data` 主流程

- [ ] **Step 1：** 若 `context.done`：return。
- [ ] **Step 2：** 首日主流程：**执行建仓**（§7），更新 `core_shares` / `grid_shares` / `ref` / `rolling_peak_close = close`。
- [ ] **Step 3：** 每日：**若未 `done`**，先更新 `rolling_peak_close = max(rolling_peak_close, close)`。
- [ ] **Step 4：** **EXIT_ALL**（§8）：若满足则清仓、`done=True`。
- [ ] **Step 5：** 若 `GRID_SUSPENDED_UP`（§5）：跳过网格。
- [ ] **Step 6：** 若尚未 `defensive` 且 §4 成立：置 `defensive=True`（打日志）。
- [ ] **Step 7：** 调用 **§0 的 `pick_grid_action_close`** 或等价内联逻辑，得到至多一笔动作；与 **`order`** / **`order_target`** 接口对齐下单；成交后更新 **活仓**、**在途配对状态**、**物理持仓假设与框架一致**。
- [ ] **Step 8：** 每次网格 **卖**成交后：`grid_shares -= sold`；**买**成交后：`grid_shares += bought`；**卖**不得多于 **T+1 允许** 与 **`grid_shares`**。
- [ ] **Step 9：** 卖后若 `grid_shares==0`：进入 **`GRID_SUSPENDED_UP`** 状态位。

**日志（最低）：** 建仓日、`ref`、每次状态迁移（防御、暂停网格、止盈）、每笔网格与 `k`。

---

## Task 3：与 SimTradeLab 订单 API 对齐

- [ ] **Step 1：** 阅读框架文档或 [`strategies/grid_trading/backtest.py`](../../../strategies/grid_trading/backtest.py)（若存在）确认 **`order` / `order_target_value`** 与 **持仓读取** 方式。
- [ ] **Step 2：** 确保 **卖出数量** 取 `min(框架可卖, grid_shares)`（减死仓保护）。
- [ ] **Step 3：** 首版若框架在回测中 **较难取单笔成交均价**：允许 `ref = 建仓当日 close` 并在日志注明近似。

---

## Task 4：`run_backtest.py` 注释与本地冒烟

- [ ] **Step 1：** 在 [`run_backtest.py`](../../../src/simtradelab/backtest/run_backtest.py) 增加注释行：`# strategy_name = 'core_grid_hybrid_v1'`。
- [ ] **Step 2：** 本地运行（示例）：

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
# 将 run_backtest 中 strategy_name 临时改为 core_grid_hybrid_v1 后：
poetry run python src/simtradelab/backtest/run_backtest.py
```

- [ ] **Step 3：** 确认 **无异常退出**、日志目录 `strategies/core_grid_hybrid_v1/stats/` 生成文件（若框架写入）。

---

## Task 5：单元测试用例（最小集）

在 `tests/unit/test_core_grid_hybrid_v1.py` 覆盖（示例名）：

- [ ] `test_geometric_prices_k1_k2`：`ref=10`，`step=0.03`，`P_sell_1==10.3`，`P_buy_1≈9.7`（允许浮点容差）。
- [ ] `test_defensive_buy_prices_use_wider_step`：防御后 `P_buy_k` 使用 `defensive_buy_step`。
- [ ] `test_pairing_after_sell_only_buy_at_P_buy_k`：在途卖档 `k=1` 时，价在 `P_buy_2` 也不买。
- [ ] `test_peak_drawdown_triggers_defensive`：峰值 100 → 收盘 84.9 → 回撤 `>0.15`。
- [ ] `test_one_trade_per_bar_counter`：同一组价格序列，断言调用下单次数 per day ≤ 1（可对 mock 计数）。

---

## Task 6：文档与提交

- [ ] **Step 1：** 更新 [`my_docs/core_grid_hybrid/README.md`](../README.md) 表格：`02-plan` 链接指向本文。
- [ ] **Step 2：** `01-design.md` 中若仍写「由 02-plan 二选一」的句子，**在实施完成后** 可在小修订中改为「见 `02-plan` §× 已定案」（可选，避免重复劳动）。
- [ ] **Step 3：** Git commit 建议：`feat(core_grid_hybrid_v1): scaffold strategy per v1.0 plan`（代码就绪时）；仅文档时：`docs(core_grid_hybrid): add v1.0 implementation plan`。

---

## 风险与回滚

| 风险 | 缓解 |
|------|------|
| 框架 T+1 与虚拟分账不一致 | Task 3 以 **`positions` + 自有 `grid_shares`** 双检；首段短回测人工对账一日 |
| 固定 `ref` 导致长期无成交 | 预期内；用 `03-optimization-summary` 披露「网格有效成交次数」 |
| `pick_grid_action` 与实盘口径偏差 | v1.0 已锁 **收盘价**；v1.1 再引入 OHLC |

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-14 | 初稿：锁死触发价、配对、防御、跳空、峰值、分账与日序；任务分解 |

---

*本计划供工程实施用，不构成投资建议。*
