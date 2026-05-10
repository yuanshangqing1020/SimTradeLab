# 多标的自适应网格策略 v3.0 — 实施计划

> **For agentic workers:** 按需使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans` 按任务逐步执行。步骤统一使用 checkbox（`- [ ]`）勾选进度。

**Goal:** 在 v2 策略 fork（`strategies/grid_multi_asset_v3/`）上实现 **M1（熊市网格语义）** 与 **M2（BEAR 防御 ETF 候选池分流）**，并扩展 Walk-Forward + Optuna 参数空间做新一轮优化；回测口径与 [v2.0 `03-optimization-summary.md`](../v2.0/03-optimization-summary.md) **时间轴、Holdout 对齐**，并增补 **2021-01～2022-12 熊市专项** 报表。**禁止**引入期货、反向 ETF、融券及任何对冲腿（见 [01-design.md](./01-design.md)）。

**Architecture:** **最小侵入**：保留 v2 `_detect_regime`、`invested_ratio`、water-filling、打分与网格主体；仅在 `_refresh_pool` 增加按 `regime` + `BEAR_UNIVERSE_MODE` 的候选池分叉；在 `_execute_grid` 增加 BEAR（默认）下按 `BEAR_GRID_MODE` 的 **有效层上限**或 **日前收盘持仓** 基准的加仓约束；在日终钩子中写入「上一日收盘可用于 M1 的持仓市值快照」。优化器沿用 `optimizer_framework.optimize_strategy`，通过 `template.py` 行内 `context.* =` 正则替换注入参数（字符串枚举已支持）。

**Tech Stack:** Python 3.8+、NumPy、Pandas、Optuna、SimTradeLab/PTrade 回测；JoinQuant 映射为可选后置任务。

---

## 文件清单

| 操作 | 路径 | 说明 |
|------|------|------|
| 创建 | `strategies/grid_multi_asset_v3/template.py` | 自 v2 复制后改 M1/M2 |
| 创建 | `strategies/grid_multi_asset_v3/backtest.py` | 与 template 同步；WF 完成后写入最优参数 |
| 创建 | `strategies/grid_multi_asset_v3/optimization/optimize_params.py` | v3 参数空间 + 与 v2 相同 WF/Holdout 配置 |
| 创建 | `strategies/grid_multi_asset_v3/optimization/results/.gitkeep` | 保证目录存在 |
| 创建 | `strategies/grid_multi_asset_v3/stats/.gitkeep` | 同上 |
| 创建 | `tests/unit/test_grid_multi_asset_v3.py` | M1/M2 纯逻辑与轻量集成测试 |
| 可选 | `strategies_jq/grid_multi_asset/README.md` | 增加 v3 行（非必达） |
| 可选 | `strategies_jq/grid_multi_asset/v3/strategy.py` | 聚宽版（非 v3.0 必达） |
| 文档 | `my_docs/grid_multi_asset/v3.0/03-optimization-summary.md` | WF 与分段回测完成后撰写 |

**说明：** `run_backtest.py` 仅通过 `strategy_name = 'grid_multi_asset_v3'` 切换策略；`BacktestConfig.strategy_path` 已解析为 `strategies/{strategy_name}/backtest.py`，**无需**改框架注册表。

---

## M1 / M2 实现要点（编码前约定）

### M2：`DEFENSIVE_ETF_POOL`

- 在 `template.py` 顶部维护 **单一常量列表**（`DEFENSIVE_ETF_POOL`），元素为 `CANDIDATE_ETFS` 的**子集**（宽基/行业龙头 ETF，与 v2 品种一致，仅缩小数量）。
- `_refresh_pool` 内：若 `context.regime == 'BEAR'` 且 `context.BEAR_UNIVERSE_MODE == 'ETF_DEFENSIVE'`，则 **股票指数成分不计入本次候选**，仅对 `DEFENSIVE_ETF_POOL` 做流动性/ST/停牌过滤与波动率打分；若该池经过滤后为空，打日志并 **保留原池**（与 v2「保留原池」降级一致）。
- `BEAR_UNIVERSE_MODE == 'SAME'` 时行为与 v2 **完全一致**。

### M1：`BEAR_GRID_MODE`

- **默认仅当 `context.regime == 'BEAR'`** 时启用 M1；`NORMAL` 时与 v2 网格相同。
- **`CAP_LAYER`：** `effective_max_layer = min(context.GRID_MAX_LAYER, context.BEAR_GRID_MAX_LAYER_CAP)`，传入 `_calc_layer`。
- **`NO_NET_ADD`：**  
  - **参照口径（固定）：** 使用 **上一交易日日终** 后、本日 `_execute_grid` 执行前可读到的 **各标的持仓市值**（`context.portfolio.positions` 中能稳定取得的市值或 `amount × 昨收` 等价量，**禁止**用当日未收盘价做「前视」）。  
  - 在 `after_trading_end`（或框架保证在下一交易日 grid 前已更新的快照）中写入 `context._prev_eod_position_value: dict[str, float]`（仅池内或全持仓，实现选一种并全文一致）。  
  - `_execute_grid` 在计算 `order_target_value` 前：对每只 active 标的，若今日目标市值 **大于** `_prev_eod_position_value.get(code, 0.0)`，则 **clamp 至该上限**（允许小于等于）；减仓不受此限制。  
- **`NORMAL`：** 不套用上述两项。

### 参数默认值（写入 `initialize`，供优化器替换）

```text
context.BEAR_UNIVERSE_MODE      = 'SAME'           # 'SAME' | 'ETF_DEFENSIVE'
context.BEAR_GRID_MODE          = 'NORMAL'         # 'NORMAL' | 'NO_NET_ADD' | 'CAP_LAYER'
context.BEAR_GRID_MAX_LAYER_CAP = 1               # 仅 CAP_LAYER 有意义；仍为整数
```

### 优化空间与组合数

- v2 的 11 个维度保持类属性列表形式不变（见 v2 `GridMultiAssetV2Params`）。  
- v3 类 **新增**：`BEAR_UNIVERSE_MODE`、`BEAR_GRID_MODE`、`BEAR_GRID_MAX_LAYER_CAP`（例如 `[0, 1, 2]`）。  
- 理论笛卡尔积为 **原 v2 组合数 × 2 × 3 × 3**（体量显著上升）。**冒烟建议：** 第一轮可将 `BEAR_GRID_MAX_LAYER_CAP` 收成单点（如 `[1]`）或暂时固定若干 v2 维为 v2 Trial 29 最优，待逻辑跑通后再放开；或在 `optimize_params.py` 中显式传入更大的 `patience`、依赖剪枝Early Stop。  

**`validate(params)`：** 继承 v2 的步长与 `BEAR_RATIO < NEUTRAL_RATIO < BULL_RATIO`；另：若 `BEAR_GRID_MODE != 'CAP_LAYER'`，仍可接受任意 `BEAR_GRID_MAX_LAYER_CAP`（忽略即可）；若 `CAP_LAYER`，建议要求 `BEAR_GRID_MAX_LAYER_CAP <= params['GRID_MAX_LAYER']`，否则 `ValueError` 拒绝 trial。

---

## Task 1：目录与 fork

**Files:**  
创建 `strategies/grid_multi_asset_v3/` 骨架；从 `grid_multi_asset_v2` **复制** `template.py`、`backtest.py`、`optimization/optimize_params.py` 作为起点。

- [ ] **Step 1:** 复制 v2 → v3，重命名文件中注释头「v3」、`backtest.py` 说明。
- [ ] **Step 2:** 新建 `optimization/results/.gitkeep`、`stats/.gitkeep`。
- [ ] **Step 3:** Commit：`chore: scaffold grid_multi_asset_v3 from v2`

---

## Task 2：TDD — 纯函数与子集测试先行

**Files:** Create `tests/unit/test_grid_multi_asset_v3.py`

建议在 `template.py` 中提取可单测函数（与 v2 同款 `exec`+mock API 加载方式）：

| 函数（建议名） | 测什么 |
|----------------|--------|
| `_effective_max_layer_for_bear(grid_max, bear_mode, cap)` | NORMAL→grid_max；CAP_LAYER→min；其他模式可返回 grid_max |
| `_defensive_etf_list(full_etfs, mode)` | `ETF_DEFENSIVE`→子集，`SAME`→full |
| `_clamp_targets_no_net_add(prev_values, targets)` | dict 层面上目标不高于 prev |

- [ ] **Step 1:** 写上述测试（先失败）。
- [ ] **Step 2:** `conda run -n SimTrade pytest tests/unit/test_grid_multi_asset_v3.py -q` 确认为红。
- [ ] **Step 3:** Commit：`test: add grid_multi_asset_v3 failing tests`

---

## Task 3：实现 template.py — M2 + M1 + 日终快照

**Files:** `strategies/grid_multi_asset_v3/template.py`

- [ ] 增加 `DEFENSIVE_ETF_POOL` 常量与 `initialize` 中三个新 `context.*` 字段。
- [ ] `_refresh_pool`：按上文 M2 分支构建 `etfs`/股票列表逻辑。
- [ ] `_execute_grid`：CAP_LAYER 使用 `effective_max_layer`；NO_NET_ADD 在下单前 clamp；NORMAL 跳过。
- [ ] `after_trading_end`：在保留原有日志前提下，更新 `context._prev_eod_position_value`（或使用你固定命名的单 dict）。
- [ ] pytest 绿灯。

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
conda run -n SimTrade python -m pytest tests/unit/test_grid_multi_asset_v2.py tests/unit/test_grid_multi_asset_v3.py -q
```

- [ ] Commit：`feat(grid_multi_asset_v3): M1 bear grid modes and M2 defensive universe`

---

## Task 4：同步 backtest.py

**Files:** `strategies/grid_multi_asset_v3/backtest.py`

- [ ] 与 `template.py` 对齐（或复制后仅改注释）；短期跑回测仅用 `backtest.py` 亦可。
- [ ] Commit：`chore(grid_multi_asset_v3): sync backtest.py`

---

## Task 5：optimize_params.py — v3 参数类与映射

**Files:** `strategies/grid_multi_asset_v3/optimization/optimize_params.py`

- [ ] 将 `GridMultiAssetV2Params` Subclass/重命名为 **`GridMultiAssetV3Params`**，保留原 11 维，追加 v3 三字段与 `validate`。
- [ ] `custom_mapping` 增加：`'BEAR_UNIVERSE_MODE': 'context.BEAR_UNIVERSE_MODE'` 等三行。
- [ ] `optimize_strategy(..., strategy_file='template.py')` 的 `strategy_name`/`path`：**确认** `_script_path` 或通过默认解析指向 `grid_multi_asset_v3/optimization/`（照抄 v2 相对路径写法即可）。
- [ ] 语法自检：`python -c "exec(open(...).read().split('if __name__')[0])"`。
- [ ] Commit：`feat(grid_multi_asset_v3): extend WF parameter space for v3`

---

## Task 6：本地短窗冒烟回测

**Files:** `src/simtradelab/backtest/run_backtest.py`（本地仅临时改 **`strategy_name` / 日期**，测完可还原或注释说明）

- [ ] `strategy_name='grid_multi_asset_v3'`，`start_date='2025-01-01'`，`end_date='2025-06-30'`，跑一次无异常。
- [ ] Commit：若你只改入口且希望入库，可加注释块说明默认仍为 v2；否则不提交 `run_backtest.py`。

---

## Task 7：Walk-Forward 全流程（长任务）

**Files:** journal / CSV / `best_params_*.json` under `optimization/results/`

- [ ] 启动：`conda run -n SimTrade python strategies/grid_multi_asset_v3/optimization/optimize_params.py`（或 `nohup`）。
- [ ] 用 v2 同款的 journal 解析脚本盯进度。
- [ ] 结束后将最优参数写入 `backtest.py`（及可选 `optimized_strategy.py`）。
- [ ] Commit 最优 JSON + backtest：`feat(grid_multi_asset_v3): WF optimal params`

---

## Task 8：三段式回测与文档

**Files:** `my_docs/grid_multi_asset/v3.0/03-optimization-summary.md`

- [ ] **Holdout**（与 `holdout_period` 一致）。  
- [ ] **全长**（与 [v2 §5.2](../v2.0/03-optimization-summary.md) **同一 `end_date`**，便于横比）。  
- [ ] **熊市专项** `2021-01-01`～`2022-12-31`（若日志需微调闭区间，与 v2 报告中用语一致）。  
- [ ] 表格对照 **v2 Trial 29** 与同窗 **v3 最优 trial**。  
- [ ] Commit：`docs(grid_multi_asset): add v3.0 optimization summary`

---

## Task 9（可选）：聚宽 README + v3 脚本

非 v3.0 必达；与 v2 Task 11 同流程。

---

## Task 10：顶层文档索引

**Files:** `my_docs/README.md`

- [ ] 将 v3.0 行的状态从「设计已定稿」更新为「WF 已完成」等实情（在完成 Task 8 后）。

---

## 快速参考命令

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
conda run -n SimTrade python -m pytest tests/unit/test_grid_multi_asset_v2.py tests/unit/test_grid_multi_asset_v3.py -v
conda run -n SimTrade python src/simtradelab/backtest/run_backtest.py   # 需先配置 strategy_name
```

---

本计划与 [01-design.md](./01-design.md) 同属 `my_docs/grid_multi_asset/v3.0/`；后续 **writing-plans** 若需在别处存档，可在此路径维护主本并仅追加链接。
