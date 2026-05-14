# 多标的网格策略 v4.0 — 实施计划

> **For agentic workers:** 可按任务顺序执行；步骤统一使用 checkbox（`- [ ]`）勾选进度。实现前对照 [01-design.md](./01-design.md)。

**Goal：** 新建 `strategies/grid_multi_asset_v4/`：以 **v2 代码为内核**（regime 三档、`invested_ratio`、单票 water-filling、网格步长与 layer），默认 **收窄 Universe 为高流动性 ETF 固定池**；**周频刷新** `invested_ratio`（换股频率仍可独立）；优化流程在 Walk-Forward 之后对 trial **强制执行 FULL / RECENT /（建议）BEAR 三段回测**，仅当 **I ∧ II ∧ III**（见设计 §2）全部满足时才标记为 **v4_eligible** 并允许写入「上线候选参数」。**默认不启用** v3 的 `BEAR_UNIVERSE_MODE` / `BEAR_GRID_MODE` 搜索空间；若三重门禁反复无解，再单列对照试验（见 Task 11）。

**Architecture：**

- **策略：** `template.py` / `backtest.py` 由 v2 **复制后改**，新增 `ETF_GRID_UNIVERSE`（或同名常量）与可选 `UNIVERSE_MODE = 'NARROW_ETF' | 'WIDE_V2'`（默认 `NARROW_ETF`）；窄池模式下 **跳过** 沪深300/中证500 成分股拉取与股票基本面分支，仅在固定 ETF 列表上做波动率/流动性过滤与 Top-`MAX_HOLD` 选股（逻辑尽量复用 v2 的 ETF 打分分支）。
- **周频 regime：** 在 `handle_data` 内用 **交易周** 或 **5 个交易日滚动** 与 `g.week_regime_interval` 对齐（实现选一种，全文一致）：满足「换周」时调用与 v2 相同的 `_detect_regime`；**非**换周且非 `REBALANCE_FREQ` 换股日时 **不** 调用 `_refresh_pool`，但 **须** 在换周时更新 `context.invested_ratio`（与换仓日逻辑同顺序：先 regime 再定仓）。若首轮实现困难，可先实现「**每个交易日** 更新 regime、仅换股日 refresh pool」作为对照分支，并在 `03-optimization-summary.md` 记一条 AB 结论。
- **三重门禁：** 不破坏 `simtradelab.backtest.optimizer_framework.optimize_strategy` 的 WF 主循环；在 **`optimization/`** 内增加 **后置评估**（独立函数或小脚本）：对已 **COMPLETE** 且 WF 目标值有效的 trial，生成临时 `backtest.py` 或内存注入参数后，用 **`BacktestRunner` + `BacktestConfig`** 分别跑 **FULL / RECENT / BEAR** 区间，解析 `report`（或与现有 stats 模块一致的指标字典）判定 I/II/III。输出：`results/gate_results_<ts>.csv`（每 trial 一行：WF 分、FULL 最大回撤、FULL 超额、FULL IR、RECENT 年化/回撤/夏普、是否 eligible）。

**Tech Stack：** Python 3.8+、Optuna、SimTradeLab `BacktestRunner`；与 v2 相同的 `template.py` 行内 `context.*` 注入方式。

---

## 文件清单

| 操作 | 路径 | 说明 |
|------|------|------|
| 创建 | `strategies/grid_multi_asset_v4/template.py` | v2 fork；窄 ETF 池 + 周频 regime |
| 创建 | `strategies/grid_multi_asset_v4/backtest.py` | 与 template 同步；门禁通过后写入候选最优参数 |
| 创建 | `strategies/grid_multi_asset_v4/optimization/optimize_params.py` | v4 参数空间；WF 调用 + 后置门禁评估入口 |
| 创建 | `strategies/grid_multi_asset_v4/optimization/gate_eval.py`（或内嵌于 optimize_params） | FULL/RECENT/BEAR 三段指标与 I/II/III 判定 |
| 创建 | `strategies/grid_multi_asset_v4/optimization/results/.gitkeep` | |
| 创建 | `strategies/grid_multi_asset_v4/stats/.gitkeep` | |
| 创建 | `tests/unit/test_grid_multi_asset_v4.py` | 窄池、周频计数、门禁纯函数 |
| 可选 | `strategies/grid_multi_asset_v4/optimization/post_select_eligible.py` | 从 journal/CSV 批量补跑门禁（中断恢复用） |
| 文档 | `my_docs/grid_multi_asset/v4.0/03-optimization-summary.md` | WF + 门禁达标情况 + 三段表 |

**说明：** `run_backtest.py` 仅需临时或永久增加 `strategy_name = 'grid_multi_asset_v4'`；框架按 `strategies/{strategy_name}/backtest.py` 解析。**勿修改** v2/v3 策略文件行为（v4 独立目录）。

---

## 常量与参数空间（编码前约定）

### 窄 ETF 池

- 在 `template.py` **顶部**维护单一列表，例如 **`NARROW_ETF_UNIVERSE`**：3～8 只，全部取自 v2/v3 已使用的 **`.SS` / `.SZ` 代码**，优先宽基 + 红利（具体代码表以 v2 `CANDIDATE_ETFS` 为母本摘抄，**注释**每只名称）。
- `MAX_HOLD` 候选须 **≤ len(NARROW_ETF_UNIVERSE)**，例如 v2 的 `[5,8,10,12,15]` 在 6 只池上改为 `[3, 4, 5, 6]` 或 `[3, 4, 5, 6, 6]` 去重后的合法升序列表；**validate** 中若 `MAX_HOLD > pool_size` 直接 `ValueError`。

### v4 参数类

- **继承 v2 的 11 维**（类名建议 `GridMultiAssetV4Params`），必要时 **删去** 在窄池下无意义的过大 `MAX_HOLD`。
- **可选** 增加离散维 `UNIVERSE_MODE`（仅当要做「宽池对照」试验时打开；默认优化可固定 `NARROW_ETF` 以控 trial 体积）。
- **可选** `REGIME_REFRESH`：`WEEKLY`（默认） / `ON_REBALANCE_ONLY`（对照）。

### 门禁阈值

- 阈值常量集中在一处（如 `gate_eval.py` 顶部或 `GateThresholds` 数据类），默认值 **与 [01-design.md §2.2](./01-design.md) 首轮建议一致**；首轮实测后仅改常量并 git 记录，同步更新 `03-optimization-summary.md` 脚注。

---

## Task 1：目录与 v2 fork

- [ ] 新建 `strategies/grid_multi_asset_v4/`，从 `grid_multi_asset_v2` **复制** `template.py`、`backtest.py`、`optimization/optimize_params.py`。
- [ ] 全局替换注释/日志中的版本标识为 v4；`optimize_params` 中 `GridMultiAssetV2Params` → `GridMultiAssetV4Params`（先保持与 v2 相同字段，窄池改造在 Task 3 完成）。
- [ ] 新建 `optimization/results/.gitkeep`、`stats/.gitkeep`。
- [ ] Commit 建议：`chore: scaffold grid_multi_asset_v4 from v2`

---

## Task 2：TDD — 可单测辅助函数先行

**Files:** `tests/unit/test_grid_multi_asset_v4.py`

建议从 `template.py` 提取或 `exec` 加载策略命名空间（与 v2/v3 单测风格一致）：

| 测试目标 | 说明 |
|----------|------|
| `MAX_HOLD` vs 池长 | `validate` 拒绝 `MAX_HOLD > len(NARROW_ETF_UNIVERSE)` |
| 窄池候选 | `_candidate_codes` 或等价函数在 `NARROW` 模式下 **不产生** 指数成分股依赖 |
| 门禁判定 | 给定伪造的 FULL/RECENT 指标 dict，**I ∧ II ∧ III** 与 relax 顺序的单元测试 |

- [ ] 先写失败测试；`conda run -n SimTrade pytest tests/unit/test_grid_multi_asset_v4.py -q` 见红。
- [ ] Commit：`test: add grid_multi_asset_v4 scaffolding tests`

---

## Task 3：实现 template.py — 窄池 + 周频 regime

**Files:** `strategies/grid_multi_asset_v4/template.py`

- [ ] 增加 `NARROW_ETF_UNIVERSE` 与 `UNIVERSE_MODE`（若启用）。
- [ ] `_refresh_pool`：`NARROW` 时仅用 ETF 池走 v2 ETF 打分/过滤；`WIDE_V2` 时 **逐字**恢复 v2 行为（供对照）。
- [ ] **周频 regime：** 实现 `_should_refresh_regime_weekly(context)` + 在 `handle_data` 中 **先于** 换股判断调用 `_detect_regime`；换股仍按 `day_counter % REBALANCE_FREQ == 0`（与 v2 首日例外一致）。
- [ ] 确认 `invested_ratio` 在周更日更新后，当日 `_execute_grid` 使用新 cap。
- [ ] `pytest` 绿灯；可选与 v2 并行：`pytest tests/unit/test_grid_multi_asset_v2.py tests/unit/test_grid_multi_asset_v4.py -q`

- [ ] Commit：`feat(grid_multi_asset_v4): narrow ETF universe + weekly regime refresh`

---

## Task 4：同步 backtest.py

- [ ] 与 `template.py` 逻辑一致；默认参数为「可跑通」的保守值（不必最优）。
- [ ] Commit：`chore(grid_multi_asset_v4): sync backtest.py`

---

## Task 5：optimize_params.py — WF 与 v4 映射

**Files:** `strategies/grid_multi_asset_v4/optimization/optimize_params.py`

- [ ] `custom_mapping` 指向 v4 `template.py` 中所有 `context.*` 字段；若新增 `UNIVERSE_MODE`/`REGIME_REFRESH`，补映射键。
- [ ] `strategy_file='template.py'` 路径相对 `grid_multi_asset_v4/optimization/` 正确。
- [ ] `patience`、剪枝与 v2 对齐或略放宽（窄池后 trial 可先做冒烟小空间）。
- [ ] Commit：`feat(grid_multi_asset_v4): wire WF optimizer to v4 template`

---

## Task 6：gate_eval — 三段回测与 I/II/III

**Files:** `strategies/grid_multi_asset_v4/optimization/gate_eval.py`（推荐独立模块，便于 `post_select` 复用）

- [ ] 定义区间常量：`FULL_START/END`、`RECENT_*`、`BEAR_*`，与 [01-design.md §2.1](./01-design.md) 一致；**FULL_END** 与 v2/v3 全长横比日对齐（当前 **`2026-04-20`**，若变更须双文档更新）。
- [ ] 提供 `inject_params_into_strategy(template_path, params) -> str` 或复用框架已有「写临时策略文件」能力；**同一参数** 依次跑三段 `BacktestConfig`（`initial_capital=500000.0`，`strategy_name='grid_multi_asset_v4'` 或指向临时文件——以_runner 支持为准）。
- [ ] 从 `report` 提取：**最大回撤**（注意符号约定：与设计表格一致）、**相对 000300 超额**、**IR**、**年化/夏普**（RECENT）；字段名若与现有 `BacktestRunner` 输出不一致，**打印一次报告 keys** 后在代码里固定映射并列在计划脚注。
- [ ] 实现 `is_v4_eligible(metrics_full, metrics_recent) -> bool` 与设计 §2.2 红线一致；输出 **失败原因** 字符串列表（便于 CSV）。
- [ ] 单元测试：伪造 metrics **边界情况**（恰好踩线）。

- [ ] Commit：`feat(grid_multi_asset_v4): triple-gate evaluation for FULL/RECENT windows`

---

## Task 7：把门禁挂入优化主流程（或半自动）

**二选一（须在 PR/总结中写明实际采用）：**

**A. 半自动（推荐先做）：** WF 跑完后，`python optimization/post_select_eligible.py`（或 `optimize_params.py --gate-only`）读取 journal 最近 **K** 个 COMPLETE trial，对每参数向量跑 `gate_eval`，输出 eligible 列表与最佳 WF 分候选。

**B. 全自动：** 在每次 trial COMPLETE 后同步调 `gate_eval`（成本高）；或仅对 WF 目标值 Top-K **异步**批处理。

- [ ] 至少实现 **A**；若实现 **B**，须设 `TOP_K` 或「仅对超过阈值的 trial 跑门禁」防爆炸。
- [ ] 生成 `results/gate_results_*.csv` + 终端摘要：**eligible 数量**；若无 eligible，打印「当前结构下三重门禁不可行」并提示按设计 **I → II → III** 放宽。
- [ ] Commit：`feat(grid_multi_asset_v4): post-WF triple-gate batch selection`

---

## Task 8：冒烟回测

**Files:** 临时改 `src/simtradelab/backtest/run_backtest.py` 或命令行等价

- [ ] `strategy_name='grid_multi_asset_v4'`，短窗如 `2025-01-01`～`2025-06-30`，无异常、有成交日志。
- [ ] 全长一次：`2019-01-01`～`2026-04-20`，与 v2 同窗对照草稿指标（不必入库 run_backtest 默认策略名，若以注释块保留 v4 切换说明可提交）。

- [ ] Commit（可选）：`chore: document grid_multi_asset_v4 in run_backtest comments`

---

## Task 9：Walk-Forward 全流程

- [ ] 冒烟：`patience` 降低或子空间固定部分维，确认 journal + 门禁 CSV 管道通。
- [ ] 正式：`conda run -n SimTrade python strategies/grid_multi_asset_v4/optimization/optimize_params.py`（或文档化 `nohup`）。
- [ ] 门禁：对最优若干 trial 跑 Task 7，选出 **v4_eligible**；将参数写入 `backtest.py` 与 `optimization/optimized_strategy.py`（若保留 v2 同款生成逻辑）。

- [ ] Commit：`feat(grid_multi_asset_v4): WF results and gated best params`

---

## Task 10：`03-optimization-summary.md`

- [ ] **WF** 统计与最优 trial（与 v2/v3 同结构）。
- [ ] **三张表：** FULL / RECENT / BEAR（BEAR 为监控披露）；标注 **I / II / III 是否逐项达标**。
- [ ] **对照：** v2 Trial 29 同窗口、同初始资金、同 FULL 截止日。
- [ ] 若门禁无解：按设计文档记录 **放宽顺序** 与最终采用红线。

- [ ] Commit：`docs(grid_multi_asset): add v4.0 optimization summary`

---

## Task 11（条件触发）：v3 机制对照或路线 C

**仅当** Task 9/10 在 **窄池 + 周频 + v2 内核** 下 **长期无 eligible** 时执行：

- [ ] 开分支试验：仅增加 v3 `BEAR_UNIVERSE_MODE=ETF_DEFENSIVE` **或** `CAP_LAYER` 小网格搜索（小参数子空间）；**不得** 默认合并进主线以控制复杂度。
- [ ] 或对「双层资金」路线 C 单独立项 `v4.1` 设计增量（超出 v4.0 本计划范围时在总结中注明）。

---

## Task 12：文档索引

**Files:** `my_docs/README.md`

- [ ] 增加或更新 v4.0 一行指向 `grid_multi_asset/v4.0/` 与状态（计划中 / WF 完成 / 门禁结论）。

---

## 快速参考命令

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
conda run -n SimTrade python -m pytest tests/unit/test_grid_multi_asset_v2.py tests/unit/test_grid_multi_asset_v4.py -q
conda run -n SimTrade python strategies/grid_multi_asset_v4/optimization/optimize_params.py
conda run -n SimTrade python src/simtradelab/backtest/run_backtest.py   # strategy_name=grid_multi_asset_v4
```

---

## 与设计文档的一致性自检

| 检查项 | 对应 |
|--------|------|
| I + II + III 同时判定 | [01-design.md §2～§3](./01-design.md) |
| FULL / RECENT / BEAR 日期 | [01-design.md §2.1](./01-design.md) |
| 首轮红线可校准 | [01-design.md §2.2](./01-design.md) |
| 无解时 I→II→III 放宽 | [01-design.md §2.2](./01-design.md) |
| v3 非默认 | [01-design.md §4](./01-design.md) |

本计划主本位于 `my_docs/grid_multi_asset/v4.0/02-plan.md`；实施中若变更架构（例如门禁并入框架核心），须回写 [01-design.md](./01-design.md) 相应小节并留提交说明。
