# 多标的网格策略 v5.0 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 `strategies/grid_multi_asset_v5/`：在 **v4 代码**（周频 regime、窄池/宽池分支骨架）上增加 **锚定 ETF + 卫星 ETF 扩展 Universe**、**换仓时锚定优先入池**、**可选 v3 熊市 Universe/网格维**；优化流程在 Walk-Forward 之后对 trial 执行 **两阶段选参**（先 **FULL 的 I+II** 可行域，再 **完整 III** 与 **FULL 年化** 排序）。三重门禁定义与口径对齐 [v4.0 `01-design.md`](../v4.0/01-design.md)；规格全文见 [v5.0 `01-design.md`](./01-design.md)。

**Architecture：** 策略侧保持 **单轨资金**：`_refresh_pool` 在 **同一候选集** 上打分排序，再用纯函数 **`build_grid_pool_anchor_first`** 把 **锚定代码**按固定顺序优先塞进 `context.pool`，剩余名额按分数顺延。**BEAR** 时可选切换候选来源或限制有效层数（端口自 v3，与 v4 的 `handle_data` 分频逻辑组合：regime 仍按 `REGIME_REFRESH`，换股仍按 `REBALANCE_FREQ`）。优化侧复用 v4 的 **Optuna + WF**；新增或扩展脚本：对结果做 **阶段 A（I+II）** 过滤与 **阶段 B（FULL 年化降序）**，**不得**用 WF 分替代门禁。

**Tech Stack：** Python 3.x、Optuna、SimTradeLab `BacktestRunner` / `BacktestConfig`、与 v4 相同的 `context.*` 行内注入。

---

## 规格覆盖自检（撰稿后）

| 设计 § | 本计划任务 |
|--------|------------|
| §1 门禁 + 可行域内 FULL 年化优先 | Task 7、Task 8 |
| §3 Universe 锚定 + 数据流顺序 | Task 2、Task 3 |
| §3 BEAR 小维 | Task 4 |
| §4 两阶段选参 | Task 8 |
| §5 单测、gate_eval、文档 | Task 2、Task 7、Task 10 |

---

## 文件清单

| 操作 | 路径 | 说明 |
|------|------|------|
| 创建 | `strategies/grid_multi_asset_v5/template.py` | v4 fork + 锚定池 +（可选）v3 BEAR 维 |
| 创建 | `strategies/grid_multi_asset_v5/backtest.py` | 与 template 同步；保守默认可跑通参数 |
| 创建 | `strategies/grid_multi_asset_v5/optimization/optimize_params.py` | `GridMultiAssetV5Params` + `V5_CUSTOM_MAPPING` |
| 创建 | `strategies/grid_multi_asset_v5/optimization/gate_eval.py` | 自 v4 复制并改 `strategy_name` 与输出前缀 `v5` |
| 创建 | `strategies/grid_multi_asset_v5/optimization/two_stage_select.py` | 阶段 A/B 入口（读 trials CSV 或 journal） |
| 创建 | `strategies/grid_multi_asset_v5/optimization/results/.gitkeep` | |
| 创建 | `strategies/grid_multi_asset_v5/stats/.gitkeep` | |
| 创建 | `tests/unit/test_grid_multi_asset_v5.py` | 锚定选股、validate、门禁纯函数 |
| 修改 | `src/simtradelab/backtest/run_backtest.py` | 增加注释行 `grid_multi_asset_v5`（**不**改当前默认 `v4`，避免误跑） |

**勿修改** `strategies/grid_multi_asset_v2|v3|v4/**` 的行为（v5 独立目录）；若必须抽 **共享纯函数** 到 `simtradelab`，须 Task 10 跑 v4 短窗回归。

---

## v5 常量与参数空间（编码前写死在 template 与 optimize_params）

### Universe（示例默认值，可按回测微调）

- **`ANCHOR_ETF_UNIVERSE`**：`['510300.SS', '510500.SS']`（注释写明中文简称）。
- **`SATELLITE_ETF_UNIVERSE`**：从 v4 `NARROW_ETF_UNIVERSE` 与 `CANDIDATE_ETFS` 并集中剔除锚定后取不少于 **8** 只，代码表与 v4 一致用 `.SS`/`.SZ`。
- **合并候选（默认模式）：** `list(dict.fromkeys(ANCHOR_ETF_UNIVERSE + SATELLITE_ETF_UNIVERSE))`，长度记 **`V5_COMB_UNIVERSE_SIZE`**，供 `validate` 使用。
- **`UNIVERSE_MODE`：** `'ANCHOR_SATELLITE'`（默认）| `'WIDE_V2'`（对照：保留 v4 宽池行为）。

### 新增 `context` 字段（须写入 `V5_CUSTOM_MAPPING`）

| 字段 | 含义 | 优化器离散候选（示例） |
|------|------|------------------------|
| `MIN_ANCHORS_IN_POOL` | 换仓时若锚定在候选集中可交易，则 `pool` 中至少保留的锚定只数 | `[1, 2]`（须 `<= len(ANCHOR_ETF_UNIVERSE)` 且 `<= MAX_HOLD`） |
| `BEAR_UNIVERSE_MODE` | 同 v3：`SAME` \| `ETF_DEFENSIVE` | `['SAME', 'ETF_DEFENSIVE']` |
| `BEAR_GRID_MODE` | 同 v3：`NORMAL` \| `NO_NET_ADD` \| `CAP_LAYER` | `['NORMAL', 'CAP_LAYER']`（首轮可砍 `NO_NET_ADD` 降维） |
| `BEAR_GRID_MAX_LAYER_CAP` | 仅当 `CAP_LAYER` 时参与 | `[0, 1]` 与 v3 语义一致 |

继承 v4 的 `REGIME_REFRESH`、`BULL_RATIO`、`NEUTRAL_RATIO`、网格维、`MAX_HOLD`、`REBALANCE_FREQ` 等。

### `GridMultiAssetV5Params.validate`

- 保留 v4：`GRID_STEP_MIN < GRID_STEP_MAX`、`BEAR_RATIO < NEUTRAL_RATIO < BULL_RATIO`。
- `MAX_HOLD <= V5_COMB_UNIVERSE_SIZE`（默认合并池长度）。
- `MIN_ANCHORS_IN_POOL <= len(ANCHOR_ETF_UNIVERSE)` 且 `MIN_ANCHORS_IN_POOL <= MAX_HOLD`。

### 门禁与回测日期

- 与 v4 `gate_eval.py`：**FULL_START/END、RECENT、BEAR、初始资金 50 万、基准 000300.SS** 一致；`FULL_END` 当前与文档对齐为 **`2026-04-20`**（若变更须同步 `01-design` 脚注与总结）。

---

## Task 1：脚手架 — 从 v4 复制 v5 目录

**Files:**
- Create: `strategies/grid_multi_asset_v5/template.py`（初版 = v4 拷贝）
- Create: `strategies/grid_multi_asset_v5/backtest.py`（初版 = v4 拷贝）
- Create: `strategies/grid_multi_asset_v5/optimization/optimize_params.py`（初版 = v4 拷贝改名）
- Create: `strategies/grid_multi_asset_v5/optimization/gate_eval.py`（初版 = v4 拷贝）
- Create: `strategies/grid_multi_asset_v5/optimization/results/.gitkeep`
- Create: `strategies/grid_multi_asset_v5/stats/.gitkeep`

- [ ] **Step 1：** 递归复制 `strategies/grid_multi_asset_v4/` → `strategies/grid_multi_asset_v5/`（含 `optimization`）。**必须排除** 名为 **`backtest_cache/`** 的目录（体积大、且为运行产物，勿进版本库或二次复制）。推荐：`rsync -a --exclude='backtest_cache' strategies/grid_multi_asset_v4/ strategies/grid_multi_asset_v5/`（首次建 `v5` 时）或等价 `cp`/`tar` 排除规则。复制后删除 `optimization/results/*.csv` 若存在（只保留 `.gitkeep`）；删除误入的嵌套副本（如 `grid_multi_asset_v5/grid_multi_asset_v4/`）。
- [ ] **Step 2：** 全文替换文件名注释与日志中的 `v4` → `v5`、`grid_multi_asset_v4` → `grid_multi_asset_v5`、`GridMultiAssetV4Params` → `GridMultiAssetV5Params`、`V4_CUSTOM_MAPPING` → `V5_CUSTOM_MAPPING`。
- [ ] **Step 3：** `gate_eval.py` 内 **`strategy_name` / 临时文件路径** 指向 `grid_multi_asset_v5`。
- [ ] **Step 4：** 运行冒烟（应等价于尚未改逻辑的 v4）：

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
conda run -n SimTrade python -c "
from pathlib import Path
p = Path('strategies/grid_multi_asset_v5/template.py')
t = p.read_text(encoding='utf-8')
assert '# strategies/grid_multi_asset_v5/template.py' in t
assert 'grid_multi_asset_v4' not in t
print('ok')
"
```

Expected: `ok`

- [ ] **Step 5：Commit**

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
git add strategies/grid_multi_asset_v5
git commit -m "chore: scaffold grid_multi_asset_v5 from v4"
```

---

## Task 2：TDD — `build_grid_pool_anchor_first` 与 `validate`

**Files:**
- Modify: `strategies/grid_multi_asset_v5/template.py`（仅增加纯函数；暂不接 `_refresh_pool`）
- Create: `tests/unit/test_grid_multi_asset_v5.py`

在 `template.py` **紧接** `_score_universe之后`、`_etf_list_for_mode` 之前插入：

```python
def build_grid_pool_anchor_first(ranked_codes, anchor_codes, max_hold):
    """ranked_codes: 分数从高到低；anchor_codes: 锚定顺序（优先级递减）。
    先在不超过 max_hold 前提下按 anchor_codes 顺序塞入「出现在 ranked_codes 中」的锚定，
    再按 ranked_codes 顺序补足名额。单轨；不去重 ranked 以外的代码。"""
    ranked = list(ranked_codes)
    in_ranked = set(ranked)
    pool = []
    for c in anchor_codes:
        if c in in_ranked and len(pool) < max_hold:
            pool.append(c)
    for c in ranked:
        if len(pool) >= max_hold:
            break
        if c not in pool:
            pool.append(c)
    return pool
```

**注意：** [v5.0 `01-design.md`](./01-design.md) 要求「至少保留锚定数」——本函数保证 **按顺序优先放入所有在 ranked 中出现的锚定（直到满额）**；`MIN_ANCHORS_IN_POOL` 的硬约束在 **`_refresh_pool` 末尾检查** 中通过 **validate +「若可交易锚定数≥min 则 pool 内锚定数≥min」** 实现，见 Task 3。

- [ ] **Step 1：Write the failing test**

在 `tests/unit/test_grid_multi_asset_v5.py`：

```python
# tests/unit/test_grid_multi_asset_v5.py
# -*- coding: utf-8 -*-
from pathlib import Path
import types
import numpy as np
import pandas as pd
import pytest

_STRATEGY_PATH = Path(__file__).parents[2] / 'strategies' / 'grid_multi_asset_v5' / 'template.py'


def _load_template_ns():
    _log = types.SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
    )
    mock_ns = {
        '__builtins__': __builtins__,
        'np': np,
        'set_benchmark': lambda *a, **kw: None,
        'set_slippage': lambda *a, **kw: None,
        'get_history': lambda *a, **kw: pd.DataFrame(),
        'get_fundamentals': lambda *a, **kw: pd.DataFrame(),
        'get_index_stocks': lambda *a, **kw: [],
        'get_stock_status': lambda *a, **kw: {},
        'order_target': lambda *a, **kw: None,
        'order_target_value': lambda *a, **kw: None,
        'log': _log,
    }
    src = _STRATEGY_PATH.read_text(encoding='utf-8')
    exec(compile(src, str(_STRATEGY_PATH), 'exec'), mock_ns)
    return mock_ns


def test_build_grid_pool_anchor_first_order():
    ns = _load_template_ns()
    fn = ns['build_grid_pool_anchor_first']
    ranked = ['512010.SS', '510300.SS', '159915.SZ', '588000.SS']
    anchors = ['510300.SS', '510500.SS']
    pool = fn(ranked, anchors, max_hold=3)
    assert pool == ['510300.SS', '512010.SS', '159915.SZ']


def test_build_grid_pool_respects_max_hold():
    ns = _load_template_ns()
    fn = ns['build_grid_pool_anchor_first']
    ranked = ['A', 'B', 'C', 'D']
    anchors = ['X', 'Y']
    assert fn(ranked, anchors, max_hold=2) == ['A', 'B']
```

- [ ] **Step 2：Run test to verify failure**

Run: `conda run -n SimTrade pytest tests/unit/test_grid_multi_asset_v5.py::test_build_grid_pool_anchor_first_order -v`

Expected: **FAIL**（`build_grid_pool_anchor_first` 尚未定义或行为不符）。

- [ ] **Step 3：** 将上文 **`build_grid_pool_anchor_first` 完整粘贴进** `template.py`。
- [ ] **Step 4：Run test to verify pass**

Run: `conda run -n SimTrade pytest tests/unit/test_grid_multi_asset_v5.py -v`

Expected: **PASS**

- [ ] **Step 5：Commit**

```bash
git add strategies/grid_multi_asset_v5/template.py tests/unit/test_grid_multi_asset_v5.py
git commit -m "feat(grid_multi_asset_v5): anchor-first pool builder + unit tests"
```

---

## Task 3：`template.py` — 扩展 Universe + `_refresh_pool` 接入锚定优先

**Files:**
- Modify: `strategies/grid_multi_asset_v5/template.py`

- [ ] **Step 1：** 在文件顶部定义 `ANCHOR_ETF_UNIVERSE`、`SATELLITE_ETF_UNIVERSE`，以及：

```python
def _combined_etf_universe_for_mode(universe_mode):
    if universe_mode == 'WIDE_V2':
        return list(CANDIDATE_ETFS)
    merged = []
    seen = set()
    for x in list(ANCHOR_ETF_UNIVERSE) + list(SATELLITE_ETF_UNIVERSE):
        if x not in seen:
            merged.append(x)
            seen.add(x)
    return merged
```

- [ ] **Step 2：** `initialize` 中增加 `context.UNIVERSE_MODE = 'ANCHOR_SATELLITE'`（及 v3 三字段默认值，与 v3 `template.py` 一致以便单测对照）。
- [ ] **Step 3：** 将 `_etf_list_for_mode` 改为调用 `_combined_etf_universe_for_mode`：当 `WIDE_V2` 时返回 `CANDIDATE_ETFS`；否则返回合并锚定+卫星列表。删除仅 6 只窄池专用逻辑。
- [ ] **Step 4：** 在 `_refresh_pool` 中，在得到 `ranked = _score_universe(...)` 之后、取 `[:max_hold]` **之前**，改为：

```python
ranked_pairs = _score_universe(vol_dict, fund_df, etfs, context.VOL_WEIGHT)
ranked_codes = [code for code, _ in ranked_pairs]
if context.UNIVERSE_MODE != 'WIDE_V2':
    new_pool = build_grid_pool_anchor_first(
        ranked_codes,
        ANCHOR_ETF_UNIVERSE,
        max_hold,
    )
    min_a = int(getattr(context, 'MIN_ANCHORS_IN_POOL', 1))
    n_anchor = len(set(new_pool) & set(ANCHOR_ETF_UNIVERSE))
    tradable_anchor = [c for c in ANCHOR_ETF_UNIVERSE if c in vol_dict]
    if len(tradable_anchor) >= min_a and n_anchor < min_a:
        log.warning(
            '锚定不足: 需要至少 %d 只，实际 %d — 保持原池' % (min_a, n_anchor))
        return
else:
    new_pool = [code for code, _ in ranked_pairs[:max_hold]]
```

（`min_a` 逻辑：当可交易锚定足够而构造池未满足时 **skip 本次换仓** 以避免破坏 II；与 [01-design §3.3](01-design.md) 一致，须在 `03-optimization-summary` 披露若长期触发。）

- [ ] **Step 5：** 跑单元测试与 v5 单文件测试：

```bash
conda run -n SimTrade pytest tests/unit/test_grid_multi_asset_v5.py -v
```

Expected: PASS

- [ ] **Step 6：Commit**

```bash
git add strategies/grid_multi_asset_v5/template.py
git commit -m "feat(grid_multi_asset_v5): ANCHOR_SATELLITE universe + anchor-first refresh"
```

---

## Task 4：接入 v3 BEAR 分支（Universe + 有效层数）

**Files:**
- Modify: `strategies/grid_multi_asset_v5/template.py`
- Read: `strategies/grid_multi_asset_v3/template.py`（`DEFENSIVE_ETF_POOL`、`_refresh_pool` 里 BEAR 候选、`NO_NET_ADD`、`_execute_grid` 里 `effective_max_layer`）

- [ ] **Step 1：** 从 v3 复制 **`DEFENSIVE_ETF_POOL`** 常量到 v5 `template.py`。
- [ ] **Step 2：** 在 `_refresh_pool` 中，在拼接 `etfs` 用于打分前，若 `context.regime == 'BEAR'` 且 `context.BEAR_UNIVERSE_MODE == 'ETF_DEFENSIVE'` 且非 `WIDE_V2`，将 ETF 候选限制为 `DEFENSIVE_ETF_POOL` 与当前合并 Universe 的交集（无交集则 **记录 warning 并 return 保留原池**，与 v3 防御性一致）。
- [ ] **Step 3：** 在 `initialize` 里增加 `context._prev_eod_position_value = {}`（供 `NO_NET_ADD` 使用，若 Task 4 启用该模式）。
- [ ] **Step 4：** 在 `_execute_grid` 中，计算 `layer` 前得到：

```python
max_layer_eff = int(context.GRID_MAX_LAYER)
if context.regime == 'BEAR' and getattr(context, 'BEAR_GRID_MODE', 'NORMAL') == 'CAP_LAYER':
    max_layer_eff = min(max_layer_eff, int(context.BEAR_GRID_MAX_LAYER_CAP))
```

然后将原 `context.GRID_MAX_LAYER` 传入 `_calc_layer` 处改为 `max_layer_eff`；`raw_w` / `max_w` 仍用 **`context.GRID_MAX_LAYER`**（与 v3 一致：_cap 只限层数不累乘权重上界时需对照 v3 — 若 v3 用 cap 同时改了 `max_w`，则逐行复制 v3 对应 15 行，**禁止猜测**）。
- [ ] **Step 5：** 在 `after_trading_end` 末尾更新 `_prev_eod_position_value`（复制 v3 同名块）。
- [ ] **Step 6：** `pytest tests/unit/test_grid_multi_asset_v5.py -v`；若有 BEAR 纯函数可测，追加断言 `max_layer_eff == 0` 当 `CAP_LAYER` 且 cap=0。

- [ ] **Step 7：Commit**

```bash
git add strategies/grid_multi_asset_v5/template.py
git commit -m "feat(grid_multi_asset_v5): optional v3 BEAR universe and grid cap"
```

---

## Task 5：同步 `backtest.py` 与 `optimized_strategy.py`

**Files:**
- Modify: `strategies/grid_multi_asset_v5/backtest.py`
- Modify: （若存在）`strategies/grid_multi_asset_v5/optimization/optimized_strategy.py`

- [ ] **Step 1：** `backtest.py` 的 `initialize` 与 `template.py` **逐字段一致**（含 `MIN_ANCHORS_IN_POOL`、`BEAR_*`）。
- [ ] **Step 2：** 跑短窗回测（可选，依赖数据）：

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
conda run -n SimTrade python -c "
from simtradelab.backtest.runner import BacktestRunner
from simtradelab.backtest.config import BacktestConfig
r = BacktestRunner()
rep = r.run(BacktestConfig(strategy_name='grid_multi_asset_v5', start_date='2024-01-01', end_date='2024-06-30', initial_capital=500000.0))
print('done', type(rep))
"
```

Expected: 无未捕获异常，`done` 打印。

- [ ] **Step 3：Commit**

```bash
git add strategies/grid_multi_asset_v5/backtest.py
git commit -m "chore(grid_multi_asset_v5): sync backtest.py with template"
```

---

## Task 6：`optimize_params.py` — V5 参数空间与 mapping

**Files:**
- Modify: `strategies/grid_multi_asset_v5/optimization/optimize_params.py`

- [ ] **Step 1：** 定义 `V5_COMB_UNIVERSE_SIZE = len(合并列表)`（与 template 顶层常量同步，可用 importlib 加载 template 只读常量避免双份维护，**或**在 optimize_params 顶部手写同一合并表并单测强制相等）。
- [ ] **Step 2：** `MAX_HOLD` 候选改为 **`[k for k in [4, 5, 6, 8, 10] if k <= V5_COMB_UNIVERSE_SIZE]`**（若合并池为 10 则去掉 >10）。
- [ ] **Step 3：** 增加离散维：`MIN_ANCHORS_IN_POOL`、`BEAR_UNIVERSE_MODE`、`BEAR_GRID_MODE`、`BEAR_GRID_MAX_LAYER_CAP`（取值见上文「参数空间」；**条件约束** 可在 `validate` 里写明：`CAP_LAYER` 时 cap 合法、`ETF_DEFENSIVE` 时合并池与防御池交集非空——若空 **raise ValueError** 令 Optuna 拒绝）。
- [ ] **Step 4：** `V5_CUSTOM_MAPPING` 补全新键到 `context.*`。
- [ ] **Step 5：单元测试** `GridMultiAssetV5Params.validate`（模板：复制 `test_grid_multi_asset_v4.py` 中 validate 段改路径）。

```bash
conda run -n SimTrade pytest tests/unit/test_grid_multi_asset_v5.py -v
```

- [ ] **Step 6：Commit**

```bash
git add strategies/grid_multi_asset_v5/optimization/optimize_params.py tests/unit/test_grid_multi_asset_v5.py
git commit -m "feat(grid_multi_asset_v5): optimizer parameter space and mapping"
```

---

## Task 7：`gate_eval.py` — strategy v5 与导出

**Files:**
- Modify: `strategies/grid_multi_asset_v5/optimization/gate_eval.py`

- [ ] **Step 1：** 所有 `grid_multi_asset_v4` 字符串改为 **`grid_multi_asset_v5`**。
- [ ] **Step 2：** 输出 CSV 前缀改为 `gate_results_v5_*.csv`（或与 v4 区分）。
- [ ] **Step 3：** 用文档最优或默认可跑通参数跑：

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
conda run -n SimTrade python strategies/grid_multi_asset_v5/optimization/gate_eval.py --help
```

Expected: 打印帮助，无 ImportError。

- [ ] **Step 4：Commit**

```bash
git add strategies/grid_multi_asset_v5/optimization/gate_eval.py
git commit -m "chore(grid_multi_asset_v5): gate_eval targets v5 strategy"
```

---

## Task 8：`two_stage_select.py` — 阶段 A（I+II）与阶段 B（FULL 年化）

**Files:**
- Create: `strategies/grid_multi_asset_v5/optimization/two_stage_select.py`

实现要求（**不得**留空实现）：

1. **输入：** Optuna `trials_*.csv` 路径或 journal；解析 `params_*` 列还原 `dict params`。
2. **阶段 A：** 对每个 trial 调用与 `gate_eval` **相同的 FULL 回测**，抽取 **I（最大回撤）** 与 **II（超额、IR）**；不达标则丢弃；**不**在此时写最终 eligible。
3. **可选 A2：** 阶段 A 通过者跑 **RECENT** 仅读 **夏普 + 最大回撤**（与设计「粗筛」一致）；不达标丢弃；仍非最终 III。
4. **阶段 A3：** 幸存者跑 **完整 RECENT** 验证 **III**；通过者进入 **eligible** 列表。
5. **阶段 B：** 对 eligible 按 **FULL 年化收益率**（与 `report` 或 stats 字典字段名 — **从 gate_eval 已用字段复制**）降序排序，打印 Top 10 与 **卡玛**。

脚本 `main` 示例：

```python
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='v5 two-stage gate filter')
    p.add_argument('--trials-csv', required=True)
    p.add_argument('--top', type=int, default=10)
    args = p.parse_args()
    raise SystemExit(main(args.trials_csv, args.top))
```

（`main` 函数体实现调用现有 `gate_eval` 内跑单段回测的函数 — **优先 refactor `gate_eval` 抽出** `run_segment(strategy_name, params, start, end) -> metrics_dict` 避免复制粘贴。）

- [ ] **Step 1：** 若需重构，在 **`gate_eval.py`** 增加 **`run_backtest_for_params(params, start, end) -> dict`**，原逻辑调用它。
- [ ] **Step 2：** 实现 `two_stage_select.py` 并用手工构造的 **1 行 trials CSV** 做冒烟（可选用已知 v5 默认参数）。
- [ ] **Step 3：Commit**

```bash
git add strategies/grid_multi_asset_v5/optimization/two_stage_select.py strategies/grid_multi_asset_v5/optimization/gate_eval.py
git commit -m "feat(grid_multi_asset_v5): two-stage gate filter and gate_eval refactor"
```

---

## Task 9：`run_backtest.py` 注释入口

**Files:**
- Modify: `src/simtradelab/backtest/run_backtest.py`

- [ ] **Step 1：** 在 `strategy_name = 'grid_multi_asset_v4'` 下一行增加：

```python
    # strategy_name = 'grid_multi_asset_v5'
```

- [ ] **Step 2：Commit**

```bash
git add src/simtradelab/backtest/run_backtest.py
git commit -m "docs(backtest): mention grid_multi_asset_v5 entry"
```

---

## Task 10：回归与交付文档

- [ ] **Step 1：** v4 短窗（确保未被破坏）：

```bash
conda run -n SimTrade python -c "
from simtradelab.backtest.runner import BacktestRunner
from simtradelab.backtest.config import BacktestConfig
r = BacktestRunner()
r.run(BacktestConfig(strategy_name='grid_multi_asset_v4', start_date='2024-01-01', end_date='2024-03-29', initial_capital=500000.0))
print('v4 ok')
"
```

Expected: `v4 ok`

- [ ] **Step 2：** 全量单测（项目允许时）：

```bash
conda run -n SimTrade pytest tests/unit/test_grid_multi_asset_v5.py tests/unit/test_grid_multi_asset_v4.py -q
```

Expected: **全部通过**

- [ ] **Step 3：** 撰写 **`my_docs/grid_multi_asset/v5.0/03-optimization-summary.md`**：在完成一次真实 `optimize_params.py` 与 `two_stage_select.py` / `gate_eval.py` 运行后，按 v4 总结结构填写 **WF 统计、最优参数、FULL/RECENT/BEAR 三表、I～III 判定、与 v2/v4 对照**。该文件在本次实现计划中 **依赖真实跑数**，可在首次优化完成后由工程师补全。
- [ ] **Step 4：Commit**

```bash
git add my_docs/grid_multi_asset/v5.0/03-optimization-summary.md
git commit -m "docs(grid_multi_asset): v5.0 optimization summary (post-run)"
```

---

## 计划自检（占位符与一致性）

- 已避免「稍后实现」「TBD」作为步骤内容；**03-optimization-summary** 明确依赖优化跑批后的实测填写。
- `MIN_ANCHORS_IN_POOL` 在 Task 3 的 `getattr(..., 1)` 与 Task 6 的优化维一致；默认 **1**。
- `BEAR` 层数逻辑要求与 **v3** 行级一致 — Task 4 要求复制而非凭记忆改写。

---

## 执行交接

**计划已保存至：** `my_docs/grid_multi_asset/v5.0/02-plan.md`

**可选执行方式：**

1. **Subagent-Driven（推荐）** — 每个 Task 单独开子代理，任务间人工复核。  
2. **Inline Execution** — 在当前会话按 Task 顺序实现，每 Task 结束跑对应 `pytest`/冒烟。

你更倾向哪一种？
