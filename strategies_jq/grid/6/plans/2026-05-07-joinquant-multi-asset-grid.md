# 聚宽多标的分钟网格策略 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在本地仓库中交付一份可粘贴到聚宽编辑器的一分钟回测策略，实现 `docs/superpowers/specs/2026-05-07-joinquant-multi-grid-design.md` 中的多标的固定百分比浅网格、季度选股、三 ETF 固定槽、收盘价触档与 T+1 可卖约束；并用可离线运行的 pytest 覆盖纯逻辑。

**Architecture:** 将「无 jqdata 依赖」的日历、档位价、穿价判定、手数与资金约束抽成 `jq_grid_pure.py`；策略主体 `multi_asset_minute_grid.py` 仅负责 `initialize` / `run_daily` 季度门闩 / `handle_data` 数据拉取与下单；选股与波动计算在季度函数内用 `get_price` + `panel=False` 批量完成。验证阶段对策略文件运行 `joinquant-skill` 的 `strategy_lint.py`。

**Tech Stack:** Python 3.x、pytest、JoinQuant API（`jqdata`）、仓库内 `useful_skills/joinquant-skill/scripts/strategy_lint.py`。

---

## 文件结构（创建 / 修改）

| 路径 | 职责 |
|------|------|
| `SimTradeLab/strategies_jq/grid/__init__.py` | 包标记（空文件即可） |
| `SimTradeLab/strategies_jq/grid/jq_grid_pure.py` | 季度边界、档位价、收盘价穿档、按手数约束的股数计算（**禁止** `from jqdata import *`） |
| `SimTradeLab/strategies_jq/grid/tests/test_jq_grid_pure.py` | 上述纯函数的单元测试 |
| `SimTradeLab/strategies_jq/grid/multi_asset_minute_grid.py` | 聚宽策略主文件（`from jqdata import *` 仅在此文件） |
| `SimTradeLab/strategies_jq/grid/README.md` | 回测设置（1m）、参数表、与 spec 差异说明 |

**不修改** `useful_skills/joinquant-skill/` 核心脚本；仅 **命令行调用** 其 lint。

---

### Task 1: 纯函数模块 `jq_grid_pure.py`（季度与档位）

**Files:**
- Create: `SimTradeLab/strategies_jq/grid/__init__.py`
- Create: `SimTradeLab/strategies_jq/grid/jq_grid_pure.py`

- [ ] **Step 1: 创建空包**

Create `SimTradeLab/strategies_jq/grid/__init__.py`:

```python
# JoinQuant multi-asset grid strategy package (local helpers + strategy file).
```

- [ ] **Step 2: 写入 `jq_grid_pure.py` 完整初版**

Create `SimTradeLab/strategies_jq/grid/jq_grid_pure.py`:

```python
# -*- coding: utf-8 -*-
"""JoinQuant 网格纯逻辑：无 jqdata 依赖，供 pytest 与策略共用。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Sequence, Tuple


def year_quarter(d: date) -> Tuple[int, int]:
    q = (d.month - 1) // 3 + 1
    return d.year, q


def is_quarter_turn_first_trading_day(curr: date, prev_trade: Optional[date]) -> bool:
    """若前一交易日与当前日不在同一 (年, 季)，则当前为进入新季度后的首个交易日。"""
    if prev_trade is None:
        return True
    return year_quarter(curr) != year_quarter(prev_trade)


def build_grid_prices(anchor: float, grid_step: float, n_levels: int) -> Tuple[List[float], List[float]]:
    """
    卖档价（由低到高）、买档价（由高到低）。
    卖档: anchor * (1 + k * step), k=1..n
    买档: anchor * (1 - k * step), k=1..n
    """
    if anchor <= 0 or grid_step <= 0 or n_levels < 1:
        return [], []
    sells = [anchor * (1 + k * grid_step) for k in range(1, n_levels + 1)]
    buys = [anchor * (1 - k * grid_step) for k in range(1, n_levels + 1)]
    return sells, buys


def crosses_down_through(prev_close: Optional[float], curr_close: float, level: float) -> bool:
    """上一分钟收盘在 level 之上，本分钟收盘在 level 之下或等于：视为向下穿过（偏买入网格一侧）。"""
    if prev_close is None:
        return False
    return prev_close > level and curr_close <= level


def crosses_up_through(prev_close: Optional[float], curr_close: float, level: float) -> bool:
    """向上穿过 level（偏卖出网格一侧）。"""
    if prev_close is None:
        return False
    return prev_close < level and curr_close >= level


def floor_to_lot(shares: int, lot: int = 100) -> int:
    if shares < lot:
        return 0
    return (shares // lot) * lot


def max_buy_shares_for_cash(cash_budget: float, price: float, lot: int = 100) -> int:
    """在预算内按 A 股一手向下取整。"""
    if price <= 0 or cash_budget <= 0:
        return 0
    return floor_to_lot(int(cash_budget // price), lot)


@dataclass(frozen=True)
class LayerBudget:
    """将单标的名义上限 C 均分到各买/卖逻辑层（实现计划采用对称 2*n 层预算）。"""
    cap_per_security: float
    n_levels: int

    def per_layer_cash(self) -> float:
        # 买侧 n 档 + 卖侧 n 档共 2*n 份现金预算（卖侧受持仓约束，买侧用现金）
        denom = 2 * self.n_levels
        if denom <= 0:
            return 0.0
        return self.cap_per_security / denom
```

- [ ] **Step 3: Commit**

```bash
cd /mnt/c/Quant-Workspace
git add SimTradeLab/strategies_jq/grid/__init__.py SimTradeLab/strategies_jq/grid/jq_grid_pure.py
git commit -F /tmp/msg.txt
```

（将 `/tmp/msg.txt` 写一行：`feat(grid): add jq_grid_pure helpers`；若环境对 `git commit -m` 注入非法参数，沿用 `-F` 文件方式。）

---

### Task 2: 单元测试 `test_jq_grid_pure.py`

**Files:**
- Create: `SimTradeLab/strategies_jq/grid/tests/test_jq_grid_pure.py`

- [ ] **Step 1: 写入失败测试文件（完整内容）**

Create `SimTradeLab/strategies_jq/grid/tests/test_jq_grid_pure.py`:

```python
# -*- coding: utf-8 -*-
from datetime import date

import pytest

from SimTradeLab.strategies_jq.grid.jq_grid_pure import (
    LayerBudget,
    build_grid_prices,
    crosses_down_through,
    crosses_up_through,
    is_quarter_turn_first_trading_day,
    max_buy_shares_for_cash,
    year_quarter,
)


def test_year_quarter():
    assert year_quarter(date(2026, 1, 1)) == (2026, 1)
    assert year_quarter(date(2026, 4, 1)) == (2026, 2)


def test_is_quarter_turn_first_trading_day():
    assert is_quarter_turn_first_trading_day(date(2026, 4, 1), date(2026, 3, 31)) is True
    assert is_quarter_turn_first_trading_day(date(2026, 4, 2), date(2026, 4, 1)) is False
    assert is_quarter_turn_first_trading_day(date(2026, 1, 2), None) is True


def test_build_grid_prices():
    sells, buys = build_grid_prices(100.0, 0.01, 2)
    assert sells == [101.0, 102.0]
    assert buys == [99.0, 98.0]


def test_crosses_down_through():
    assert crosses_down_through(100.5, 99.0, 100.0) is True
    assert crosses_down_through(100.0, 100.0, 100.0) is False
    assert crosses_down_through(None, 99.0, 100.0) is False


def test_crosses_up_through():
    assert crosses_up_through(99.5, 100.5, 100.0) is True
    assert crosses_up_through(100.0, 100.0, 100.0) is False


def test_max_buy_shares_for_cash():
    assert max_buy_shares_for_cash(10000, 59.0) == 100
    assert max_buy_shares_for_cash(5000, 300.0) == 0


def test_layer_budget_per_layer():
    lb = LayerBudget(cap_per_security=80000, n_levels=4)
    assert lb.per_layer_cash() == pytest.approx(10000.0)
```

- [ ] **Step 2: 在仓库根添加可导入路径并运行 pytest（应失败：包路径）**

若 `SimTradeLab` 尚未作为包安装，在仓库根执行（任选其一，计划推荐 pytest 配置）：

在仓库根创建或修改 `pytest.ini`：

```ini
[pytest]
pythonpath = .
```

Create `pytest.ini` at `/mnt/c/Quant-Workspace/pytest.ini` with the above content if missing.

Run:

```bash
cd /mnt/c/Quant-Workspace
pytest SimTradeLab/strategies_jq/grid/tests/test_jq_grid_pure.py -v
```

**Expected:** 若 `SimTradeLab` 无 `__init__.py`，则添加 `SimTradeLab/__init__.py`（空文件）与 `SimTradeLab/strategies_jq/__init__.py`（空文件）后重跑直至 **PASS**。

- [ ] **Step 3: 测试通过后 commit**

```bash
git add SimTradeLab/strategies_jq/grid/tests/test_jq_grid_pure.py pytest.ini SimTradeLab/__init__.py SimTradeLab/strategies_jq/__init__.py
git commit -F /tmp/msg2.txt
```

`/tmp/msg2.txt` 内容：`test(grid): add jq_grid_pure unit tests`

---

### Task 3: 策略骨架 — `initialize` 与全局参数

**Files:**
- Create: `SimTradeLab/strategies_jq/grid/multi_asset_minute_grid.py`

- [ ] **Step 1: 创建策略文件头部与 `initialize`（完整可复制块）**

以下代码依赖聚宽运行时注入的 `jqdata` 符号；**仅**在本文件顶层 `from jqdata import *`。

Create `SimTradeLab/strategies_jq/grid/multi_asset_minute_grid.py`:

```python
# -*- coding: utf-8 -*-
"""
多标的分钟网格 — 设计见 docs/superpowers/specs/2026-05-07-joinquant-multi-grid-design.md

回测请在聚宽中选择「分钟」频率。初始资金在 Web 端设置（如 50 万）。
"""
from jqdata import *

import datetime as dt

from SimTradeLab.strategies_jq.grid.jq_grid_pure import (
    LayerBudget,
    build_grid_prices,
    crosses_down_through,
    crosses_up_through,
    is_quarter_turn_first_trading_day,
    max_buy_shares_for_cash,
)

# —— 可调参数（与 spec 对齐）——
BENCHMARK = '000300.XSHG'
INDEX_HS300 = '000300.XSHG'
INDEX_ZZ500 = '000905.XSHG'

FIXED_ETFS = [
    '510300.XSHG',
    '510500.XSHG',
    '159915.XSHE',
]

N_TOTAL_MIN = 20
N_TOTAL_MAX = 40
N_TOTAL_TARGET = 30

VOL_WINDOW = 30
LIQ_WINDOW = 60
LIQ_MIN_AVG_MONEY = 5e7
LIQ_MIN_QUANTILE = 0.30
LISTING_MIN_DAYS = 120
MAX_SUSPEND_RATIO = 0.15
MAX_LIMIT_MOVE_DAYS = 8

GRID_STEP = 0.009
GRID_LEVELS = 4

LIMIT_NEAR_PCT = 0.002
ORDER_STALE_MINUTES = 5


def initialize(context):
    set_benchmark(BENCHMARK)
    set_option('use_real_price', True)

    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.001,
        open_commission=0.0003, close_commission=0.0003,
        close_today_commission=0, min_commission=5,
    ), type='stock')
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0,
        open_commission=0.0002, close_commission=0.0002,
        close_today_commission=0, min_commission=5,
    ), type='fund')

    set_slippage(PriceRelatedSlippage(0.001), type='stock')
    set_slippage(PriceRelatedSlippage(0.0005), type='fund')

    g.etf_list = list(FIXED_ETFS)
    g.n_total_target = N_TOTAL_TARGET
    g.grid_step = GRID_STEP
    g.grid_levels = GRID_LEVELS
    g.prev_trade_date = None
    g.securities = []
    g.anchor = {}
    g.prev_minute_close = {}
    g.last_quarter_rebalance_date = None

    run_daily(quarter_rebalance_gate, time='09:30')
```

说明：`quarter_rebalance_gate` 在 Task 4 实现；分钟逻辑在 `handle_data`。

- [ ] **Step 2: Commit**

`git add SimTradeLab/strategies_jq/grid/multi_asset_minute_grid.py && git commit -F /tmp/msg3.txt`  
内容：`feat(grid): scaffold multi_asset_minute_grid initialize`

---

### Task 4: 季度门闩与股票筛选

**Files:**
- Modify: `SimTradeLab/strategies_jq/grid/multi_asset_minute_grid.py`（追加函数，不删改上节已约定符号）

- [ ] **Step 1: 追加 `quarter_rebalance_gate` 与选股辅助函数**

在 `multi_asset_minute_grid.py` 末尾追加（整段粘贴；若已存在占位则替换为完整实现）：

```python
def quarter_rebalance_gate(context):
    """日频 9:30：判断是否季度首交易日，若是则执行换池与锚价重置。"""
    cur = context.current_dt.date()
    prev = g.prev_trade_date
    if not is_quarter_turn_first_trading_day(cur, prev):
        return
    if g.last_quarter_rebalance_date == cur:
        return
    rebalance_quarter(context)
    g.last_quarter_rebalance_date = cur


def before_trading_start(context):
    """维护上一交易日（自然日不等价于交易日）。"""
    td = get_trade_days(end_date=context.current_dt.date(), count=2)
    if len(td) >= 2:
        g.prev_trade_date = dt.datetime.strptime(str(td[-2]), '%Y-%m-%d').date()
    elif len(td) == 1:
        g.prev_trade_date = None


def _merge_index_universe(as_of_date):
    u300 = set(get_index_stocks(INDEX_HS300, as_of_date))
    u500 = set(get_index_stocks(INDEX_ZZ500, as_of_date))
    return sorted(u300 | u500)


def _avg_daily_money(ser_close, ser_money, liq_window):
    df = ser_close.to_frame('close').join(ser_money.to_frame('money'), how='inner')
    if len(df) < liq_window:
        return None
    tail = df.iloc[-liq_window:]
    am = (tail['money'] / tail['close']).replace([float('inf')], float('nan')).dropna()
    if len(am) == 0:
        return None
    return float(tail['money'].mean())


def _volatility_std(ser_close, vol_window):
    if len(ser_close) < vol_window + 1:
        return None
    r = ser_close.pct_change().dropna()
    if len(r) < vol_window:
        return None
    return float(r.iloc[-vol_window:].std())


def _count_limit_like_moves(ser_close, ser_high, ser_low, vol_window):
    cnt = 0
    for i in range(-vol_window, 0):
        c = float(ser_close.iloc[i])
        h = float(ser_high.iloc[i])
        l = float(ser_low.iloc[i])
        if c <= 0:
            continue
        if abs(h - l) / c < 1e-6 and (abs(h - c) / c < 1e-6 or abs(l - c) / c < 1e-6):
            cnt += 1
        elif abs(h - c) / c < 0.001 or abs(l - c) / c < 0.001:
            cnt += 1
    return cnt


def screen_stocks(end_trade_date, index_asof_date, n_pick, log_fn):
    """
    end_trade_date: 日线数据窗口的结束日，**必须 <= 调仓日前一交易日**（无未来函数）。
    index_asof_date: 取指数成分用的日期，一般用调仓当日或 end_trade_date（与聚宽 `get_index_stocks` 约定一致）。
    返回长度为 <= n_pick 的股票列表（按波动从高到低）。
    """
    import pandas as pd

    end_d = end_trade_date
    universe = _merge_index_universe(index_asof_date)
    if len(universe) == 0:
        return []

    need = max(VOL_WINDOW, LIQ_WINDOW) + 5
    raw = get_price(
        universe,
        end_date=end_d,
        count=need,
        frequency='daily',
        fields=['close', 'high', 'low', 'volume', 'money'],
        panel=False,
        skip_paused=False,
        fq='pre',
    )
    if raw is None or len(raw) == 0:
        return []

    rows = []
    grouped = raw.groupby('code')
    all_med = []
    for code, grp in grouped:
        grp = grp.sort_values('time')
        if len(grp) < max(VOL_WINDOW, LIQ_WINDOW) + 1:
            continue
        close = grp['close']
        vol = _volatility_std(close, VOL_WINDOW)
        if vol is None:
            continue
        avg_money = _avg_daily_money(close, grp['money'], LIQ_WINDOW)
        if avg_money is None or avg_money < LIQ_MIN_AVG_MONEY:
            continue
        all_med.append(avg_money)

    if len(all_med) == 0:
        return []
    thr_med = float(pd.Series(all_med).quantile(LIQ_MIN_QUANTILE))

    cand = []
    for code, grp in grouped:
        grp = grp.sort_values('time')
        if len(grp) < max(VOL_WINDOW, LIQ_WINDOW) + 1:
            continue
        close = grp['close']
        high = grp['high']
        low = grp['low']
        vol = _volatility_std(close, VOL_WINDOW)
        avg_money = _avg_daily_money(close, grp['money'], LIQ_WINDOW)
        if vol is None or avg_money is None:
            continue
        if avg_money < max(LIQ_MIN_AVG_MONEY, thr_med):
            continue
        if _count_limit_like_moves(close, high, low, VOL_WINDOW) > MAX_LIMIT_MOVE_DAYS:
            continue
        info = get_security_info(code)
        if info is None or info.start_date > end_trade_date - dt.timedelta(days=LISTING_MIN_DAYS):
            continue
        cd = get_current_data()[code]
        if cd.is_st:
            continue
        name = cd.name or ''
        if 'ST' in name or '*' in name:
            continue
        susp = float((grp['volume'].iloc[-VOL_WINDOW:] == 0).mean())
        if susp > MAX_SUSPEND_RATIO:
            continue
        cand.append((code, vol))

    cand.sort(key=lambda x: -x[1])
    return [c for c, _ in cand[:n_pick]]


def rebalance_quarter(context):
    """季度换池：合并 ETF + 股票，设定锚价（优先 9:31 分钟 open，否则昨收）。"""
    import pandas as pd

    d = context.current_dt.date()
    n_stock_target = min(max(g.n_total_target - len(g.etf_list), 0), N_TOTAL_MAX - len(g.etf_list))
    n_stock_target = max(n_stock_target, 0)

    prev_days = get_trade_days(end_date=d, count=2)
    if len(prev_days) < 2:
        log.warn('not enough trade days for screen')
        stocks = []
    else:
        end_trade = dt.datetime.strptime(str(prev_days[-2]), '%Y-%m-%d').date()
        stocks = screen_stocks(end_trade, d, n_stock_target, log.info)
    g.securities = list(g.etf_list) + list(stocks)
    log.info('quarter rebalance %s securities=%s' % (d, g.securities))

    g.anchor.clear()
    g.prev_minute_close.clear()

    for s in list(context.portfolio.positions.keys()):
        if s not in g.securities:
            order_target(s, 0)

    for s in g.securities:
        px_open = None
        try:
            bar = get_bars(s, count=1, unit='1m', fields=['open'], include_now=True, end_dt=context.current_dt)
            if bar is not None and len(bar) > 0:
                px_open = float(bar['open'][-1])
        except Exception:
            px_open = None
        if px_open is None or px_open <= 0:
            h = attribute_history(s, 1, '1d', ['close'], skip_paused=False)
            px_open = float(h['close'][-1]) if h is not None and len(h['close']) > 0 else None
        if px_open is None or px_open <= 0:
            log.warn('skip anchor %s' % s)
            continue
        g.anchor[s] = px_open
        g.prev_minute_close[s] = None
        log.info('anchor %s = %s' % (s, px_open))
```

- [ ] **Step 2: 在聚宽或本地静态检查语法**

本地（无 jqdata 会失败属预期）：`python -m py_compile SimTradeLab/strategies_jq/grid/multi_asset_minute_grid.py` 可能因 `from jqdata import *` 失败。**可跳过**；以聚宽编辑器语法检查为准。

- [ ] **Step 3: Commit**

`feat(grid): add quarterly stock screen and rebalance`

---

### Task 5: 分钟 `handle_data`、限价、撤单、涨跌停门控

**Files:**
- Modify: `SimTradeLab/strategies_jq/grid/multi_asset_minute_grid.py`

- [ ] **Step 1: 实现 `_near_limit_block`, `_cancel_stale_orders`, `handle_data`**

追加到同一文件：

```python
def _is_stock(security):
    return security.endswith('.XSHE') or security.endswith('.XSHG')


def _near_limit_block(security, side_buy):
    """距涨停/跌停过近：side_buy=True 时若为买入则拦截；卖出侧用于放行减仓。"""
    cd = get_current_data()[security]
    if not cd.high_limit or not cd.low_limit:
        return False
    last = cd.last_price
    if last is None or last <= 0:
        return False
    hi = cd.high_limit
    lo = cd.low_limit
    if hi and last >= hi * (1 - LIMIT_NEAR_PCT):
        return side_buy
    if lo and last <= lo * (1 + LIMIT_NEAR_PCT):
        return not side_buy
    return False


def _cancel_stale_orders(context, stale_minutes):
    """撤销超过 stale_minutes 的未完成限价单。"""
    now = context.current_dt
    for o in get_open_orders():
        if o is None:
            continue
        add = o.add_time
        if add is None:
            continue
        if (now - add).total_seconds() > stale_minutes * 60:
            cancel_order(o)


def handle_data(context, data):
    _cancel_stale_orders(context, ORDER_STALE_MINUTES)

    if not g.securities:
        return

    tv = context.portfolio.total_value
    n = len(g.securities)
    if n <= 0:
        return
    cap = tv / float(n)
    lb = LayerBudget(cap_per_security=cap, n_levels=g.grid_levels)
    layer_cash = lb.per_layer_cash()

    for s in g.securities:
        if s not in g.anchor:
            continue
        if not data.can_trade(s):
            continue
        cur = data.current(s)
        curr_close = cur.close
        if curr_close is None or curr_close <= 0:
            continue

        anchor = g.anchor[s]
        sells, buys = build_grid_prices(anchor, g.grid_step, g.grid_levels)
        prev = g.prev_minute_close.get(s)

        pos = context.portfolio.positions[s]
        closeable = int(pos.closeable_amount) if _is_stock(s) else int(pos.total_amount)
        total_amt = int(pos.total_amount)

        for bp in buys:
            if crosses_down_through(prev, float(curr_close), float(bp)):
                if _near_limit_block(s, side_buy=True):
                    break
                cash = context.portfolio.available_cash
                want = max_buy_shares_for_cash(min(layer_cash, cash * 0.95), float(bp))
                if want > 0:
                    order(s, want, style=LimitOrderStyle(float(bp)))

        for sp in sells:
            if crosses_up_through(prev, float(curr_close), float(sp)):
                if _near_limit_block(s, side_buy=False):
                    break
                sellable = closeable if _is_stock(s) else total_amt
                lot = max_buy_shares_for_cash(layer_cash * float(sp), float(sp))
                amt = min(sellable, lot) if lot > 0 else 0
                if amt > 0:
                    order(s, -amt, style=LimitOrderStyle(float(sp)))

        g.prev_minute_close[s] = float(curr_close)
```

- [ ] **Step 2: 自检与 spec 对照**

- 触价：仅用 `prev` 与 `curr_close`，未用 high/low。  
- T+1：卖出量用 `closeable_amount`（股票）；ETF 用 `total_amount`。  
- 单票名义：用 `layer_cash` 与 `min(layer_cash, cash)` 约束单笔；若需更严格「持仓市值 ≤ cap」，在后续迭代对 `order_target_value` 做尾盘校正（**本计划 YAGNI：不加入**）。

- [ ] **Step 3: Commit**

`feat(grid): add minute handle_data grid crossing and orders`

---

### Task 6: `README.md` 与 Lint

**Files:**
- Create: `SimTradeLab/strategies_jq/grid/README.md`

- [ ] **Step 1: README 完整内容**

```markdown
# 多标的分钟网格（聚宽）

## 使用方式

1. 打开 `multi_asset_minute_grid.py`，全选复制到聚宽「投资策略」编辑器。
2. 回测设置：**分钟线**、初始资金建议 50 万、起止日期自定。
3. 首次回测前在聚宽内确认 `get_bars` / `get_price` 行为与当前 API 一致。

## 参数一览

| 变量 | 默认 | 含义 |
|------|------|------|
| N_TOTAL_TARGET | 30 | 目标总只数（含 3 ETF），实际在 [20,40] 由筛选结果截断 |
| GRID_STEP | 0.009 | 统一档距 |
| GRID_LEVELS | 4 | 上下各档位数 |
| VOL_WINDOW / LIQ_WINDOW | 30 / 60 | 波动与流动性窗口 |

## 与设计 spec 的差异 / 风险

- 涨跌停「一字」识别使用简化启发式，可能与真实涨跌停计数有偏差；以回测日志为准调 `MAX_LIMIT_MOVE_DAYS`。
- 单票名义上限以「每层现金预算 + 限价股数」软约束实现；极端价格下若需硬顶，可再加日终 `order_target_value` 校正。

## Lint

在仓库根目录执行：

\`\`\`bash
python useful_skills/joinquant-skill/scripts/strategy_lint.py SimTradeLab/strategies_jq/grid/multi_asset_minute_grid.py
\`\`\`
```

- [ ] **Step 2: 运行 Lint**

```bash
cd /mnt/c/Quant-Workspace
python useful_skills/joinquant-skill/scripts/strategy_lint.py SimTradeLab/strategies_jq/grid/multi_asset_minute_grid.py
```

**Expected:** 无 `JQ001` 类错误；若有「缺少 g.」类警告，在策略中确保跨日状态均挂在 `g.` 上（本计划已使用 `g.*`）。

- [ ] **Step 3: Commit**

`docs(grid): add README and lint instructions`

---

### Task 7: 回测验证清单（人工，在聚宽 Web）

**Files:** 无代码变更

- [ ] **Step 1:** 运行 2019–2021（偏震荡）与 2018、2022（压力）两段回测，记录最大回撤、年化、月均收益波动。  
- [ ] **Step 2:** 将 `GRID_STEP` 改为 `0.008` 与 `0.01` 各跑一次，对比换手与佣金占比。  
- [ ] **Step 3:** 在日志中确认每季度首日出现 `quarter rebalance` 与每只 `anchor` 打印。

---

## Plan self-review（对照 spec）

| Spec 章节 | 对应任务 |
|-----------|----------|
| 2 ETF 三槽 + 20–40 只 + 均分 | Task3 参数；Task4 `g.securities`；Task5 `cap = tv/n` |
| 3 季度 300∪500 选股与过滤 | Task4 `screen_stocks` / `rebalance_quarter` |
| 4.1 分钟频率 | README + 聚宽 UI |
| 4.2 锚价季度重置 | Task4 `rebalance_quarter` |
| 4.3 档距与深度 | `jq_grid_pure.build_grid_prices` + `GRID_*` |
| 4.4 收盘价触档 | Task5 `prev` / `curr_close` |
| 4.5 T+1 / ETF | Task5 `closeable_amount` vs `total_amount` |
| 4.5 涨跌停 | Task5 `_near_limit_block` |
| 6 撤单 | Task5 `_cancel_stale_orders` |
| 6 佣金滑点 | Task3 `set_order_cost` / `set_slippage` |
| 7 验证 | Task7 |

**已知缺口（已显式接受 YAGNI）：**  
- `before_trading_start` 与 `quarter_rebalance_gate` 的调用顺序依赖聚宽引擎；若发现 9:30 门闩早于 `prev_trade_date` 更新，将 `g.prev_trade_date` 维护移入 `quarter_rebalance_gate` 开头并用 `get_trade_days` 取前两日。计划在首次聚宽联调时 **单步验证并补丁**。

**Placeholder 扫描：** 本计划未使用「TBD / TODO / 稍后实现」类措辞；可执行命令与代码块已写全。

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-07-joinquant-multi-asset-grid.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — 每个 Task 派生子代理执行，任务间人工复核，迭代快  

**2. Inline Execution** — 本会话内按 Task 顺序实现，批量提交并在检查点停顿复核  

**Which approach?**
