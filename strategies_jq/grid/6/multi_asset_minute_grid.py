# -*- coding: utf-8 -*-
"""
多标的分钟网格 — 设计见 docs/superpowers/specs/2026-05-07-joinquant-multi-grid-design.md

【聚宽】单文件粘贴：新建策略 → 清空默认代码 → 全选粘贴本文件 → 回测选「分钟」、资金在网页设置。

【本地】jq_grid_pure.py 与下方「纯逻辑」区块保持一致，供 pytest。

锚价：季度调仓日 run_daily(9:30) 内取当日日线 open（attribute_history 1d），失败则用当日日线 close。

分钟穿档（与聚宽引擎一致）：每次 handle_data 触发时，官方约定 `data[code]` 为**上一根已走完的分钟 K**，
用其 `close` 与 `g.prev_minute_close`（存的是该标的上一根同样来源的 close）做上下穿判断；停牌看
`get_current_data()[code].paused`。涨跌停距离用 `get_current_data().last_price` 快照，与 K 线 close 职责不同。

同一根 K 的 close 若一次穿多档，会对各档分别尝试下单（可能多笔）。
"""
from jqdata import *

import datetime as dt
from datetime import date
from typing import List, Optional, Tuple

# ========== 纯逻辑（与 jq_grid_pure.py 保持一致，供聚宽单文件运行）==========


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
    """上一根 K 收盘在 level 之上，本根收盘在 level 之下或等于：向下穿过（偏买侧网格）。"""
    if prev_close is None:
        return False
    return prev_close > level and curr_close <= level


def crosses_up_through(prev_close: Optional[float], curr_close: float, level: float) -> bool:
    """上一根在 level 之下，本根在 level 之上或等于：向上穿过（偏卖侧网格）。"""
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


class LayerBudget(object):
    """将单标的名义上限 C 均分到各买/卖逻辑层（对称 2*n 层预算）。不用 dataclasses，兼容聚宽策略环境。"""

    __slots__ = ('cap_per_security', 'n_levels')

    def __init__(self, cap_per_security, n_levels):
        self.cap_per_security = float(cap_per_security)
        self.n_levels = int(n_levels)

    def per_layer_cash(self):
        denom = 2 * self.n_levels
        if denom <= 0:
            return 0.0
        return self.cap_per_security / denom


# ========== 策略参数与回调 ==========

BENCHMARK = '000300.XSHG'
INDEX_HS300 = '000300.XSHG'
INDEX_ZZ500 = '000905.XSHG'

FIXED_ETFS = [
    '510300.XSHG',
    '510500.XSHG',
    '159915.XSHE',
]

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
    set_option('avoid_future_data', True)

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


def before_trading_start(context):
    """维护上一交易日。"""
    td = get_trade_days(end_date=context.current_dt.date(), count=2)
    if len(td) >= 2:
        g.prev_trade_date = dt.datetime.strptime(str(td[-2]), '%Y-%m-%d').date()
    elif len(td) == 1:
        g.prev_trade_date = None


def quarter_rebalance_gate(context):
    """日频 9:30：冷启动或季度首个交易日换池与锚价重置。"""
    cur = context.current_dt.date()
    prev = g.prev_trade_date
    cold = len(g.securities) == 0
    roll = is_quarter_turn_first_trading_day(cur, prev)
    if not cold and not roll:
        return
    if g.last_quarter_rebalance_date == cur:
        return
    rebalance_quarter(context)
    g.last_quarter_rebalance_date = cur


def _merge_index_universe(as_of_date):
    u300 = set(get_index_stocks(INDEX_HS300, as_of_date))
    u500 = set(get_index_stocks(INDEX_ZZ500, as_of_date))
    return sorted(u300 | u500)


def _avg_daily_money(ser_close, ser_money, liq_window):
    df = ser_close.to_frame('close').join(ser_money.to_frame('money'), how='inner')
    if len(df) < liq_window:
        return None
    tail = df.iloc[-liq_window:]
    if (tail['close'] <= 0).any():
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


def _listing_start_date(info):
    if info is None or not hasattr(info, 'start_date'):
        return None
    sd = info.start_date
    if sd is None:
        return None
    if hasattr(sd, 'date') and callable(getattr(sd, 'date', None)):
        try:
            return sd.date()
        except Exception:
            pass
    if isinstance(sd, dt.date) and not isinstance(sd, dt.datetime):
        return sd
    if isinstance(sd, str):
        return dt.datetime.strptime(sd[:10], '%Y-%m-%d').date()
    return None


def screen_stocks(end_trade_date, index_asof_date, n_pick, log_fn):
    """
    end_trade_date: 日线数据窗口结束日（<= 调仓日前一交易日）。
    index_asof_date: 指数成分查询日（调仓当日）。
    """
    import pandas as pd

    end_d = end_trade_date
    universe = _merge_index_universe(index_asof_date)
    if len(universe) == 0:
        return []

    # 日线拉取条数：须 >= 波动/流动性窗口（误删 need 会导致聚宽运行 NameError）
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
        sd = _listing_start_date(info)
        if sd is None or sd > end_trade_date - dt.timedelta(days=LISTING_MIN_DAYS):
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
    """季度换池：合并 ETF + 股票，设定锚价。"""
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
        h1 = attribute_history(s, 1, '1d', ['open', 'close'], skip_paused=False, fq='pre')
        if h1 is not None and len(h1['open']) > 0:
            px_open = float(h1['open'][-1])
        if px_open is None or px_open <= 0:
            if h1 is not None and len(h1['close']) > 0:
                px_open = float(h1['close'][-1])
        if px_open is None or px_open <= 0:
            log.warn('skip anchor %s' % s)
            continue
        g.anchor[s] = px_open
        g.prev_minute_close[s] = None
        log.info('anchor %s = %s' % (s, px_open))


def _is_stock(security):
    return security.endswith('.XSHE') or security.endswith('.XSHG')


def _near_limit_block(security, side_buy):
    """距涨跌停过近则拦截加仓方向；用 get_current_data 快照价，与穿档用的 K 线 close 分开。"""
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
    oo = get_open_orders()
    if not oo:
        return
    for o in oo.values():
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
        try:
            bar = data[s]
        except KeyError:
            continue
        cd = get_current_data()[s]
        if cd.paused:
            continue
        curr_close = float(bar.close)
        if curr_close != curr_close or curr_close <= 0:
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
