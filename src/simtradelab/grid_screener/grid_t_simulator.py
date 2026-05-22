"""纯做T存钱罐网格回测 — 移植自 my_docs/grid_friendly_screener/04-另一个思路2/jq.py。

默认使用未复权日线收盘价（对齐聚宽 use_real_price=True），盘前按 adj_a 缩放网格基准价。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from simtradelab.grid_screener.config import GridTParams


@dataclass
class GridTResult:
    grid_t_profit_yuan: float
    grid_t_profit_rate: float
    grid_t_harvest_count: int
    grid_t_buy_count: int
    grid_t_sell_count: int
    grid_t_init_shares: int
    grid_t_active_days: int
    grid_t_bad_bar_count: int
    grid_t_veto: bool
    grid_t_veto_reason: str


@dataclass
class _GridTState:
    base_price: float = 0.0
    position: int = 0
    cash: float = 0.0
    grid_t_profit: float = 0.0
    harvest_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    init_done: bool = False
    init_shares: int = 0
    active_days: int = 0
    last_adj_a: float | None = None


def _shares_for_amount(amount: float, price: float, lot_size: int) -> int:
    if price <= 0 or amount <= 0:
        return 0
    return int(amount / price / lot_size) * lot_size


def _sell_fee(trade_value: float, params: GridTParams, *, is_etf: bool) -> float:
    comm = max(params.min_commission, trade_value * params.close_commission)
    tax = 0.0 if is_etf else trade_value * params.close_tax
    return comm + tax


def _valid_price(price: float, min_price: float) -> bool:
    return bool(np.isfinite(price) and not math.isnan(price) and price >= min_price)


def _sanitize_window(window: pd.DataFrame, params: GridTParams) -> tuple[pd.DataFrame, int]:
    """剔除无效 OHLC（非正、非有限、 high<low ），避免前复权脏数据虚增利润。"""
    min_p = params.min_valid_price
    keep: list[pd.Series] = []
    bad = 0
    for _, row in window.iterrows():
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        if not all(_valid_price(x, min_p) for x in (o, h, l, c)):
            bad += 1
            continue
        if h + 1e-12 < l:
            bad += 1
            continue
        keep.append(row)
    if not keep:
        return window.iloc[0:0].copy(), bad
    return pd.DataFrame(keep), bad


def _apply_adj_factor(state: _GridTState, adj_a: float | None) -> None:
    if adj_a is None or not np.isfinite(adj_a) or adj_a <= 0:
        return
    if state.init_done and state.last_adj_a is not None and adj_a != state.last_adj_a:
        state.base_price *= state.last_adj_a / adj_a
    state.last_adj_a = adj_a


def _process_upward_grids(
    state: _GridTState,
    price: float,
    params: GridTParams,
    *,
    is_etf: bool,
) -> None:
    step = params.grid_step
    cap = params.max_grid_steps_per_price
    n = 0
    while price >= state.base_price * (1.0 + step):
        shares = _shares_for_amount(params.trade_amount, price, params.lot_size)
        if shares > 0 and state.position >= shares:
            trade_value = shares * price
            cost_value = shares * state.base_price
            fee = _sell_fee(trade_value, params, is_etf=is_etf)
            state.grid_t_profit += trade_value - cost_value - fee
            state.position -= shares
            state.harvest_count += 1
            state.sell_count += 1
        state.base_price *= 1.0 + step
        n += 1
        if n >= cap:
            break


def _process_downward_grids(
    state: _GridTState,
    price: float,
    params: GridTParams,
) -> None:
    step = params.grid_step
    cap = params.max_grid_steps_per_price
    n = 0
    while price <= state.base_price * (1.0 - step):
        shares = _shares_for_amount(params.trade_amount, price, params.lot_size)
        cost_needed = shares * price
        if shares > 0 and state.cash >= cost_needed:
            state.position += shares
            state.cash -= cost_needed
            state.buy_count += 1
        state.base_price *= 1.0 - step
        n += 1
        if n >= cap:
            break


def _try_initial_buy(state: _GridTState, price: float, params: GridTParams) -> None:
    shares = _shares_for_amount(params.initial_amount, price, params.lot_size)
    cost = shares * price
    if shares <= 0 or state.cash < cost:
        return
    state.position += shares
    state.cash -= cost
    state.base_price = price
    state.init_done = True
    state.init_shares = shares


def _day_prices(row: pd.Series, params: GridTParams) -> list[float]:
    c = float(row["close"])
    if not params.use_intraday_path:
        return [c]
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    if c >= o:
        return [o, l, h, c]
    return [o, h, l, c]


def _process_price(state: _GridTState, price: float, params: GridTParams, *, is_etf: bool) -> None:
    if not _valid_price(price, params.min_valid_price):
        return
    if not state.init_done:
        _try_initial_buy(state, price, params)
        return
    _process_upward_grids(state, price, params, is_etf=is_etf)
    _process_downward_grids(state, price, params)


def _nan_result(bad_bar_count: int = 0, *, veto: bool = False, reason: str = "") -> GridTResult:
    nan = float("nan")
    return GridTResult(nan, nan, 0, 0, 0, 0, 0, bad_bar_count, veto, reason)


def simulate_grid_t_profit(
    window: pd.DataFrame,
    params: GridTParams,
    *,
    is_etf: bool = False,
    min_active_days: int = 2,
) -> GridTResult:
    """对单标的 OHLCV 窗口运行存钱罐网格回测。"""
    if window.empty or len(window) < 2:
        return _nan_result()

    raw_len = len(window)
    clean, bad_bar_count = _sanitize_window(window, params)
    bad_ratio = bad_bar_count / raw_len if raw_len else 1.0
    if bad_ratio > params.max_bad_bar_ratio:
        return _nan_result(bad_bar_count, veto=True, reason="bad_price_bars")
    if len(clean) < max(min_active_days, 2):
        return _nan_result(bad_bar_count, veto=True, reason="insufficient_valid_bars")

    cash_budget = params.initial_amount * params.reserve_cash_ratio
    state = _GridTState(cash=cash_budget)

    for _, row in clean.iterrows():
        adj_a = None
        if "adj_a" in row.index:
            val = row["adj_a"]
            if val is not None and np.isfinite(val):
                adj_a = float(val)
        _apply_adj_factor(state, adj_a)

        for price in _day_prices(row, params):
            _process_price(state, price, params, is_etf=is_etf)

        if state.init_done:
            state.active_days += 1

    if not state.init_done or state.init_shares <= 0:
        return _nan_result(bad_bar_count, veto=True, reason="init_failed")

    profit_rate = state.grid_t_profit / params.initial_amount
    return GridTResult(
        grid_t_profit_yuan=state.grid_t_profit,
        grid_t_profit_rate=profit_rate,
        grid_t_harvest_count=state.harvest_count,
        grid_t_buy_count=state.buy_count,
        grid_t_sell_count=state.sell_count,
        grid_t_init_shares=state.init_shares,
        grid_t_active_days=state.active_days,
        grid_t_bad_bar_count=bad_bar_count,
        grid_t_veto=False,
        grid_t_veto_reason="",
    )
