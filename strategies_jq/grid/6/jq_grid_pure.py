# -*- coding: utf-8 -*-
"""JoinQuant 网格纯逻辑：无 jqdata 依赖，供本地 pytest。

逻辑须与 multi_asset_minute_grid.py 中「========== 纯逻辑」区块保持一致。"""
from __future__ import annotations

from datetime import date
from typing import List, Optional, Tuple


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
    """将单标的名义上限 C 均分到各买/卖逻辑层（对称 2*n 层预算）。与 multi_asset_minute_grid.py 保持一致。"""

    __slots__ = ('cap_per_security', 'n_levels')

    def __init__(self, cap_per_security, n_levels):
        self.cap_per_security = float(cap_per_security)
        self.n_levels = int(n_levels)

    def per_layer_cash(self):
        denom = 2 * self.n_levels
        if denom <= 0:
            return 0.0
        return self.cap_per_security / denom
