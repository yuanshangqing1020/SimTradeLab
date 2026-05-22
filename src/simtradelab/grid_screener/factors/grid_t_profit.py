"""纯做T存钱罐网格收益因子 — 见 my_docs/grid_friendly_screener/04-另一个思路2/思路.md"""

from __future__ import annotations

import numpy as np

from simtradelab.grid_screener.config import GridTParams
from simtradelab.grid_screener.context import FactorContext
from simtradelab.grid_screener.grid_t_simulator import simulate_grid_t_profit


def _gtp(ctx: FactorContext) -> GridTParams:
    return ctx.params.grid_t


_ETF_CODE_PREFIXES = ("51", "50", "15", "16", "18")


def _is_etf(meta) -> bool:
    if meta.asset_type == "etf":
        return True
    code = str(meta.symbol).split(".")[0]
    return code.startswith(_ETF_CODE_PREFIXES)


class GridTProfitFactor:
    """对单标的运行存钱罐网格回测，输出累计落袋现金及衍生指标。"""

    name = "grid_t_profit"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        nan = float("nan")
        if ctx.insufficient:
            return _empty_result(nan)

        p = _gtp(ctx)
        is_etf = _is_etf(ctx.meta)
        result = simulate_grid_t_profit(
            ctx.window,
            p,
            is_etf=is_etf,
            min_active_days=ctx.params.n_min_valid,
        )

        if result.grid_t_veto or not np.isfinite(result.grid_t_profit_yuan):
            out = _empty_result(nan)
            out["grid_t_veto"] = result.grid_t_veto
            out["grid_t_veto_reason"] = result.grid_t_veto_reason
            out["grid_t_bad_bar_count"] = result.grid_t_bad_bar_count
            return out

        active = max(result.grid_t_active_days, 1)
        trading_years = active / 250.0
        profit_per_day = result.grid_t_profit_yuan / active
        profit_ann = (
            result.grid_t_profit_yuan / trading_years if trading_years > 0 else nan
        )
        harvest_per_250d = result.grid_t_harvest_count / trading_years if trading_years > 0 else nan

        return {
            "grid_t_profit_yuan": result.grid_t_profit_yuan,
            "grid_t_profit_rate": result.grid_t_profit_rate,
            "grid_t_profit_per_day": profit_per_day,
            "grid_t_profit_ann_yuan": profit_ann,
            "grid_t_harvest_count": result.grid_t_harvest_count,
            "grid_t_harvest_per_250d": harvest_per_250d,
            "grid_t_buy_count": result.grid_t_buy_count,
            "grid_t_sell_count": result.grid_t_sell_count,
            "grid_t_init_shares": result.grid_t_init_shares,
            "grid_t_active_days": result.grid_t_active_days,
            "grid_t_bad_bar_count": result.grid_t_bad_bar_count,
            "grid_t_veto": False,
            "grid_t_veto_reason": "",
        }


def _empty_result(nan: float) -> dict[str, object]:
    return {
        "grid_t_profit_yuan": nan,
        "grid_t_profit_rate": nan,
        "grid_t_profit_per_day": nan,
        "grid_t_profit_ann_yuan": nan,
        "grid_t_harvest_count": 0,
        "grid_t_harvest_per_250d": nan,
        "grid_t_buy_count": 0,
        "grid_t_sell_count": 0,
        "grid_t_init_shares": 0,
        "grid_t_active_days": 0,
        "grid_t_bad_bar_count": 0,
        "grid_t_veto": False,
        "grid_t_veto_reason": "",
    }


GRID_T_FACTORS = (GridTProfitFactor(),)
