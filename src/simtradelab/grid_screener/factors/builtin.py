from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from simtradelab.grid_screener.context import FactorContext
from simtradelab.grid_screener.labels import history_insufficient_flags
from simtradelab.grid_screener.metrics import (
    acf1_returns,
    gap_tail_ratio,
    intraday_extreme_ratio,
    mean_abs_overnight_gap,
    ols_log_close_trend,
    range_time_ratio,
    realized_vol_ann,
    variance_ratio_lm,
    vol_comfort_score,
)


def _nan_map(keys: tuple[str, ...]) -> dict[str, float]:
    v = float("nan")
    return dict.fromkeys(keys, v)


class MetaFactor:
    name = "meta"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        return {
            "symbol": ctx.meta.symbol,
            "name": ctx.meta.name,
            "asset_type": ctx.meta.asset_type,
        }


class SampleQualityFactor:
    name = "sample_quality"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        eff = int(ctx.close.size) if not ctx.window.empty else 0
        hist_short, insuff = history_insufficient_flags(
            eff, ctx.params.window_trading_days, ctx.params.n_min_valid
        )
        return {
            "effective_days": eff,
            "history_short": hist_short,
            "insufficient_data": insuff,
        }


class TrendFactor:
    name = "trend"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        if ctx.insufficient or ctx.close.size < 3:
            return _nan_map(("trend_t", "trend_r2"))
        t_stat, r2 = ols_log_close_trend(ctx.log_close)
        return {"trend_t": t_stat, "trend_r2": r2}


class VarianceRatioFactor:
    name = "variance_ratio"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        if ctx.insufficient:
            return {"variance_ratio": float("nan")}
        return {"variance_ratio": variance_ratio_lm(ctx.log_close, q=2)}


class Acf1Factor:
    name = "acf1"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        if ctx.insufficient:
            return {"acf1_ret": float("nan")}
        return {"acf1_ret": acf1_returns(ctx.r1)}


class VolatilityFactor:
    name = "volatility"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        if ctx.insufficient:
            return {
                "rv_ann": float("nan"),
                "vol_comfort_score": float("nan"),
                "vol_band": "unknown",
            }
        rv = realized_vol_ann(ctx.r1)
        vc = vol_comfort_score(rv, ctx.params.sigma_low, ctx.params.sigma_high)
        return {
            "rv_ann": rv,
            "vol_comfort_score": vc,
            "vol_band": _vol_band(rv, ctx.params),
        }


class GapFactor:
    name = "gap"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        if ctx.insufficient:
            return _nan_map(("mean_abs_gap", "gap_tail_ratio", "intraday_extreme_ratio"))
        p = ctx.params
        return {
            "mean_abs_gap": mean_abs_overnight_gap(ctx.open, ctx.close),
            "gap_tail_ratio": gap_tail_ratio(ctx.open, ctx.close, p.gap_tail_delta),
            "intraday_extreme_ratio": intraday_extreme_ratio(
                ctx.open, ctx.high, ctx.low, p.intraday_extreme_delta
            ),
        }


class RangeRegimeFactor:
    name = "range_regime"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        if ctx.insufficient:
            return {"range_time_ratio": float("nan")}
        p = ctx.params
        c = ctx.close
        ms = pd.Series(c).rolling(p.range_ma_short, min_periods=p.range_ma_short).mean().to_numpy()
        ml = pd.Series(c).rolling(p.range_ma_long, min_periods=p.range_ma_long).mean().to_numpy()
        return {
            "range_time_ratio": range_time_ratio(
                c, ms, ml, p.range_band_price_vs_long, p.range_band_spread_vs_long
            ),
        }


class GridScoreFactor:
    """可选：固定权重综合分，依赖前述因子列已在 ctx.outputs 中。"""

    name = "grid_score"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        row = ctx.outputs
        return {"grid_friendly_score": _grid_friendly_score(row)}


def _vol_band(rv: float, params: Any) -> str:
    if not np.isfinite(rv):
        return "unknown"
    if rv < params.sigma_low:
        return "vol_low"
    if rv > params.sigma_high:
        return "vol_high"
    return "vol_mid"


def _grid_friendly_score(row: dict[str, object]) -> float:
    if row.get("insufficient_data"):
        return float("nan")
    tt = float(row.get("trend_t") or 0.0)
    rtr = float(row.get("range_time_ratio") or 0.0)
    vc = float(row.get("vol_comfort_score") or 0.0)
    vr = float(row.get("variance_ratio") or 1.0)
    gtr = float(row.get("gap_tail_ratio") or 0.0)
    if not all(map(math.isfinite, (tt, rtr, vc, vr, gtr))):
        return float("nan")
    s = 40.0 * rtr + 25.0 * vc - min(30.0, 0.35 * abs(tt))
    if vr > 1.0:
        s -= (vr - 1.0) * 18.0
    s -= gtr * 22.0
    return float(s)


BUILTIN_FACTORS = (
    MetaFactor(),
    SampleQualityFactor(),
    TrendFactor(),
    VarianceRatioFactor(),
    Acf1Factor(),
    VolatilityFactor(),
    GapFactor(),
    RangeRegimeFactor(),
    GridScoreFactor(),
)
