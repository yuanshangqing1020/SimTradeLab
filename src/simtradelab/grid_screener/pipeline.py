from __future__ import annotations

import numpy as np
import pandas as pd

from simtradelab.grid_screener.config import ScreenerParams, UniverseItem
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
from simtradelab.grid_screener.preprocess import normalize_ohlcv, slice_window
from simtradelab.grid_screener.scoring import grid_friendly_score


def compute_screener_row(raw: pd.DataFrame, meta: UniverseItem, params: ScreenerParams) -> dict[str, object]:
    df0 = normalize_ohlcv(raw)
    if df0.empty:
        return _empty_row(meta, effective_days=0)

    win = min(params.window_trading_days, len(df0))
    df = slice_window(df0, win)
    c = df["close"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    log_c = np.log(c)
    r1 = np.diff(log_c)

    eff = int(c.size)
    hist_short, insuff = history_insufficient_flags(eff, params.window_trading_days, params.n_min_valid)
    if insuff:
        base = _empty_row(meta, effective_days=eff)
        base["history_short"] = False
        base["insufficient_data"] = True
        base["grid_friendly_score"] = float("nan")
        return base

    trend_t, trend_r2 = ols_log_close_trend(log_c)
    vr = variance_ratio_lm(log_c, q=2)
    rho1 = acf1_returns(r1)
    rv = realized_vol_ann(r1)
    vcomf = vol_comfort_score(rv, params.sigma_low, params.sigma_high)

    mag = mean_abs_overnight_gap(o, c)
    gtr = gap_tail_ratio(o, c, params.gap_tail_delta)

    ms = pd.Series(c).rolling(params.range_ma_short, min_periods=params.range_ma_short).mean().to_numpy()
    ml = pd.Series(c).rolling(params.range_ma_long, min_periods=params.range_ma_long).mean().to_numpy()
    rtr = range_time_ratio(c, ms, ml, params.range_band_price_vs_long, params.range_band_spread_vs_long)
    ier = intraday_extreme_ratio(o, h, l, params.intraday_extreme_delta)

    row: dict[str, object] = {
        "symbol": meta.symbol,
        "name": meta.name,
        "asset_type": meta.asset_type,
        "effective_days": eff,
        "history_short": hist_short,
        "insufficient_data": False,
        "trend_t": trend_t,
        "trend_r2": trend_r2,
        "variance_ratio": vr,
        "acf1_ret": rho1,
        "rv_ann": rv,
        "vol_comfort_score": vcomf,
        "mean_abs_gap": mag,
        "gap_tail_ratio": gtr,
        "intraday_extreme_ratio": ier,
        "range_time_ratio": rtr,
    }
    row["vol_band"] = _vol_band(rv, params)
    row["grid_friendly_score"] = grid_friendly_score(row)
    return row


def _vol_band(rv: float, params: ScreenerParams) -> str:
    if not np.isfinite(rv):
        return "unknown"
    if rv < params.sigma_low:
        return "vol_low"
    if rv > params.sigma_high:
        return "vol_high"
    return "vol_mid"


def _empty_row(meta: UniverseItem, effective_days: int) -> dict[str, object]:
    nan = float("nan")
    return {
        "symbol": meta.symbol,
        "name": meta.name,
        "asset_type": meta.asset_type,
        "effective_days": effective_days,
        "history_short": False,
        "insufficient_data": True,
        "trend_t": nan,
        "trend_r2": nan,
        "variance_ratio": nan,
        "acf1_ret": nan,
        "rv_ann": nan,
        "vol_comfort_score": nan,
        "mean_abs_gap": nan,
        "gap_tail_ratio": nan,
        "intraday_extreme_ratio": nan,
        "range_time_ratio": nan,
        "vol_band": "unknown",
        "grid_friendly_score": nan,
    }
