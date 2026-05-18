# tests/unit/test_grid_screener_metrics.py
import math

import numpy as np
import pandas as pd

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


def test_normalize_ohlcv_drops_nan_close_and_sorts():
    df = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [np.nan, 10.0, 11.0],
            "volume": [100, 100, 100],
        },
        index=pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
    )
    got = normalize_ohlcv(df)
    assert list(got.index.date) == [pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-01-02").date()]
    assert got["close"].tolist() == [10.0, 11.0]


def test_slice_window_truncates_last_w_rows():
    df = pd.DataFrame(
        {"close": np.arange(10.0, 20.0)},
        index=pd.date_range("2020-01-01", periods=10, freq="B"),
    )
    got = slice_window(df, 5)
    assert len(got) == 5
    assert got["close"].iloc[-1] == 19.0


def test_ols_flat_series_near_zero_t():
    lc = np.log(np.ones(30))
    t_stat, r2 = ols_log_close_trend(lc)
    assert abs(t_stat) < 0.5
    assert r2 < 1e-6


def test_variance_ratio_random_walk_near_one():
    rng = np.random.default_rng(0)
    steps = rng.normal(0.0, 0.01, size=500)
    lc = np.cumsum(steps)
    vr = variance_ratio_lm(lc, q=2)
    assert 0.85 <= vr <= 1.15


def test_acf1_alternating_negative():
    r = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01], dtype=float)
    rho = acf1_returns(r)
    assert rho < -0.3


def test_realized_vol_positive():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.015, size=200)
    rv = realized_vol_ann(r)
    assert rv > 0.1
    assert math.isfinite(rv)


def test_mean_abs_gap_positive():
    o = np.array([10.0, 10.5, 10.2], dtype=float)
    c = np.array([10.0, 10.0, 10.4], dtype=float)
    g = mean_abs_overnight_gap(o, c)
    assert g > 0


def test_vol_comfort_mid_band_one():
    assert vol_comfort_score(0.20, sigma_low=0.10, sigma_high=0.40) == 1.0


def test_range_time_ratio_normalizes():
    close = np.linspace(100.0, 100.01, 80)
    ma_long = pd.Series(close).rolling(60, min_periods=60).mean().to_numpy()
    ma_short = pd.Series(close).rolling(20, min_periods=20).mean().to_numpy()
    r = range_time_ratio(close, ma_short, ma_long, b=0.10, b2=0.10)
    assert 0.0 <= r <= 1.0


def test_gap_tail_ratio_bounded():
    o = np.array([10.0, 10.1, 10.2], dtype=float)
    c = np.array([10.0, 10.05, 10.15], dtype=float)
    gr = gap_tail_ratio(o, c, 0.5)
    assert 0.0 <= gr <= 1.0


def test_intraday_extreme_ratio_bounded():
    o = np.ones(20, dtype=float)
    h = np.ones(20) * 1.05
    l = np.ones(20) * 0.95
    r = intraday_extreme_ratio(o, h, l, 0.02)
    assert 0.0 <= r <= 1.0
