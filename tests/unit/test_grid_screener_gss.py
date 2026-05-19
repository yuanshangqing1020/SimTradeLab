import numpy as np
import pandas as pd

from simtradelab.grid_screener.config import RunConfig, UniverseItem
from simtradelab.grid_screener.engine import compute_row
from simtradelab.grid_screener.metrics import adx_last, atr_ratio, hurst_exponent, price_percentile


def _ohlcv(n: int = 260, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1.0 + rng.normal(0.0, 0.012, size=n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.02, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.02, n))
    vol = rng.uniform(5e6, 2e7, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _gss_cfg() -> RunConfig:
    return RunConfig.model_validate(
        {
            "preset": "grid_suitability_v1",
            "params": {"window_trading_days": 252, "n_min_valid": 120},
        }
    )


def test_gss_preset_resolves():
    cfg = _gss_cfg()
    assert "gss_score" in cfg.factors
    assert cfg.params.gss.adtv_min_yuan == 1e8


def test_gss_full_row_keys():
    df = _ohlcv()
    meta = UniverseItem(symbol="510300.SH", name="沪深300ETF", asset_type="etf")
    row = compute_row(df, meta, _gss_cfg())
    for k in (
        "hurst",
        "adx",
        "hv_ann",
        "atr_ratio",
        "adtv_yuan",
        "liquidity_ok",
        "gss_score",
        "gss_veto",
    ):
        assert k in row
    assert row["gss_veto"] in (True, False)


def test_gss_st_veto():
    df = _ohlcv()
    meta = UniverseItem(symbol="600001.SH", name="*ST测试", asset_type="stock")
    row = compute_row(df, meta, _gss_cfg())
    assert row.get("st_flag") is True
    assert row.get("gss_veto") is True
    assert row.get("gss_score") == 0.0


def test_metrics_atr_adx_finite():
    df = _ohlcv(300)
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    assert np.isfinite(atr_ratio(h, l, c))
    assert np.isfinite(adx_last(h, l, c))
    assert np.isfinite(hurst_exponent(c))
    assert 0.0 <= price_percentile(c) <= 1.0
