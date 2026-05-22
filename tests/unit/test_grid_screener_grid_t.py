import numpy as np
import pandas as pd

from simtradelab.grid_screener.config import GridTParams, RunConfig, UniverseItem
from simtradelab.grid_screener.engine import compute_row
from simtradelab.grid_screener.grid_t_simulator import simulate_grid_t_profit


def _ohlc_from_close(closes: list[float], idx: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    if idx is None:
        idx = pd.date_range("2020-01-01", periods=len(c), freq="B")
    o = np.roll(c, 1)
    o[0] = c[0]
    spread = np.maximum(c * 0.005, 0.01)
    return pd.DataFrame(
        {
            "open": o,
            "high": c + spread,
            "low": c - spread,
            "close": c,
            "volume": np.full(len(c), 1e7),
        },
        index=idx,
    )


def _gtp_cfg() -> RunConfig:
    return RunConfig.model_validate(
        {
            "preset": "grid_t_profit_v1",
            "params": {"window_trading_days": 252, "n_min_valid": 30},
        }
    )


def test_grid_t_preset_resolves():
    cfg = _gtp_cfg()
    assert "grid_t_profit" in cfg.factors
    assert cfg.params.grid_t.grid_step == 0.03


def test_grid_t_oscillation_produces_harvests():
    """价格围绕 30 元在 ±3% 间来回，应多次触发卖网落袋。"""
    closes = [30.0]
    for _ in range(40):
        closes.append(closes[-1] * 1.03)
        closes.append(closes[-1] / 1.03)
    df = _ohlc_from_close(closes)
    params = GridTParams(
        initial_amount=100_000,
        grid_step=0.03,
        trade_amount=10_000,
        use_intraday_path=True,
    )
    result = simulate_grid_t_profit(df, params, is_etf=False)
    assert result.grid_t_harvest_count >= 5
    assert result.grid_t_profit_yuan > 0
    assert result.grid_t_profit_rate > 0


def test_grid_t_monotone_rally_still_non_negative_profit():
    """单边上涨仍会逢高卖出，落袋利润不应为负。"""
    closes = [10.0 * (1.03**i) for i in range(60)]
    df = _ohlc_from_close(closes)
    params = GridTParams(initial_amount=100_000, grid_step=0.03, trade_amount=10_000)
    result = simulate_grid_t_profit(df, params, is_etf=False)
    assert result.grid_t_profit_yuan >= 0
    assert result.grid_t_harvest_count >= 1


def test_grid_t_etf_no_stamp_tax():
    closes = [50.0]
    for _ in range(20):
        closes.append(closes[-1] * 1.04)
        closes.append(closes[-1] / 1.04)
    df = _ohlc_from_close(closes)
    params = GridTParams(initial_amount=100_000, grid_step=0.03, trade_amount=10_000)
    stock = simulate_grid_t_profit(df, params, is_etf=False)
    etf = simulate_grid_t_profit(df, params, is_etf=True)
    assert etf.grid_t_profit_yuan >= stock.grid_t_profit_yuan


def test_grid_t_factor_in_pipeline():
    closes = [30.0]
    for _ in range(30):
        closes.append(closes[-1] * 1.03)
        closes.append(closes[-1] / 1.03)
    df = _ohlc_from_close(closes)
    meta = UniverseItem(symbol="002430.SZ", name="杭氧股份", asset_type="stock")
    row = compute_row(df, meta, _gtp_cfg())
    for k in (
        "grid_t_profit_yuan",
        "grid_t_profit_rate",
        "grid_t_harvest_count",
        "grid_t_active_days",
    ):
        assert k in row
    assert row["grid_t_harvest_count"] >= 1
    assert float(row["grid_t_profit_yuan"]) > 0


def test_grid_t_insufficient_data():
    df = _ohlc_from_close([100.0, 101.0, 102.0])
    meta = UniverseItem(symbol="600000.SH", name="测试", asset_type="stock")
    cfg = RunConfig.model_validate(
        {
            "preset": "grid_t_profit_v1",
            "params": {"window_trading_days": 252, "n_min_valid": 500},
        }
    )
    row = compute_row(df, meta, cfg)
    assert row.get("insufficient_data") is True
    assert np.isnan(float(row["grid_t_profit_yuan"]))


def test_grid_t_rejects_mostly_negative_prices():
    """前复权脏数据（大量负价）应否决，避免在极低假价建仓。"""
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    close = np.concatenate([np.full(100, -1.0), np.full(20, 5.0)])
    df = _ohlc_from_close(list(close), idx=idx)
    params = GridTParams(
        initial_amount=100_000,
        grid_step=0.03,
        trade_amount=10_000,
        max_bad_bar_ratio=0.05,
        min_valid_price=0.5,
    )
    result = simulate_grid_t_profit(df, params, is_etf=False, min_active_days=10)
    assert result.grid_t_veto is True
    assert result.grid_t_veto_reason == "bad_price_bars"
    assert not np.isfinite(result.grid_t_profit_yuan)


def test_grid_t_preset_uses_unadjusted_prices():
    cfg = _gtp_cfg()
    assert cfg.resolved_fq() is None
    assert cfg.params.grid_t.use_intraday_path is False
