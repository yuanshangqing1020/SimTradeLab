import numpy as np
import pandas as pd

from simtradelab.grid_screener.config import ScreenerParams, UniverseItem
from simtradelab.grid_screener.explain import explain_row
from simtradelab.grid_screener.pipeline import compute_screener_row
from simtradelab.grid_screener.report import rows_to_sorted_frame


def _synth(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    vol = np.full(n, 1e6)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_compute_screener_row_has_expected_keys():
    df = _synth(600)
    meta = UniverseItem(symbol="000001.SZ", name="Ping An Bank", asset_type="stock")
    params = ScreenerParams(window_trading_days=500, n_min_valid=200)
    row = compute_screener_row(df, meta, params)
    for k in (
        "symbol",
        "name",
        "asset_type",
        "effective_days",
        "history_short",
        "insufficient_data",
        "trend_t",
        "trend_r2",
        "variance_ratio",
        "acf1_ret",
        "rv_ann",
        "vol_comfort_score",
        "mean_abs_gap",
        "gap_tail_ratio",
        "intraday_extreme_ratio",
        "range_time_ratio",
        "grid_friendly_score",
    ):
        assert k in row
    assert row["insufficient_data"] in (True, False)
    assert row["effective_days"] <= 500


def test_explain_emits_zh_strings():
    row = {
        "trend_t": 5.0,
        "variance_ratio": 1.3,
        "acf1_ret": 0.2,
        "vol_band": "vol_low",
        "gap_tail_ratio": 0.4,
        "range_time_ratio": 0.8,
        "insufficient_data": False,
    }
    lines = explain_row(row)
    assert isinstance(lines, list) and len(lines) >= 1
    assert all(isinstance(s, str) for s in lines)


def test_rows_to_sorted_frame_sorts():
    rows = [
        {"symbol": "A", "range_time_ratio": 0.1, "trend_t": 0.5},
        {"symbol": "B", "range_time_ratio": 0.9, "trend_t": 0.1},
    ]
    df = rows_to_sorted_frame(rows)
    assert df.iloc[0]["symbol"] == "B"


def test_write_csv_chunked_splits(tmp_path):
    from simtradelab.grid_screener.report import write_csv_chunked

    df = pd.DataFrame({"a": range(250)})
    out = tmp_path / "rep.csv"
    paths = write_csv_chunked(df, out, chunk_rows=100)
    assert len(paths) == 3
    assert (tmp_path / "rep_part0001.csv").is_file()


def test_write_csv_chunked_single_file_when_small(tmp_path):
    from simtradelab.grid_screener.report import write_csv_chunked

    df = pd.DataFrame({"a": range(50)})
    out = tmp_path / "rep.csv"
    paths = write_csv_chunked(df, out, chunk_rows=500)
    assert len(paths) == 1
    assert paths[0] == str(out.resolve())
