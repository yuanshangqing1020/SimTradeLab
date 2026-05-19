import numpy as np
import pandas as pd

from simtradelab.grid_screener.config import RunConfig, ScreenerParams, UniverseItem
from simtradelab.grid_screener.engine import compute_row
from simtradelab.grid_screener.explain import explain_row
from simtradelab.grid_screener.report import rows_to_sorted_frame
from simtradelab.grid_screener.sort_spec import SortKey, SortSpec


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


def _cfg(**kwargs) -> RunConfig:
    base = {
        "factors": [
            "meta",
            "sample_quality",
            "trend",
            "variance_ratio",
            "acf1",
            "volatility",
            "gap",
            "range_regime",
            "grid_score",
        ],
        "params": {"window_trading_days": 500, "n_min_valid": 200},
    }
    base.update(kwargs)
    return RunConfig.model_validate(base)


def test_compute_row_has_expected_keys():
    df = _synth(600)
    meta = UniverseItem(symbol="000001.SZ", name="Ping An Bank", asset_type="stock")
    row = compute_row(df, meta, _cfg())
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


def test_preset_grid_friendly_v1():
    cfg = RunConfig.model_validate({"preset": "grid_friendly_v1"})
    assert "grid_score" in cfg.factors
    assert cfg.sort_spec().keys[0].field == "range_time_ratio"


def test_sort_spec_multi_key():
    rows = [
        {"symbol": "A", "range_time_ratio": 0.1, "trend_t": 0.5},
        {"symbol": "B", "range_time_ratio": 0.9, "trend_t": 0.1},
    ]
    spec = SortSpec(keys=[SortKey(field="range_time_ratio", ascending=False)])
    df = rows_to_sorted_frame(rows, spec)
    assert df.iloc[0]["symbol"] == "B"


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


def test_format_explanations_for_export_normalizes_punct():
    from simtradelab.grid_screener.explain import format_explanations_for_export

    s = format_explanations_for_export(
        [
            "大跳空占比偏高：隔夜跳开可能放大滑点与挂单风险。",
            "区间震荡时间占比较高：与网格友好方向更一致（仍需结合趋势项）。",
        ]
    )
    assert "\uff1a" not in s
    assert "\u3002" not in s
    assert " | " in s
    assert "(" in s and ")" in s


def test_lookup_stock_name_aliases():
    from simtradelab.grid_screener.data_path import lookup_stock_name

    m = {"502011.SH": "测试LOF"}
    assert lookup_stock_name(m, "502011.SS") == "测试LOF"


def test_format_export_table_rounds_floats():
    from simtradelab.grid_screener.report import format_export_table

    df = pd.DataFrame(
        {
            "symbol": ["X"],
            "trend_t": [1.6044569443876195],
            "effective_days": [1250],
            "history_short": [False],
        }
    )
    got = format_export_table(df, float_decimals=4)
    assert got["trend_t"].iloc[0] == 1.6045
    assert got["effective_days"].iloc[0] == 1250


def test_write_csv_limits_float_width(tmp_path):
    from simtradelab.grid_screener.report import format_export_table, write_csv

    df = format_export_table(pd.DataFrame({"x": [1.23456789], "y": [9]}))
    out = tmp_path / "rep.csv"
    write_csv(df, out)
    text = out.read_text(encoding="utf-8-sig")
    assert "1.2346" in text
    assert "1.23456789" not in text


def test_write_csv_strips_null_bytes_for_excel(tmp_path):
    from simtradelab.grid_screener.report import write_csv

    df = pd.DataFrame({"symbol": ["159001.SZ"], "name": ["保证金\x00\x00"], "x": [1.0]})
    out = tmp_path / "rep.csv"
    write_csv(df, out)
    raw = out.read_bytes()
    assert b"\x00" not in raw
    assert "保证金" in raw.decode("utf-8-sig")


def test_factor_registry_unknown_raises():
    from simtradelab.grid_screener.factors.registry import default_registry

    with __import__("pytest").raises(KeyError):
        default_registry().get("not_a_factor")
