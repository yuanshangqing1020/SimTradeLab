from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# CSV 导出：写入文件时限定小数位数（避免 to_csv 仍打印过长浮点表示）
EXPORT_FLOAT_DECIMALS = 4

_STRING_EXPORT_COLUMNS = frozenset({"symbol", "name", "asset_type", "explanations", "vol_band"})


def format_export_table(df: pd.DataFrame, float_decimals: int = EXPORT_FLOAT_DECIMALS) -> pd.DataFrame:
    """供 CSV 使用：缩短浮点小数位；字符串/布尔/整数列不动。"""
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if col in _STRING_EXPORT_COLUMNS:
            continue
        s = out[col]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_integer_dtype(s):
            continue
        if pd.api.types.is_float_dtype(s):
            out[col] = s.round(float_decimals)
            continue
        if s.dtype == object:
            continue
        if pd.api.types.is_numeric_dtype(s):
            out[col] = pd.to_numeric(s, errors="coerce").round(float_decimals)
    return out


def rows_to_sorted_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "grid_friendly_score" in df.columns:
        df = df.sort_values(
            by=["grid_friendly_score", "range_time_ratio"],
            ascending=[False, False],
            na_position="last",
        )
    elif "range_time_ratio" in df.columns and "trend_t" in df.columns:
        df = df.sort_values(by=["range_time_ratio", "trend_t"], ascending=[False, True], na_position="last")
    return df.reset_index(drop=True)


def write_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    float_decimals: int = EXPORT_FLOAT_DECIMALS,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%.{0}f".format(max(0, int(float_decimals)))
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format=fmt)
