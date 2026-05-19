from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from simtradelab.grid_screener.sort_spec import SortSpec

EXPORT_FLOAT_DECIMALS = 4

_STRING_EXPORT_COLUMNS = frozenset({"symbol", "name", "asset_type", "explanations", "vol_band"})


def format_export_table(df: pd.DataFrame, float_decimals: int = EXPORT_FLOAT_DECIMALS) -> pd.DataFrame:
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


def rows_to_sorted_frame(rows: list[dict[str, Any]], sort_spec: SortSpec | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    spec = sort_spec or SortSpec()
    return spec.apply(df)


def write_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    float_decimals: int = EXPORT_FLOAT_DECIMALS,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%.{0}f".format(max(0, int(float_decimals)))
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format=fmt)
