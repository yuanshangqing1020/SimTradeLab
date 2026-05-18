from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_ohlcv_csv(path: str | Path, date_col: str = "date") -> pd.DataFrame:
    """Expect columns: date, open, high, low, close, volume (case-insensitive)."""
    p = Path(path)
    df = pd.read_csv(p)
    lower = {c.lower(): c for c in df.columns}
    lower_keys = set(lower.keys())
    for need in ("open", "high", "low", "close", "volume"):
        if need not in lower_keys:
            raise KeyError("missing column: {0}".format(need))
    dc = lower.get(date_col.lower())
    if dc is None:
        dc = lower.get("datetime") or lower.get("time")
    if dc is None:
        raise KeyError("need date column: date/datetime/time")
    rename = {
        lower["open"]: "open",
        lower["high"]: "high",
        lower["low"]: "low",
        lower["close"]: "close",
        lower["volume"]: "volume",
    }
    out = df.rename(columns=rename).copy()
    out.index = pd.to_datetime(df[dc])
    return out[["open", "high", "low", "close", "volume"]]
