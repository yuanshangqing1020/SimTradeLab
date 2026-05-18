from __future__ import annotations

import pandas as pd


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without close; sort ascending by index."""
    need = ["open", "high", "low", "close", "volume"]
    for c in need:
        if c not in df.columns:
            raise KeyError("missing column: {0}".format(c))
    out = df.loc[df["close"].notna(), need].copy()
    out.sort_index(inplace=True)
    return out


def slice_window(df: pd.DataFrame, w: int) -> pd.DataFrame:
    if w < 1:
        raise ValueError("w must be >= 1")
    return df.iloc[-w:].copy()
