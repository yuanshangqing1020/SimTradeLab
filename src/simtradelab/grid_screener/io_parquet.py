from __future__ import annotations

import pandas as pd


def ohlcv_from_stock_parquet_df(df: pd.DataFrame) -> pd.DataFrame:
    """将 `storage.load_stock` 返回的 DataFrame 规范为 grid_screener 所需列。"""
    if df.empty:
        return df
    need_price = ("open", "high", "low", "close")
    for c in need_price:
        if c not in df.columns:
            raise KeyError("missing column: {0}".format(c))
    out = df.loc[:, list(need_price)].copy()
    if "volume" in df.columns:
        out["volume"] = df["volume"]
    else:
        out["volume"] = 0.0
    return out[["open", "high", "low", "close", "volume"]]
