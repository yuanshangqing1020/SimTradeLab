from __future__ import annotations

import os
from typing import Literal

import pandas as pd

from simtradelab.grid_screener.data_path import resolve_stock_data_root
from simtradelab.grid_screener.preprocess import normalize_ohlcv
from simtradelab.ptrade import storage
from simtradelab.ptrade.adj_cache import (
    _adj_cache_path,
    _calculate_adj_factors_from_events,
    _parquet_to_adj_cache,
)

FqType = Literal["pre", "post"] | None

_PRICE_COLS = ("open", "high", "low", "close")


def apply_adj_factors(
    stock_df: pd.DataFrame,
    adj_factors: pd.DataFrame,
) -> pd.DataFrame:
    """前/后复权：adj_a * 价 + adj_b（与 PtradeAPI._apply_adj_factors 一致）。"""
    common_idx = stock_df.index.intersection(adj_factors.index)
    if len(common_idx) == 0:
        return stock_df

    adjusted_df = stock_df.copy()
    adj_a = adj_factors.loc[common_idx, "adj_a"]
    adj_b = adj_factors.loc[common_idx, "adj_b"]
    for col in _PRICE_COLS:
        if col in adjusted_df.columns:
            adjusted_df.loc[common_idx, col] = adj_a * adjusted_df.loc[common_idx, col] + adj_b
    return adjusted_df


class MarketDataSession:
    """与回测同源 Parquet + 复权因子缓存。"""

    def __init__(
        self,
        data_path: str | None,
        market: str,
        fq: FqType = "pre",
    ) -> None:
        self.data_root = resolve_stock_data_root(data_path, market)
        self.fq: FqType = fq if fq in ("pre", "post", None) else None
        self._adj_pre: dict[str, pd.DataFrame] | None = None
        self._adj_post: dict[str, pd.DataFrame] | None = None

    def list_symbols(self) -> list[str]:
        return sorted(storage.list_stocks(self.data_root))

    def _load_adj_cache(self, kind: str) -> dict[str, pd.DataFrame]:
        path = _adj_cache_path(self.data_root, kind)
        if os.path.exists(path):
            cached = _parquet_to_adj_cache(path)
            if cached:
                return cached
        return {}

    @property
    def adj_pre_cache(self) -> dict[str, pd.DataFrame]:
        if self._adj_pre is None:
            self._adj_pre = self._load_adj_cache("pre")
        return self._adj_pre

    @property
    def adj_post_cache(self) -> dict[str, pd.DataFrame]:
        if self._adj_post is None:
            self._adj_post = self._load_adj_cache("post")
        return self._adj_post

    def _adj_factors_for_symbol(self, symbol: str, stock_df: pd.DataFrame) -> pd.DataFrame | None:
        if self.fq == "pre":
            cache = self.adj_pre_cache
            if symbol in cache:
                return cache[symbol]
        elif self.fq == "post":
            cache = self.adj_post_cache
            if symbol in cache:
                return cache[symbol]
        else:
            return None

        ex = storage.load_exrights(self.data_root, symbol)
        events = ex.get("exrights_events") if ex else None
        return _calculate_adj_factors_from_events(symbol, stock_df, events)

    def load_ohlcv(self, symbol: str, as_of: str | None = None) -> pd.DataFrame:
        raw = storage.load_stock(self.data_root, symbol)
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = normalize_ohlcv(raw)
        if self.fq in ("pre", "post"):
            adj = self._adj_factors_for_symbol(symbol, df)
            if adj is not None:
                df = apply_adj_factors(df, adj)

        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            df = df.loc[df.index <= cutoff]
        return df
