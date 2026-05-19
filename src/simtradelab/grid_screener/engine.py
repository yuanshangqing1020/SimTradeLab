from __future__ import annotations

import pandas as pd

from simtradelab.grid_screener.config import RunConfig, UniverseItem
from simtradelab.grid_screener.context import FactorContext
from simtradelab.grid_screener.factors.registry import FactorRegistry, default_registry
from simtradelab.grid_screener.market_data import MarketDataSession
from simtradelab.grid_screener.preprocess import normalize_ohlcv, slice_window
from simtradelab.grid_screener.io_csv import read_ohlcv_csv


def build_factor_context(
    raw: pd.DataFrame,
    meta: UniverseItem,
    params,
) -> FactorContext:
    df0 = normalize_ohlcv(raw)
    if df0.empty:
        win = df0
    else:
        win_len = min(params.window_trading_days, len(df0))
        win = slice_window(df0, win_len)
    return FactorContext(meta=meta, params=params, window=win)


def compute_row(
    raw: pd.DataFrame,
    meta: UniverseItem,
    cfg: RunConfig,
    registry: FactorRegistry | None = None,
) -> dict[str, object]:
    reg = registry or default_registry()
    ctx = build_factor_context(raw, meta, cfg.params)
    row: dict[str, object] = {}
    for factor in reg.resolve(cfg.resolved_factors()):
        chunk = factor.compute(ctx)
        row.update(chunk)
        ctx.outputs.update(chunk)
    return row


def run_symbol_parquet(
    session: MarketDataSession,
    meta: UniverseItem,
    cfg: RunConfig,
    registry: FactorRegistry | None = None,
) -> dict[str, object]:
    raw = session.load_ohlcv(meta.symbol, as_of=cfg.as_of)
    return compute_row(raw, meta, cfg, registry)


def run_symbol_csv(path: str, meta: UniverseItem, cfg: RunConfig, registry: FactorRegistry | None = None) -> dict[str, object]:
    raw = read_ohlcv_csv(path)
    if cfg.as_of is not None:
        cutoff = pd.Timestamp(cfg.as_of)
        raw = raw.loc[raw.index <= cutoff]
    return compute_row(raw, meta, cfg, registry)
