"""与回测 DataServer 一致的数据根目录解析（含 cn 子目录、旧版 data 扁平迁移）。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from simtradelab.ptrade.market_profile import get_market_profile
from simtradelab.service.data_server import _migrate_legacy_data


def _symbol_name_lookup_keys(symbol: str) -> list[str]:
    """为 metadata 与 parquet 代码格式不一致时准备候选键（如 .SH / .SS）。"""
    symbol = str(symbol).strip()
    if not symbol:
        return []
    keys: list[str] = [symbol]
    if "." in symbol:
        base, suf = symbol.split(".", 1)
        suf_u = suf.upper()
        keys.append(base)
        if suf_u == "SS":
            keys.extend((base + ".SH", base + ".XSHG"))
        elif suf_u == "SH":
            keys.extend((base + ".SS", base + ".XSHG"))
        elif suf_u == "SZ":
            keys.append(base + ".XSHE")
        elif suf_u == "XSHE":
            keys.append(base + ".SZ")
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def lookup_stock_name(name_map: dict[str, str], symbol: str) -> str:
    """按精确代码与常见别名从 ``load_stock_name_map`` 结果中取名称；无则返回空串。"""
    for k in _symbol_name_lookup_keys(symbol):
        if k in name_map:
            return name_map[k]
    return ""


def resolve_stock_data_root(data_path: str | None, market: str) -> str:
    """解析出含 `stocks/*.parquet` 的目录，逻辑对齐 `DataServer.__init__`。

    Args:
        data_path: 与 `BacktestConfig.data_path` 相同含义的根（一般为项目 `data/`）；
                   为 None 时使用 `utils.paths.get_data_path()`（或环境变量）。
        market: 与回测一致，如 ``CN``。
    """
    from simtradelab.utils.paths import get_data_path

    profile = get_market_profile(market)
    base_path = Path(data_path).resolve() if data_path else get_data_path()
    _migrate_legacy_data(base_path)
    candidate = base_path / profile.data_dir_name
    resolved = candidate if candidate.exists() else base_path
    return str(resolved)


def load_stock_name_map(data_root: str) -> dict[str, str]:
    """与 DataServer 相同来源：`metadata/stock_metadata`。"""
    from simtradelab.ptrade import storage

    stock_metadata_data = storage.load_metadata(data_root, "stock_metadata")
    if not stock_metadata_data or "data" not in stock_metadata_data:
        return {}
    df = pd.DataFrame(stock_metadata_data["data"])
    if df.empty or "symbol" not in df.columns or "stock_name" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for i in range(len(df)):
        sym = str(df["symbol"].iloc[i]).strip()
        if not sym:
            continue
        raw = df["stock_name"].iloc[i]
        if pd.isna(raw):
            continue
        out[sym] = str(raw).strip()
    return out
