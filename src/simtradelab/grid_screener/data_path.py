"""与回测 DataServer 一致的数据根目录解析（含 cn 子目录、旧版 data 扁平迁移）。"""

from __future__ import annotations

from pathlib import Path

from simtradelab.ptrade.market_profile import get_market_profile
from simtradelab.service.data_server import _migrate_legacy_data


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
    import pandas as pd

    from simtradelab.ptrade import storage

    stock_metadata_data = storage.load_metadata(data_root, "stock_metadata")
    if not stock_metadata_data or "data" not in stock_metadata_data:
        return {}
    df = pd.DataFrame(stock_metadata_data["data"])
    if df.empty or "symbol" not in df.columns or "stock_name" not in df.columns:
        return {}
    return dict(zip(df["symbol"].astype(str), df["stock_name"].astype(str)))
