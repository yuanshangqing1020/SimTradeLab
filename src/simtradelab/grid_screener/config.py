from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def _default_data_path() -> str | None:
    """与 `BacktestConfig` 一致：默认 None，解析时走 `get_data_path()`。"""
    return None


class ScreenerParams(BaseModel):
    """Defaults aligned with `01-design.md` §–§4."""

    window_trading_days: int = Field(default=1250, ge=50)
    n_min_valid: int = Field(default=500, ge=10)
    sigma_low: float = Field(default=0.10, gt=0)
    sigma_high: float = Field(default=0.40, gt=0)
    gap_tail_delta: float = Field(default=0.01, gt=0)
    range_ma_long: int = Field(default=60, ge=5)
    range_ma_short: int = Field(default=20, ge=2)
    range_band_price_vs_long: float = Field(default=0.05, gt=0)
    range_band_spread_vs_long: float = Field(default=0.03, gt=0)
    intraday_extreme_delta: float = Field(default=0.02, gt=0)
    enable_composite: bool = False


class UniverseItem(BaseModel):
    symbol: str
    name: str = ""
    asset_type: Literal["stock", "etf"]


class RunConfig(BaseModel):
    """行情来源：

    - **默认**：不写 ``ohlcv_glob`` 时，与回测相同使用 ``data_path`` + ``market``，
      经 ``DataServer`` 同款路径解析后，用 ``storage.list_stocks`` / ``storage.load_stock`` 遍历 ``stocks/*.parquet``。
    - **演示 / 外部 CSV**：设置 ``ohlcv_glob`` 时走 CSV（需 ``universe`` 或 ``discover_glob``）。
    """

    model_config = {"extra": "ignore"}

    as_of: str | None = None
    params: ScreenerParams = Field(default_factory=ScreenerParams)
    universe: list[UniverseItem] = Field(default_factory=list)
    # 与 BacktestConfig.data_path 一致；JSON 可省略，表示使用项目 data/（或 SIMTRADELAB_DATA_PATH）
    data_path: str | None = Field(default_factory=_default_data_path)
    market: str = Field(default="CN", description="与 BacktestConfig.market 一致")
    ohlcv_glob: str | None = None
    discover_glob: bool = False
    default_asset_type: Literal["stock", "etf"] = "stock"
    etf_symbols_path: str | None = None
    output_csv: str = "grid_screener_report.csv"
    composite_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _csv_requires_universe_or_discover(self) -> RunConfig:
        if self.ohlcv_glob is not None and str(self.ohlcv_glob).strip() != "":
            if not self.universe and not self.discover_glob:
                raise ValueError("使用 ohlcv_glob 时请在 universe 中列出标的，或设置 discover_glob=true")
        return self


def load_run_config(path: str | Path) -> RunConfig:
    p = Path(path)
    return RunConfig.model_validate_json(p.read_text(encoding="utf-8"))


def load_etf_symbol_set(path: str | None) -> set[str]:
    if path is None or str(path).strip() == "":
        return set()
    p = Path(path)
    if not p.is_file():
        return set()
    lines = p.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")}
