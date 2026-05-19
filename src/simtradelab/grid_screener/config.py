from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from simtradelab.grid_screener.sort_spec import SortKey, SortSpec


def _default_data_path() -> str | None:
    return None


class ScreenerParams(BaseModel):
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


_PRESETS_DIR = Path(__file__).resolve().parent / "presets"


def _load_preset(name: str) -> dict[str, Any]:
    path = _PRESETS_DIR / "{0}.json".format(name)
    if not path.is_file():
        raise ValueError("unknown preset: {0!r} (expected {1})".format(name, path))
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_preset(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """浅合并：override 中非 None 的顶层键覆盖 preset。"""
    out = dict(base)
    for k, v in override.items():
        if k == "preset":
            continue
        if v is not None:
            out[k] = v
    if "params" in base and "params" in override and isinstance(override["params"], dict):
        merged_params = dict(base.get("params") or {})
        merged_params.update(override["params"])
        out["params"] = merged_params
    return out


class RunConfig(BaseModel):
    """v2 筛选运行配置。见 ``my_docs/grid_friendly_screener/01-design.md``。"""

    model_config = {"extra": "ignore"}

    preset: str | None = None
    as_of: str | None = None
    params: ScreenerParams = Field(default_factory=ScreenerParams)
    universe: list[UniverseItem] = Field(default_factory=list)
    data_path: str | None = Field(default_factory=_default_data_path)
    market: str = Field(default="CN")
    fq: Literal["pre", "post", "none"] | None = Field(
        default="pre",
        description="pre/post 复权；none 为未复权原始价",
    )
    factors: list[str] = Field(default_factory=list)
    sort: list[SortKey] = Field(default_factory=list)
    explain: str | None = Field(default="grid_default")
    ohlcv_glob: str | None = None
    discover_glob: bool = False
    default_asset_type: Literal["stock", "etf"] = "stock"
    etf_symbols_path: str | None = None
    output_csv: str = "grid_screener_report.csv"

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        preset_name = data.get("preset")
        if not preset_name:
            return data
        base = _load_preset(str(preset_name))
        return _merge_preset(base, data)

    @model_validator(mode="after")
    def _csv_requires_universe_or_discover(self) -> RunConfig:
        if self.ohlcv_glob is not None and str(self.ohlcv_glob).strip() != "":
            if not self.universe and not self.discover_glob:
                raise ValueError("使用 ohlcv_glob 时请在 universe 中列出标的，或设置 discover_glob=true")
        return self

    @model_validator(mode="after")
    def _default_factors_if_empty(self) -> RunConfig:
        if self.factors:
            return self
        return self.model_copy(
            update={
                "factors": [
                    "meta",
                    "sample_quality",
                    "trend",
                    "variance_ratio",
                    "acf1",
                    "volatility",
                    "gap",
                    "range_regime",
                ]
            }
        )

    def resolved_factors(self) -> list[str]:
        names = list(self.factors)
        if self.params.enable_composite and "grid_score" not in names:
            names.append("grid_score")
        return names

    def sort_spec(self) -> SortSpec:
        return SortSpec(keys=list(self.sort))

    def resolved_fq(self) -> str | None:
        if self.fq is None or self.fq == "none":
            return None
        return self.fq


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
