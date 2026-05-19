from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class SortKey(BaseModel):
    field: str
    ascending: bool = False


class SortSpec(BaseModel):
    keys: list[SortKey] = Field(default_factory=list)

    @classmethod
    def from_config(cls, raw: list[dict[str, Any]] | None) -> SortSpec:
        if not raw:
            return cls()
        return cls(keys=[SortKey.model_validate(k) for k in raw])

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if not self.keys:
            if "symbol" in df.columns:
                return df.sort_values("symbol").reset_index(drop=True)
            return df.reset_index(drop=True)
        by = [k.field for k in self.keys]
        asc = [k.ascending for k in self.keys]
        return df.sort_values(by=by, ascending=asc, na_position="last").reset_index(drop=True)
