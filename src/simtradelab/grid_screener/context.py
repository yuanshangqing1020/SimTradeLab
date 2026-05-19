from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from simtradelab.grid_screener.config import ScreenerParams, UniverseItem


@dataclass
class FactorContext:
    """单标的筛选上下文：窗口化行情 + 累积输出。"""

    meta: UniverseItem
    params: ScreenerParams
    window: pd.DataFrame
    outputs: dict[str, object] = field(default_factory=dict)

    @property
    def insufficient(self) -> bool:
        return bool(self.outputs.get("insufficient_data"))

    @property
    def close(self) -> np.ndarray:
        return self.window["close"].to_numpy(dtype=float)

    @property
    def open(self) -> np.ndarray:
        return self.window["open"].to_numpy(dtype=float)

    @property
    def high(self) -> np.ndarray:
        return self.window["high"].to_numpy(dtype=float)

    @property
    def low(self) -> np.ndarray:
        return self.window["low"].to_numpy(dtype=float)

    @property
    def log_close(self) -> np.ndarray:
        c = self.close
        return np.log(c)

    @property
    def r1(self) -> np.ndarray:
        return np.diff(self.log_close)
