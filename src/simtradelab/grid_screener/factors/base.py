from __future__ import annotations

from typing import Protocol, runtime_checkable

from simtradelab.grid_screener.context import FactorContext


@runtime_checkable
class Factor(Protocol):
    """筛选因子：向 ``ctx.outputs`` 贡献一列或多列。"""

    name: str

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        ...
