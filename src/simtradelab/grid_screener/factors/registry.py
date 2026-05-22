from __future__ import annotations

from simtradelab.grid_screener.factors.base import Factor
from simtradelab.grid_screener.factors.builtin import BUILTIN_FACTORS
from simtradelab.grid_screener.factors.grid_t_profit import GRID_T_FACTORS
from simtradelab.grid_screener.factors.gss import GSS_FACTORS


class FactorRegistry:
    def __init__(self) -> None:
        self._by_name: dict[str, Factor] = {}

    def register(self, factor: Factor) -> None:
        self._by_name[factor.name] = factor

    def get(self, name: str) -> Factor:
        if name not in self._by_name:
            raise KeyError("unknown factor: {0!r}. Registered: {1}".format(name, sorted(self._by_name)))
        return self._by_name[name]

    def resolve(self, names: list[str]) -> list[Factor]:
        seen: set[str] = set()
        out: list[Factor] = []
        for n in names:
            if n in seen:
                continue
            seen.add(n)
            out.append(self.get(n))
        return out

    def names(self) -> list[str]:
        return sorted(self._by_name)


def default_registry() -> FactorRegistry:
    reg = FactorRegistry()
    for f in (*BUILTIN_FACTORS, *GSS_FACTORS, *GRID_T_FACTORS):
        reg.register(f)
    return reg
