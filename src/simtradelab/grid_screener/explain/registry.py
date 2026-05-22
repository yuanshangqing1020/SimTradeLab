from __future__ import annotations

from collections.abc import Callable
from typing import Any

from simtradelab.grid_screener.explain.grid_default import explain_grid_default
from simtradelab.grid_screener.explain.gtp_default import explain_gtp_default
from simtradelab.grid_screener.explain.gss_default import explain_gss_default

ExplainFn = Callable[[dict[str, Any]], list[str]]

_REGISTRY: dict[str, ExplainFn] = {
    "grid_default": explain_grid_default,
    "gss_default": explain_gss_default,
    "gtp_default": explain_gtp_default,
}


def get_explain(name: str | None) -> ExplainFn | None:
    if name is None or str(name).strip() == "":
        return None
    key = str(name).strip()
    if key not in _REGISTRY:
        raise KeyError("unknown explain ruleset: {0!r}".format(key))
    return _REGISTRY[key]
