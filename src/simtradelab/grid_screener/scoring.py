from __future__ import annotations

import math
from typing import Any


def grid_friendly_score(row: dict[str, Any]) -> float:
    """固定权重综合分：越高越偏「震荡、波动适中、跳空与强趋势惩罚」。不足样本为 nan。"""
    if row.get("insufficient_data"):
        return float("nan")
    tt = float(row.get("trend_t") or 0.0)
    rtr = float(row.get("range_time_ratio") or 0.0)
    vc = float(row.get("vol_comfort_score") or 0.0)
    vr = float(row.get("variance_ratio") or 1.0)
    gtr = float(row.get("gap_tail_ratio") or 0.0)

    if not all(map(math.isfinite, (tt, rtr, vc, vr, gtr))):
        return float("nan")

    s = 40.0 * rtr
    s += 25.0 * vc
    s -= min(30.0, 0.35 * abs(tt))
    if vr > 1.0:
        s -= (vr - 1.0) * 18.0
    s -= gtr * 22.0
    return float(s)
