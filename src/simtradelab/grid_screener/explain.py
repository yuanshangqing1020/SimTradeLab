from __future__ import annotations

from typing import Any


def explain_row(row: dict[str, Any]) -> list[str]:
    if row.get("insufficient_data"):
        return ["有效交易日不足：分项未计算或不可靠。"]
    out: list[str] = []
    tt = float(row.get("trend_t") or 0.0)
    if abs(tt) > 2.5:
        out.append("趋势检验较强：窗口内对数价格可能存在显著漂移，网格假设偏弱。")

    vr = float(row.get("variance_ratio") or 1.0)
    rho = float(row.get("acf1_ret") or 0.0)
    if vr > 1.05 and rho > 0.05:
        out.append("收益持续性偏高：方差比率与一阶自相关同向不利震荡网格。")

    band = str(row.get("vol_band") or "")
    if band == "vol_low":
        out.append("实现波动偏低：理论格子空间可能不足。")
    elif band == "vol_high":
        out.append("实现波动偏高：执行与假突破风险上升。")

    gtr = float(row.get("gap_tail_ratio") or 0.0)
    if gtr > 0.1:
        out.append("大跳空占比偏高：隔夜跳开可能放大滑点与挂单风险。")

    rtr = float(row.get("range_time_ratio") or 0.0)
    if rtr > 0.55 and abs(tt) < 2.0:
        out.append("区间震荡时间占比较高：与网格友好方向更一致（仍需结合趋势项）。")

    if not out:
        out.append("未触发强提示：请结合分项与基本面/流动性人工复核。")
    return out
