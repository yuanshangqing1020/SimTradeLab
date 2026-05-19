from __future__ import annotations

from typing import Any


def explain_gss_default(row: dict[str, Any]) -> list[str]:
    if row.get("insufficient_data"):
        return ["有效交易日不足：GSS 分项未计算。"]
    if row.get("gss_veto"):
        reason = str(row.get("gss_veto_reason") or "veto")
        mapping = {
            "low_liquidity": "日均成交额低于阈值：流动性一票否决。",
            "st": "ST 或退市风险标的：安全性一票否决。",
            "price_percentile_high": "价格处于窗口内高位分位：警惕高位网格接盘。",
            "insufficient_data": "样本不足。",
        }
        parts = [mapping.get(p, p) for p in reason.split("|") if p]
        return parts if parts else ["触发 GSS 一票否决。"]

    out: list[str] = []
    hurst = float(row.get("hurst") or 0.5)
    if hurst < 0.45:
        out.append("赫斯特指数偏低：价格序列偏均值回归，利于网格。")
    elif hurst > 0.55:
        out.append("赫斯特指数偏高：趋势持续性较强，网格易踏空或套牢。")

    adx = float(row.get("adx") or 0.0)
    if adx > 30:
        out.append("ADX 偏高：趋势市特征明显，震荡网格假设偏弱。")

    hv = float(row.get("hv_ann") or 0.0)
    if hv < 0.15:
        out.append("年化波动偏低：网格套利空间可能不足。")
    elif hv > 0.45:
        out.append("年化波动偏高：注意跳空与执行风险。")

    pct = float(row.get("price_percentile") or 0.5)
    if pct > 0.65:
        out.append("价格接近窗口高位：单边下跌时网格补仓风险上升。")

    if not out:
        out.append("未触发强提示：请结合 GSS 分项与资金规模人工复核。")
    return out
