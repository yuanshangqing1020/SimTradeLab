from __future__ import annotations

from typing import Any


def explain_gtp_default(row: dict[str, Any]) -> list[str]:
    if row.get("insufficient_data"):
        return ["有效交易日不足：网格做T回测未运行。"]
    if row.get("grid_t_veto"):
        reason = str(row.get("grid_t_veto_reason") or "veto")
        mapping = {
            "bad_price_bars": "无效或异常 OHLC（如前复权负价）过多：结果已否决，避免虚增做T利润。",
            "insufficient_valid_bars": "剔除异常 K 线后有效样本不足。",
            "init_failed": "未能以合规价格完成初始建仓。",
        }
        return [mapping.get(reason, "网格做T回测被否决：{0}".format(reason))]

    profit = float(row.get("grid_t_profit_yuan") or 0.0)
    if not profit or profit != profit:
        return ["未能完成初始建仓或样本不足：无做T落袋利润。"]

    out: list[str] = []
    rate = float(row.get("grid_t_profit_rate") or 0.0)
    harvest = int(row.get("grid_t_harvest_count") or 0)
    per_250d = float(row.get("grid_t_harvest_per_250d") or 0.0)
    active = int(row.get("grid_t_active_days") or 0)
    bad = int(row.get("grid_t_bad_bar_count") or 0)

    out.append(
        "窗口内累计落袋现金约 {0:.0f} 元（相对底仓 {1:.1%}），有效回测 {2} 个交易日。".format(
            profit, rate, active
        )
    )
    out.append("基于未复权日线收盘价；与聚宽分钟级回测的绝对值会有差距，宜作横向排序参考。")

    if bad > 0:
        out.append("已剔除 {0} 根异常 K 线（非正价或 high<low）。".format(bad))

    if per_250d >= 80:
        out.append("年化收割频次高：震荡充分，网格做T空间较大。")
    elif per_250d >= 30:
        out.append("年化收割频次中等：具备一定网格做T空间。")
    elif harvest > 0:
        out.append("年化收割频次偏低：波动或振幅可能不足以支撑密集网格。")
    else:
        out.append("窗口内未触发有效卖网收割：不适合当前步长的网格做T。")

    if rate >= 1.0:
        out.append("落袋利润已超过初始底仓金额：请结合流动性人工复核。")

    return out
