#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽 grid_etf 回测日志分析器。

解析 grid_etf_v*.txt 中的交易日志，生成可视化报告（含资金利用率、分标的持仓市值）。

用法:
    python analyze_backtest_log.py --log grid_etf_v0.8.txt
    python analyze_backtest_log.py --log grid_etf_v0.8.txt --initial-cash 500000
    python analyze_backtest_log.py --log grid_etf_v0.8.txt --output out/

输出（--output 为目录时）:
    backtest_analysis.png  - 主报告（做T利润 / 资金利用率等）
    position_charts.png    - 各标的持仓市值（5 个子图折线，分开展示）
    summary_table.png      - 做T统计 + 末持仓市值
    position_table.png     - 分标的持仓市值专表
    position_summary.csv   - 各标的末持仓/日均/最高/最低市值
    position_daily.csv     - 各标的每日持仓市值（宽表）
    trades.csv             - 交易明细
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 日志解析
# ---------------------------------------------------------------------------

LOG_LINE_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"【(?P<action>[^】]+)】"
    r"(?P<name>[^|(]+)\((?P<code>[^)]+)\)\s*\|\s*"
    r"(?P<rest>.+)"
)
PROFIT_RE = re.compile(r"(?:做T净赚|释放利润):([\d.]+)")
LEVEL_RE = re.compile(r"(?:等级|剩余等级):(-?\d+)")
SHARES_RE = re.compile(r"(?:买入|卖出|减仓):(\d+)股")
PRICE_RE = re.compile(r"价格:([\d.]+)")
POSITION_RE = re.compile(r"总持仓:(\d+)股|调仓至:(\d+)股")

ACTION_COLORS = {
    "重装建仓": "#6366f1",
    "加仓": "#ef4444",
    "止盈": "#22c55e",
    "止盈（部分）": "#16a34a",
    "破顶": "#f59e0b",
    "清仓": "#94a3b8",
    "清仓-碎股": "#64748b",
    "等级校准": "#f97316",
    "底仓到账": "#06b6d4",
}


def _is_take_profit(action: str) -> bool:
    return action == "止盈" or action.startswith("止盈")


TICKER_LINE_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
]

# 与策略 set_order_cost 一致：ETF 万一免五
COMMISSION_RATE = 0.0001
MIN_COMMISSION = 0.1
DEFAULT_INITIAL_CASH = 500_000.0


@dataclass
class TradeRecord:
    timestamp: datetime
    action: str
    name: str
    code: str
    price: Optional[float] = None
    shares: Optional[int] = None
    position: Optional[int] = None
    profit: Optional[float] = None
    level: Optional[int] = None


@dataclass
class ParsedLog:
    trades: list[TradeRecord] = field(default_factory=list)
    encoding: str = "gbk"


def _parse_int(match: Optional[re.Match]) -> Optional[int]:
    if not match:
        return None
    for g in match.groups():
        if g is not None:
            return int(g)
    return None


def parse_log(log_path: Path, encoding: str = "gbk") -> ParsedLog:
    """解析聚宽回测日志，自动尝试 gbk / utf-8。"""
    text = None
    used_encoding = encoding
    for enc in (encoding, "utf-8", "gb18030"):
        try:
            text = log_path.read_text(encoding=enc)
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"无法解码日志文件: {log_path}")

    trades: list[TradeRecord] = []
    for line in text.splitlines():
        m = LOG_LINE_RE.search(line)
        if not m:
            continue
        rest = m.group("rest")
        trades.append(
            TradeRecord(
                timestamp=datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S"),
                action=m.group("action"),
                name=m.group("name").strip(),
                code=m.group("code").strip(),
                price=float(PRICE_RE.search(rest).group(1)) if PRICE_RE.search(rest) else None,
                shares=int(SHARES_RE.search(rest).group(1)) if SHARES_RE.search(rest) else None,
                position=_parse_int(POSITION_RE.search(rest)),
                profit=float(PROFIT_RE.search(rest).group(1)) if PROFIT_RE.search(rest) else None,
                level=int(LEVEL_RE.search(rest).group(1)) if LEVEL_RE.search(rest) else None,
            )
        )
    return ParsedLog(trades=trades, encoding=used_encoding)


# ---------------------------------------------------------------------------
# 指标汇总
# ---------------------------------------------------------------------------

def trades_to_dataframe(trades: list[TradeRecord]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.__dict__ for t in trades])
    df["date"] = df["timestamp"].dt.date
    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["label"] = df["name"] + "\n" + df["code"]
    return df


def build_summary(df: pd.DataFrame, position_summary: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []
    for code, g in df.groupby("code"):
        name = g["name"].iloc[0]
        profit_rows = g[g["profit"].notna()]
        row = {
            "标的": f"{name} ({code})",
            "重装建仓": int((g["action"] == "重装建仓").sum()),
            "加仓": int((g["action"] == "加仓").sum()),
            "止盈": int(g["action"].apply(_is_take_profit).sum()),
            "破顶": int((g["action"] == "破顶").sum()),
            "做T次数": int(len(profit_rows)),
            "做T利润(元)": round(profit_rows["profit"].sum(), 2),
            "单次均值(元)": round(profit_rows["profit"].mean(), 2) if len(profit_rows) else 0.0,
            "末次等级": g["level"].dropna().iloc[-1] if g["level"].notna().any() else None,
        }
        if position_summary is not None and not position_summary.empty:
            pos = position_summary[position_summary["code"] == code]
            if not pos.empty:
                p = pos.iloc[0]
                row["末持仓(股)"] = int(p["末持仓(股)"])
                row["末市值(元)"] = int(p["末市值(元)"])
                row["日均市值(元)"] = int(p["日均市值(元)"])
                row["期末占比(%)"] = float(p["期末占比(%)"])
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("做T利润(元)", ascending=False)
    return summary


def cumulative_profit_series(df: pd.DataFrame) -> pd.DataFrame:
    profit_df = df[df["profit"].notna()].copy()
    if profit_df.empty:
        return pd.DataFrame()
    profit_df = profit_df.sort_values("timestamp")
    profit_df["cum_profit"] = profit_df.groupby("code")["profit"].cumsum()
    profit_df["cum_total"] = profit_df["profit"].cumsum()
    return profit_df


def _trade_fee(trade_value: float) -> float:
    return max(MIN_COMMISSION, trade_value * COMMISSION_RATE)


def _trade_share_delta(row: pd.Series, old_position: int) -> int:
    """根据日志字段推断本次成交的股数变化（正=买入，负=卖出）。"""
    new_pos = row["position"]
    shares = row["shares"]
    if pd.notna(new_pos):
        return int(new_pos) - old_position
    if pd.notna(shares):
        signed = int(shares)
        return signed if row["action"] in ("重装建仓", "加仓") else -signed
    return 0


def _apply_trade_to_state(
    row: pd.Series,
    positions: dict[str, int],
    last_prices: dict[str, float],
    cash: float,
) -> float:
    """应用单笔成交，更新 cash / positions / last_prices。"""
    code = str(row["code"])
    price = row["price"]
    old_pos = positions.get(code, 0)
    action = str(row["action"])

    if action in ("清仓", "清仓-碎股"):
        delta = -old_pos if old_pos else 0
    else:
        delta = _trade_share_delta(row, old_pos)

    if pd.notna(price) and delta != 0:
        trade_value = abs(delta) * float(price)
        fee = _trade_fee(trade_value)
        if delta > 0:
            cash -= trade_value + fee
        else:
            cash += trade_value - fee

    if action in ("清仓", "清仓-碎股"):
        positions[code] = 0
    elif pd.notna(row["position"]):
        positions[code] = int(row["position"])
    elif delta:
        positions[code] = old_pos + delta

    if pd.notna(price):
        last_prices[code] = float(price)

    return cash


def _snapshot_tickers(
    timestamp: datetime,
    names: dict[str, str],
    positions: dict[str, int],
    last_prices: dict[str, float],
) -> list[dict]:
    """记录各标的当前持仓快照。"""
    rows: list[dict] = []
    for code in sorted(set(names) | set(positions)):
        shares = int(positions.get(code, 0))
        price = float(last_prices.get(code, 0.0))
        rows.append(
            {
                "timestamp": timestamp,
                "code": code,
                "name": names.get(code, code),
                "shares": shares,
                "price": price if price > 0 else None,
                "market_value": shares * price if price > 0 else 0.0,
            }
        )
    return rows


def replay_portfolio(df: pd.DataFrame, initial_cash: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """回放交易日志，返回 (组合汇总, 分标的持仓快照)。

    说明:
    - 聚宽日志不含账户快照，持仓市值用各标的末次成交价 mark-to-market。
    - 未交易时段价格沿用最近一次成交价，因此为近似值。
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    ordered = df.sort_values("timestamp").copy()
    names = {str(c): n for c, n in zip(ordered["code"], ordered["name"])}
    positions: dict[str, int] = {}
    last_prices: dict[str, float] = {}
    cash = float(initial_cash)
    portfolio_records: list[dict] = []
    ticker_records: list[dict] = []

    for _, row in ordered.iterrows():
        cash = _apply_trade_to_state(row, positions, last_prices, cash)
        ticker_records.extend(
            _snapshot_tickers(row["timestamp"], names, positions, last_prices)
        )

        position_value = sum(
            positions.get(ticker, 0) * last_prices.get(ticker, 0.0) for ticker in positions
        )
        total_assets = cash + position_value
        utilization = position_value / total_assets if total_assets > 0 else 0.0

        portfolio_records.append(
            {
                "timestamp": row["timestamp"],
                "cash": cash,
                "position_value": position_value,
                "total_assets": total_assets,
                "utilization": utilization,
                "idle_cash": cash,
                "idle_ratio": cash / total_assets if total_assets > 0 else 0.0,
            }
        )

    portfolio = pd.DataFrame(portfolio_records)
    ticker_positions = pd.DataFrame(ticker_records)
    return portfolio, ticker_positions


def build_portfolio_series(df: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    """根据交易日志回放现金与持仓，估算资金利用率。"""
    portfolio, _ = replay_portfolio(df, initial_cash)
    return portfolio


def build_position_summary(
    ticker_positions: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """汇总各标的持仓股数与市值（末态 / 日均 / 区间）。"""
    if ticker_positions.empty:
        return pd.DataFrame()

    daily_long = (
        ticker_positions.set_index("timestamp")
        .groupby("code")[["shares", "price", "market_value", "name"]]
        .resample("D")
        .last()
        .reset_index()
    )
    end_assets = portfolio["total_assets"].iloc[-1] if not portfolio.empty else 0.0

    rows = []
    for code, g in daily_long.groupby("code"):
        g = g.sort_values("timestamp")
        name = g["name"].iloc[-1]
        end_row = g.iloc[-1]
        held = g[g["shares"] > 0]
        rows.append(
            {
                "code": code,
                "标的": f"{name} ({code})",
                "末持仓(股)": int(end_row["shares"]),
                "末价格": round(float(end_row["price"]), 3) if pd.notna(end_row["price"]) else None,
                "末市值(元)": round(float(end_row["market_value"]), 0),
                "日均市值(元)": round(g["market_value"].mean(), 0),
                "最高市值(元)": round(g["market_value"].max(), 0),
                "最低市值(元)": round(held["market_value"].min(), 0) if not held.empty else 0.0,
                "期末占比(%)": round(end_row["market_value"] / end_assets * 100, 1)
                if end_assets > 0
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("末市值(元)", ascending=False)


def daily_ticker_market_value(ticker_positions: pd.DataFrame) -> pd.DataFrame:
    """按日汇总各标的市值（宽表，列=code）。"""
    if ticker_positions.empty:
        return pd.DataFrame()
    daily_long = (
        ticker_positions.set_index("timestamp")
        .groupby("code")["market_value"]
        .resample("D")
        .last()
        .reset_index()
    )
    wide = daily_long.pivot(index="timestamp", columns="code", values="market_value").fillna(0.0)
    return wide.reset_index()


def daily_portfolio_series(portfolio: pd.DataFrame) -> pd.DataFrame:
    """按日取最后一次快照，用于绘制资金利用率曲线。"""
    if portfolio.empty:
        return portfolio
    daily = portfolio.set_index("timestamp").resample("D").last().dropna(how="all")
    return daily.reset_index()


def monthly_activity(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["加仓", "止盈", "破顶"]
    pivot = (
        df[df["action"].isin(cols)]
        .groupby(["year_month", "action"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=cols, fill_value=0)
    )
    return pivot


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
        }
    )


def render_report(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    portfolio: pd.DataFrame,
    initial_cash: float,
    ticker_positions: Optional[pd.DataFrame] = None,
) -> None:
    _setup_matplotlib()
    position_summary = build_position_summary(ticker_positions, portfolio) if ticker_positions is not None else None
    summary = build_summary(df, position_summary)
    cum = cumulative_profit_series(df)
    monthly = monthly_activity(df)
    daily_portfolio = daily_portfolio_series(portfolio)

    start = df["timestamp"].min().strftime("%Y-%m-%d")
    end = df["timestamp"].max().strftime("%Y-%m-%d")
    total_profit = df["profit"].sum()
    total_trades = len(df)
    profit_trades = df["profit"].notna().sum()

    fig = plt.figure(figsize=(18, 17))
    fig.suptitle(f"{title}\n回测区间: {start} ~ {end}", fontsize=16, fontweight="bold", y=0.985)

    gs = fig.add_gridspec(4, 2, height_ratios=[1.4, 1.0, 1.0, 0.95], hspace=0.40, wspace=0.28)

    # 1) 各标的累计做T利润曲线
    ax1 = fig.add_subplot(gs[0, :])
    if not cum.empty:
        for code, g in cum.groupby("code"):
            label = f"{g['name'].iloc[0]} ({code})"
            ax1.plot(g["timestamp"], g["cum_profit"], linewidth=1.8, label=label)
        total_by_ts = cum.groupby("timestamp")["profit"].sum().cumsum()
        ax1.plot(
            total_by_ts.index,
            total_by_ts.values,
            color="#111827",
            linewidth=2.5,
            linestyle="--",
            label=f"合计 ({total_profit:,.0f} 元)",
        )
    ax1.set_title("各标的累计做T利润", fontsize=13, pad=10)
    ax1.set_ylabel("累计利润 (元)")
    ax1.legend(loc="upper left", ncol=3, fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # 2) 各标的做T利润柱状图
    ax2 = fig.add_subplot(gs[1, 0])
    if not summary.empty:
        y_pos = np.arange(len(summary))
        profits = summary["做T利润(元)"].values
        bars = ax2.barh(y_pos, profits, color="#22c55e", alpha=0.85)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([s.split(" (")[0] for s in summary["标的"]], fontsize=10)
        ax2.invert_yaxis()
        for bar, val in zip(bars, profits):
            ax2.text(bar.get_width() + max(profits) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:,.0f}", va="center", fontsize=9)
    ax2.set_title("各标的累计做T利润对比", fontsize=13)
    ax2.set_xlabel("利润 (元)")
    ax2.grid(True, axis="x", alpha=0.25)

    # 3) 操作类型分布（横向条形图，避免饼图标签重叠）
    ax3 = fig.add_subplot(gs[1, 1])
    action_counts = df["action"].value_counts().sort_values(ascending=True)
    total_actions = int(action_counts.sum())
    y_pos = np.arange(len(action_counts))
    bar_colors = [ACTION_COLORS.get(a, "#94a3b8") for a in action_counts.index]
    bars = ax3.barh(y_pos, action_counts.values, color=bar_colors, alpha=0.88, height=0.72)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(action_counts.index, fontsize=10)
    ax3.set_xlabel("笔数")
    ax3.set_title("交易操作类型占比", fontsize=13)
    ax3.grid(True, axis="x", alpha=0.25)
    xmax = action_counts.max() if len(action_counts) else 1
    for bar, val in zip(bars, action_counts.values):
        pct = val / total_actions * 100 if total_actions else 0
        ax3.text(
            bar.get_width() + xmax * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )
    ax3.set_xlim(0, xmax * 1.22)

    # 4) 资金利用率 / 闲置资金
    ax4 = fig.add_subplot(gs[2, :])
    if not daily_portfolio.empty:
        ts = daily_portfolio["timestamp"]
        utilization_pct = daily_portfolio["utilization"] * 100
        idle_wan = daily_portfolio["idle_cash"] / 10_000

        ax4.fill_between(
            ts,
            0,
            utilization_pct,
            color="#3b82f6",
            alpha=0.25,
            label="持仓市值占比",
        )
        ax4.plot(ts, utilization_pct, color="#1d4ed8", linewidth=1.6)
        ax4.set_ylabel("资金利用率 (%)", color="#1d4ed8")
        ax4.set_ylim(0, max(100, utilization_pct.max() * 1.15))
        ax4.tick_params(axis="y", labelcolor="#1d4ed8")
        ax4.grid(True, alpha=0.25)

        ax4b = ax4.twinx()
        ax4b.plot(ts, idle_wan, color="#f97316", linewidth=1.4, linestyle="--", label="闲置资金")
        ax4b.set_ylabel("闲置资金 (万元)", color="#f97316")
        ax4b.tick_params(axis="y", labelcolor="#f97316")

        avg_util = daily_portfolio["utilization"].mean() * 100
        avg_idle_wan = daily_portfolio["idle_cash"].mean() / 10_000
        end_util = daily_portfolio["utilization"].iloc[-1] * 100
        end_idle_wan = daily_portfolio["idle_cash"].iloc[-1] / 10_000
        ax4.set_title(
            "资金利用率与闲置资金 "
            f"(初始 {initial_cash/10_000:,.0f} 万 | 日均利用率 {avg_util:.1f}% | "
            f"日均闲置 {avg_idle_wan:,.1f} 万 | 期末利用率 {end_util:.1f}% | "
            f"期末闲置 {end_idle_wan:,.1f} 万)",
            fontsize=13,
            pad=10,
        )
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax4.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right")

        lines_a, labels_a = ax4.get_legend_handles_labels()
        lines_b, labels_b = ax4b.get_legend_handles_labels()
        ax4.legend(lines_a + lines_b, labels_a + labels_b, loc="upper left", fontsize=9)
    else:
        ax4.set_title("资金利用率与闲置资金", fontsize=13)
        ax4.text(0.5, 0.5, "无可用数据", ha="center", va="center", transform=ax4.transAxes)

    # 5) 月度交易活跃度
    ax5 = fig.add_subplot(gs[3, 0])
    if not monthly.empty:
        x = np.arange(len(monthly))
        bottom = np.zeros(len(monthly))
        for action in monthly.columns:
            vals = monthly[action].values
            ax5.bar(x, vals, bottom=bottom, label=action, color=ACTION_COLORS.get(action, "#94a3b8"), width=0.85)
            bottom += vals
        tick_step = max(1, len(monthly) // 12)
        ax5.set_xticks(x[::tick_step])
        ax5.set_xticklabels(monthly.index[::tick_step], rotation=45, ha="right", fontsize=8)
    ax5.set_title("月度交易活跃度 (加仓 / 止盈 / 破顶)", fontsize=13)
    ax5.set_ylabel("笔数")
    ax5.legend(loc="upper right", fontsize=9)
    ax5.grid(True, axis="y", alpha=0.25)

    # 6) 各标的操作次数分组柱状图
    ax6 = fig.add_subplot(gs[3, 1])
    if not summary.empty:
        codes_short = [s.split("(")[0].strip() for s in summary["标的"]]
        x = np.arange(len(codes_short))
        width = 0.2
        for i, action in enumerate(["加仓", "止盈", "破顶"]):
            offset = (i - 1) * width
            ax6.bar(
                x + offset,
                summary[action].values,
                width,
                label=action,
                color=ACTION_COLORS.get(action, "#94a3b8"),
            )
        ax6.set_xticks(x)
        ax6.set_xticklabels(codes_short, fontsize=9)
        ax6.legend(fontsize=9)
    ax6.set_title("各标的网格操作次数", fontsize=13)
    ax6.set_ylabel("笔数")
    ax6.grid(True, axis="y", alpha=0.25)

    # 页脚摘要
    footer = (
        f"总交易 {total_trades} 笔 | 落袋 {profit_trades} 笔 | "
        f"累计做T利润 {total_profit:,.2f} 元 | "
        f"日均 {total_profit / max((df['timestamp'].max() - df['timestamp'].min()).days, 1):.1f} 元"
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=11, color="#374151")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_position_charts(
    ticker_positions: pd.DataFrame,
    position_summary: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """各标的持仓市值：5 个子图纵向排列，折线 + 浅色填充。"""
    _setup_matplotlib()
    daily_mv = daily_ticker_market_value(ticker_positions)
    if daily_mv.empty:
        return

    codes = [c for c in daily_mv.columns if c != "timestamp"]
    if position_summary is not None and not position_summary.empty:
        ordered = position_summary.sort_values("末市值(元)", ascending=False)["code"].tolist()
        codes = [c for c in ordered if c in codes] + [c for c in codes if c not in ordered]

    n = len(codes)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.6 * n + 1.2), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(f"{title} — 各标的持仓市值（估）", fontsize=15, fontweight="bold", y=0.995)
    ts = daily_mv["timestamp"]

    for i, code in enumerate(codes):
        ax = axes[i]
        name = ticker_positions.loc[ticker_positions["code"] == code, "name"].iloc[0]
        values_wan = daily_mv[code].values / 10_000
        color = TICKER_LINE_COLORS[i % len(TICKER_LINE_COLORS)]

        ax.fill_between(ts, 0, values_wan, color=color, alpha=0.12)
        ax.plot(ts, values_wan, color=color, linewidth=1.8, marker="", label=name)

        end_wan = values_wan[-1] if len(values_wan) else 0
        end_shares = 0
        if position_summary is not None and not position_summary.empty:
            row = position_summary[position_summary["code"] == code]
            if not row.empty:
                end_shares = int(row.iloc[0]["末持仓(股)"])
                end_mv = float(row.iloc[0]["末市值(元)"])
                end_wan = end_mv / 10_000

        ax.set_ylabel("万元", fontsize=9)
        ax.set_title(
            f"{name} ({code})  |  期末 {end_shares:,} 股 ≈ {end_wan:,.1f} 万",
            fontsize=11,
            loc="left",
            pad=6,
        )
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_xlim(ts.min(), ts.max())
        ymax = max(values_wan.max() * 1.12, 0.5)
        ax.set_ylim(0, ymax)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    axes[-1].set_xlabel("日期")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_summary_table(summary: pd.DataFrame, output_path: Path) -> None:
    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(max(14, 0.9 * len(summary.columns)), max(2.5, 0.45 * len(summary) + 1.2)))
    ax.axis("off")
    display_df = summary.copy()
    for col in display_df.columns:
        if col.endswith("(元)"):
            display_df[col] = display_df[col].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "")
        elif col.endswith("(%)") and col != "期末占比(%)":
            pass
        elif col == "期末占比(%)":
            display_df[col] = display_df[col].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        elif col == "单次均值(元)":
            display_df[col] = display_df[col].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(fontweight="bold")
    ax.set_title("分标的统计摘要（含持仓市值）", fontsize=14, fontweight="bold", pad=16)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_position_table(position_summary: pd.DataFrame, output_path: Path) -> None:
    _setup_matplotlib()
    display_df = position_summary.drop(columns=["code"], errors="ignore").copy()
    for col in ["末市值(元)", "日均市值(元)", "最高市值(元)", "最低市值(元)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda v: f"{v:,.0f}")
    if "末价格" in display_df.columns:
        display_df["末价格"] = display_df["末价格"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    if "期末占比(%)" in display_df.columns:
        display_df["期末占比(%)"] = display_df["期末占比(%)"].map(lambda v: f"{v:.1f}")

    fig, ax = plt.subplots(figsize=(14, max(2.2, 0.42 * len(display_df) + 1.0)))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(fontweight="bold")
    ax.set_title("分标的持仓市值", fontsize=14, fontweight="bold", pad=16)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_console_summary(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    encoding: str,
    portfolio: pd.DataFrame,
    initial_cash: float,
    position_summary: Optional[pd.DataFrame] = None,
) -> None:
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    days = max((end - start).days, 1)
    daily_portfolio = daily_portfolio_series(portfolio)
    print("=" * 60)
    print("grid_etf 回测日志分析")
    print("=" * 60)
    print(f"日志编码: {encoding}")
    print(f"回测区间: {start:%Y-%m-%d} ~ {end:%Y-%m-%d} ({days} 天)")
    print(f"总交易笔数: {len(df)}")
    print(f"落袋笔数: {df['profit'].notna().sum()}")
    print(f"累计做T利润: {df['profit'].sum():,.2f} 元")
    print(f"日均做T利润: {df['profit'].sum() / days:,.2f} 元")
    if not daily_portfolio.empty:
        end_assets = daily_portfolio["total_assets"].iloc[-1]
        end_position = daily_portfolio.iloc[-1]["position_value"] if "position_value" in daily_portfolio else 0
        print(
            f"资金利用率: 日均 {daily_portfolio['utilization'].mean() * 100:.1f}% | "
            f"区间 {daily_portfolio['utilization'].min() * 100:.1f}% ~ "
            f"{daily_portfolio['utilization'].max() * 100:.1f}% | "
            f"期末 {daily_portfolio['utilization'].iloc[-1] * 100:.1f}%"
        )
        print(
            f"闲置资金(估): 初始 {initial_cash:,.0f} 元 | "
            f"日均 {daily_portfolio['idle_cash'].mean():,.0f} 元 | "
            f"最低 {daily_portfolio['idle_cash'].min():,.0f} 元 | "
            f"最高 {daily_portfolio['idle_cash'].max():,.0f} 元 | "
            f"期末 {daily_portfolio['idle_cash'].iloc[-1]:,.0f} 元"
        )
        print(
            f"组合总资产(估): 期末 {end_assets:,.0f} 元 | "
            f"期末持仓市值 {end_position:,.0f} 元"
        )
    print()
    print("【做T统计】")
    print(summary.to_string(index=False))
    if position_summary is not None and not position_summary.empty:
        print()
        print("【持仓市值】")
        pos_display = position_summary.drop(columns=["code"], errors="ignore").copy()
        print(pos_display.to_string(index=False))
        total_mv = position_summary["末市值(元)"].sum()
        print(f"\n  合计末持仓市值: {total_mv:,.0f} 元")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

REPORT_FILENAME = "backtest_analysis.png"
SUMMARY_FILENAME = "summary_table.png"
POSITION_FILENAME = "position_table.png"
POSITION_CHARTS_FILENAME = "position_charts.png"
POSITION_DAILY_FILENAME = "position_daily.csv"
POSITION_SUMMARY_FILENAME = "position_summary.csv"
CSV_FILENAME = "trades.csv"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}


def _is_output_directory(path: Path) -> bool:
    """判断 --output 是否应视为输出目录而非单个 PNG 文件。"""
    raw = str(path)
    if raw.endswith(("/", "\\")):
        return True
    if path.exists() and path.is_dir():
        return True
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return True
    return False


def resolve_output_paths(
    output: Path,
    summary_output: Optional[Path] = None,
    csv_output: Optional[Path] = None,
    position_output: Optional[Path] = None,
    position_charts_output: Optional[Path] = None,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    """根据 --output 解析输出文件路径。"""
    if _is_output_directory(output):
        out_dir = output
        report = out_dir / REPORT_FILENAME
        summary = summary_output or out_dir / SUMMARY_FILENAME
        csv = csv_output or out_dir / CSV_FILENAME
        position_table = position_output or out_dir / POSITION_FILENAME
        position_charts = position_charts_output or out_dir / POSITION_CHARTS_FILENAME
        position_daily = out_dir / POSITION_DAILY_FILENAME
        position_summary_csv = out_dir / POSITION_SUMMARY_FILENAME
    else:
        report = output
        parent = output.parent if str(output.parent) else Path(".")
        summary = summary_output or parent / SUMMARY_FILENAME
        csv = csv_output or parent / CSV_FILENAME
        position_table = position_output or parent / POSITION_FILENAME
        position_charts = position_charts_output or parent / POSITION_CHARTS_FILENAME
        position_daily = parent / POSITION_DAILY_FILENAME
        position_summary_csv = parent / POSITION_SUMMARY_FILENAME
    return report, summary, csv, position_table, position_charts, position_daily, position_summary_csv


def _default_log_path(here: Path) -> Path:
    """按目录名 (v0.x) 自动匹配同目录下的回测日志。"""
    named = here / f"grid_etf_{here.name}.txt"
    if named.exists():
        return named
    matches = sorted(here.glob("grid_etf_v*.txt"))
    if matches:
        return matches[0]
    return named


def main() -> None:
    here = Path(__file__).resolve().parent
    default_log = _default_log_path(here)
    parser = argparse.ArgumentParser(description="分析 grid_etf 聚宽回测日志并生成图表")
    parser.add_argument(
        "--log",
        type=Path,
        default=default_log,
        help=f"回测日志 txt 路径 (默认: {default_log.name})",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=DEFAULT_INITIAL_CASH,
        help=f"回测初始资金 (默认: {DEFAULT_INITIAL_CASH:,.0f}，本策略回测 50 万)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=here / "out",
        help="输出目录，或主报告 PNG 文件路径（目录时生成 backtest_analysis.png 等）",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="统计摘要表 PNG 输出路径（默认与 --output 同目录）",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="解析后的交易明细 CSV 输出路径（默认与 --output 同目录）",
    )
    parser.add_argument("--encoding", default="gbk", help="日志文件编码 (默认 gbk)")
    parser.add_argument("--show", action="store_true", help="保存后弹出 matplotlib 窗口")
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"日志文件不存在: {args.log}")

    parsed = parse_log(args.log, encoding=args.encoding)
    if not parsed.trades:
        raise SystemExit(f"未能从日志中解析到任何交易记录: {args.log}")

    df = trades_to_dataframe(parsed.trades)
    portfolio, ticker_positions = replay_portfolio(df, initial_cash=args.initial_cash)
    position_summary = build_position_summary(ticker_positions, portfolio)
    summary = build_summary(df, position_summary)

    report_path, summary_path, csv_path, position_table_path, position_charts_path, position_daily_path, position_summary_path = (
        resolve_output_paths(
            args.output,
            summary_output=args.summary_output,
            csv_output=args.csv_output,
        )
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if not position_summary.empty:
        position_summary.to_csv(position_summary_path, index=False, encoding="utf-8-sig")
    daily_mv = daily_ticker_market_value(ticker_positions)
    if not daily_mv.empty:
        daily_mv.to_csv(position_daily_path, index=False, encoding="utf-8-sig")

    title = args.log.stem.replace("_", " ")
    render_report(
        df,
        report_path,
        title=title,
        portfolio=portfolio,
        initial_cash=args.initial_cash,
        ticker_positions=ticker_positions,
    )
    render_summary_table(summary, summary_path)
    if not position_summary.empty:
        render_position_table(position_summary, position_table_path)
        render_position_charts(
            ticker_positions,
            position_summary,
            position_charts_path,
            title=title,
        )
    print_console_summary(
        df, summary, parsed.encoding, portfolio, args.initial_cash, position_summary
    )

    print(f"\n已保存: {report_path}")
    print(f"已保存: {summary_path}")
    if not position_summary.empty:
        print(f"已保存: {position_table_path}")
        print(f"已保存: {position_charts_path}")
        print(f"已保存: {position_summary_path}")
    if not daily_mv.empty:
        print(f"已保存: {position_daily_path}")
    print(f"已保存: {csv_path}")

    if args.show:
        _setup_matplotlib()
        img = plt.imread(report_path)
        fig, ax = plt.subplots(figsize=(18, 14))
        ax.imshow(img)
        ax.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
