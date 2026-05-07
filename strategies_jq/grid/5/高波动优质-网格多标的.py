# -*- coding: utf-8 -*-
"""
高波动 + 优质标的 等权网格（股票 + ETF）

- 资金规模：按 50 万总底仓思路设计（等权目标权重；回测起始资金建议 ≥50 万）。
- 持仓数量：每周在沪深300 ∪ 中证500 ∪ 主流 ETF 中选股，最多 50 只。
- 选股：股票侧偏「高波动 + ROE/估值/市值」综合分；ETF 侧偏「高波动 + 成交额流动性」。
- 网格：以 20 日均线为中枢，按近端年化波动率自适应步长（约 1.8%～8%），在价低于中枢时加仓、高于时减仓；
  单标的目标权重在等权基础上按层数 ±12% 浮动（层数上限 3），权重归一化后下单。

参数见下方常量；可直接粘贴聚宽研究/回测编辑器运行。
"""

from jqdata import *
import numpy as np
import pandas as pd


# —— 资金与持仓 —— #
TARGET_BOOK = 500000.0   # 策略目标资金规模（用于说明与软约束；实际按账户总资产比例下单）
MAX_HOLD = 50            # 最多同时网格的标的数
GRID_MAX_LAYER = 3       # 相对中枢最多偏离层数
LAYER_FRACTION = 0.12    # 每层相对等权基准的权重增减比例

# —— 流动性 ETF（与指数成分合并打分，不保证全部入选）—— #
CANDIDATE_ETFS = [
    '510300.XSHG', '510500.XSHG', '159915.XSHE', '512880.XSHG', '512690.XSHG',
    '512010.XSHG', '515050.XSHG', '512480.XSHG', '159949.XSHE', '588000.XSHG',
    '512170.XSHG', '512760.XSHG', '159792.XSHE', '513100.XSHG', '513050.XSHG',
]


def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)

    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.001,
        open_commission=0.0003, close_commission=0.0003,
        close_today_commission=0, min_commission=5,
    ), type='stock')
    set_slippage(PriceRelatedSlippage(0.00246))

    g.pool = []
    g.target_book = TARGET_BOOK

    run_weekly(weekly_select, weekday=1, time='09:31', force=False)
    run_daily(daily_grid, time='14:50')


def weekly_select(context):
    """每周第一个交易日：更新标的池；卖出被调出标的。"""
    date = context.current_dt.date()
    stocks = list(set(
        get_index_stocks('000300.XSHG', date)
        + get_index_stocks('000905.XSHG', date)
    ))
    stocks = [s for s in stocks if s.endswith('.XSHE') or s.endswith('.XSHG')]

    etfs = [e for e in CANDIDATE_ETFS if e not in stocks]

    picked = build_universe(stocks, etfs, date)
    if not picked:
        log.info('警告: 本周未选出标的，保持原池')
        return

    old = set(g.pool)
    new = set(picked)
    for s in old - new:
        order_target(s, 0)
        log.info('调出网格池 %s' % s)

    g.pool = picked
    log.info('本周网格池 %d 只: %s' % (len(g.pool), ','.join(g.pool[:8]) + ('...' if len(g.pool) > 8 else '')))


def build_universe(stock_list, etf_list, date):
    """综合波动与质量/流动性打分，取前 MAX_HOLD。"""
    # 注意：开盘调度点 get_current_data() 往往只含部分标的，用「不在 current 就剔除」会导致全池被清空。
    current = get_current_data()
    stock_list = filter_tradeable_light(stock_list, current)
    etf_list = filter_tradeable_light(etf_list, current)

    scores = []

    # —— 股票：基本面 + 波动 —— #
    if stock_list:
        q = query(
            valuation.code,
            valuation.market_cap,
            valuation.pe_ratio,
            indicator.roe,
        ).filter(
            valuation.code.in_(stock_list),
            valuation.market_cap > 3e9,
            valuation.pe_ratio > 0,
            valuation.pe_ratio < 120,
        )
        # 财报 ROE 可能为 NaN，不在 SQL 里写死 >0，避免整表被滤空
        fdf = get_fundamentals(q, date=date)
        if fdf is not None and len(fdf) > 0:
            codes = fdf['code'].tolist()
            vols = batch_realized_vol(codes, date)
            fdf = fdf.set_index('code')
            for c in codes:
                if c not in fdf.index or c not in vols:
                    continue
                roe = fdf.loc[c, 'roe']
                pe = fdf.loc[c, 'pe_ratio']
                mcap = fdf.loc[c, 'market_cap']
                if pe is None or mcap is None:
                    continue
                if not (np.isfinite(pe) and np.isfinite(mcap)):
                    continue
                if roe is None or (isinstance(roe, float) and not np.isfinite(roe)):
                    roe = 0.0
                vol = vols[c]
                if not np.isfinite(vol) or vol <= 0:
                    continue
                inv_pe = 1.0 / max(pe, 1.0)
                scores.append((c, 'stock', vol, roe, inv_pe, mcap))

    # —— ETF：波动 + 成交额 —— #
    if etf_list:
        vols_e = batch_realized_vol(etf_list, date)
        money20 = batch_avg_money(etf_list, date)
        m_rank = {}
        if money20:
            m_rank = pd.Series(money20).rank(pct=True).to_dict()
        for c in etf_list:
            if c not in vols_e:
                continue
            vol = vols_e[c]
            if not np.isfinite(vol) or vol <= 0:
                continue
            liq = float(m_rank.get(c, 0.5))
            scores.append((c, 'etf', vol, liq, 0.0, 0.0))

    if not scores:
        return fallback_vol_top(stock_list, etf_list, date)

    df = pd.DataFrame(scores, columns=['code', 'kind', 'vol', 'a', 'b', 'c'])

    df['vol_pct'] = df['vol'].rank(pct=True)
    # 股票: a=roe, b=1/pe, c=市值（大市值偏「稳」）
    if (df['kind'] == 'stock').any():
        stk = df['kind'] == 'stock'
        df.loc[stk, 'qual_pct'] = (
            df.loc[stk, 'a'].rank(pct=True) * 0.45
            + df.loc[stk, 'b'].rank(pct=True) * 0.35
            + df.loc[stk, 'c'].rank(pct=True) * 0.20
        )
    if (df['kind'] == 'etf').any():
        et = df['kind'] == 'etf'
        df.loc[et, 'qual_pct'] = df.loc[et, 'a'].rank(pct=True)

    df['score'] = df['vol_pct'] * 0.62 + df['qual_pct'].fillna(0.5) * 0.38
    df = df.sort_values('score', ascending=False)

    out = df['code'].head(MAX_HOLD).tolist()
    return out


def fallback_vol_top(stock_list, etf_list, date):
    """打分失败时：按已实现波动率从高到低取满池（保证有成交）。"""
    cand = (stock_list or []) + (etf_list or [])
    if not cand:
        return []
    vols = batch_realized_vol(cand, date)
    if not vols:
        return cand[:MAX_HOLD]
    ranked = sorted(vols.keys(), key=lambda k: vols[k], reverse=True)
    return ranked[:MAX_HOLD]


def filter_tradeable_light(codes, current):
    """仅剔除明确 ST / 停牌；未出现在 current 中的标的保留（避免空池）。"""
    res = []
    for s in codes:
        if s in current:
            d = current[s]
            if d.paused or d.is_st:
                continue
            name = getattr(d, 'name', '') or ''
            if 'ST' in name or '*' in name:
                continue
        res.append(s)
    return res


def batch_realized_vol(codes, date):
    """约 60 日年化已实现波动率。"""
    if not codes:
        return {}
    end = date.strftime('%Y-%m-%d')
    try:
        px = get_price(
            codes, end_date=end, count=65, frequency='daily',
            fields=['close'], panel=False, skip_paused=False,
        )
    except Exception:
        return {}
    if px is None or len(px) == 0:
        return {}
    px = _normalize_price_frame(px)
    if px is None or 'code' not in px.columns:
        return {}
    out = {}
    for code, grp in px.groupby('code'):
        closes = grp['close'].dropna()
        if len(closes) < 30:
            continue
        r = closes.pct_change().dropna()
        if len(r) < 20:
            continue
        sig = float(r.iloc[-60:].std()) if len(r) >= 60 else float(r.std())
        out[code] = float(sig * np.sqrt(250.0))
    return out


def batch_avg_money(codes, date):
    """近 20 日日均成交额（元）。"""
    if not codes:
        return {}
    end = date.strftime('%Y-%m-%d')
    try:
        px = get_price(
            codes, end_date=end, count=25, frequency='daily',
            fields=['money'], panel=False, skip_paused=False,
        )
    except Exception:
        return {}
    if px is None or len(px) == 0:
        return {}
    px = _normalize_price_frame(px)
    if px is None or 'code' not in px.columns:
        return {}
    out = {}
    for code, grp in px.groupby('code'):
        m = grp['money'].dropna()
        if len(m) < 10:
            continue
        out[code] = float(m.iloc[-20:].mean())
    return out


def _normalize_price_frame(px):
    """统一多标的 get_price(panel=False) 的长表格式，确保有 code 列。"""
    if px is None or len(px) == 0:
        return None
    if 'code' in px.columns:
        return px
    if isinstance(px.index, pd.MultiIndex):
        px = px.reset_index()
        for col in list(px.columns):
            if not isinstance(col, str):
                continue
            lc = col.lower()
            if lc in ('code', 'security'):
                if col != 'code':
                    px = px.rename(columns={col: 'code'})
                break
    return px if 'code' in px.columns else None


def daily_grid(context):
    """尾盘：按中枢与波动步长计算层数，归一化权重后调仓。"""
    if not g.pool:
        return

    cd = get_current_data()
    N = len(g.pool)
    if N == 0:
        return

    layers = []
    active = []

    for s in g.pool:
        if s in cd and cd[s].paused:
            continue
        h = attribute_history(s, 30, '1d', ['close'], skip_paused=True, df=True)
        if h is None or len(h) < 22:
            continue
        closes = h['close'].astype(float)
        close = float(closes.iloc[-1])
        ma = float(closes.iloc[-20:].mean())
        rets = closes.pct_change().dropna()
        if len(rets) < 10:
            continue
        vol = float(rets.iloc[-min(60, len(rets)):].std() * np.sqrt(250.0))
        step = max(0.018, min(0.08, vol * 0.45))
        denom = max(close * step, 1e-6)
        raw = (ma - close) / denom
        layer = int(np.clip(int(np.floor(raw + 0.5)), -GRID_MAX_LAYER, GRID_MAX_LAYER))
        layers.append(layer)
        active.append(s)

    if not active:
        return

    raw_weights = []
    for s, layer in zip(active, layers):
        w = (1.0 / N) * (1.0 + LAYER_FRACTION * float(layer))
        raw_weights.append(max(w, 1e-6))

    sw = sum(raw_weights)
    tv = context.portfolio.total_value
    # 50 万底仓：总资产高于目标时，仅按目标规模分配网格权重，其余为现金
    cap = min(tv, max(getattr(g, 'target_book', TARGET_BOOK), 1000.0))
    for s, rw in zip(active, raw_weights):
        tgt = cap * (rw / sw)
        order_target_value(s, tgt)


def after_trading_end(context):
    if g.pool:
        n = 0
        for s in g.pool:
            p = context.portfolio.positions.get(s)
            if p is not None and p.total_amount > 0:
                n += 1
        log.info('日终持仓 %d 只，总资产 %.0f' % (n, context.portfolio.total_value))
