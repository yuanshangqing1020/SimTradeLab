# -*- coding: utf-8 -*-
"""
core_grid_hybrid v1 — JoinQuant 版

与 SimTradeLab ``strategies/core_grid_hybrid_v1/backtest.py`` 中
``regime_mode='trend_sizing'`` 逻辑对齐（慢均线 z、峰值回撤限仓、再平衡死区、整体止盈）。

- 代码规范：510300.XSHG（上交所 ETF）；基准 000300.XSHG。
- 聚宽回测请在「交易」里将 ETF 按基金费率设佣；本文件与 v5 对齐使用 ``type='fund'`` 的一档费率。
- 规格与设计说明：``my_docs/core_grid_hybrid/``。
"""

SECURITY = '510300.XSHG'
BENCHMARK_CODE = '000300.XSHG'


def initialize(context):
    set_benchmark(BENCHMARK_CODE)
    set_option('use_real_price', True)

    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5,
    ), type='stock')
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0,
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5,
    ), type='fund')

    set_slippage(PriceRelatedSlippage(0.00246))

    g.symbol = SECURITY
    g.take_profit_ret = 9.99

    g.ma_slow_days = 120
    g.ma_min_days = 20
    g.ts_z_cut = -0.078
    g.ts_w_floor = 0.52
    g.ts_w_cap = 0.99
    g.ts_w_mid = 0.68
    g.ts_w_slope = 3.5
    g.ts_rebalance_band = 0.026
    g.ts_peak_dd_cut = 0.10
    g.ts_w_peak_stress = 0.32

    g.eod_closes = []
    g.eod_last_d = None
    g.ts_last_w = None
    g.ts_peak_close = None
    g.done = False

    run_daily(rebalance_trend_sizing, time='14:55')


def _ma_tail_mean(hist, n, min_n=20):
    if hist is None:
        return None
    L = len(hist)
    if L < min_n:
        return None
    use = n if L >= n else L
    s = 0.0
    for x in hist[-use:]:
        s += x
    return s / float(use)


def _append_eod_close(context, close):
    d = context.current_dt.date()
    if g.eod_last_d == d:
        return
    g.eod_last_d = d
    g.eod_closes.append(close)
    if len(g.eod_closes) > 450:
        g.eod_closes = g.eod_closes[-450:]


def _last_price(context, security):
    cd = get_current_data()
    if security in cd:
        p = cd[security].last_price
        if p is not None and p > 0:
            return float(p)
    df = attribute_history(security, 1, '1d', ['close'])
    if df is not None and len(df) > 0:
        v = float(df['close'].iloc[-1])
        if v > 0:
            return v
    return None


def _try_take_profit(context, security):
    sc = float(context.portfolio.starting_cash)
    if sc <= 0:
        return
    ret_end = context.portfolio.total_value / sc - 1.0
    if ret_end < g.take_profit_ret:
        return
    log.info('EXIT_ALL 触发 ret=%.4f >= %.4f', ret_end, g.take_profit_ret)
    order_target(security, 0)
    pos = context.portfolio.positions.get(security)
    amt = int(pos.total_amount) if pos is not None else 0
    if amt <= 0:
        g.done = True


def rebalance_trend_sizing(context):
    if g.done:
        return
    sym = g.symbol
    close = _last_price(context, sym)
    if close is None or close != close or close <= 0:
        return

    hist_before = list(g.eod_closes)
    slow_n = int(g.ma_slow_days)
    min_n = int(g.ma_min_days)
    ma = _ma_tail_mean(hist_before, slow_n, min_n=min_n)
    _append_eod_close(context, close)
    if ma is None or ma <= 0:
        _try_take_profit(context, sym)
        return

    z = (close - ma) / ma
    zc = float(g.ts_z_cut)
    wf = float(g.ts_w_floor)
    wc = float(g.ts_w_cap)
    wm = float(g.ts_w_mid)
    ws = float(g.ts_w_slope)
    if z < zc:
        w = 0.0
    else:
        w = wm + z * ws
        if w < wf:
            w = wf
        if w > wc:
            w = wc

    pk = g.ts_peak_close
    if pk is None or close > pk:
        pk = close
    g.ts_peak_close = pk
    if pk > 0:
        pdd = (pk - close) / pk
        pc = float(g.ts_peak_dd_cut)
        if pdd > pc:
            w = min(w, float(g.ts_w_peak_stress))

    pv = float(context.portfolio.total_value)
    if pv <= 0:
        return

    pos = context.portfolio.positions.get(sym)
    phys = float(pos.total_amount) if pos is not None else 0.0
    cur_stock = phys * close
    cur_w = cur_stock / pv
    band = float(g.ts_rebalance_band)
    if band > 0.0 and abs(w - cur_w) < band:
        _try_take_profit(context, sym)
        return

    order_target_value(sym, pv * w)

    lw = g.ts_last_w
    if lw is None or abs(w - lw) >= 0.04:
        log.info('TGT_W z=%.4f w=%.3f pv=%.0f', z, w, pv)
        g.ts_last_w = w

    _try_take_profit(context, sym)
