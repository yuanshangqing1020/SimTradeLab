# strategies/core_grid_hybrid_v1/backtest.py
# -*- coding: utf-8 -*-
"""
核心仓 + 网格混合策略 v1.0 — SimTradeLab 回测

规格：my_docs/core_grid_hybrid/v1.0/01-design.md
实施：my_docs/core_grid_hybrid/v1.0/02-plan.md

网格基准：固定 ref 几何卖档（方案 A）+ 网格卖后回补：ref 买档 或「上一网格卖价」回撤 buy_step；
暂停期间用峰值跟踪辅助回补。首次建仓按「当前现金 × initial_position_ratio」预算（勿用 starting_cash 以免再入时重复满额）。

说明：默认 regime_mode='trend_sizing'，按相对慢均线的 z-score 连续调仓（order_target_value）；几何网格相关纯函数仍保留供单测与后续叠加。
"""
import math

# ---------------------------------------------------------------------------
# 纯函数（单测入口）
# ---------------------------------------------------------------------------


def grid_sell_price(ref, k, grid_step):
    """第 k 档卖出参考价：ref * (1 + grid_step)^k，k >= 1。"""
    if k < 1:
        raise ValueError("k must be >= 1")
    return ref * (1.0 + grid_step) ** k


def grid_buy_price(ref, k, step):
    """第 k 档买入参考价：ref * (1 - step)^k，k >= 1。step 为 grid_step 或 defensive_buy_step。"""
    if k < 1:
        raise ValueError("k must be >= 1")
    return ref * (1.0 - step) ** k


def grid_sell_prices(ref, grid_step, max_k):
    hi = max_k if max_k is not None else 50
    return [grid_sell_price(ref, k, grid_step) for k in range(1, hi + 1)]


def grid_buy_prices(ref, step, max_k, defensive=False):
    _ = defensive
    hi = max_k if max_k is not None else 50
    return [grid_buy_price(ref, k, step) for k in range(1, hi + 1)]


def should_enter_defensive(rolling_peak_close, close, threshold):
    """峰值回撤 (peak - close) / peak >= threshold。"""
    if rolling_peak_close <= 0:
        return False
    return (rolling_peak_close - close) / rolling_peak_close >= threshold


def should_cancel_pair_after_sell_close(close, ref, grid_step, last_pair_k, max_grid_level):
    """卖后等回补时：若收盘已上破「下一档」卖价，解除回补义务（避免单边上涨永久锁死网格）。"""
    if last_pair_k is None:
        return False
    hi = max_grid_level if max_grid_level is not None else 50
    nk = last_pair_k + 1
    if nk > hi:
        return False
    return close >= grid_sell_price(ref, nk, grid_step)


def should_cancel_pair_after_buy_close(close, ref, buy_step, last_pair_k, max_grid_level):
    """买后等卖出时：若收盘已跌破「下一档」买价，解除卖出义务（避免单边下跌永久锁死）。"""
    if last_pair_k is None:
        return False
    hi = max_grid_level if max_grid_level is not None else 50
    nk = last_pair_k + 1
    if nk > hi:
        return False
    return close <= grid_buy_price(ref, nk, buy_step)


def pick_grid_action_close(
    close,
    ref,
    grid_step,
    buy_step,
    round_active,
    last_pair_side,
    last_pair_k,
    max_grid_level,
    last_open_sell_k,
    last_grid_sell_price,
    allow_neutral_sell=True,
    suspend_peak_close=None,
):
    """
    至多一笔：('buy', k) / ('sell', k) / None。
    last_open_sell_k：中性新卖单从 k>floor 起算。
    last_grid_sell_price：最近一笔网格卖出的收盘价；卖后等回补时除 ref 几何买档外，
    还允许 close <= last_grid_sell_price*(1-buy_step)（相对卖价回撤，否则牛市常年无买单）。
    allow_neutral_sell：False 时不从「中性」min_k_sell 开卖，用于活仓已清零后的暂停，
    否则 min_k_sell 恒优先于 trail_buy，会把回补信号挡掉。
    suspend_peak_close：暂停网格期间跟踪的收盘新高；当 close <= 峰值*(1-buy_step) 也触发回补，
    避免牛市长期不破「末笔卖价」导致数年无买。
    """
    hi = max_grid_level if max_grid_level is not None else 50
    floor_s = last_open_sell_k if last_open_sell_k is not None else 0

    if round_active and last_pair_side == 'sell' and last_pair_k is not None:
        hit_ref = close <= grid_buy_price(ref, last_pair_k, buy_step)
        trail_ls = (
            last_grid_sell_price is not None
            and last_grid_sell_price > 0
            and close <= last_grid_sell_price * (1.0 - buy_step)
        )
        trail_pk = (
            suspend_peak_close is not None
            and suspend_peak_close > 0
            and close <= suspend_peak_close * (1.0 - buy_step)
        )
        if hit_ref or trail_ls or trail_pk:
            return ('buy', last_pair_k)
        return None

    if round_active and last_pair_side == 'buy' and last_pair_k is not None:
        if close >= grid_sell_price(ref, last_pair_k, grid_step):
            return ('sell', last_pair_k)
        return None

    min_k_sell = None
    for k in range(floor_s + 1, hi + 1):
        if close >= grid_sell_price(ref, k, grid_step):
            min_k_sell = k
            break

    min_k_buy = None
    for k in range(1, hi + 1):
        if close <= grid_buy_price(ref, k, buy_step):
            min_k_buy = k
            break

    trail_ls = (
        last_grid_sell_price is not None
        and last_grid_sell_price > 0
        and close <= last_grid_sell_price * (1.0 - buy_step)
    )
    trail_pk = (
        suspend_peak_close is not None
        and suspend_peak_close > 0
        and close <= suspend_peak_close * (1.0 - buy_step)
    )
    trail_buy = trail_ls or trail_pk

    if not allow_neutral_sell:
        min_k_sell = None

    if min_k_sell is not None and min_k_buy is not None:
        return ('sell', min_k_sell)
    if min_k_sell is not None:
        return ('sell', min_k_sell)
    if min_k_buy is not None:
        return ('buy', min_k_buy)
    if trail_buy:
        return ('buy', 1)
    return None


# ---------------------------------------------------------------------------
# 策略生命周期
# ---------------------------------------------------------------------------


def initialize(context):
    set_benchmark('000300.SS')

    context.symbol = '510300.SS'
    # 趋势为主时略留现金，降低满仓波动；网格打开时可再提高。
    context.initial_position_ratio = 0.94
    context.core_ratio = 1.0
    context.grid_step = 0.028
    context.grid_lot = 1000
    context.defensive_trigger_drawdown = 0.14
    context.defensive_buy_step = 0.045
    context.take_profit_ret = 9.99
    context.max_grid_level = None

    # regime_mode='trend_sizing': 相对慢均线的连续仓位（非全进全出，减轻踏空与深套）
    context.regime_mode = 'trend_sizing'
    context.ma_slow_days = 120
    context.ma_min_days = 20
    context.ts_z_cut = -0.078
    context.ts_w_floor = 0.52
    context.ts_w_cap = 0.99
    context.ts_w_mid = 0.68
    context.ts_w_slope = 3.5
    context.ts_rebalance_band = 0.026
    context.ts_peak_dd_cut = 0.10
    context.ts_w_peak_stress = 0.32

    context.ref = None
    context.core_shares_intended = 0
    context.grid_shares = 0
    context.round_active = False
    context.last_pair_side = None
    context.last_pair_k = None
    context.defensive = False
    context.built = False
    context.done = False
    context.grid_suspended = False
    context.rolling_peak_close = None
    context.last_open_sell_k = 0
    context.last_grid_sell_price = None
    context.suspend_peak_close = None

    context.eod_closes = []
    context.eod_last_d = None
    context.ts_last_w = None
    context.ts_peak_close = None

    log.info('core_grid_hybrid_v1 初始化 symbol=%s', context.symbol)


def _floor_lot(n, lot=100):
    return int(math.floor(n / lot)) * lot


def _ma_tail_mean(hist, n, min_n=20):
    """尾部 n 日均值；历史不足 n 时用已有长度（至少 min_n）以免长年空舱。"""
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
    if context.eod_last_d == d:
        return
    context.eod_last_d = d
    context.eod_closes.append(close)
    if len(context.eod_closes) > 450:
        context.eod_closes = context.eod_closes[-450:]


def _handle_trend_sizing(context, sym, close):
    hist_before = list(context.eod_closes)
    slow_n = int(context.ma_slow_days)
    min_n = int(getattr(context, 'ma_min_days', 20))
    ma = _ma_tail_mean(hist_before, slow_n, min_n=min_n)
    _append_eod_close(context, close)
    if ma is None or ma <= 0:
        return
    z = (close - ma) / ma
    zc = float(context.ts_z_cut)
    wf = float(context.ts_w_floor)
    wc = float(context.ts_w_cap)
    wm = float(context.ts_w_mid)
    ws = float(context.ts_w_slope)
    if z < zc:
        w = 0.0
    else:
        w = wm + z * ws
        if w < wf:
            w = wf
        if w > wc:
            w = wc
    pk = context.ts_peak_close
    if pk is None or close > pk:
        pk = close
    context.ts_peak_close = pk
    if pk > 0:
        pdd = (pk - close) / pk
        pc = float(context.ts_peak_dd_cut)
        if pdd > pc:
            w = min(w, float(context.ts_w_peak_stress))
    pv = float(context.portfolio.portfolio_value)
    if pv <= 0:
        return
    pos = context.portfolio.positions.get(sym)
    phys = float(pos.amount) if pos is not None else 0.0
    cur_stock = phys * close
    cur_w = cur_stock / pv
    band = float(getattr(context, 'ts_rebalance_band', 0.0))
    if band > 0.0 and abs(w - cur_w) < band:
        _try_take_profit(context, sym)
        return
    order_target_value(sym, pv * w)
    lw = context.ts_last_w
    if lw is None or abs(w - lw) >= 0.04:
        log.info('TGT_W z=%.4f w=%.3f pv=%.0f', z, w, pv)
        context.ts_last_w = w
    _try_take_profit(context, sym)


def _try_take_profit(context, sym):
    """整体止盈：达到 take_profit_ret 则清仓；若已无仓则标记 done。"""
    ret_end = context.portfolio.returns
    if ret_end < context.take_profit_ret:
        return
    log.info('EXIT_ALL 触发 ret=%.4f >= %.4f', ret_end, context.take_profit_ret)
    order_target(sym, 0)
    pos_after = context.portfolio.positions.get(sym)
    if pos_after is None or pos_after.amount <= 0:
        context.done = True


def handle_data(context, data):
    sym = context.symbol
    if context.done:
        return

    try:
        bar = data[sym]
    except Exception:
        return

    close = float(bar.close)
    if close != close or close <= 0:  # NaN
        return

    if context.regime_mode == 'trend_sizing':
        _handle_trend_sizing(context, sym, close)
        return

    log.warning('未知 regime_mode=%s，跳过', getattr(context, 'regime_mode', None))
