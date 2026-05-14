# strategies/core_grid_hybrid_v1/backtest.py
# -*- coding: utf-8 -*-
"""
核心仓 + 网格混合策略 v1.0 — SimTradeLab 回测

规格：my_docs/core_grid_hybrid/v1.0/01-design.md
实施：my_docs/core_grid_hybrid/v1.0/02-plan.md

网格基准：固定 ref 几何卖档（方案 A）+ 网格卖后回补：ref 买档 或「上一网格卖价」回撤 buy_step。
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
):
    """
    至多一笔：('buy', k) / ('sell', k) / None。
    last_open_sell_k：中性新卖单从 k>floor 起算。
    last_grid_sell_price：最近一笔网格卖出的收盘价；卖后等回补时除 ref 几何买档外，
    还允许 close <= last_grid_sell_price*(1-buy_step)（相对卖价回撤，否则牛市常年无买单）。
    """
    hi = max_grid_level if max_grid_level is not None else 50
    floor_s = last_open_sell_k if last_open_sell_k is not None else 0

    if round_active and last_pair_side == 'sell' and last_pair_k is not None:
        hit_ref = close <= grid_buy_price(ref, last_pair_k, buy_step)
        hit_trail = (
            last_grid_sell_price is not None
            and last_grid_sell_price > 0
            and close <= last_grid_sell_price * (1.0 - buy_step)
        )
        if hit_ref or hit_trail:
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

    trail_buy = (
        last_grid_sell_price is not None
        and last_grid_sell_price > 0
        and close <= last_grid_sell_price * (1.0 - buy_step)
    )

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
    context.initial_position_ratio = 0.5
    context.core_ratio = 0.5
    context.grid_step = 0.03
    context.grid_lot = 1000
    context.defensive_trigger_drawdown = 0.15
    context.defensive_buy_step = 0.05
    context.take_profit_ret = 0.35
    context.max_grid_level = None

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

    log.info('core_grid_hybrid_v1 初始化 symbol=%s', context.symbol)


def _floor_lot(n, lot=100):
    return int(math.floor(n / lot)) * lot


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

    if not context.built:
        cash0 = context.portfolio.starting_cash
        deployable = cash0 * context.initial_position_ratio
        target_shares = _floor_lot(deployable / close)
        if target_shares < 100:
            log.warning('建仓预算不足一手，跳过 symbol=%s', sym)
            context.built = True
            return

        core_alloc = _floor_lot(target_shares * context.core_ratio)
        grid_alloc = target_shares - core_alloc
        if grid_alloc < 0:
            grid_alloc = 0
            core_alloc = target_shares

        log.info(
            'INITIAL_BUILD 买 %s 股价≈%.4f deployable=%.2f target=%d core=%d grid=%d',
            sym,
            close,
            deployable,
            target_shares,
            core_alloc,
            grid_alloc,
        )
        order(sym, target_shares)

        context.ref = close
        context.core_shares_intended = core_alloc
        context.grid_shares = grid_alloc
        context.rolling_peak_close = close
        context.built = True
        return

    # 峰值
    if context.rolling_peak_close is not None:
        context.rolling_peak_close = max(context.rolling_peak_close, close)

    if not context.defensive and context.rolling_peak_close is not None:
        if should_enter_defensive(
            context.rolling_peak_close,
            close,
            context.defensive_trigger_drawdown,
        ):
            context.defensive = True
            log.info('进入 DEFENSIVE_DOWN')

    buy_step = context.defensive_buy_step if context.defensive else context.grid_step
    if context.ref is None:
        return

    # 解除 stale 配对（不比 02-plan 冲突：属固定锚定下「趋势踏空」处理，否则长年无网格）
    if context.round_active and context.last_pair_side == 'sell' and context.last_pair_k is not None:
        if should_cancel_pair_after_sell_close(
            close, context.ref, context.grid_step, context.last_pair_k, context.max_grid_level
        ):
            nk = context.last_pair_k + 1
            log.info(
                'PAIR_CANCEL 收盘上破下一卖档(k=%d)，解除对 k=%d 的回补等待',
                nk,
                context.last_pair_k,
            )
            context.last_open_sell_k = context.last_pair_k
            context.round_active = False
            context.last_pair_side = None
            context.last_pair_k = None
    if context.round_active and context.last_pair_side == 'buy' and context.last_pair_k is not None:
        if should_cancel_pair_after_buy_close(
            close, context.ref, buy_step, context.last_pair_k, context.max_grid_level
        ):
            nk = context.last_pair_k + 1
            log.info(
                'PAIR_CANCEL 收盘跌破下一买档(k=%d)，解除对 k=%d 的反弹卖出等待',
                nk,
                context.last_pair_k,
            )
            context.round_active = False
            context.last_pair_side = None
            context.last_pair_k = None

    if context.grid_suspended:
        _try_take_profit(context, sym)
        return

    action = pick_grid_action_close(
        close=close,
        ref=context.ref,
        grid_step=context.grid_step,
        buy_step=buy_step,
        round_active=context.round_active,
        last_pair_side=context.last_pair_side,
        last_pair_k=context.last_pair_k,
        max_grid_level=context.max_grid_level,
        last_open_sell_k=context.last_open_sell_k,
        last_grid_sell_price=context.last_grid_sell_price,
    )

    if action is None:
        _try_take_profit(context, sym)
        return

    side, k = action
    pos = context.portfolio.positions.get(sym)
    enable = int(pos.enable_amount) if pos is not None else 0
    physical = int(pos.amount) if pos is not None else 0

    if side == 'sell':
        cap = min(context.grid_shares, context.grid_lot, enable, physical)
        cap = _floor_lot(cap)
        if cap < 100:
            _try_take_profit(context, sym)
            return
        oid = order(sym, -cap)
        if oid is None:
            _try_take_profit(context, sym)
            return
        context.grid_shares -= cap
        context.round_active = not context.round_active
        if context.round_active:
            context.last_pair_side = 'sell'
            context.last_pair_k = k
            context.last_open_sell_k = k
        else:
            context.last_pair_side = None
            context.last_pair_k = None
            context.last_open_sell_k = k
        if context.grid_shares == 0:
            context.grid_suspended = True
            context.round_active = False
            context.last_pair_side = None
            context.last_pair_k = None
            context.last_grid_sell_price = None
            log.info('GRID sell k=%d qty=%d close=%.4f -> GRID_SUSPENDED_UP（清零在途配对）', k, cap, close)
        else:
            context.last_grid_sell_price = close
            log.info('GRID sell k=%d qty=%d close=%.4f round_active=%s', k, cap, close, context.round_active)
        _try_take_profit(context, sym)
        return

    # buy
    lot = _floor_lot(context.grid_lot)
    if lot < 100:
        _try_take_profit(context, sym)
        return
    cost_upper = lot * close * 1.01
    if context.portfolio.cash < cost_upper:
        log.info('现金不足跳过买入 k=%d', k)
        _try_take_profit(context, sym)
        return
    oid = order(sym, lot)
    if oid is None:
        _try_take_profit(context, sym)
        return

    context.grid_shares += lot
    context.round_active = not context.round_active
    context.last_grid_sell_price = None
    if context.round_active:
        context.last_pair_side = 'buy'
        context.last_pair_k = k
    else:
        context.last_pair_side = None
        context.last_pair_k = None
        context.last_open_sell_k = 0
    log.info('GRID buy k=%d qty=%d close=%.4f round_active=%s', k, lot, close, context.round_active)

    _try_take_profit(context, sym)
