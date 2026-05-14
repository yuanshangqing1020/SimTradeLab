# strategies/core_grid_hybrid_v1/backtest.py
# -*- coding: utf-8 -*-
"""
核心仓 + 网格混合策略 v1.0 — SimTradeLab 回测

规格：my_docs/core_grid_hybrid/v1.0/01-design.md
实施：my_docs/core_grid_hybrid/v1.0/02-plan.md

网格基准：固定锚定价 ref（方案 A）。日线收盘触发，每 bar 最多一笔网格。
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


def pick_grid_action_close(
    close,
    ref,
    grid_step,
    buy_step,
    round_active,
    last_pair_side,
    last_pair_k,
    max_grid_level,
):
    """
    在单根 K 线收盘价下，按 02-plan §3～§6 至多产生一笔网格意向：('buy', k) / ('sell', k) / None。
    buy_step：非防御 = grid_step；防御 = defensive_buy_step。
    """
    hi = max_grid_level if max_grid_level is not None else 50

    if round_active and last_pair_side == 'sell' and last_pair_k is not None:
        if close <= grid_buy_price(ref, last_pair_k, buy_step):
            return ('buy', last_pair_k)
        return None

    if round_active and last_pair_side == 'buy' and last_pair_k is not None:
        if close >= grid_sell_price(ref, last_pair_k, grid_step):
            return ('sell', last_pair_k)
        return None

    min_k_sell = None
    for k in range(1, hi + 1):
        if close >= grid_sell_price(ref, k, grid_step):
            min_k_sell = k
            break

    min_k_buy = None
    for k in range(1, hi + 1):
        if close <= grid_buy_price(ref, k, buy_step):
            min_k_buy = k
            break

    if min_k_sell is not None and min_k_buy is not None:
        return ('sell', min_k_sell)
    if min_k_sell is not None:
        return ('sell', min_k_sell)
    if min_k_buy is not None:
        return ('buy', min_k_buy)
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
    context.take_profit_ret = 0.20
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

    log.info('core_grid_hybrid_v1 初始化 symbol=%s', context.symbol)


def _floor_lot(n: float, lot: int = 100) -> int:
    return int(math.floor(n / lot)) * lot


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

    # 整体止盈（用组合总收益率 = 总资产相对初始资金）
    ret = context.portfolio.returns
    if ret >= context.take_profit_ret:
        log.info('EXIT_ALL 触发 ret=%.4f >= %.4f', ret, context.take_profit_ret)
        order_target(sym, 0)
        pos_after = context.portfolio.positions.get(sym)
        if pos_after is None or pos_after.amount <= 0:
            context.done = True
        return

    if context.grid_suspended:
        return

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

    action = pick_grid_action_close(
        close=close,
        ref=context.ref,
        grid_step=context.grid_step,
        buy_step=buy_step,
        round_active=context.round_active,
        last_pair_side=context.last_pair_side,
        last_pair_k=context.last_pair_k,
        max_grid_level=context.max_grid_level,
    )

    if action is None:
        return

    side, k = action
    pos = context.portfolio.positions.get(sym)
    enable = int(pos.enable_amount) if pos is not None else 0
    physical = int(pos.amount) if pos is not None else 0

    if side == 'sell':
        cap = min(context.grid_shares, context.grid_lot, enable, physical)
        cap = _floor_lot(cap)
        if cap < 100:
            return
        oid = order(sym, -cap)
        if oid is None:
            return
        context.grid_shares -= cap
        context.round_active = not context.round_active
        if context.round_active:
            context.last_pair_side = 'sell'
            context.last_pair_k = k
        else:
            context.last_pair_side = None
            context.last_pair_k = None
        log.info('GRID sell k=%d qty=%d close=%.4f round_active=%s', k, cap, close, context.round_active)
        if context.grid_shares == 0:
            context.grid_suspended = True
            log.info('GRID_SUSPENDED_UP 活仓已空')
        return

    # buy
    lot = _floor_lot(context.grid_lot)
    if lot < 100:
        return
    cost_upper = lot * close * 1.01
    if context.portfolio.cash < cost_upper:
        log.info('现金不足跳过买入 k=%d', k)
        return
    oid = order(sym, lot)
    if oid is None:
        return

    context.grid_shares += lot
    context.round_active = not context.round_active
    if context.round_active:
        context.last_pair_side = 'buy'
        context.last_pair_k = k
    else:
        context.last_pair_side = None
        context.last_pair_k = None
    log.info('GRID buy k=%d qty=%d close=%.4f round_active=%s', k, lot, close, context.round_active)
