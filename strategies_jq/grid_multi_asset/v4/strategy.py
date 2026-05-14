# -*- coding: utf-8 -*-
"""
多标的自适应网格策略 v4.0（JoinQuant 版）

移植自 SimTradeLab strategies/grid_multi_asset_v4/template.py / backtest.py。
Walk-Forward 最优参数 **Trial 185**（`best_params_20260513_120438.json`，与 SimTradeLab 一致）。

v4 相对 v2：
  - **窄 ETF 池** `NARROW_ETF_UNIVERSE`（默认 6 只）或 **全量 ETF** `WIDE_V2`（`g.UNIVERSE_MODE`）
  - **周频 regime**：`g.REGIME_REFRESH=WEEKLY` 时，换仓日或每 **5** 个交易日刷新大盘状态与 `g.invested_ratio`；换股仍按 `REBALANCE_FREQ`

PTrade → JoinQuant 映射同 `v1`/`v2` 文件头说明。
"""
import numpy as np
import pandas as pd

CANDIDATE_ETFS = [
    '510300.XSHG', '510500.XSHG', '159915.XSHE', '512880.XSHG', '512690.XSHG',
    '512010.XSHG', '515050.XSHG', '512480.XSHG', '159949.XSHE', '588000.XSHG',
    '512170.XSHG', '512760.XSHG', '159792.XSHE', '513100.XSHG', '513050.XSHG',
]

NARROW_ETF_UNIVERSE = [
    '510300.XSHG',
    '510500.XSHG',
    '159915.XSHE',
    '512010.XSHG',
    '513100.XSHG',
    '588000.XSHG',
]

TARGET_CAPITAL = 500000.0
BENCHMARK_CODE = '000300.XSHG'


def _regime_refresh_day(day_counter, rebalance_freq, regime_refresh):
    is_rebalance = (day_counter == 1) or (day_counter % rebalance_freq == 0)
    if regime_refresh == 'ON_REBALANCE_ONLY':
        return is_rebalance
    return is_rebalance or ((day_counter - 1) % 5 == 0)


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

    # Walk-Forward Trial 185（与 SimTradeLab best_params_20260513_120438.json 一致）
    g.MAX_HOLD             = 6
    g.GRID_STEP_VOL_FACTOR = 0.6
    g.GRID_STEP_MIN        = 0.02
    g.GRID_STEP_MAX        = 0.05
    g.GRID_MAX_LAYER       = 2
    g.LAYER_FRACTION       = 0.08
    g.VOL_WEIGHT           = 0.80
    g.REBALANCE_FREQ       = 20
    g.BULL_RATIO           = 0.70
    g.NEUTRAL_RATIO        = 0.50
    g.BEAR_RATIO           = 0.45

    g.UNIVERSE_MODE   = 'NARROW_ETF'
    g.REGIME_REFRESH  = 'WEEKLY'

    g.pool           = []
    g.day_counter    = 0
    g.regime         = 'NEUTRAL'
    g.invested_ratio = g.NEUTRAL_RATIO

    run_daily(trade, time='14:50')


def trade(context):
    g.day_counter += 1
    dc = g.day_counter
    rf = g.REBALANCE_FREQ
    if _regime_refresh_day(dc, rf, g.REGIME_REFRESH):
        _detect_regime(context)
    if dc == 1 or dc % rf == 0:
        _refresh_pool(context)
    _execute_grid(context)


def after_trading_end(context):
    held = sum(1 for p in context.portfolio.positions.values() if p.total_amount > 0)
    log.info('日终 | %s | 投入%.0f%% | %s | 总资产: %.0f | 网格池: %d只 | 持仓: %d只 | 现金: %.0f' % (
        g.regime,
        g.invested_ratio * 100,
        g.UNIVERSE_MODE,
        context.portfolio.total_value,
        len(g.pool),
        held,
        context.portfolio.cash,
    ))


def _calc_vol_from_prices(prices):
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 22:
        return None
    rets = np.diff(arr) / arr[:-1]
    valid = rets[np.isfinite(rets)]
    if len(valid) < 20:
        return None
    vol = float(valid[-20:].std() * np.sqrt(250.0))
    return vol if vol > 0 else None


def _calc_layer(price, ma20, step, max_layer):
    raw = (ma20 - price) / (price * step)
    return int(np.clip(int(np.floor(raw + 0.5)), -max_layer, max_layer))


def _normalize_weights(raw_weights):
    if not raw_weights:
        return []
    total = sum(raw_weights)
    if total <= 0:
        n = len(raw_weights)
        return [1.0 / n] * n
    return [w / total for w in raw_weights]


def _calc_regime(prices):
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 250:
        return 'NEUTRAL'
    price_now = arr[-1]
    ma120 = arr[-120:].mean()
    ma250 = arr[-250:].mean()
    above_120 = price_now > ma120
    above_250 = price_now > ma250
    if above_120 and above_250:
        return 'BULL'
    if (not above_120) and (not above_250):
        return 'BEAR'
    return 'NEUTRAL'


def _apply_weight_cap(norm_w, max_w, iterations=3):
    result = list(norm_w)
    for _ in range(iterations):
        capped_idx = [i for i, w in enumerate(result) if w > max_w + 1e-12]
        if not capped_idx:
            break
        excess = sum(result[i] - max_w for i in capped_idx)
        uncapped_idx = [i for i, w in enumerate(result) if w < max_w - 1e-12]
        uncapped_total = sum(result[i] for i in uncapped_idx)
        if uncapped_total < 1e-12:
            clipped = [min(w, max_w) for w in result]
            s = sum(clipped)
            result = [w / s for w in clipped] if s > 0 else clipped
            break
        new_result = list(result)
        for i in capped_idx:
            new_result[i] = max_w
        for i in uncapped_idx:
            new_result[i] = result[i] * (1.0 + excess / uncapped_total)
        result = new_result
    return result


def _etf_list_for_mode(universe_mode):
    if universe_mode == 'WIDE_V2':
        return list(CANDIDATE_ETFS)
    return list(NARROW_ETF_UNIVERSE)


def _max_hold_cap(universe_mode):
    return len(_etf_list_for_mode(universe_mode))


def _score_universe(vol_dict, fund_df, etf_codes, vol_weight):
    if not vol_dict:
        return []

    etf_set = set(etf_codes)
    records = []

    stock_codes = [c for c in vol_dict if c not in etf_set]
    if stock_codes and fund_df is not None and len(fund_df) > 0:
        fd = fund_df.set_index('code') if 'code' in fund_df.columns else fund_df
        for code in stock_codes:
            if code not in vol_dict or code not in fd.index:
                continue
            row = fd.loc[code]
            pe = float(row['pe_ratio']) if 'pe_ratio' in fd.columns else None
            roe = float(row['roe']) if 'roe' in fd.columns else 0.0
            mcap = float(row['market_cap']) if 'market_cap' in fd.columns else None
            if pe is None or mcap is None:
                continue
            if not (np.isfinite(pe) and 0 < pe < 120):
                continue
            if not (np.isfinite(mcap) and mcap >= 30):
                continue
            roe = roe if np.isfinite(roe) else 0.0
            records.append({
                'code': code, 'kind': 'stock',
                'vol': vol_dict[code],
                'roe': roe,
                'inv_pe': 1.0 / max(pe, 1.0),
                'mcap': mcap,
            })

    for code in etf_codes:
        if code in vol_dict:
            records.append({
                'code': code, 'kind': 'etf',
                'vol': vol_dict[code],
                'roe': 0.0, 'inv_pe': 0.0, 'mcap': 0.0,
            })

    if not records:
        return []

    df = pd.DataFrame(records)
    df['vol_pct'] = df['vol'].rank(pct=True)
    df['qual_pct'] = 0.5

    stock_mask = df['kind'] == 'stock'
    if stock_mask.any():
        stk = df[stock_mask]
        df.loc[stock_mask, 'qual_pct'] = (
            stk['roe'].rank(pct=True) * 0.45
            + stk['inv_pe'].rank(pct=True) * 0.35
            + stk['mcap'].rank(pct=True) * 0.20
        )

    df['score'] = df['vol_pct'] * vol_weight + df['qual_pct'] * (1.0 - vol_weight)
    df = df.sort_values('score', ascending=False)
    return list(zip(df['code'], df['score']))


def _detect_regime(context):
    try:
        price_df = history(260, '1d', 'close', [BENCHMARK_CODE], df=True)
        if price_df is None or BENCHMARK_CODE not in price_df.columns:
            log.warning('_detect_regime: 无沪深300行情，保持当前状态')
            return
        prices = price_df[BENCHMARK_CODE].dropna().values
    except Exception as exc:
        log.warning('_detect_regime history 失败: %s，保持当前状态' % str(exc))
        return

    g.regime = _calc_regime(prices)
    ratio_map = {
        'BULL':    g.BULL_RATIO,
        'NEUTRAL': g.NEUTRAL_RATIO,
        'BEAR':    g.BEAR_RATIO,
    }
    g.invested_ratio = ratio_map[g.regime]
    log.info('大盘状态: %s | 投入比例: %.0f%%' % (g.regime, g.invested_ratio * 100))


def _refresh_pool(context):
    etfs = _etf_list_for_mode(g.UNIVERSE_MODE)
    stocks = []
    if g.UNIVERSE_MODE == 'WIDE_V2':
        stocks = list(set(
            get_index_stocks('000300.XSHG') + get_index_stocks('000905.XSHG')
        ))

    if not stocks and not etfs:
        log.warning('候选池为空，保留原池')
        return

    current_data = get_current_data()
    stocks = [s for s in stocks
              if not current_data[s].paused and not current_data[s].is_st]
    etfs = [e for e in etfs if not current_data[e].paused]

    if not stocks and not etfs:
        log.warning('ST/停牌过滤后候选池为空，保留原池')
        return

    fund_df = None
    if stocks:
        try:
            q = query(
                valuation.code,
                valuation.pe_ratio,
                valuation.market_cap,
                indicator.roe,
            ).filter(valuation.code.in_(stocks))
            raw = get_fundamentals(q)
            if raw is not None and len(raw) > 0:
                raw = raw.dropna(subset=['pe_ratio', 'market_cap'])
                raw = raw[(raw['pe_ratio'] > 0) & (raw['pe_ratio'] < 120)]
                raw = raw[raw['market_cap'] >= 30]
                stocks = [s for s in stocks if s in raw['code'].values]
                fund_df = raw
        except Exception as exc:
            log.warning('get_fundamentals 失败: %s，跳过基本面过滤' % str(exc))

    all_active = stocks + etfs
    if not all_active:
        log.warning('有效候选池为空，保留原池')
        return

    cap = _max_hold_cap(g.UNIVERSE_MODE)
    max_hold = min(int(g.MAX_HOLD), cap)

    vol_dict = {}
    try:
        price_df = history(26, '1d', 'close', all_active, df=True)
        if price_df is not None and len(price_df) > 0:
            for code in all_active:
                if code not in price_df.columns:
                    continue
                prices = price_df[code].dropna().values
                v = _calc_vol_from_prices(prices)
                if v is not None:
                    vol_dict[code] = v
    except Exception as exc:
        log.warning('history(vol) 失败: %s' % str(exc))

    if not vol_dict:
        log.warning('波动率计算全部失败，保留原池')
        return

    ranked = _score_universe(vol_dict, fund_df, etfs, g.VOL_WEIGHT)
    new_pool = [code for code, _ in ranked[:max_hold]]

    old_set = set(g.pool)
    new_set = set(new_pool)
    for code in old_set - new_set:
        order_target(code, 0)
        log.info('调出网格池: %s' % code)

    g.pool = new_pool
    log.info('网格池更新 %d只: %s%s' % (
        len(g.pool),
        ','.join(g.pool[:5]),
        '...' if len(g.pool) > 5 else '',
    ))


def _execute_grid(context):
    if not g.pool:
        return

    N = len(g.pool)

    try:
        price_df = history(31, '1d', 'close', g.pool, df=True)
    except Exception as exc:
        log.warning('_execute_grid history 失败: %s' % str(exc))
        return

    layers = []
    active = []

    for code in g.pool:
        if price_df is None or code not in price_df.columns:
            continue
        prices = price_df[code].dropna().values
        if len(prices) < 22:
            continue
        price = float(prices[-1])
        if not (np.isfinite(price) and price > 0):
            continue
        ma20 = float(prices[-20:].mean())
        vol = _calc_vol_from_prices(prices)
        if vol is None:
            continue
        step = float(np.clip(
            vol * g.GRID_STEP_VOL_FACTOR,
            g.GRID_STEP_MIN,
            g.GRID_STEP_MAX,
        ))
        layer = _calc_layer(price, ma20, step, g.GRID_MAX_LAYER)
        layers.append(layer)
        active.append(code)

    if not active:
        return

    raw_w = [max((1.0 / N) * (1.0 + g.LAYER_FRACTION * float(lyr)), 1e-9)
             for lyr in layers]
    norm_w = _normalize_weights(raw_w)

    max_w = (1.0 / N) * (1.0 + g.LAYER_FRACTION * g.GRID_MAX_LAYER)
    norm_w = _apply_weight_cap(norm_w, max_w)

    tv = context.portfolio.total_value
    cap = tv * g.invested_ratio
    cap = min(cap, TARGET_CAPITAL)

    for code, w in zip(active, norm_w):
        target_val = cap * w
        if target_val < 1e-6:
            continue
        px_series = price_df[code].dropna()
        if len(px_series) == 0:
            continue
        last_px = float(px_series.iloc[-1])
        if not (np.isfinite(last_px) and last_px > 0):
            continue
        if target_val < last_px * 100:
            log.debug('跳过 %s: 目标金额 %.0f < 1手金额 %.0f' % (code, target_val, last_px * 100))
            continue
        order_target_value(code, target_val)
