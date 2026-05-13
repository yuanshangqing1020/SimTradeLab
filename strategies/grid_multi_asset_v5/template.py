# strategies/grid_multi_asset_v5/template.py
# -*- coding: utf-8 -*-
"""
多标的自适应网格策略 v5

在 v2 内核之上：
  - 默认 ANCHOR_SATELLITE：锚定 ETF + 卫星 ETF 合并池，换仓时锚定优先入池
  - 可选 WIDE_V2：成分股 + 全 ETF 母本（与 v2 一致）
  - 大盘 regime：WEEKLY 时每 5 个交易日刷新 invested_ratio；换股仍仅 REBALANCE_FREQ

参数：由 optimization/optimize_params.py Walk-Forward 注入
"""
import numpy as np

# 全市场 ETF 母本（WIDE_V2 与 v2 一致）
CANDIDATE_ETFS = [
    '510300.SS', '510500.SS', '159915.SZ', '512880.SS', '512690.SS',
    '512010.SS', '515050.SS', '512480.SS', '159949.SZ', '588000.SS',
    '512170.SS', '512760.SS', '159792.SZ', '513100.SS', '513050.SS',
]

# 锚定：贴近沪深300 / 宽基（换仓时优先保留）
ANCHOR_ETF_UNIVERSE = [
    '510300.SS',  # 沪深300ETF
    '510500.SS',  # 中证500ETF
]
_ANCHOR_SET = frozenset(ANCHOR_ETF_UNIVERSE)

# v4 窄池六只（历史对照；卫星构建时仍会用到其中非锚定标的）
NARROW_ETF_UNIVERSE = [
    '510300.SS',  # 沪深300ETF
    '510500.SS',  # 中证500ETF
    '159915.SZ',  # 创业板ETF
    '512010.SS',  # 医药ETF
    '513100.SS',  # 纳指ETF
    '588000.SS',  # 科创50ETF
]


def _build_satellite_etf_universe():
    """NARROW ∪ CANDIDATE 去锚定、保序去重，供 ANCHOR_SATELLITE 卫星腿。"""
    out = []
    seen = set()
    for x in list(NARROW_ETF_UNIVERSE) + list(CANDIDATE_ETFS):
        if x in _ANCHOR_SET or x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


SATELLITE_ETF_UNIVERSE = _build_satellite_etf_universe()

NARROW_ETF_POOL_SIZE = len(NARROW_ETF_UNIVERSE)

TARGET_CAPITAL = 500000.0


def _combined_etf_universe_for_mode(universe_mode):
    if universe_mode == 'WIDE_V2':
        return list(CANDIDATE_ETFS)
    merged = []
    seen = set()
    for x in list(ANCHOR_ETF_UNIVERSE) + list(SATELLITE_ETF_UNIVERSE):
        if x not in seen:
            merged.append(x)
            seen.add(x)
    return merged


V5_COMBINED_POOL_SIZE = len(_combined_etf_universe_for_mode('ANCHOR_SATELLITE'))


def _regime_refresh_day(day_counter, rebalance_freq, regime_refresh):
    """是否在本交易日刷新大盘 regime（进而更新 invested_ratio）。"""
    is_rebalance = (day_counter == 1) or (day_counter % rebalance_freq == 0)
    if regime_refresh == 'ON_REBALANCE_ONLY':
        return is_rebalance
    # WEEKLY：换仓日一定刷新；其余每 5 个交易日刷新一次
    return is_rebalance or ((day_counter - 1) % 5 == 0)


def initialize(context):
    set_benchmark('000300.SS')
    set_slippage(slippage=0.00246)

    # ── WF 最优（Trial 185，见 results/best_params_20260513_120438.json）── #
    context.MAX_HOLD             = 6
    context.GRID_STEP_VOL_FACTOR = 0.6
    context.GRID_STEP_MIN        = 0.02
    context.GRID_STEP_MAX        = 0.05
    context.GRID_MAX_LAYER       = 2
    context.LAYER_FRACTION       = 0.08
    context.VOL_WEIGHT           = 0.80
    context.REBALANCE_FREQ       = 20

    context.BULL_RATIO    = 0.70
    context.NEUTRAL_RATIO = 0.50
    context.BEAR_RATIO    = 0.45

    # ANCHOR_SATELLITE | WIDE_V2 | NARROW_ETF（窄池 6 只，与 v4 对齐仅作对照）
    context.UNIVERSE_MODE = 'ANCHOR_SATELLITE'
    context.MIN_ANCHORS_IN_POOL = 1
    # WEEKLY | ON_REBALANCE_ONLY
    context.REGIME_REFRESH = 'WEEKLY'

    context.pool           = []
    context.day_counter    = 0
    context.regime         = 'NEUTRAL'
    context.invested_ratio = context.NEUTRAL_RATIO


def handle_data(context, data):
    context.day_counter += 1
    dc = context.day_counter
    rf = context.REBALANCE_FREQ
    if _regime_refresh_day(dc, rf, context.REGIME_REFRESH):
        _detect_regime(context)
    if dc == 1 or dc % rf == 0:
        _refresh_pool(context)
    _execute_grid(context)


def after_trading_end(context, data):
    held = sum(1 for p in context.portfolio.positions.values() if p.amount > 0)
    log.info('日终 | %s | 投入%.0f%% | %s | 总资产: %.0f | 网格池: %d只 | 持仓: %d只 | 现金: %.0f' % (
        context.regime,
        context.invested_ratio * 100,
        context.UNIVERSE_MODE,
        context.portfolio.portfolio_value,
        len(context.pool),
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


def _score_universe(vol_dict, fund_df, etf_codes, vol_weight):
    import pandas as pd

    if not vol_dict:
        return []

    etf_set = set(etf_codes)
    records = []

    stock_codes = [c for c in vol_dict if c not in etf_set]
    if stock_codes and fund_df is not None and len(fund_df) > 0:
        if 'code' in fund_df.columns:
            fd = fund_df.set_index('code')
        else:
            fd = fund_df
        for code in stock_codes:
            if code not in vol_dict or code not in fd.index:
                continue
            row = fd.loc[code]
            pe   = float(row['pe_ttm'])    if 'pe_ttm'    in fd.columns else None
            roe  = float(row['roe'])        if 'roe'        in fd.columns else 0.0
            mcap = float(row['total_value']) if 'total_value' in fd.columns else None
            if pe is None or mcap is None:
                continue
            if not (np.isfinite(pe) and 0 < pe < 120):
                continue
            if not (np.isfinite(mcap) and mcap >= 3e9):
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


def build_grid_pool_anchor_first(ranked_codes, anchor_codes, max_hold):
    """ranked_codes: 分数从高到低；anchor_codes: 锚定顺序（优先级递减）。
    先按顺序放入「出现在 ranked_codes 中」的锚定（不超过 max_hold），
    再按 ranked_codes 顺序补足名额。"""
    ranked = list(ranked_codes)
    in_ranked = set(ranked)
    pool = []
    for c in anchor_codes:
        if c in in_ranked and len(pool) < max_hold:
            pool.append(c)
    for c in ranked:
        if len(pool) >= max_hold:
            break
        if c not in pool:
            pool.append(c)
    return pool


def _etf_list_for_mode(universe_mode):
    if universe_mode == 'WIDE_V2':
        return list(CANDIDATE_ETFS)
    if universe_mode == 'NARROW_ETF':
        return list(NARROW_ETF_UNIVERSE)
    return _combined_etf_universe_for_mode('ANCHOR_SATELLITE')


def _max_hold_cap(universe_mode):
    return len(_etf_list_for_mode(universe_mode))


def _detect_regime(context):
    try:
        hist = get_history(260, '1d', 'close', ['000300.SS'])
        prices = hist['000300.SS'].dropna().values
    except Exception as exc:
        log.warning('_detect_regime get_history 失败: %s，保持当前状态' % str(exc))
        return

    context.regime = _calc_regime(prices)
    ratio_map = {
        'BULL':    context.BULL_RATIO,
        'NEUTRAL': context.NEUTRAL_RATIO,
        'BEAR':    context.BEAR_RATIO,
    }
    context.invested_ratio = ratio_map[context.regime]
    log.info('大盘状态: %s | 投入比例: %.0f%%' % (
        context.regime, context.invested_ratio * 100))


def _refresh_pool(context):
    etfs = _etf_list_for_mode(context.UNIVERSE_MODE)
    stocks = []
    if context.UNIVERSE_MODE == 'WIDE_V2':
        stocks = list(set(
            get_index_stocks('000300.SS') + get_index_stocks('000905.SS')
        ))
    all_cands = stocks + etfs

    if not all_cands:
        log.warning('候选池为空，保留原池')
        return

    st_map   = get_stock_status(all_cands, 'ST')
    halt_map = get_stock_status(all_cands, 'HALT')
    stocks = [s for s in stocks
              if not st_map.get(s, False) and not halt_map.get(s, False)]
    etfs = [e for e in etfs   # noqa: redefined
            if not st_map.get(e, False) and not halt_map.get(e, False)]

    if not stocks and not etfs:
        log.warning('ST/停牌过滤后候选池为空，保留原池')
        return

    fund_df = None
    if stocks:
        try:
            raw = get_fundamentals(stocks, 'valuation', ['pe_ttm', 'total_value', 'roe'])
            if raw is not None and len(raw) > 0:
                if 'code' not in raw.columns and raw.index.name == 'code':
                    raw = raw.reset_index()
                raw = raw.dropna(subset=['pe_ttm', 'total_value'])
                raw = raw[(raw['pe_ttm'] > 0) & (raw['pe_ttm'] < 120)]
                raw = raw[raw['total_value'] >= 3e9]
                if 'code' in raw.columns:
                    stocks = [s for s in stocks if s in raw['code'].values]
                fund_df = raw
        except Exception as exc:
            log.warning('get_fundamentals 失败: %s，跳过基本面过滤' % str(exc))

    all_active = stocks + etfs
    if not all_active:
        log.warning('有效候选池为空，保留原池')
        return

    cap = _max_hold_cap(context.UNIVERSE_MODE)
    max_hold = min(int(context.MAX_HOLD), cap)

    vol_dict = {}
    try:
        hist = get_history(26, '1d', 'close', all_active)
        if hist is not None and len(hist) > 0:
            for code in all_active:
                if code not in hist.columns:
                    continue
                prices = hist[code].dropna().values
                v = _calc_vol_from_prices(prices)
                if v is not None:
                    vol_dict[code] = v
    except Exception as exc:
        log.warning('get_history(vol) 失败: %s' % str(exc))

    if not vol_dict:
        log.warning('波动率计算全部失败，保留原池')
        return

    ranked_pairs = _score_universe(vol_dict, fund_df, etfs, context.VOL_WEIGHT)
    ranked_codes = [code for code, _ in ranked_pairs]

    if context.UNIVERSE_MODE == 'WIDE_V2':
        new_pool = [code for code, _ in ranked_pairs[:max_hold]]
    elif context.UNIVERSE_MODE == 'ANCHOR_SATELLITE':
        new_pool = build_grid_pool_anchor_first(
            ranked_codes, ANCHOR_ETF_UNIVERSE, max_hold,
        )
        min_a = int(getattr(context, 'MIN_ANCHORS_IN_POOL', 1))
        tradable_anchor = [c for c in ANCHOR_ETF_UNIVERSE if c in vol_dict]
        n_anchor_in = sum(1 for c in new_pool if c in _ANCHOR_SET)
        if len(tradable_anchor) >= min_a and n_anchor_in < min_a:
            log.warning(
                '锚定不足: 目标至少 %d 只锚定，实际入池 %d — 保留原池' % (min_a, n_anchor_in),
            )
            return
    else:
        new_pool = [code for code, _ in ranked_pairs[:max_hold]]

    old_set = set(context.pool)
    new_set = set(new_pool)
    for code in old_set - new_set:
        order_target(code, 0)
        log.info('调出网格池: %s' % code)

    context.pool = new_pool
    log.info('网格池更新 %d只: %s%s' % (
        len(context.pool),
        ','.join(context.pool[:5]),
        '...' if len(context.pool) > 5 else '',
    ))


def _execute_grid(context):
    if not context.pool:
        return

    N = len(context.pool)

    try:
        hist = get_history(31, '1d', 'close', context.pool)
    except Exception as exc:
        log.warning('_execute_grid get_history 失败: %s' % str(exc))
        return

    layers = []
    active = []

    for code in context.pool:
        if hist is None or code not in hist.columns:
            continue
        prices = hist[code].dropna().values
        if len(prices) < 22:
            continue
        price = float(prices[-1])
        if not (np.isfinite(price) and price > 0):
            continue
        ma20  = float(prices[-20:].mean())
        vol   = _calc_vol_from_prices(prices)
        if vol is None:
            continue
        step = float(np.clip(
            vol * context.GRID_STEP_VOL_FACTOR,
            context.GRID_STEP_MIN,
            context.GRID_STEP_MAX,
        ))
        layer = _calc_layer(price, ma20, step, context.GRID_MAX_LAYER)
        layers.append(layer)
        active.append(code)

    if not active:
        return

    raw_w  = [max((1.0 / N) * (1.0 + context.LAYER_FRACTION * float(lyr)), 1e-9)
              for lyr in layers]
    norm_w = _normalize_weights(raw_w)

    max_w = (1.0 / N) * (1.0 + context.LAYER_FRACTION * context.GRID_MAX_LAYER)
    norm_w = _apply_weight_cap(norm_w, max_w)

    tv  = context.portfolio.portfolio_value
    cap = tv * context.invested_ratio
    cap = min(cap, TARGET_CAPITAL)

    for code, w in zip(active, norm_w):
        order_target_value(code, cap * w)
