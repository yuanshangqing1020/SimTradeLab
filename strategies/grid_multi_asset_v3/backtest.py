# strategies/grid_multi_asset_v3/backtest.py
# -*- coding: utf-8 -*-
"""
多标的自适应网格策略 v3 — 直接回测版

参数：**Walk-Forward 最优 Trial 357** · `optimization/results/best_params_20260510_232314.json` · `optimization/optimized_strategy.py`  
初始资金 50 万；入口：`run_backtest.py` · `strategy_name='grid_multi_asset_v3'`。
"""
import numpy as np

# ── ETF 候选池（固定，与 v1 相同）────────────────────────────────────────────── #
CANDIDATE_ETFS = [
    '510300.SS', '510500.SS', '159915.SZ', '512880.SS', '512690.SS',
    '512010.SS', '515050.SS', '512480.SS', '159949.SZ', '588000.SS',
    '512170.SS', '512760.SS', '159792.SZ', '513100.SS', '513050.SS',
]

# BEAR + ETF_DEFENSIVE 时使用（CANDIDATE_ETFS 子集：宽基/核心行业）
DEFENSIVE_ETF_POOL = [
    '510300.SS',
    '510500.SS',
    '159915.SZ',
    '588000.SS',
    '512880.SS',
]

TARGET_CAPITAL = 500000.0  # 策略目标资金规模（绝对上限）

# NO_NET_ADD：仅当标的昨日日终持仓市值 > EPS 时才限制不得高于昨收持仓
_NET_ADD_EPS_POSITION_VALUE = 1e-6


def initialize(context):
    set_benchmark('000300.SS')
    set_slippage(slippage=0.00246)

    # ── WF 最优基线（Trial 357，见 results/best_params_20260510_232314.json）── #
    context.MAX_HOLD             = 12    # 最多持仓标的数
    context.GRID_STEP_VOL_FACTOR = 0.45  # 步长 = clip(vol * factor, min, max)
    context.GRID_STEP_MIN        = 0.01  # 步长下限
    context.GRID_STEP_MAX        = 0.05  # 步长上限
    context.GRID_MAX_LAYER       = 2     # 最大偏离层数
    context.LAYER_FRACTION       = 0.08  # 每层权重增减幅度
    context.VOL_WEIGHT           = 0.65  # 波动率在综合打分中的权重
    context.REBALANCE_FREQ       = 10    # 重新选股间隔（交易日）

    # ── v2 新增参数 ──────────────────────────────────────────────────────── #
    context.BULL_RATIO    = 0.70  # 牛市总投入比例
    context.NEUTRAL_RATIO = 0.50  # 震荡总投入比例
    context.BEAR_RATIO    = 0.45  # 熊市总投入比例

    # ── v3 新增（optimizer 可调） ─────────────────────────────────────── #
    context.BEAR_UNIVERSE_MODE      = 'SAME'       # SAME | ETF_DEFENSIVE
    context.BEAR_GRID_MODE          = 'CAP_LAYER'  # NORMAL | NO_NET_ADD | CAP_LAYER
    context.BEAR_GRID_MAX_LAYER_CAP = 0

    # ── 运行时状态 ──────────────────────────────────────────────────────── #
    context.pool           = []         # 当前活跃网格池
    context.day_counter    = 0          # 交易日计数器
    context.regime         = 'NEUTRAL'  # 大盘状态
    context.invested_ratio = context.NEUTRAL_RATIO  # 当前投入比例
    context._prev_eod_position_value = {}  # 上一交易日收盘持仓市值快照，供 NO_NET_ADD


def handle_data(context, data):
    context.day_counter += 1
    if context.day_counter == 1 or context.day_counter % context.REBALANCE_FREQ == 0:
        _detect_regime(context)   # 先判断大盘趋势
        _refresh_pool(context)    # 再选股换仓
    _execute_grid(context)


def after_trading_end(context, data):
    held = sum(1 for p in context.portfolio.positions.values() if p.amount > 0)
    log.info('日终 | %s | 投入%.0f%% | 总资产: %.0f | 网格池: %d只 | 持仓: %d只 | 现金: %.0f' % (
        context.regime,
        context.invested_ratio * 100,
        context.portfolio.portfolio_value,
        len(context.pool),
        held,
        context.portfolio.cash,
    ))
    # 下一交易日 _execute_grid 使用的「昨收持仓市值」（无前视）
    snap = {}
    positions = getattr(context.portfolio, 'positions', {}) or {}
    for code, pos in positions.items():
        amt = getattr(pos, 'amount', 0) or 0
        if amt <= 0:
            continue
        mv = getattr(pos, 'market_value', None)
        if mv is None or not (isinstance(mv, (int, float)) and np.isfinite(float(mv))):
            lp = getattr(pos, 'last_sale_price', 0)
            mv = float(amt) * float(lp if lp else 0.0)
        snap[code] = float(mv)
    context._prev_eod_position_value = snap


# ── 纯数学函数（无 PTrade 依赖，可单元测试）─────────────────────────────────── #

def _calc_vol_from_prices(prices):
    """计算年化已实现波动率。（与 v1 完全相同）"""
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
    """计算当前网格层数。（与 v1 完全相同）"""
    raw = (ma20 - price) / (price * step)
    return int(np.clip(int(np.floor(raw + 0.5)), -max_layer, max_layer))


def _normalize_weights(raw_weights):
    """将原始权重列表归一化到 sum=1。（与 v1 完全相同）"""
    if not raw_weights:
        return []
    total = sum(raw_weights)
    if total <= 0:
        n = len(raw_weights)
        return [1.0 / n] * n
    return [w / total for w in raw_weights]


def _calc_regime(prices):
    """【v2 新增】纯数学：根据沪深300价格序列判断大盘趋势状态。

    Input:  numpy array，建议至少 250 根 K 线
    Output: 'BULL' / 'NEUTRAL' / 'BEAR'

    规则：
      价格 > MA120 且 > MA250 → BULL
      价格 < MA120 且 < MA250 → BEAR
      其他（含数据不足）       → NEUTRAL
    """
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
    """【v2 新增】Water-filling 算法：截断超额权重，将多余部分按比例分配给未超额标的。

    Input:  norm_w    - list of floats（归一化权重，sum = 1.0）
            max_w     - float，单标的权重上限
            iterations - 最大迭代次数
    Output: list of floats（sum = 1.0，典型情况下每项 ≤ max_w）

    收敛保证：当 N * max_w ≥ 1.0（生产场景中恒成立，因 max_w ≥ 1/N）时，
    每次迭代后所有曾被截断的权重恰好等于 max_w，且不超额，通常 1 次迭代即收敛。
    """
    result = list(norm_w)
    for _ in range(iterations):
        capped_idx = [i for i, w in enumerate(result) if w > max_w + 1e-12]
        if not capped_idx:
            break  # 已收敛，提前退出
        excess = sum(result[i] - max_w for i in capped_idx)
        uncapped_idx = [i for i, w in enumerate(result) if w < max_w - 1e-12]
        uncapped_total = sum(result[i] for i in uncapped_idx)
        if uncapped_total < 1e-12:
            # 退化情形：全部权重超 cap（N*max_w < 1.0），等比例截断后归一化
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


def _effective_max_layer_bear(regime, bear_grid_mode, grid_max_layer, bear_cap):
    """BEAR + CAP_LAYER 时裁减网格层上限；否则沿用 grid_max_layer。"""
    if regime != 'BEAR' or bear_grid_mode != 'CAP_LAYER':
        return int(grid_max_layer)
    return int(min(int(grid_max_layer), int(bear_cap)))


def _etf_list_for_refresh(regime, bear_universe_mode, candidate_etfs, defensive_etfs):
    """M2：BEAR + ETF_DEFENSIVE 时仅返回防御 ETF 列表。"""
    if regime == 'BEAR' and bear_universe_mode == 'ETF_DEFENSIVE':
        return list(defensive_etfs)
    return list(candidate_etfs)


def _apply_no_net_add_targets(prev_by_code, target_by_code):
    """NO_NET_ADD：若昨日该标的有仓，则当日目标市值不得高于昨收持仓市值。"""
    out = {}
    prev_by_code = prev_by_code or {}
    for code, tgt in target_by_code.items():
        t = float(tgt)
        if not np.isfinite(t):
            out[code] = 0.0
            continue
        prev = prev_by_code.get(code)
        pv = float(prev) if prev is not None and np.isfinite(prev) else 0.0
        if pv > _NET_ADD_EPS_POSITION_VALUE:
            out[code] = min(t, pv)
        else:
            out[code] = t
    return out


def _score_universe(vol_dict, fund_df, etf_codes, vol_weight):
    """综合打分。（与 v1 完全相同）"""
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


# ── API 依赖函数（需在 SimTradeLab 运行环境中执行）────────────────────────────── #

def _detect_regime(context):
    """【v2 新增】拉取沪深300历史，调用 _calc_regime，更新 context。
    仅在换股日调用，避免每日重复拉 260 根 K 线。
    """
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
    """重新选股。v3：BEAR + ETF_DEFENSIVE 时不用指数成分股，仅用防御 ETF 列表。"""
    etfs = _etf_list_for_refresh(
        context.regime,
        context.BEAR_UNIVERSE_MODE,
        CANDIDATE_ETFS,
        DEFENSIVE_ETF_POOL,
    )
    if context.regime == 'BEAR' and context.BEAR_UNIVERSE_MODE == 'ETF_DEFENSIVE':
        stocks = []
    else:
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
    etfs = [e for e in etfs
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

    ranked = _score_universe(vol_dict, fund_df, etfs, context.VOL_WEIGHT)
    new_pool = [code for code, _ in ranked[:context.MAX_HOLD]]

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
    """每日收盘前执行网格：v2 invested_ratio + 权重上限；v3 M1 CAP_LAYER / NO_NET_ADD。"""
    if not context.pool:
        return

    N = len(context.pool)

    eff_layer = _effective_max_layer_bear(
        context.regime,
        context.BEAR_GRID_MODE,
        context.GRID_MAX_LAYER,
        getattr(context, 'BEAR_GRID_MAX_LAYER_CAP', 0),
    )
    layer_for_cap = (
        eff_layer
        if (context.regime == 'BEAR' and context.BEAR_GRID_MODE == 'CAP_LAYER')
        else context.GRID_MAX_LAYER
    )

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
        layer = _calc_layer(price, ma20, step, eff_layer)
        layers.append(layer)
        active.append(code)

    if not active:
        return

    # 原始权重（网格层数加权）
    raw_w  = [max((1.0 / N) * (1.0 + context.LAYER_FRACTION * float(lyr)), 1e-9)
              for lyr in layers]
    norm_w = _normalize_weights(raw_w)

    max_w = (1.0 / N) * (1.0 + context.LAYER_FRACTION * layer_for_cap)
    norm_w = _apply_weight_cap(norm_w, max_w)

    tv  = context.portfolio.portfolio_value
    cap = tv * context.invested_ratio
    cap = min(cap, TARGET_CAPITAL)

    target_by_code = {code: cap * w for code, w in zip(active, norm_w)}

    if context.regime == 'BEAR' and context.BEAR_GRID_MODE == 'NO_NET_ADD':
        prev = getattr(context, '_prev_eod_position_value', {}) or {}
        target_by_code = _apply_no_net_add_targets(prev, target_by_code)

    for code, val in target_by_code.items():
        order_target_value(code, val)
