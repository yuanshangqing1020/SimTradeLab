# strategies/grid_multi_asset/backtest.py
# -*- coding: utf-8 -*-
"""
多标的自适应网格策略

- 资金规模: 50万（TARGET_CAPITAL 软约束上限）
- 持仓数量: 10~50只（由 context.MAX_HOLD 控制）
- 标的: 沪深300+中证500 动态成分股 + 固定 ETF 候选池
- 网格步长: clip(vol * FACTOR, MIN, MAX)，默认区间 1%~4%（优化候选含 5%）
- 参数: 由 optimization/optimize_params.py Walk-Forward 自动调参
"""
import numpy as np

# ── ETF 候选池（固定）─────────────────────────────────────────────────────── #
CANDIDATE_ETFS = [
    '510300.SS', '510500.SS', '159915.SZ', '512880.SS', '512690.SS',
    '512010.SS', '515050.SS', '512480.SS', '159949.SZ', '588000.SS',
    '512170.SS', '512760.SS', '159792.SZ', '513100.SS', '513050.SS',
]

TARGET_CAPITAL = 500000.0  # 策略目标资金规模（网格分配上限）


def initialize(context):
    set_benchmark('000300.SS')
    set_slippage(slippage=0.00246)

    # ── 可调参数（optimizer 通过 context.* regex 注入）────────────────── #
    context.MAX_HOLD             = 20    # 最多持仓标的数
    context.GRID_STEP_VOL_FACTOR = 0.45  # 步长 = clip(vol * factor, min, max)
    context.GRID_STEP_MIN        = 0.01  # 步长下限 1%
    context.GRID_STEP_MAX        = 0.04  # 步长上限 4%
    context.GRID_MAX_LAYER       = 3     # 最大偏离层数
    context.LAYER_FRACTION       = 0.12  # 每层权重增减幅度 ±12%
    context.VOL_WEIGHT           = 0.62  # 波动率在综合打分中的权重
    context.REBALANCE_FREQ       = 5     # 重新选股间隔（交易日）

    # ── 运行时状态 ──────────────────────────────────────────────────────── #
    context.pool        = []  # 当前活跃网格池（股票代码列表）
    context.day_counter = 0   # 交易日计数器


def handle_data(context, data):
    # 假定日频回测（handle_data 每交易日调用一次）
    context.day_counter += 1
    if context.day_counter == 1 or context.day_counter % context.REBALANCE_FREQ == 0:
        _refresh_pool(context)
    _execute_grid(context)


def after_trading_end(context, data):
    held = sum(1 for p in context.portfolio.positions.values() if p.amount > 0)
    log.info('日终 | 总资产: %.0f | 网格池: %d只 | 持仓: %d只 | 现金: %.0f' % (
        context.portfolio.portfolio_value,
        len(context.pool),
        held,
        context.portfolio.cash,
    ))


# ── 纯数学函数（无 PTrade 依赖，可单元测试）─────────────────────────────────── #

def _calc_vol_from_prices(prices):
    """计算年化已实现波动率。
    Input:  numpy 数组，至少 22 根 K 线（1根用于第一个 ret）
    Output: float（年化 vol）或 None（数据不足/vol 为零）
    """
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
    """计算当前网格层数。
    layer > 0: 价格低于中枢（超配信号）
    layer < 0: 价格高于中枢（欠配信号）
    layer = 0: 价格在中枢附近
    """
    raw = (ma20 - price) / (price * step)
    return int(np.clip(int(np.floor(raw + 0.5)), -max_layer, max_layer))


def _normalize_weights(raw_weights):
    """将原始权重列表归一化到 sum=1。
    若总和 <= 0，则返回等权；输入为空则返回空列表。
    """
    if not raw_weights:
        return []
    total = sum(raw_weights)
    if total <= 0:
        n = len(raw_weights)
        return [1.0 / n] * n
    return [w / total for w in raw_weights]


def _score_universe(vol_dict, fund_df, etf_codes, vol_weight):
    """综合打分，返回按得分降序排列的 [(code, score), ...]。

    vol_dict:   {code: annualized_vol}
    fund_df:    DataFrame with columns ['code','pe_ttm','total_value','roe']，可为 None
    etf_codes:  ETF 代码列表（只用 vol，无基本面）
    vol_weight: 波动率在总分中的权重（0~1）
    """
    import pandas as pd

    if not vol_dict:
        return []

    etf_set = set(etf_codes)
    records = []

    # 股票侧：波动率 + 基本面
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
            pe   = float(row['pe_ttm'])   if 'pe_ttm'   in fd.columns else None
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

    # ETF 侧：只用波动率
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
    df['qual_pct'] = 0.5  # ETF 默认值

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

def _refresh_pool(context):
    """重新选股：从动态指数成分+ETF中，按综合打分选 Top-MAX_HOLD。"""
    # 1. 获取候选列表
    stocks = list(set(
        get_index_stocks('000300.SS') + get_index_stocks('000905.SS')
    ))
    etfs = list(CANDIDATE_ETFS)
    all_cands = stocks + etfs

    if not all_cands:
        log.warning('候选池为空，保留原池')
        return

    # 2. 过滤 ST / 停牌
    st_map   = get_stock_status(all_cands, 'ST')
    halt_map = get_stock_status(all_cands, 'HALT')
    stocks = [s for s in stocks
              if not st_map.get(s, False) and not halt_map.get(s, False)]
    etfs = [e for e in etfs
            if not st_map.get(e, False) and not halt_map.get(e, False)]

    if not stocks and not etfs:
        log.warning('ST/停牌过滤后候选池为空，保留原池')
        return

    # 3. 基本面数据（股票侧）
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

    # 4. 计算波动率（批量）
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

    # 5. 综合打分，取 Top-N
    ranked = _score_universe(vol_dict, fund_df, etfs, context.VOL_WEIGHT)
    new_pool = [code for code, _ in ranked[:context.MAX_HOLD]]

    # 6. 清仓已调出标的
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
    """每日收盘前：计算各标的网格层数，归一化权重后 order_target_value。"""
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
        step  = float(np.clip(
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

    tv  = context.portfolio.portfolio_value
    cap = min(tv, max(TARGET_CAPITAL, 1000.0))
    for code, w in zip(active, norm_w):
        order_target_value(code, cap * w)
