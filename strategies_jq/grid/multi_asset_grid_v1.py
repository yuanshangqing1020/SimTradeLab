# -*- coding: utf-8 -*-
"""
多标的自适应网格策略 v1.0（JoinQuant 版）

移植自 SimTradeLab strategies/grid_multi_asset_best/backtest.py
Walk-Forward 调参结果（2019-2024）：
  Holdout 2025-2026：年化 +11.99%，夏普 0.70，最大回撤 -15.21%

参数说明（已注入最优值，可在 initialize 中手动修改）：
  MAX_HOLD             = 50    最多持仓标的数
  GRID_STEP_VOL_FACTOR = 0.60  步长 = clip(vol × factor, min, max)
  GRID_STEP_MIN        = 0.01  步长下限 1%
  GRID_STEP_MAX        = 0.05  步长上限 5%
  GRID_MAX_LAYER       = 3     最大偏离层数
  LAYER_FRACTION       = 0.08  层间权重增量（保守）
  VOL_WEIGHT           = 0.80  波动率在综合打分中的权重
  REBALANCE_FREQ       = 20    换股间隔（交易日，约月频）

PTrade → JoinQuant 主要差异：
  代码格式：.SS/.SZ → .XSHG/.XSHE
  全局变量：context.xxx → g.xxx
  行情数据：get_history() → history(df=True)
  基本面：get_fundamentals(list,...) → get_fundamentals(query(...))
  ST/停牌：get_stock_status() → get_current_data()[code].is_st / .paused
  市值单位：JoinQuant market_cap 单位为亿元（SimTradeLab 为元）
  定时执行：handle_data → run_daily(func, time='14:50')
  持仓字段：p.amount → p.total_amount
  组合字段：portfolio.portfolio_value → portfolio.total_value
"""
import numpy as np
import pandas as pd

# ── ETF 候选池（JoinQuant 代码格式）──────────────────────────────────────── #
CANDIDATE_ETFS = [
    '510300.XSHG',  # 沪深300ETF
    '510500.XSHG',  # 中证500ETF
    '159915.XSHE',  # 创业板ETF
    '512880.XSHG',  # 证券ETF
    '512690.XSHG',  # 酒ETF
    '512010.XSHG',  # 医疗ETF
    '515050.XSHG',  # 5G ETF
    '512480.XSHG',  # 半导体ETF
    '159949.XSHE',  # 创业板50ETF
    '588000.XSHG',  # 科创50ETF
    '512170.XSHG',  # 医疗ETF（华宝）
    '512760.XSHG',  # 芯片ETF
    '159792.XSHE',  # 恒生科技ETF（QDII，可能有数据限制）
    '513100.XSHG',  # 纳指ETF（QDII）
    '513050.XSHG',  # 中概互联ETF（QDII）
]

TARGET_CAPITAL = 500000.0  # 策略目标资金规模（网格分配上限）


# ════════════════════════════════════════════════════════════════════════════ #
#  生命周期函数
# ════════════════════════════════════════════════════════════════════════════ #

def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)          # 使用真实价格，避免未来函数偏差

    # 手续费：股票万三 + 印花税千一（卖出）
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5,
    ), type='stock')
    # 基金/ETF 手续费
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0,
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5,
    ), type='fund')

    set_slippage(PriceRelatedSlippage(0.00246))  # 双边滑点 0.246%

    # ── 策略参数（Walk-Forward 最优值）──────────────────────────────────── #
    g.MAX_HOLD             = 50
    g.GRID_STEP_VOL_FACTOR = 0.60
    g.GRID_STEP_MIN        = 0.01
    g.GRID_STEP_MAX        = 0.05
    g.GRID_MAX_LAYER       = 3
    g.LAYER_FRACTION       = 0.08
    g.VOL_WEIGHT           = 0.80
    g.REBALANCE_FREQ       = 20

    # ── 运行时状态 ──────────────────────────────────────────────────────── #
    g.pool        = []
    g.day_counter = 0

    run_daily(trade, time='14:50')  # 收盘前10分钟执行


def trade(context):
    """每日 14:50 执行：按频率换股 + 执行网格。"""
    g.day_counter += 1
    if g.day_counter == 1 or g.day_counter % g.REBALANCE_FREQ == 0:
        _refresh_pool(context)
    _execute_grid(context)


def after_trading_end(context, data):
    held = sum(1 for p in context.portfolio.positions.values() if p.total_amount > 0)
    log.info('日终 | 总资产: %.0f | 网格池: %d只 | 持仓: %d只 | 现金: %.0f' % (
        context.portfolio.total_value,
        len(g.pool),
        held,
        context.portfolio.cash,
    ))


# ════════════════════════════════════════════════════════════════════════════ #
#  纯数学函数（逻辑与 SimTradeLab 版完全一致）
# ════════════════════════════════════════════════════════════════════════════ #

def _calc_vol_from_prices(prices):
    """年化已实现波动率（20日收益率标准差 × sqrt(250)）。"""
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
    """网格层数：正数=价格低于中枢（超配），负数=价格高于中枢（欠配）。"""
    raw = (ma20 - price) / (price * step)
    return int(np.clip(int(np.floor(raw + 0.5)), -max_layer, max_layer))


def _normalize_weights(raw_weights):
    """权重归一化到 sum=1，若总和<=0 则等权。"""
    if not raw_weights:
        return []
    total = sum(raw_weights)
    if total <= 0:
        n = len(raw_weights)
        return [1.0 / n] * n
    return [w / total for w in raw_weights]


def _score_universe(vol_dict, fund_df, etf_codes, vol_weight):
    """综合打分，返回按得分降序排列的 [(code, score), ...]。

    stock 得分 = vol_pct × vol_weight + qual_pct × (1 - vol_weight)
    qual_pct   = ROE_pct×0.45 + 1/PE_pct×0.35 + mcap_pct×0.20
    ETF 得分   = vol_pct × vol_weight + 0.5 × (1 - vol_weight)
    """
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
            row   = fd.loc[code]
            pe    = float(row['pe_ratio'])   if 'pe_ratio'   in fd.columns else None
            roe   = float(row['roe'])        if 'roe'        in fd.columns else 0.0
            # JoinQuant market_cap 单位：亿元；过滤 >= 30 亿（等价 3e9 元）
            mcap  = float(row['market_cap']) if 'market_cap' in fd.columns else None
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
    df['vol_pct']  = df['vol'].rank(pct=True)
    df['qual_pct'] = 0.5  # ETF 默认中位数

    stock_mask = df['kind'] == 'stock'
    if stock_mask.any():
        stk = df[stock_mask]
        df.loc[stock_mask, 'qual_pct'] = (
            stk['roe'].rank(pct=True)    * 0.45
            + stk['inv_pe'].rank(pct=True) * 0.35
            + stk['mcap'].rank(pct=True)   * 0.20
        )

    df['score'] = df['vol_pct'] * vol_weight + df['qual_pct'] * (1.0 - vol_weight)
    df = df.sort_values('score', ascending=False)
    return list(zip(df['code'], df['score']))


# ════════════════════════════════════════════════════════════════════════════ #
#  API 依赖函数
# ════════════════════════════════════════════════════════════════════════════ #

def _refresh_pool(context):
    """重新选股：动态指数成分 + ETF，综合打分取 Top-MAX_HOLD。"""
    # 1. 候选列表
    stocks = list(set(
        get_index_stocks('000300.XSHG') + get_index_stocks('000905.XSHG')
    ))
    etfs = list(CANDIDATE_ETFS)

    if not stocks and not etfs:
        log.warning('候选池为空，保留原池')
        return

    # 2. 过滤 ST / 停牌（JQ：get_current_data()）
    current_data = get_current_data()
    stocks = [s for s in stocks
              if not current_data[s].paused and not current_data[s].is_st]
    etfs   = [e for e in etfs if not current_data[e].paused]

    if not stocks and not etfs:
        log.warning('ST/停牌过滤后候选池为空，保留原池')
        return

    # 3. 基本面数据（JQ：query 方式）
    fund_df = None
    if stocks:
        try:
            q = query(
                valuation.code,
                valuation.pe_ratio,
                valuation.market_cap,   # 单位：亿元
                indicator.roe           # 净资产收益率（%，如 15 表示 15%）
            ).filter(valuation.code.in_(stocks))
            raw = get_fundamentals(q)
            if raw is not None and len(raw) > 0:
                raw = raw.dropna(subset=['pe_ratio', 'market_cap'])
                raw = raw[(raw['pe_ratio'] > 0) & (raw['pe_ratio'] < 120)]
                raw = raw[raw['market_cap'] >= 30]   # 30亿元以上
                stocks  = [s for s in stocks if s in raw['code'].values]
                fund_df = raw
        except Exception as exc:
            log.warning('get_fundamentals 失败: %s，跳过基本面过滤' % str(exc))

    # 4. 计算波动率（JQ：history(df=True) 返回 DataFrame，行=日期，列=code）
    all_active = stocks + etfs
    if not all_active:
        log.warning('有效候选池为空，保留原池')
        return

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

    # 5. 综合打分，取 Top-N
    ranked   = _score_universe(vol_dict, fund_df, etfs, g.VOL_WEIGHT)
    new_pool = [code for code, _ in ranked[:g.MAX_HOLD]]

    # 6. 清仓调出标的
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
    """计算各标的网格层数，归一化权重后 order_target_value。"""
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
        ma20  = float(prices[-20:].mean())
        vol   = _calc_vol_from_prices(prices)
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

    raw_w  = [max((1.0 / N) * (1.0 + g.LAYER_FRACTION * float(lyr)), 1e-9)
              for lyr in layers]
    norm_w = _normalize_weights(raw_w)

    tv  = context.portfolio.total_value
    cap = min(tv, max(TARGET_CAPITAL, 1000.0))
    for code, w in zip(active, norm_w):
        order_target_value(code, cap * w)
