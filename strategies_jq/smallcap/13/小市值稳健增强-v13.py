# -*- coding: utf-8 -*-
# 小市值稳健增强 v13（相对 v12 的优化方向，对应回测图末期大回撤与后期波动放大）
# 1) 组合净值高点回撤熔断：超过阈值则清仓并转入货基 511880，待 399101 站上 MA20 后再恢复股票仓。
# 2) 指数趋势：昨收低于 MA20 时减少持股数量（集中降风险），与原有 MA10 差值动态仓叠加取下限。
# 3) ATR 移动止损：持仓跟踪最高价，跌破「高点 − k×ATR」则卖出单票（目录内 ATR 思路）。
# 4) 周调仓后按目标列表等权分配总权益（减少权重漂移）。
# 5) 保留：399101 池、国九、季节空仓、涨停持有、资金流与质量动量低波打分（波动项略加重）。
#
# 参数请按回测区间在聚宽内微调；货基需为可交易代码。

from jqdata import *
from jqdata import finance
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta


def initialize(context):
    set_option('avoid_future_data', True)
    set_benchmark('399101.XSHE')
    set_option('use_real_price', True)
    set_slippage(FixedSlippage(3 / 10000))
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0.001,
            open_commission=2.5 / 10000,
            close_commission=2.5 / 10000,
            close_today_commission=0,
            min_commission=5,
        ),
        type='stock',
    )
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'info')

    g.trading_signal = True
    g.run_stoploss = True
    g.filter_audit = False
    g.adjust_num = True

    g.hold_list = []
    g.yesterday_HL_list = []
    g.target_list = []
    g.limitup_stocks = []

    g.pass_periods = [
        ('01-01', '01-31'),
        ('04-01', '04-30'),
        ('12-15', '12-31'),
    ]

    g.min_mv = 0
    g.max_mv = 1e10
    g.stock_num = 4
    g.candidate_pool = 100
    g.reason_to_sell = ''
    g.stoploss_strategy = 3
    g.stoploss_limit = 0.08
    g.stoploss_market = 0.94
    g.lowest = 1.0
    g.highest = 80.0

    g.use_breadth_stop = True
    g.mf_weight = 0.25
    g.vol_score_weight = 1.15
    g._last_pass_state = None

    g.etf = '511880.XSHG'
    g.nav_peak = None
    g.dd_circuit_ratio = 0.16
    g.risk_circuit = False
    g.ma_trend_days = 20
    g.ma_weak_reduce = 1

    g.atr_period = 14
    g.atr_mult = 2.3
    g.stock_high = {}

    run_daily(prepare_stock_list, '9:05')
    run_daily(emergency_risk_flatten, '9:36')
    run_weekly(weekly_sell_task, 3, '10:30')
    run_weekly(weekly_buy_task, 3, '10:31')
    run_daily(trade_afternoon, time='14:20', reference_security='399101.XSHE')
    run_daily(sell_stocks, time='10:00')
    run_daily(sell_stocks, time='14:00')
    run_daily(close_account, '14:50')


def prepare_stock_list(context):
    g.hold_list = []
    g.limitup_stocks = []
    tv = context.portfolio.total_value
    if g.nav_peak is None:
        g.nav_peak = tv
    else:
        g.nav_peak = max(g.nav_peak, tv)
    dd = tv / max(g.nav_peak, 1e-9) - 1.0
    if (not g.risk_circuit) and dd <= -g.dd_circuit_ratio:
        g.risk_circuit = True
        log.info('组合回撤%.1f%%触发熔断，后续转防御直至指数站上MA%d' % (dd * 100, g.ma_trend_days))

    for position in list(context.portfolio.positions.values()):
        s = position.security
        g.hold_list.append(s)
        if s == g.etf:
            continue
        px = position.price
        g.stock_high[s] = max(g.stock_high.get(s, px), px)
    for s in list(g.stock_high.keys()):
        if s not in g.hold_list:
            del g.stock_high[s]

    if g.hold_list:
        df = get_price(
            g.hold_list,
            end_date=context.previous_date,
            frequency='daily',
            fields=['close', 'high_limit', 'low_limit'],
            count=1,
            panel=False,
            fill_paused=False,
        )
        df = df[df['close'] == df['high_limit']]
        g.yesterday_HL_list = list(df.code)
    else:
        g.yesterday_HL_list = []

    check_pass_period(context)
    try_exit_risk_circuit(context)


def emergency_risk_flatten(context):
    if not g.trading_signal or not g.risk_circuit:
        return
    for s, pos in list(context.portfolio.positions.items()):
        if s == g.etf:
            continue
        if pos and pos.closeable_amount > 0:
            order_target_value(s, 0)
    order_target_value(g.etf, context.portfolio.total_value * 0.98)


def try_exit_risk_circuit(context):
    if not g.risk_circuit:
        return
    try:
        idx = get_price(
            '399101.XSHE',
            end_date=context.previous_date,
            count=g.ma_trend_days + 5,
            frequency='daily',
            fields=['close'],
        )
        if idx is None or len(idx) < g.ma_trend_days + 1:
            return
        c = idx['close'].values
        ma = np.mean(c[-g.ma_trend_days :])
        if c[-1] > ma:
            g.risk_circuit = False
            g.nav_peak = context.portfolio.total_value
            log.info('指数站上MA%d，解除熔断' % g.ma_trend_days)
    except Exception:
        pass


def _date_in_pass_period(d):
    m, day = d.month, d.day
    for start_s, end_s in g.pass_periods:
        sm, sd = int(start_s[:2]), int(start_s[3:5])
        em, ed = int(end_s[:2]), int(end_s[3:5])
        if sm != em:
            continue
        if m == sm and sd <= day <= ed:
            return True
    return False


def check_pass_period(context):
    d = context.current_dt.date()
    is_pass = _date_in_pass_period(d)
    g.trading_signal = not is_pass
    prev = g._last_pass_state
    if prev is not None and prev != is_pass:
        if is_pass:
            log.info('进入空仓避险期: %s' % d)
        else:
            log.info('结束空仓避险期，恢复交易: %s' % d)
    g._last_pass_state = is_pass


def index_below_ma(context):
    try:
        idx = get_price(
            '399101.XSHE',
            end_date=context.previous_date,
            count=g.ma_trend_days + 2,
            frequency='daily',
            fields=['close'],
        )
        if idx is None or len(idx) < g.ma_trend_days + 1:
            return False
        c = idx['close'].values
        return c[-1] < np.mean(c[-g.ma_trend_days :])
    except Exception:
        return False


def adjust_stock_num(context):
    if not g.adjust_num:
        return g.stock_num
    ma_para = 10
    today = context.previous_date
    start_date = today - datetime.timedelta(days=ma_para * 3)
    index_df = get_price(
        '399101.XSHE', start_date=start_date, end_date=today, frequency='daily'
    )
    index_df['ma'] = index_df['close'].rolling(window=ma_para).mean()
    last_row = index_df.iloc[-1]
    diff = float(last_row['close'] - last_row['ma'])
    if diff >= 500:
        n = 3
    elif 200 <= diff < 500:
        n = 3
    elif -200 <= diff < 200:
        n = 4
    elif -500 <= diff < -200:
        n = 5
    else:
        n = 6
    if index_below_ma(context):
        n = max(2, n - g.ma_weak_reduce)
    return n


def _safe_z(s):
    s = pd.to_numeric(s, errors='coerce')
    if s.notna().sum() < 5:
        return pd.Series(0.0, index=s.index)
    mu = s.mean()
    sig = s.std()
    if sig == 0 or np.isnan(sig):
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sig).fillna(0.0)


def _attach_money_flow_score(codes, context):
    if not codes:
        return pd.Series(dtype=float)
    try:
        days = get_trade_days(end_date=context.previous_date, count=3)
        if len(days) < 2:
            return pd.Series(0.0, index=codes)
        mf_date = days[-2]
        parts = []
        for i in range(0, len(codes), 400):
            sub = codes[i : i + 400]
            df = get_money_flow(
                security_list=sub,
                end_date=mf_date,
                count=1,
                fields=['sec_code', 'net_amount_main'],
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                parts.append(df)
        if not parts:
            return pd.Series(0.0, index=codes)
        mf = pd.concat(parts, ignore_index=True).drop_duplicates(subset=['sec_code'])
        s = mf.set_index('sec_code')['net_amount_main'].reindex(codes).fillna(0.0)
        return _safe_z(s)
    except Exception:
        return pd.Series(0.0, index=codes)


def get_stock_list(context):
    MKT_index = '399101.XSHE'
    initial_list = filter_stocks(context, get_index_stocks(MKT_index))
    if not initial_list:
        return []

    q = query(
        valuation.code,
        valuation.market_cap,
        indicator.roe,
        income.np_parent_company_owners,
        income.net_profit,
        income.operating_revenue,
    ).filter(
        valuation.code.in_(initial_list),
        valuation.market_cap.between(g.min_mv, g.max_mv),
        income.np_parent_company_owners > 0,
        income.net_profit > 0,
        income.operating_revenue > 1e8,
    ).order_by(valuation.market_cap.asc()).limit(g.candidate_pool)

    df = get_fundamentals(q)
    if df is None or df.empty:
        return []
    if g.filter_audit:
        df = df[df['code'].apply(lambda x: filter_audit(context, x))]

    codes = list(df['code'])
    if not codes:
        return []

    px = get_price(
        codes,
        end_date=context.previous_date,
        frequency='daily',
        fields=['close'],
        count=25,
        panel=False,
        fill_paused=False,
    )
    if px is None or px.empty:
        return codes[: g.stock_num]

    mom = {}
    vol = {}
    for c in codes:
        sub = px[px['code'] == c].sort_values('time')
        closes = sub['close'].values.astype(float)
        if len(closes) < 21:
            mom[c] = 0.0
            vol[c] = 0.05
            continue
        r20 = closes[-2] / max(closes[-22], 1e-9) - 1.0
        rets = np.diff(closes[-21:]) / np.clip(closes[-21:-1], 1e-9, None)
        mom[c] = float(r20)
        vol[c] = float(np.std(rets)) if len(rets) > 1 else 0.05

    d = df.set_index('code')
    d['mom20'] = pd.Series(mom).reindex(d.index).fillna(0.0)
    d['vol20'] = pd.Series(vol).reindex(d.index).fillna(0.05)
    d['roe'] = pd.to_numeric(d['roe'], errors='coerce').fillna(0.0)

    z_roe = _safe_z(d['roe'])
    z_mom = _safe_z(d['mom20'])
    z_vol = _safe_z(d['vol20'])
    z_mf = _attach_money_flow_score(list(d.index), context).reindex(d.index).fillna(0.0)

    d['score'] = z_roe + z_mom - g.vol_score_weight * z_vol + g.mf_weight * z_mf
    d = d.sort_values(['score', 'market_cap'], ascending=[False, True])
    ranked = list(d.index)

    last_prices = history(1, unit='1d', field='close', security_list=ranked)
    out = []
    for c in ranked:
        if c in g.hold_list:
            out.append(c)
            continue
        try:
            p = last_prices[c][-1]
        except Exception:
            continue
        if g.lowest <= p <= g.highest:
            out.append(c)

    return filter_recent_extreme_movements(context, out)


def filter_recent_extreme_movements(context, stock_list):
    if not stock_list:
        return []
    end_date = context.previous_date
    df = get_price(
        stock_list,
        end_date=end_date,
        frequency='daily',
        fields=['close', 'high_limit', 'low_limit', 'volume'],
        count=3,
        panel=False,
        fill_paused=False,
    )
    exclude = set()
    for stock in stock_list:
        sd = df[df['code'] == stock]
        if len(sd) < 3:
            exclude.add(stock)
            continue
        has_up = (sd['close'] == sd['high_limit']).any()
        has_dn = (sd['close'] == sd['low_limit']).any()
        y = sd.iloc[-1]
        y_not_up = y['close'] != y['high_limit']
        if (has_up or has_dn) and y_not_up:
            exclude.add(stock)
    return [s for s in stock_list if s not in exclude]


def _atr_last(context, stock):
    try:
        bars = get_bars(
            stock,
            count=g.atr_period + 1,
            unit='1d',
            fields=['high', 'low', 'close'],
            include_now=False,
            df=True,
        )
        if bars is None or len(bars) < g.atr_period + 1:
            return None
        h, low, c = bars['high'], bars['low'], bars['close']
        pc = c.shift(1)
        tr = pd.concat([h - low, (h - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
        return float(tr.rolling(g.atr_period).mean().iloc[-1])
    except Exception:
        return None


def weekly_sell_task(context):
    check_pass_period(context)

    if not g.trading_signal:
        g.target_list = []
        for s in list(g.hold_list):
            pos = context.portfolio.positions.get(s)
            if pos:
                close_position(pos)
        return

    if g.risk_circuit:
        g.target_list = [g.etf]
        for s in list(g.hold_list):
            if s == g.etf:
                continue
            pos = context.portfolio.positions.get(s)
            if pos:
                close_position(pos)
        log.info('熔断期周调仓：保留/买入货基')
        return

    new_num = adjust_stock_num(context)
    if g.stock_num != new_num:
        g.stock_num = new_num
        log.info('持仓数量调整为 %s' % new_num)

    g.target_list = get_stock_list(context)[: g.stock_num]
    log.info('本周目标: %s' % g.target_list)

    sell_list = [
        s for s in g.hold_list if s not in g.target_list and s not in g.yesterday_HL_list
    ]
    for s in sell_list:
        pos = context.portfolio.positions.get(s)
        if pos:
            close_position(pos)


def weekly_buy_task(context):
    check_pass_period(context)
    if not g.trading_signal:
        return
    if g.risk_circuit:
        order_target_value(g.etf, context.portfolio.total_value * 0.98)
        return
    if not g.target_list:
        return
    rebalance_to_targets(context, g.target_list)


def rebalance_to_targets(context, target_list):
    target_list = [s for s in target_list if s and s != g.etf]
    if not target_list:
        return
    tv = context.portfolio.total_value
    w = 0.98 / len(target_list)
    for s in target_list:
        if s in g.yesterday_HL_list and s in context.portfolio.positions:
            continue
        order_target_value(s, tv * w)


def check_limit_up(context):
    now_time = context.current_dt
    if not g.yesterday_HL_list:
        return
    for stock in g.yesterday_HL_list:
        pos = context.portfolio.positions.get(stock)
        if not pos:
            continue
        cd = get_price(
            stock,
            end_date=now_time,
            frequency='1m',
            fields=['close', 'high_limit'],
            skip_paused=False,
            fq='pre',
            count=1,
            panel=False,
            fill_paused=True,
        )
        if cd is None or cd.empty:
            continue
        if cd.iloc[0, 0] < cd.iloc[0, 1]:
            log.info('涨停打开卖出 %s' % stock)
            close_position(pos)
            g.reason_to_sell = 'limitup'
            g.limitup_stocks.append(stock)


def check_remain_amount(context):
    if g.reason_to_sell == 'limitup':
        g.hold_list = [p.security for p in context.portfolio.positions.values()]
        if len(g.hold_list) < g.stock_num and not g.risk_circuit and g.target_list:
            rebalance_to_targets(context, g.target_list)
        g.reason_to_sell = ''
    elif g.reason_to_sell == 'stoploss':
        g.reason_to_sell = ''


def trade_afternoon(context):
    if g.trading_signal and not g.risk_circuit:
        check_limit_up(context)
        check_remain_amount(context)


def sell_stocks(context):
    if not g.run_stoploss:
        return
    positions = context.portfolio.positions

    if g.stoploss_strategy in (1, 3):
        for stock in list(positions.keys()):
            if stock == g.etf:
                continue
            pos = positions.get(stock)
            if not pos:
                continue
            price = pos.price
            avg = pos.avg_cost
            if price >= avg * 2:
                order_target_value(stock, 0)
                log.info('翻倍止盈 %s' % stock)
            elif price < avg * (1 - g.stoploss_limit):
                order_target_value(stock, 0)
                g.reason_to_sell = 'stoploss'
                log.info('止损 %s' % stock)
                continue
            hi = g.stock_high.get(stock, max(price, avg))
            atr = _atr_last(context, stock)
            if atr is not None and hi - g.atr_mult * atr > 0:
                stop_line = hi - g.atr_mult * atr
                if price < stop_line:
                    order_target_value(stock, 0)
                    log.info('ATR止损 %s' % stock)
                    if stock in g.stock_high:
                        del g.stock_high[stock]

    if g.use_breadth_stop and g.stoploss_strategy in (2, 3):
        try:
            stock_df = get_price(
                security=get_index_stocks('399101.XSHE'),
                end_date=context.previous_date,
                frequency='daily',
                fields=['close', 'open'],
                count=1,
                panel=False,
            )
            ratio = (stock_df['close'] / stock_df['open']).mean()
            if ratio <= g.stoploss_market:
                g.reason_to_sell = 'stoploss'
                log.info('广度止损 平均收盘/开盘=%.4f' % ratio)
                positions = context.portfolio.positions
                for stock in list(positions.keys()):
                    if stock != g.etf:
                        order_target_value(stock, 0)
        except Exception:
            pass


def close_account(context):
    check_pass_period(context)
    if not g.trading_signal and g.hold_list:
        for stock in list(g.hold_list):
            pos = context.portfolio.positions.get(stock)
            if pos:
                close_position(pos)
                log.info('空仓期清仓 %s' % stock)


def filter_stocks(context, stock_list):
    current_data = get_current_data()
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    out = []
    for stock in stock_list:
        if current_data[stock].paused:
            continue
        if current_data[stock].is_st:
            continue
        if '退' in current_data[stock].name:
            continue
        if stock.startswith('30') or stock.startswith('68') or stock.startswith('8') or stock.startswith('4'):
            continue
        if not (stock in context.portfolio.positions or last_prices[stock][-1] < current_data[stock].high_limit):
            continue
        if not (stock in context.portfolio.positions or last_prices[stock][-1] > current_data[stock].low_limit):
            continue
        info = get_security_info(stock)
        if info is None:
            continue
        if context.previous_date - info.start_date < timedelta(days=375):
            continue
        out.append(stock)
    return out


def filter_audit(context, code):
    lstd = context.previous_date
    last_year = (lstd.replace(year=lstd.year - 3, month=1, day=1)).strftime('%Y-%m-%d')
    q = query(finance.STK_AUDIT_OPINION).filter(
        finance.STK_AUDIT_OPINION.code == code,
        finance.STK_AUDIT_OPINION.pub_date >= last_year,
    )
    df = finance.run_query(q)
    if df is None or df.empty:
        return True
    df['report_type'] = df['report_type'].astype(str)
    bad = df['report_type'].str.contains(r'2|3|4|5')
    return not bad.any()


def close_position(position):
    security = position.security
    order = order_target_value(security, 0)
    if order is not None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            if security in g.stock_high:
                del g.stock_high[security]
            return True
    return False
