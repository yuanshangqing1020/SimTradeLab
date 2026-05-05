# -*- coding: utf-8 -*-
# 小市值近期适配 v15（相对 v14：针对「近几个月」常见失效点做结构微调，而非再叠熔断类风控）
#
# v14 可能问题简述：
# - 固定成交额阈值在「小票+缩量」阶段容易误杀仍有效的最小市值票；
# - 周度等权会把强势股减到与弱势股同权，窄市/主题行情下拖累近期收益；
# - 近 3 日「只要碰过涨跌停且昨非涨停就剔除」偏严，易把活跃小票整池排掉；
# - 广度止损阈值偏紧时，在指数单日分化大时容易集体清仓。
#
# v15 调整：
# 1) 流动性：在小市值候选池内按近 20 日日均成交额做分位数，剔除最差一档（默认下 18%），
#    持仓股永远保留；全市场缩量导致池过小时自动放弃该过滤。
# 2) 买回 v12 的「仅对未持仓标的按现金均分买入」，强势股不被周度强行减仓。
# 3) 打分：20 日动量 + 0.4×10 日动量（更贴近近期节奏），略减轻波动惩罚系数。
# 4) 资金流：最近 3 个交易日主力净流入求和再 z-score，减弱单日噪声。
# 5) 极端 K：仅当近 3 日出现过跌停、且昨日非涨停时才剔除（放过「仅涨停」的强趋势形态）。
# 6) 广度止损略放宽（默认 0.93），可在 initialize 关闭。
#
# 回测请在聚宽校验；参数集中在 initialize。

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
    g.candidate_pool = 130
    g.reason_to_sell = ''
    g.stoploss_strategy = 3
    g.stoploss_limit = 0.10
    g.stoploss_market = 0.93
    g.lowest = 1.0
    g.highest = 80.0

    g.use_breadth_stop = True
    g.mf_weight = 0.38
    g.mom10_weight = 0.4
    g.vol_penalty = 0.88
    g._last_pass_state = None

    g.liquidity_quantile_drop = 0.18
    g.liquidity_min_keep = 40

    run_daily(prepare_stock_list, '9:05')
    run_weekly(weekly_sell_task, 3, '10:30')
    run_weekly(weekly_buy_task, 3, '10:31')
    run_daily(trade_afternoon, time='14:20', reference_security='399101.XSHE')
    run_daily(sell_stocks, time='10:00')
    run_daily(sell_stocks, time='14:00')
    run_daily(close_account, '14:50')


def prepare_stock_list(context):
    g.hold_list = []
    g.limitup_stocks = []
    for position in list(context.portfolio.positions.values()):
        g.hold_list.append(position.security)

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
        return 3
    if 200 <= diff < 500:
        return 3
    if -200 <= diff < 200:
        return 4
    if -500 <= diff < -200:
        return 5
    return 6


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
        days = get_trade_days(end_date=context.previous_date, count=6)
        if len(days) < 2:
            return pd.Series(0.0, index=codes)
        end_d = days[-2]
        parts = []
        for i in range(0, len(codes), 400):
            sub = codes[i : i + 400]
            df = get_money_flow(
                security_list=sub,
                end_date=end_d,
                count=3,
                fields=['sec_code', 'net_amount_main'],
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                parts.append(df)
        if not parts:
            return pd.Series(0.0, index=codes)
        mf = pd.concat(parts, ignore_index=True)
        agg = mf.groupby('sec_code')['net_amount_main'].sum()
        s = agg.reindex(codes).fillna(0.0)
        return _safe_z(s)
    except Exception:
        return pd.Series(0.0, index=codes)


def _dollar_volume_series(context, codes):
    out = {}
    if not codes:
        return out
    try:
        px = get_price(
            codes,
            end_date=context.previous_date,
            frequency='daily',
            fields=['close', 'volume'],
            count=22,
            panel=False,
            fill_paused=False,
        )
        if px is None or px.empty:
            return out
        for c in codes:
            sub = px[px['code'] == c].sort_values('time')
            if len(sub) < 15:
                out[c] = 0.0
                continue
            cl = sub['close'].values.astype(float)
            vol = sub['volume'].values.astype(float)
            out[c] = float((cl[-21:-1] * vol[-21:-1]).mean())
    except Exception:
        pass
    return out


def _liquidity_quantile_filter(context, codes):
    if not codes:
        return []
    keep = set(g.hold_list)
    vols = _dollar_volume_series(context, codes)
    if len(vols) < 10:
        return list(codes)
    s = pd.Series([float(vols.get(c, 0.0)) for c in codes], index=codes)
    q = float(s.quantile(g.liquidity_quantile_drop))
    ok = [c for c in codes if c in keep or float(s.loc[c]) >= q]
    if len(ok) < g.liquidity_min_keep:
        return list(codes)
    return ok


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

    codes = _liquidity_quantile_filter(context, codes)
    df = df[df['code'].isin(codes)]
    if df.empty:
        return []

    codes = list(df['code'])
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

    mom20 = {}
    mom10 = {}
    vol = {}
    for c in codes:
        sub = px[px['code'] == c].sort_values('time')
        closes = sub['close'].values.astype(float)
        if len(closes) < 21:
            mom20[c] = 0.0
            mom10[c] = 0.0
            vol[c] = 0.05
            continue
        mom20[c] = float(closes[-2] / max(closes[-22], 1e-9) - 1.0)
        if len(closes) >= 12:
            mom10[c] = float(closes[-2] / max(closes[-12], 1e-9) - 1.0)
        else:
            mom10[c] = 0.0
        rets = np.diff(closes[-21:]) / np.clip(closes[-21:-1], 1e-9, None)
        vol[c] = float(np.std(rets)) if len(rets) > 1 else 0.05

    d = df.set_index('code')
    d['mom20'] = pd.Series(mom20).reindex(d.index).fillna(0.0)
    d['mom10'] = pd.Series(mom10).reindex(d.index).fillna(0.0)
    d['vol20'] = pd.Series(vol).reindex(d.index).fillna(0.05)
    d['roe'] = pd.to_numeric(d['roe'], errors='coerce').fillna(0.0)

    z_roe = _safe_z(d['roe'])
    z_m20 = _safe_z(d['mom20'])
    z_m10 = _safe_z(d['mom10'])
    z_vol = _safe_z(d['vol20'])
    z_mf = _attach_money_flow_score(list(d.index), context).reindex(d.index).fillna(0.0)

    d['score'] = (
        z_roe
        + z_m20
        + g.mom10_weight * z_m10
        - g.vol_penalty * z_vol
        + g.mf_weight * z_mf
    )
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
        has_dn = (sd['close'] == sd['low_limit']).any()
        y = sd.iloc[-1]
        y_not_up = y['close'] != y['high_limit']
        if has_dn and y_not_up:
            exclude.add(stock)
    return [s for s in stock_list if s not in exclude]


def weekly_sell_task(context):
    check_pass_period(context)

    if g.trading_signal:
        new_num = adjust_stock_num(context)
        if g.stock_num != new_num:
            g.stock_num = new_num
            log.info('持仓数量调整为 %s' % new_num)

        g.target_list = get_stock_list(context)[: g.stock_num]
        log.info('本周目标: %s' % g.target_list)

        sell_list = [
            s
            for s in g.hold_list
            if s not in g.target_list and s not in g.yesterday_HL_list
        ]
        for s in sell_list:
            pos = context.portfolio.positions.get(s)
            if pos:
                close_position(pos)
    else:
        g.target_list = []
        for s in list(g.hold_list):
            pos = context.portfolio.positions.get(s)
            if pos:
                close_position(pos)


def weekly_buy_task(context):
    check_pass_period(context)
    if not g.trading_signal or not g.target_list:
        return
    buy_security(context, g.target_list)


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
        else:
            log.debug('涨停持有 %s' % stock)


def check_remain_amount(context):
    if g.reason_to_sell == 'limitup':
        g.hold_list = [p.security for p in context.portfolio.positions.values()]
        if len(g.hold_list) < g.stock_num:
            extra = [s for s in g.target_list if s not in g.limitup_stocks][
                : max(0, g.stock_num - len(g.hold_list))
            ]
            if extra:
                buy_security(context, extra)
        g.reason_to_sell = ''
    elif g.reason_to_sell == 'stoploss':
        g.reason_to_sell = ''


def trade_afternoon(context):
    if g.trading_signal:
        check_limit_up(context)
        check_remain_amount(context)


def sell_stocks(context):
    if not g.run_stoploss:
        return
    positions = context.portfolio.positions

    if g.stoploss_strategy in (1, 3):
        for stock in list(positions.keys()):
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
            return True
    return False


def open_position(security, value):
    o = order_target_value(security, value)
    return o is not None and o.filled > 0


def buy_security(context, target_list):
    target_list = [s for s in target_list if s]
    if not target_list:
        return
    stocks_to_buy = [s for s in target_list if s not in context.portfolio.positions]
    if not stocks_to_buy:
        return
    value = context.portfolio.cash / len(stocks_to_buy)
    for stock in stocks_to_buy:
        if open_position(stock, value):
            log.info('买入[%s] %.2f 元' % (stock, value))
