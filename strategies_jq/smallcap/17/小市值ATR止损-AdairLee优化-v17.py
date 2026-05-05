# -*- coding: utf-8 -*-
# 基于 smallcap/7「小市值ATR止损策略-AdairLee」修复与轻量优化（SimTradeLab → 17）
#
# 修复：
# - buy_security：未持仓时用 positions[stock] 会 KeyError，改为先判断是否在持仓或 total_amount==0。
# - filter_audit：finance.run_query 结果为空时不再对空 DataFrame 做列运算。
# - 空仓月份：原逻辑仅 buy ETF，未先卖股票；现先清仓非 ETF 再买入货基。
# - adjust_stock_num：移除函数内「单票超 30% 减仓」副作用（不应在计算持仓数时下单）。
# - check_limit_up / close_account：持仓与字典访问加防护。
#
# 轻量优化：
# - 空仓日历与目录内国九策略对齐：1 月、4 月、12 月 15 日～31 日（可按需改回仅 1、4 月）。
# - ATR 倍数略放宽至 2.2（减少震荡市误扫，可在 initialize 调回 2.0）。
# - 移除未使用的 jqfactor 导入。

from jqdata import *
from jqdata import finance
import numpy as np
import pandas as pd
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
    log.set_level('strategy', 'debug')

    g.trading_signal = True
    g.run_stoploss = True
    g.filter_audit = False
    g.adjust_num = False

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
    g.max_mv = 1e8
    g.stock_num = 3
    g.reason_to_sell = ''
    g.stoploss_strategy = 1
    g.stoploss_limit = 0.09
    g.stoploss_market = 0.05
    g.highest = 50
    g.lowest = 1
    g.etf = '511880.XSHG'

    g.atr_period = 14
    g.atr_multiplier = 2.2
    g.stock_highest_price = {}
    g.stock_entry_date = {}

    run_daily(prepare_stock_list, '9:05')
    run_daily(trade_afternoon, time='14:20', reference_security='399101.XSHE')
    run_daily(sell_stocks, time='10:00')
    run_daily(sell_stocks, time='14:00')
    run_daily(close_account, '14:50')
    run_weekly(weekly_adjustment, 2, '10:00')


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


def today_is_trading(context):
    return not _date_in_pass_period(context.current_dt.date())


def prepare_stock_list(context):
    g.hold_list = []
    g.limitup_stocks = []
    for position in list(context.portfolio.positions.values()):
        stock = position.security
        g.hold_list.append(stock)
        current_price = position.price
        if stock not in g.stock_highest_price:
            g.stock_highest_price[stock] = current_price
            g.stock_entry_date[stock] = context.current_dt.date()
        else:
            g.stock_highest_price[stock] = max(g.stock_highest_price[stock], current_price)

    for stock in list(g.stock_highest_price.keys()):
        if stock not in g.hold_list:
            del g.stock_highest_price[stock]
            if stock in g.stock_entry_date:
                del g.stock_entry_date[stock]

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

    g.trading_signal = today_is_trading(context)


def get_stock_list(context):
    MKT_index = '399101.XSHE'
    initial_list = filter_stocks(context, get_index_stocks(MKT_index))
    q = query(
        valuation.code,
        valuation.market_cap,
        income.np_parent_company_owners,
        income.net_profit,
        income.operating_revenue,
    ).filter(
        valuation.code.in_(initial_list),
        valuation.market_cap.between(g.min_mv, g.max_mv),
        income.np_parent_company_owners > 0,
        income.net_profit > 0,
        income.operating_revenue > 1e8,
    ).order_by(valuation.market_cap.asc()).limit(g.stock_num * 3)

    df = get_fundamentals(q)
    if g.filter_audit is True and df is not None and not df.empty:
        before = len(df)
        df = df[df['code'].apply(lambda x: filter_audit(context, x))]
        log.info('审计过滤后剩余 %s 只（原 %s）' % (len(df), before))

    if df is None or df.empty:
        log.info('无适合股票，买入ETF')
        return [g.etf]

    final_list = list(df.code)
    last_prices = history(1, unit='1d', field='close', security_list=final_list)
    return [
        stock
        for stock in final_list
        if stock in g.hold_list or last_prices[stock][-1] >= g.lowest
    ]


def weekly_adjustment(context):
    if g.trading_signal:
        new_num = adjust_stock_num(context)
        if new_num == 0:
            buy_security(context, [g.etf])
            log.info('MA指示指数大跌，持有银华日利ETF')
        else:
            if g.stock_num != new_num:
                g.stock_num = new_num
                log.info('持仓数量修改为%s' % new_num)
            g.target_list = get_stock_list(context)[: g.stock_num]
            log.info(str(g.target_list))

            sell_list = [
                stock
                for stock in g.hold_list
                if stock not in g.target_list and stock not in g.yesterday_HL_list
            ]
            hold_list = [
                stock
                for stock in g.hold_list
                if stock in g.target_list or stock in g.yesterday_HL_list
            ]
            log.info('已持有[%s]' % str(hold_list))
            log.info('卖出[%s]' % str(sell_list))

            for stock in sell_list:
                pos = context.portfolio.positions.get(stock)
                if pos:
                    close_position(pos)

            buy_security(context, g.target_list)
    else:
        for stock in list(g.hold_list):
            if stock == g.etf:
                continue
            pos = context.portfolio.positions.get(stock)
            if pos:
                close_position(pos)
        buy_security(context, [g.etf])
        log.info('空仓期：已清仓股票并持有银华日利')


def check_limit_up(context):
    now_time = context.current_dt
    if not g.yesterday_HL_list:
        return
    for stock in g.yesterday_HL_list:
        pos = context.portfolio.positions.get(stock)
        if not pos or pos.closeable_amount <= 0:
            continue
        current_data = get_price(
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
        if current_data is None or current_data.empty:
            continue
        if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
            log.info('[%s]涨停打开，卖出' % stock)
            close_position(pos)
            g.reason_to_sell = 'limitup'
            g.limitup_stocks.append(stock)
        else:
            log.debug('[%s]涨停，继续持有' % stock)


def check_remain_amount(context):
    if g.reason_to_sell == 'limitup':
        g.hold_list = [p.security for p in context.portfolio.positions.values()]
        if len(g.hold_list) < g.stock_num:
            num_stocks_to_buy = min(
                len(g.limitup_stocks), g.stock_num - len(context.portfolio.positions)
            )
            target_list = [
                stock for stock in g.target_list if stock not in g.limitup_stocks
            ][: max(num_stocks_to_buy, 0)]
            log.info(
                '有余额可用%s元。买入%s'
                % (round(context.portfolio.cash, 2), str(target_list))
            )
            buy_security(context, target_list)
        g.reason_to_sell = ''
    elif g.reason_to_sell == 'stoploss':
        log.info(
            '止损后余额%s元，买入%s'
            % (round(context.portfolio.cash, 2), str(g.etf))
        )
        buy_security(context, [g.etf])
        g.reason_to_sell = ''


def trade_afternoon(context):
    if g.trading_signal:
        check_limit_up(context)
        check_remain_amount(context)


def calculate_atr(stock, period=14, end_date=None):
    try:
        history_data = get_price(
            stock,
            end_date=end_date,
            frequency='daily',
            fields=['high', 'low', 'close'],
            count=period + 1,
            panel=False,
            fill_paused=True,
        )
        if history_data is None or len(history_data) < period + 1:
            return None
        high = history_data['high'].values
        low = history_data['low'].values
        close = history_data['close'].values
        tr_list = []
        for i in range(1, len(history_data)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr_list.append(max(tr1, tr2, tr3))
        if len(tr_list) >= period:
            return float(np.mean(tr_list[-period:]))
        return None
    except Exception as e:
        log.error('计算ATR出错: %s, %s' % (stock, e))
        return None


def sell_stocks(context):
    if not g.run_stoploss:
        return
    current_positions = context.portfolio.positions

    for stock in list(current_positions.keys()):
        if stock == g.etf:
            continue
        try:
            pos = current_positions.get(stock)
            if not pos:
                continue
            current_price = pos.price
            if stock not in g.stock_highest_price:
                g.stock_highest_price[stock] = current_price
                g.stock_entry_date[stock] = context.current_dt.date()
            g.stock_highest_price[stock] = max(g.stock_highest_price[stock], current_price)

            atr = calculate_atr(stock, g.atr_period, context.previous_date)
            if atr is not None and atr > 0:
                highest_price = g.stock_highest_price[stock]
                atr_stop_price = highest_price - atr * g.atr_multiplier
                if current_price <= atr_stop_price:
                    order_target_value(stock, 0)
                    log.info(
                        'ATR止损卖出 %s: 现价=%.2f 最高=%.2f ATR=%.2f 止损价=%.2f'
                        % (stock, current_price, highest_price, atr, atr_stop_price)
                    )
                    g.reason_to_sell = 'stoploss'
                    continue
        except Exception as e:
            log.error('ATR止损计算出错: %s, %s' % (stock, e))

    if g.stoploss_strategy in (1, 3):
        for stock in list(current_positions.keys()):
            if stock == g.etf:
                continue
            pos = current_positions.get(stock)
            if not pos:
                continue
            price = pos.price
            avg_cost = pos.avg_cost
            if price >= avg_cost * 2:
                order_target_value(stock, 0)
                log.info('收益100%%止盈, 卖出%s' % stock)
            elif price < avg_cost * (1 - g.stoploss_limit):
                order_target_value(stock, 0)
                log.info('收益止损, 卖出%s' % stock)
                g.reason_to_sell = 'stoploss'

    if g.stoploss_strategy in (2, 3):
        try:
            stock_df = get_price(
                security=get_index_stocks('399101.XSHE'),
                end_date=context.previous_date,
                frequency='daily',
                fields=['close', 'open'],
                count=1,
                panel=False,
            )
            down_ratio = abs((stock_df['close'] / stock_df['open'] - 1).mean())
            if down_ratio >= g.stoploss_market:
                g.reason_to_sell = 'stoploss'
                log.info('大盘惨跌, 平均降幅%.2f%%' % (down_ratio * 100))
                for s in list(current_positions.keys()):
                    if s != g.etf:
                        order_target_value(s, 0)
        except Exception as e:
            log.error('市场止损计算出错: %s' % e)


def adjust_stock_num(context):
    if g.adjust_num is not True:
        return g.stock_num
    ma_para = 10
    today = context.previous_date
    start_date = today - timedelta(days=ma_para * 2)
    index_df = get_price(
        '399101.XSHE', start_date=start_date, end_date=today, frequency='daily'
    )
    index_df['ma'] = index_df['close'].rolling(window=ma_para).mean()
    last_row = index_df.iloc[-1]
    diff = last_row['close'] - last_row['ma']
    max_diff = (index_df['close'] - index_df['ma']).max()
    min_diff = (index_df['close'] - index_df['ma']).min()
    if max_diff == min_diff:
        normalized_diff = 0.5
    else:
        normalized_diff = (diff - min_diff) / (max_diff - min_diff)

    if normalized_diff >= 0.8:
        result = 10
    elif normalized_diff >= 0.6:
        result = 8
    elif normalized_diff >= 0.4:
        result = 6
    elif normalized_diff >= 0.2:
        result = 4
    else:
        result = 2
    result = max(1, min(int(result), 10))

    returns = index_df['close'].pct_change().dropna()
    if len(returns) > 0:
        volatility = returns.std() * np.sqrt(252)
        if volatility > 0.2:
            result = int(result * 0.8)
        else:
            result = int(result * 1.2)
    result = max(1, min(int(result), 10))

    if len(index_df) >= 22:
        recent_return = index_df['close'].iloc[-1] / index_df['close'].iloc[-22] - 1
        if recent_return > 0.1:
            result = int(result * 1.2)
        elif recent_return < -0.1:
            result = int(result * 0.8)
    return max(1, min(int(result), 10))


def filter_stocks(context, stock_list):
    current_data = get_current_data()
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    filtered_stocks = []
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
        filtered_stocks.append(stock)
    return filtered_stocks


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
    df = df.copy()
    df['report_type'] = df['report_type'].astype(str)
    contains_nums = df['report_type'].str.contains(r'2|3|4|5')
    return not contains_nums.any()


def order_target_value_(security, value):
    return order_target_value(security, value)


def open_position(context, security, value):
    order = order_target_value_(security, value)
    if order is not None and order.filled > 0:
        cd = get_current_data()
        g.stock_highest_price[security] = cd[security].last_price
        g.stock_entry_date[security] = context.current_dt.date()
        return True
    return False


def close_position(position):
    security = position.security
    order = order_target_value_(security, 0)
    if order is not None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            if security in g.stock_highest_price:
                del g.stock_highest_price[security]
            if security in g.stock_entry_date:
                del g.stock_entry_date[security]
            return True
    return False


def _need_open(context, stock):
    if stock not in context.portfolio.positions:
        return True
    return context.portfolio.positions[stock].total_amount == 0


def buy_security(context, target_list):
    target_num = len(target_list)
    if target_num <= 0:
        return
    for stock in target_list:
        if not _need_open(context, stock):
            continue
        cash = context.portfolio.cash
        if cash <= 0:
            break
        remaining = sum(1 for s in target_list if _need_open(context, s))
        if remaining <= 0:
            break
        value = cash / remaining
        if open_position(context, stock, value):
            log.info('买入[%s]（%s元）' % (stock, value))
        if len(context.portfolio.positions) >= target_num:
            break


def close_account(context):
    if not g.trading_signal:
        if len(g.hold_list) == 0:
            return
        if g.hold_list == [g.etf] and len(context.portfolio.positions) <= 1:
            return
        for stock in list(g.hold_list):
            if stock == g.etf:
                continue
            pos = context.portfolio.positions.get(stock)
            if pos:
                close_position(pos)
                log.info('卖出[%s]' % stock)
