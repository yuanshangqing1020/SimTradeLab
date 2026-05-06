# 克隆自聚宽文章：https://www.joinquant.com/post/62562
# 标题：四大行 轮动做T，已实盘
# 作者：ddq1999

import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as scs
import scipy.optimize as sco
import talib as tl
from datetime import timedelta

# 银行股票池：工行、农行、中行、建行
bank_stocks = ['601398.XSHG', '601288.XSHG', '601939.XSHG', '601988.XSHG']

# 初始化参数
def initialize(context):
    # 设置股票池为空（动态调整）
    set_universe([])
    g.riskbench = '000300.XSHG'  # 沪深300作为基准

    # 使用真实价格
    set_option('use_real_price', True)
    
     # 滑点设置（实际交易滑点）
    set_slippage(PriceRelatedSlippage(0.001))
    

    set_option('use_real_price', True)
    set_option('avoid_future_data', True)  # 防未来函数
    # 交易成本设置（保持原始）
    set_order_cost(OrderCost(
        open_tax=0,  # 买入无印花税
        close_tax=0.0005,  # 卖出印花税万5
        open_commission=0.0001,  # 买入佣金万一
        close_commission=0.0001,  # 卖出佣金万一
        close_today_commission=0,  # 无实际意义，保留0
        min_commission=5  # 最低佣金5元
    ), type='stock')
    
    
    # 价差阈值（0.5%）
    g.inter = 0.004
    # 初始化止损标记（Python 3需显式定义）
    g.is_stop = False
    # 记录每只股票的买入日期
    g.buy_dates = {}

# 每天交易前调用
def before_trading_start(context):
    # 获取前一交易日收盘价（Python 3中history函数返回DataFrame时兼容处理）
    g.df_last = history(
        1, 
        unit='1d', 
        field='close', 
        security_list=bank_stocks, 
        df=False, 
        skip_paused=True, 
        fq='pre'
    )

# 每个单位时间调用（按分钟回测时每分钟触发）
def handle_data(context, data):
    raito = []
    # 计算当前价格与前一日收盘价的比值（涨幅比例）
    for code in bank_stocks:
        # Python 3中列表索引和除法兼容
        raito.append(data[code].close / g.df_last[code][-1])
    
    # 空仓时的开仓逻辑
    if not context.portfolio.positions:  # Python 3中keys()返回视图，直接判断是否为空更简洁
        if max(raito) - min(raito) > g.inter:
            # 找到涨幅最小的股票索引
            min_index = raito.index(min(raito))
            # 满仓买入
            order_value(bank_stocks[min_index], context.portfolio.total_value)
            # 记录买入日期
            g.buy_dates[bank_stocks[min_index]] = context.current_dt.date()
            g.is_stop = True
    # 持仓时的调仓逻辑
    else:
        # Python 3中dict.keys()返回视图，需转换为列表取第一个元素
        code = list(context.portfolio.positions.keys())[0]
        index = bank_stocks.index(code)
        # 当前持仓涨幅与最小涨幅的差值超过阈值时调仓
        if raito[index] - min(raito) > g.inter:
            # 检查是否是今天买入的股票
            if code in g.buy_dates and g.buy_dates[code] == context.current_dt.date():
                # 如果是今天买入的，不能卖出，跳过调仓
                log.info(f"股票{code}是今天买入的，不能卖出，跳过调仓")
                return
            
            # 清空当前持仓
            order_target(code, 0)
            # 从买入日期记录中移除
            if code in g.buy_dates:
                del g.buy_dates[code]
            
            # 买入新的涨幅最小的股票
            min_index = raito.index(min(raito))
            order_value(bank_stocks[min_index], context.portfolio.total_value)
            # 记录新买入股票的日期
            g.buy_dates[bank_stocks[min_index]] = context.current_dt.date()
            g.is_stop = True

# 每天交易后调用，记录持仓情况
def after_trading_end(context):
    # 记录四只股票的持仓数量（Python 3中字符串格式化使用f-string更简洁）
    if bank_stocks[0] in context.portfolio.positions and context.portfolio.positions[bank_stocks[0]].total_amount > 0:
        record(code0=context.portfolio.positions[bank_stocks[0]].total_amount)
    if bank_stocks[1] in context.portfolio.positions and context.portfolio.positions[bank_stocks[1]].total_amount > 0:
        record(code1=context.portfolio.positions[bank_stocks[1]].total_amount)
    if bank_stocks[2] in context.portfolio.positions and context.portfolio.positions[bank_stocks[2]].total_amount > 0:
        record(code2=context.portfolio.positions[bank_stocks[2]].total_amount)
    if bank_stocks[3] in context.portfolio.positions and context.portfolio.positions[bank_stocks[3]].total_amount > 0:
        record(code3=context.portfolio.positions[bank_stocks[3]].total_amount)
    
    # 清理已不再持仓的股票的买入日期记录
    current_holdings = set(context.portfolio.positions.keys())
    for code in list(g.buy_dates.keys()):
        if code not in current_holdings:
            del g.buy_dates[code]