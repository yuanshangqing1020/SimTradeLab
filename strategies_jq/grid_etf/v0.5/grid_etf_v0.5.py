# 导入聚宽函数库
from jqdata import *
import math

def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    
    # 5大不相关核心资产
    g.etf_list = [
        '513100.XSHG',  # 纳指ETF
        '518880.XSHG',  # 黄金ETF
        '510500.XSHG',  # 中证500
        '510300.XSHG',  # 300ETF
        '159915.XSHE'   # 创业板
        #'512010.XSHG',  # 医药ETF
        #'513050.XSHG'   # 中概互联
    ]

    # 【V4 核心参数】
    g.buy_drop = 0.03     
    g.sell_rise = 0.03    
    g.base_value = 50000   # 底仓5万元
    g.grid_value = 5000    # 每格交易5000元
    
    g.max_buy_level = 10   
    g.max_sell_level = -8  

    g.grid_status = {}
    g.grid_t_profits = {ticker: 0.0 for ticker in g.etf_list}
    
    # 获取标的名字映射
    g.stock_names = {}
    for ticker in g.etf_list:
        info = get_security_info(ticker)
        g.stock_names[ticker] = info.display_name if info else ticker

    # 设定手续费 (ETF万一免五)
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0,
        open_commission=0.0001, close_commission=0.0001,
        close_today_commission=0, min_commission=0.1
    ), type='fund')

def handle_data(context, data):
    current_yields = {}

    for ticker in g.etf_list:
        # 定义显示名称变量：名称 + 代码
        display_name = f"{g.stock_names[ticker]} ({ticker})"
        
        current_price = data[ticker].price
        if math.isnan(current_price) or current_price == 0:
            if ticker in g.grid_status:
                current_yields[f"{g.stock_names[ticker]}_做T利润"] = g.grid_t_profits[ticker]
            continue
            
        # ================= 阶段一：初始建仓 =================
        if ticker not in g.grid_status:
            base_shares = int(g.base_value / current_price / 100) * 100
            grid_shares = int(g.grid_value / current_price / 100) * 100
            if grid_shares == 0: grid_shares = 100
            
            if context.portfolio.available_cash > base_shares * current_price:
                res = order(ticker, base_shares)
                if res:
                    g.grid_status[ticker] = {
                        'ref_price': current_price,
                        'grid_shares': grid_shares,
                        'level': 0
                    }
                    total_amount = context.portfolio.positions[ticker].total_amount
                    log.info(f"🚀【重装建仓】{display_name} | 价格:{current_price:.3f} | 买入:{base_shares}股 | 总持仓:{total_amount}股")

        # ================= 阶段二：网格交易 =================
        else:
            status = g.grid_status[ticker]
            ref_price = status['ref_price']
            level = status['level']
            grid_shares = status['grid_shares']
            
            # 1. 触发向下买入
            if current_price <= ref_price * (1 - g.buy_drop):
                if level < g.max_buy_level:
                    if context.portfolio.available_cash > grid_shares * current_price:
                        res = order(ticker, grid_shares)
                        if res:
                            status['ref_price'] = current_price 
                            status['level'] += 1
                            total_amount = context.portfolio.positions[ticker].total_amount
                            log.info(f"🔴【加仓】{display_name} | 价格:{current_price:.3f} | 买入:{grid_shares}股 | 总持仓:{total_amount}股 | 等级:{status['level']}")

            # 2. 触发向上卖出 或 破顶平移
            elif current_price >= ref_price * (1 + g.sell_rise):
                # A. 正常网格卖出止盈
                if level > g.max_sell_level:
                    position = context.portfolio.positions[ticker]
                    if position.closeable_amount >= grid_shares:
                        res = order(ticker, -grid_shares)
                        if res:
                            # 计算做T利润
                            trade_value = grid_shares * current_price
                            cost_value = grid_shares * ref_price
                            fee = max(0.1, trade_value * 0.0001)
                            profit = trade_value - cost_value - fee
                            g.grid_t_profits[ticker] += profit
                            
                            status['ref_price'] = current_price
                            status['level'] -= 1
                            total_amount = context.portfolio.positions[ticker].total_amount
                            log.info(f"🟢【止盈】{display_name} | 价格:{current_price:.3f} | 卖出:{grid_shares}股 | 总持仓:{total_amount}股 | 做T净赚:{profit:.2f} | 剩余等级:{status['level']}")
                
                # B. 破顶升维
                else:
                    old_shares = context.portfolio.positions[ticker].total_amount
                    target_shares = int(g.base_value / current_price / 100) * 100
                    sell_diff = old_shares - target_shares
                    
                    if sell_diff > 0:
                        res = order_target(ticker, target_shares)
                        if res:
                            trade_value = sell_diff * current_price
                            cost_value = sell_diff * ref_price
                            fee = max(0.1, trade_value * 0.0001)
                            profit = trade_value - cost_value - fee
                            g.grid_t_profits[ticker] += profit
                            log.info(f"🌟【破顶】{display_name} | 价格:{current_price:.3f} | 减仓:{sell_diff}股 | 总持仓:{target_shares}股 | 释放利润:{profit:.2f}")
                    else:
                        order_target(ticker, target_shares)
                        log.info(f"🌟【破顶】{display_name} | 价格:{current_price:.3f} | 调仓至:{target_shares}股")

                    status['ref_price'] = current_price
                    status['level'] = 0
                    new_grid_shares = int(g.grid_value / current_price / 100) * 100
                    status['grid_shares'] = new_grid_shares if new_grid_shares > 0 else 100

        # 记录图表
        plot_key = f"{g.stock_names[ticker]}_做T利润"
        current_yields[plot_key] = g.grid_t_profits[ticker]

    if current_yields:
        record(**current_yields)