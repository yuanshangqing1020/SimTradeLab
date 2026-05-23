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
        '512100.XSHG',  # 中证1000
        #'510300.XSHG',  # 300ETF
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
    g.early_break_level = -5   # v0.7: level≤-5 且再涨则提前破顶，重算 grid_shares

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


def _calc_grid_shares(price):
    """按当前价计算每格股数（100 股取整）。"""
    if price <= 0:
        return 100
    shares = int(g.grid_value / price / 100) * 100
    return shares if shares > 0 else 100


def _refresh_grid_shares(status, price):
    """v0.7: 每次成交后按现价重算 grid_shares，避免长期牛市后格过大。"""
    status['grid_shares'] = _calc_grid_shares(price)


def handle_data(context, data):
    current_yields = {}

    for ticker in g.etf_list:
        display_name = f"{g.stock_names[ticker]} ({ticker})"
        
        current_price = data[ticker].price
        if math.isnan(current_price) or current_price == 0:
            if ticker in g.grid_status:
                current_yields[f"{g.stock_names[ticker]}_做T利润"] = g.grid_t_profits[ticker]
            continue
            
        # ================= 阶段一：初始建仓 =================
        if ticker not in g.grid_status:
            base_shares = int(g.base_value / current_price / 100) * 100
            grid_shares = _calc_grid_shares(current_price)
            
            need_cash = base_shares * current_price
            if context.portfolio.available_cash > need_cash:
                res = order(ticker, base_shares)
                if res:
                    g.grid_status[ticker] = {
                        'ref_price': current_price,
                        'grid_shares': grid_shares,
                        'level': 0
                    }
                    total_amount = context.portfolio.positions[ticker].total_amount
                    log.info(f"🚀【重装建仓】{display_name} | 价格:{current_price:.3f} | 买入:{base_shares}股 | 总持仓:{total_amount}股")
            else:
                log.warn(
                    f"⚠️【建仓跳过】{display_name} | 可用现金:{context.portfolio.available_cash:.0f} "
                    f"< 需:{need_cash:.0f} | 价格:{current_price:.3f}"
                )

        # ================= 阶段二：网格交易 =================
        else:
            status = g.grid_status[ticker]
            ref_price = status['ref_price']
            level = status['level']
            grid_shares = status['grid_shares']
            
            # 1. 触发向下买入
            if current_price <= ref_price * (1 - g.buy_drop):
                if level < g.max_buy_level:
                    need_cash = grid_shares * current_price
                    if context.portfolio.available_cash > need_cash:
                        res = order(ticker, grid_shares)
                        if res:
                            status['ref_price'] = current_price 
                            status['level'] += 1
                            _refresh_grid_shares(status, current_price)
                            total_amount = context.portfolio.positions[ticker].total_amount
                            log.info(
                                f"🔴【加仓】{display_name} | 价格:{current_price:.3f} | 买入:{grid_shares}股 "
                                f"| 总持仓:{total_amount}股 | 等级:{status['level']}"
                            )
                    else:
                        log.warn(
                            f"⚠️【加仓跳过】{display_name} | 可用现金:{context.portfolio.available_cash:.0f} "
                            f"< 需:{need_cash:.0f} | 价格:{current_price:.3f} | 等级:{level} "
                            f"| 每格:{grid_shares}股"
                        )
                else:
                    log.warn(
                        f"⚠️【加仓跳过】{display_name} | 等级已满:{level}>={g.max_buy_level} "
                        f"| 价格:{current_price:.3f} | ref:{ref_price:.3f}"
                    )

            # 2. 触发向上卖出 或 破顶平移
            elif current_price >= ref_price * (1 + g.sell_rise):
                # A. 正常网格卖出止盈（level > early_break_level 时才逐格卖）
                if level > g.early_break_level:
                    position = context.portfolio.positions[ticker]
                    closeable = position.closeable_amount
                    if closeable >= grid_shares:
                        res = order(ticker, -grid_shares)
                        if res:
                            trade_value = grid_shares * current_price
                            cost_value = grid_shares * ref_price
                            fee = max(0.1, trade_value * 0.0001)
                            profit = trade_value - cost_value - fee
                            g.grid_t_profits[ticker] += profit
                            
                            status['ref_price'] = current_price
                            status['level'] -= 1
                            _refresh_grid_shares(status, current_price)
                            total_amount = context.portfolio.positions[ticker].total_amount
                            log.info(
                                f"🟢【止盈】{display_name} | 价格:{current_price:.3f} | 卖出:{grid_shares}股 "
                                f"| 总持仓:{total_amount}股 | 做T净赚:{profit:.2f} | 剩余等级:{status['level']}"
                            )
                    else:
                        log.warn(
                            f"⚠️【止盈跳过】{display_name} | 可卖:{closeable}股 < 需卖:{grid_shares}股 "
                            f"| 价格:{current_price:.3f} | 等级:{level} | ref:{ref_price:.3f} "
                            f"| 触发价:{ref_price * (1 + g.sell_rise):.3f}"
                        )
                
                # B. 破顶升维（卖档用尽 level≤-8，或 v0.7 提前破顶 level≤-5）
                else:
                    old_shares = context.portfolio.positions[ticker].total_amount
                    target_shares = int(g.base_value / current_price / 100) * 100
                    sell_diff = old_shares - target_shares
                    break_tag = '提前' if level > g.max_sell_level else '卖尽'
                    
                    if sell_diff > 0:
                        res = order_target(ticker, target_shares)
                        if res:
                            trade_value = sell_diff * current_price
                            cost_value = sell_diff * ref_price
                            fee = max(0.1, trade_value * 0.0001)
                            profit = trade_value - cost_value - fee
                            g.grid_t_profits[ticker] += profit
                            log.info(
                                f"🌟【破顶-{break_tag}】{display_name} | 价格:{current_price:.3f} "
                                f"| 减仓:{sell_diff}股 | 总持仓:{target_shares}股 | 释放利润:{profit:.2f} | 等级:{level}"
                            )
                    else:
                        order_target(ticker, target_shares)
                        log.info(
                            f"🌟【破顶-{break_tag}】{display_name} | 价格:{current_price:.3f} "
                            f"| 调仓至:{target_shares}股 | 等级:{level}"
                        )

                    status['ref_price'] = current_price
                    status['level'] = 0
                    _refresh_grid_shares(status, current_price)

        plot_key = f"{g.stock_names[ticker]}_做T利润"
        current_yields[plot_key] = g.grid_t_profits[ticker]

    if current_yields:
        record(**current_yields)
