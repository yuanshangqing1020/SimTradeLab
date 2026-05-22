# 导入聚宽函数库
import jqdata
import math

def initialize(context):
    # ================= 1. 动态股票池设置 =================
    g.securities =[
        '002430.XSHE',   # 杭氧股份
        '002626.XSHE',   # 金达威
        '600141.XSHG',   # 兴发集团
        '000422.XSHE',   # 湖北宜化
        '513180.XSHG',   # 恒生科技ETF
        '512890.XSHG'    # 红利低波ETF
    ]
    
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_slippage(FixedSlippage(0.02)) 
    
    # 交易手续费
    set_order_cost(OrderCost(close_tax=0.0005, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')
    
    # ================= 2. 策略参数设置 =================
    g.initial_amount = 100000     # 初始底仓金额：10万
    g.grid_step = 0.03            # 网格步长：3%
    g.trade_amount = 10000        # 每次网格交易金额：10000元
    
    g.base_prices = {sec: 0.0 for sec in g.securities}
    g.init_done = {sec: False for sec in g.securities}
    g.last_factors = {sec: None for sec in g.securities} 
    
    # ================= 3. 【全新】做T存钱罐准备 =================
    # 这个字典就像一个只进不出的存钱罐，只记录你每次网格卖出时，实实在在装进口袋的现金利润
    g.grid_t_profits = {sec: 0.0 for sec in g.securities}

    g.stock_names = {}
    for sec in g.securities:
        info = get_security_info(sec)
        g.stock_names[sec] = info.display_name if info else sec

def before_trading_start(context):
    # 每天盘前自动获取复权因子，防止股票分红送转导致网格错乱
    for sec in g.securities:
        current_factor_df = get_price(sec, count=1, end_date=context.current_dt, fields=['factor'])
        if not current_factor_df.empty:
            current_factor = current_factor_df['factor'].values[0]
            if g.init_done[sec] and g.last_factors[sec] is not None:
                if current_factor != g.last_factors[sec]:
                    ratio = g.last_factors[sec] / current_factor 
                    g.base_prices[sec] *= ratio
            g.last_factors[sec] = current_factor

def handle_data(context, data):
    current_yields = {}

    for sec in g.securities:
        current_price = data[sec].price
        
        if math.isnan(current_price) or current_price <= 0 or data[sec].paused:
            plot_key = f"{g.stock_names[sec]}_纯做T现金利润"
            current_yields[plot_key] = g.grid_t_profits[sec]
            continue
        
        # ================= 4. 初始建仓 =================
        if not g.init_done[sec]:
            shares_to_buy = int(g.initial_amount / current_price / 100) * 100
            if shares_to_buy > 0 and context.portfolio.available_cash >= shares_to_buy * current_price:
                order(sec, shares_to_buy)
                g.base_prices[sec] = current_price
                g.init_done[sec] = True
        
        # ================= 5. 网格交易 =================
        else:
            # 向上突破（触发卖出，落袋为安）
            while current_price >= g.base_prices[sec] * (1 + g.grid_step):
                shares_to_sell = int(g.trade_amount / current_price / 100) * 100
                if shares_to_sell > 0 and context.portfolio.positions[sec].closeable_amount >= shares_to_sell:
                    order(sec, -shares_to_sell)
                    
                    # 【核心算法：计算绝对落袋利润】
                    # 卖出得到的总现金
                    trade_value = shares_to_sell * current_price
                    # 当初这批货锚定的基准成本价
                    cost_value = shares_to_sell * g.base_prices[sec]
                    # 手续费（卖出双边加印花税）
                    fee = max(5, trade_value * 0.0003) + trade_value * 0.0005 
                    
                    # 单笔做T利润 = 卖出得到的钱 - 锚定成本 - 手续费
                    # 大概每次赚：10000 * 3% - 手续费 ≈ 290多元
                    single_t_profit = trade_value - cost_value - fee
                    
                    # 把赚到的现金塞进存钱罐
                    g.grid_t_profits[sec] += single_t_profit
                    log.info(f"【收割】{g.stock_names[sec]} 做T成功，落袋现金利润：{single_t_profit:.2f}元")
                    
                g.base_prices[sec] = g.base_prices[sec] * (1 + g.grid_step)

            # 向下突破（触发买入，囤积便宜筹码）
            while current_price <= g.base_prices[sec] * (1 - g.grid_step):
                shares_to_buy = int(g.trade_amount / current_price / 100) * 100
                cost_needed = shares_to_buy * current_price
                if shares_to_buy > 0 and context.portfolio.available_cash >= cost_needed:
                    order(sec, shares_to_buy)
                    # 买入时不计算利润，等未来涨上去卖出时再提款
                g.base_prices[sec] = g.base_prices[sec] * (1 - g.grid_step)

        # ================= 6. 记录图表 =================
        plot_key = f"{g.stock_names[sec]}_纯做T现金利润"
        current_yields[plot_key] = g.grid_t_profits[sec]

    if current_yields:
        record(**current_yields)