# 导入聚宽函数库
import jqdata

def initialize(context):
    # ================= 1. 动态股票池设置 =================
    # 你可以在这里随意增删修改股票，支持1~10只股票
    # 比如：杭氧股份、金达威、贵州茅台、宁德时代等
    g.securities =[
        '002430.XSHE', # 杭氧股份
        '002626.XSHE', # 金达威
        '600141.XSHG', # 兴发集团
        '000422.XSHE'  # 湖北宜化
    ]
    
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_slippage(FixedSlippage(0.02)) 
    
    # 交易手续费：印花税千分之0.5，佣金万分之3，最低5元
    set_order_cost(OrderCost(close_tax=0.0005, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')
    
    # ================= 2. 策略参数设置 =================
    g.initial_amount = 100000     # 单只股票初始建仓金额：20万
    g.grid_step = 0.03            # 网格步长：3%
    g.trade_volume = 500          # 每次网格交易股数：500股
    
    # 初始化各标的的状态字典
    g.base_prices = {sec: 0.0 for sec in g.securities}
    g.init_done = {sec: False for sec in g.securities}

    # ================= 3. 高级收益率统计与绘图准备 =================
    # 自动获取股票的中文名，方便在图表上显示（比如直接显示“杭氧股份_收益%”）
    g.stock_names = {}
    for sec in g.securities:
        info = get_security_info(sec)
        g.stock_names[sec] = info.display_name if info else sec

    # 为每只股票建立“虚拟账本”，分配40万初始资金（20万底仓 + 20万补仓现金）
    g.init_cash_per_sec = 400000.0 
    g.virtual_cash = {sec: g.init_cash_per_sec for sec in g.securities}
    g.last_amount = {sec: 0 for sec in g.securities} 

def handle_data(context, data):
    # 建立一个空字典，用于动态收集当前时刻所有股票的收益率
    current_yields = {}

    for sec in g.securities:
        # 如果股票停牌，收益率沿用上一次的，但不进行交易逻辑
        if data[sec].paused:
            continue
            
        current_price = data[sec].price
        
        # ================= 4. 初始建仓逻辑 =================
        if not g.init_done[sec]:
            shares_to_buy = int(g.initial_amount / current_price / 100) * 100
            # 确保主账户有足够现金
            if context.portfolio.available_cash >= shares_to_buy * current_price:
                order(sec, shares_to_buy)
                g.base_prices[sec] = current_price
                g.init_done[sec] = True
                log.info(f"【初始建仓】{g.stock_names[sec]} 买入 {shares_to_buy}股，价格：{g.base_prices[sec]}元")
        
        # ================= 5. 网格交易逻辑 =================
        else:
            # 向上突破（卖出）
            while current_price >= g.base_prices[sec] * (1 + g.grid_step):
                if context.portfolio.positions[sec].closeable_amount >= g.trade_volume:
                    order(sec, -g.trade_volume)
                    log.info(f"【网格卖出】{g.stock_names[sec]} 触发卖出。")
                g.base_prices[sec] = g.base_prices[sec] * (1 + g.grid_step)

            # 向下突破（买入）
            while current_price <= g.base_prices[sec] * (1 - g.grid_step):
                cost_needed = g.trade_volume * current_price
                if context.portfolio.available_cash >= cost_needed:
                    order(sec, g.trade_volume)
                    log.info(f"【网格买入】{g.stock_names[sec]} 触发买入。")
                g.base_prices[sec] = g.base_prices[sec] * (1 - g.grid_step)

        # ================= 6. 独立收益率精确计算（含手续费） =================
        current_amount = context.portfolio.positions[sec].total_amount
        diff_amount = current_amount - g.last_amount[sec]
        
        # 如果持仓发生了变化，更新虚拟账本的现金（含手续费估算）
        if diff_amount != 0:
            trade_value = abs(diff_amount) * current_price
            # 估算手续费：买入收佣金，卖出收佣金+印花税
            if diff_amount > 0: # 买入
                fee = max(5, trade_value * 0.0003)
            else: # 卖出
                fee = max(5, trade_value * 0.0003) + trade_value * 0.0005
                
            # 虚拟账本现金变动 = -(股票数量变动 * 价格 + 交易手续费)
            g.virtual_cash[sec] -= (diff_amount * current_price + fee)
            g.last_amount[sec] = current_amount
            
        # 计算该股票当前总资产与收益率
        stock_value = current_amount * current_price
        total_asset = stock_value + g.virtual_cash[sec]
        yield_rate = (total_asset / g.init_cash_per_sec - 1.0) * 100
        
        # 将该股票的中文名和收益率存入字典中
        # 例如：{'杭氧股份_收益%': 5.2, '金达威_收益%': -1.3}
        plot_key = f"{g.stock_names[sec]}_收益%"
        current_yields[plot_key] = yield_rate

    # ================= 7. 动态绘制所有股票的收益曲线 =================
    # 如果字典中有数据，使用 **kwargs 字典解包技术一次性全部记录
    if current_yields:
        record(**current_yields)