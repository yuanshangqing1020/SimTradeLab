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
        #'512100.XSHG',  # 中证1000
        #'510300.XSHG',  # 300ETF
        '159915.XSHE'   # 创业板
        #'512010.XSHG',  # 医药ETF
        #'513050.XSHG'   # 中概互联
    ]

    # 【V4 核心参数】
    g.buy_drop = 0.03
    g.sell_rise = 0.03
    g.base_value = 80000   # 底仓5万元
    g.grid_value = 5000    # 每格交易5000元

    # 【v0.8 绝对格数 level】
    # level = 当前持有格数；建仓后 = base_level（≈10）；卖减买加；level=0 即清仓
    g.base_level = max(1, int(g.base_value / g.grid_value))
    g.max_extra_buy = 10
    g.max_level = g.base_level + g.max_extra_buy

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


def _calc_base_shares(price):
    """按当前价计算底仓股数（100 股取整）。"""
    if price <= 0:
        return 100
    shares = int(g.base_value / price / 100) * 100
    return shares if shares > 0 else 100


def _refresh_grid_shares(status, price):
    status['grid_shares'] = _calc_grid_shares(price)


def _get_position(context, ticker):
    """安全读取持仓，避免访问不存在的 positions[ticker] 触发聚宽 WARNING。"""
    if ticker not in context.portfolio.positions:
        return 0, 0
    pos = context.portfolio.positions[ticker]
    return pos.total_amount, pos.closeable_amount


def _clear_grid_status(ticker, display_name, note=''):
    if ticker not in g.grid_status:
        return False
    level = g.grid_status[ticker]['level']
    del g.grid_status[ticker]
    suffix = f" | {note}" if note else ''
    log.info(f"⬜【清仓】{display_name} | 等级:{level} | 网格状态已清除，待重装{suffix}")
    return True


def _clear_grid_if_empty(context, ticker, display_name):
    """level=0 且持仓为 0 时退出网格，下次走重装建仓。"""
    total_amount, _ = _get_position(context, ticker)
    if g.grid_status[ticker]['level'] == 0 and total_amount == 0:
        return _clear_grid_status(ticker, display_name)
    return False


def _flatten_remainder(context, ticker, display_name):
    """level 已减至 0 但仍有零碎持仓（底仓股数≠10×grid 取整误差）时卖清。"""
    total_amount, closeable = _get_position(context, ticker)
    if total_amount <= 0 or closeable <= 0:
        return False
    res = order(ticker, -closeable)
    if res:
        log.info(
            f"⬜【清仓-碎股】{display_name} | 卖出剩余:{closeable}股 | 等级:0"
        )
    _clear_grid_status(ticker, display_name, note='level=0 碎股清盘')
    return True


def _share_grids_from_amount(total_amount, grid_shares):
    """按持仓估算格数；不足一整格但有货时视为 1 格，避免误触发 level=0 碎股清仓。"""
    if total_amount <= 0 or grid_shares <= 0:
        return 0
    share_grids = total_amount // grid_shares
    if share_grids == 0 and total_amount >= 100:
        return 1
    return share_grids


def _sync_level_from_shares(status, total_amount):
    """底仓到账后，按实际股数同步绝对格数。"""
    grid_shares = status.get('grid_shares') or 100
    synced = _share_grids_from_amount(total_amount, grid_shares)
    if synced <= 0:
        status['level'] = 0
    else:
        status['level'] = min(g.max_level, synced)


def _sync_level_after_trade(status, total_amount):
    """成交后按实际持仓重算 level（替代盲目 level±1，防止部分卖出脱节）。"""
    grid_shares = status.get('grid_shares') or 100
    if total_amount <= 0:
        status['level'] = 0
        return
    status['level'] = min(g.max_level, _share_grids_from_amount(total_amount, grid_shares))


def _calibrate_level_from_position(status, total_amount, grid_shares):
    """level 明显高于持仓可支撑格数时下调。"""
    if total_amount <= 0 or grid_shares <= 0:
        return False
    share_grids = _share_grids_from_amount(total_amount, grid_shares)
    old_level = status['level']
    if old_level <= share_grids:
        return False
    status['level'] = share_grids
    return old_level != share_grids


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
            base_shares = _calc_base_shares(current_price)
            grid_shares = _calc_grid_shares(current_price)
            
            need_cash = base_shares * current_price
            if context.portfolio.available_cash > need_cash:
                res = order(ticker, base_shares)
                if res:
                    g.grid_status[ticker] = {
                        'ref_price': current_price,
                        'grid_shares': grid_shares,
                        'level': 0,
                        'pending_base': True,
                    }
                    total_amount, _ = _get_position(context, ticker)
                    log.info(
                        f"🚀【重装建仓】{display_name} | 价格:{current_price:.3f} | 买入:{base_shares}股 "
                        f"| 总持仓:{total_amount}股 | 待确认底仓"
                    )
            #else:
            #    log.warn(
            #        f"⚠️【建仓跳过】{display_name} | 可用现金:{context.portfolio.available_cash:.0f} "
            #        f"< 需:{need_cash:.0f} | 价格:{current_price:.3f}"
            #    )

        # ================= 阶段二：网格交易 =================
        else:
            status = g.grid_status[ticker]
            ref_price = status['ref_price']
            level = status['level']
            grid_shares = status['grid_shares']
            pending_base = status.get('pending_base', False)
            total_amount, closeable = _get_position(context, ticker)

            # 底仓买单尚未到账：512100 等上市首日常见 weeks 延迟；确认前按 level=0 跑网格
            if pending_base and total_amount > 0:
                status['pending_base'] = False
                _sync_level_from_shares(status, total_amount)
                _refresh_grid_shares(status, current_price)
                level = status['level']
                log.info(
                    f"✅【底仓到账】{display_name} | 总持仓:{total_amount}股 | 等级:{level}"
                )

            # level 高于持仓能支撑的格数 → 校准（512100 2020 后卖侧停摆根因）
            if not pending_base and total_amount > 0:
                if _calibrate_level_from_position(status, total_amount, grid_shares):
                    level = status['level']
                    log.info(
                        f"⚠️【等级校准】{display_name} | 总持仓:{total_amount}股 "
                        f"| 每格:{grid_shares}股 | 等级→{level}"
                    )

            # level=0 且非待确认底仓：应已清仓；若仍有碎股则卖清后退出
            if level <= 0 and not pending_base:
                if total_amount > 0:
                    _flatten_remainder(context, ticker, display_name)
                elif ticker in g.grid_status:
                    _clear_grid_status(ticker, display_name)
                continue

            # 持仓为 0：待确认底仓时继续跑网格（与 v0.7 一致）；否则视为异常
            if total_amount == 0:
                if pending_base:
                    pass
                else:
                    _clear_grid_status(
                        ticker, display_name,
                        note=f'持仓已空（记录等级:{level}）'
                    )
                    continue

            
            # 1. 触发向下买入（买加）
            if current_price <= ref_price * (1 - g.buy_drop):
                if level < g.max_level:
                    need_cash = grid_shares * current_price
                    if context.portfolio.available_cash > need_cash:
                        res = order(ticker, grid_shares)
                        if res:
                            status['ref_price'] = current_price
                            status['level'] += 1
                            _refresh_grid_shares(status, current_price)
                            total_amount, _ = _get_position(context, ticker)
                            log.info(
                                f"🔴【加仓】{display_name} | 价格:{current_price:.3f} | 买入:{grid_shares}股 "
                                f"| 总持仓:{total_amount}股 | 等级:{status['level']}"
                            )
                    #else:
                    #    log.warn(
                    #        f"⚠️【加仓跳过】{display_name} | 可用现金:{context.portfolio.available_cash:.0f} "
                    #        f"< 需:{need_cash:.0f} | 价格:{current_price:.3f} | 等级:{level} "
                    #        f"| 每格:{grid_shares}股"
                    #    )
                #else:
                #    log.warn(
                #        f"⚠️【加仓跳过】{display_name} | 等级已满:{level}>={g.max_level} "
                #        f"| 价格:{current_price:.3f} | ref:{ref_price:.3f}"
                #    )

            # 2. 触发向上卖出（卖减）
            elif current_price >= ref_price * (1 + g.sell_rise):
                sell_shares = grid_shares
                if closeable < grid_shares:
                    sell_shares = int(closeable / 100) * 100
                if sell_shares >= 100:
                    res = order(ticker, -sell_shares)
                    if res:
                        trade_value = sell_shares * current_price
                        cost_value = sell_shares * ref_price
                        fee = max(0.1, trade_value * 0.0001)
                        profit = trade_value - cost_value - fee
                        g.grid_t_profits[ticker] += profit
                        
                        status['ref_price'] = current_price
                        _refresh_grid_shares(status, current_price)
                        total_amount, _ = _get_position(context, ticker)
                        _sync_level_after_trade(status, total_amount)
                        partial = '（部分）' if sell_shares < grid_shares else ''
                        log.info(
                            f"🟢【止盈{partial}】{display_name} | 价格:{current_price:.3f} | 卖出:{sell_shares}股 "
                            f"| 总持仓:{total_amount}股 | 做T净赚:{profit:.2f} | 剩余等级:{status['level']}"
                        )
                        if status['level'] <= 0:
                            if total_amount > 0:
                                _flatten_remainder(context, ticker, display_name)
                            else:
                                _clear_grid_if_empty(context, ticker, display_name)
                    #else:
                    #    log.warn(
                    #        f"⚠️【止盈跳过】{display_name} | 可卖:{closeable}股 < 需卖:{grid_shares}股 "
                    #        f"| 价格:{current_price:.3f} | 等级:{level} | ref:{ref_price:.3f} "
                    #        f"| 触发价:{ref_price * (1 + g.sell_rise):.3f}"
                    #    )
                #else:
                #    log.warn(
                #        f"⚠️【止盈跳过】{display_name} | 已清仓等级:0 "
                #        f"| 价格:{current_price:.3f} | ref:{ref_price:.3f}"
                #    )

        if ticker in g.grid_status:
            plot_key = f"{g.stock_names[ticker]}_做T利润"
            current_yields[plot_key] = g.grid_t_profits[ticker]

    if current_yields:
        record(**current_yields)
