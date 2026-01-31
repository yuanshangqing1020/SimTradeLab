import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simtradelab.backtest.runner import BacktestRunner
from simtradelab.backtest.config import BacktestConfig

if __name__ == '__main__':
    # 策略名称 (strategies/目录下对应的文件夹名)
    strategy_name = 'grid_trading_002626'

    # 回测周期
    start_date = '2025-01-01'
    end_date = '2025-06-30'  # 跑半年验证网格效果

    print(f"准备运行网格策略: {strategy_name}")
    print(f"回测周期: {start_date} 至 {end_date}")

    # 创建配置
    config = BacktestConfig(
        strategy_name=strategy_name,
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000.0,
        enable_logging=True,
        enable_charts=True
    )

    # 运行回测
    runner = BacktestRunner()
    report = runner.run(config=config)
