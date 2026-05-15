# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 Kay
#
# This file is part of SimTradeLab, dual-licensed under AGPL-3.0 and a
# commercial license. See LICENSE-COMMERCIAL.md or contact kayou@duck.com
#
"""
本地回测入口 - 配置与启动

简化的入口文件，仅保留配置参数
"""


import sys

# 确保控制台 UTF-8 编码和实时输出（兼容 Windows）
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8')

from simtradelab.backtest.runner import BacktestRunner
from simtradelab.backtest.config import BacktestConfig


if __name__ == '__main__':
    # ==================== 回测配置 ====================

    # 策略：v5 Trial 190 默认参数见 strategies/grid_multi_asset_v5/backtest.py
    strategy_name = 'core_grid_hybrid_v1'
    # strategy_name = 'grid_multi_asset_v4'
    # strategy_name = 'grid_multi_asset_v3'
    # strategy_name = 'grid_multi_asset_v2'
    # strategy_name = 'core_grid_hybrid_v1'  # 见 strategies/core_grid_hybrid_v1/backtest.py

    # 全长口径与 my_docs v1/v2/v3 总结 §5.2 对齐（2019-01-01～2026-04-20）
    # 复现 v3 Holdout 时改为 end_date='2026-03-31'（与 optimize_params holdout_period 一致）
    start_date = '2019-01-01'
    end_date = '2026-04-20'

    # ==================== 启动回测 ====================

    # 创建配置
    config = BacktestConfig(
        strategy_name=strategy_name,
        start_date=start_date,
        end_date=end_date,
        initial_capital=500000.0  # 策略设计容量 50 万
    )

    # 运行回测
    runner = BacktestRunner()
    report = runner.run(config=config)
