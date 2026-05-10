# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 Kay
#
# This file is part of SimTradeLab, dual-licensed under AGPL-3.0 and a
# commercial license. See LICENSE-COMMERCIAL.md or contact kayou@duck.com
#
"""
本地批量回测入口：按策略列表顺序逐个执行回测
"""

import sys

# 确保控制台 UTF-8 编码和实时输出（兼容 Windows）
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8')

from simtradelab.backtest.config import BacktestConfig
from simtradelab.backtest.runner import BacktestRunner


def run_batch_backtests(
    strategy_names: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
) -> dict[str, dict]:
    """
    顺序执行多个策略回测。

    回测结果文件（日志/图表/导出）仍由 BacktestConfig 的 log_dir 规则决定，
    即自动写入每个策略目录下的 stats 子目录。
    """
    total = len(strategy_names)
    results: dict[str, dict] = {}

    for idx, strategy_name in enumerate(strategy_names, 1):
        print(f"\n[{idx}/{total}] 开始回测策略: {strategy_name}")
        # 每个策略使用独立Runner，避免跨策略缓存/状态污染结果
        runner = BacktestRunner()
        config = BacktestConfig(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )
        report = runner.run(config=config)
        results[strategy_name] = report
        print(f"[{idx}/{total}] 策略完成: {strategy_name}")

    return results


if __name__ == '__main__':
    # ==================== 批量回测配置 ====================

    # 策略名称列表（按顺序执行）
    strategy_names = [
        'bollinger_mean_reversion',
        'compliant_small_cap',
        'compliant_small_cap_noincome',
    ]

    # 回测周期
    start_date = '2016-01-01'
    end_date = '2026-04-20'

    # 初始资金
    initial_capital = 100000.0

    # ==================== 启动批量回测 ====================
    run_batch_backtests(
        strategy_names=strategy_names,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )
