# strategies/grid_multi_asset_v1/optimization/optimize_params.py
# -*- coding: utf-8 -*-
"""
多标的自适应网格策略 v1 - Walk-Forward 参数优化器

参数空间: 4×3×2×2×3×3×3×3 = 3,888 组合
优化期: 2019-01-01 ~ 2024-12-31（6年，覆盖多轮牛熊）
留存期: 2025-01-01 ~ 2026-03-31（样本外泛化验证）

运行方式:
    cd /mnt/c/Quant-Workspace/SimTradeLab
    conda run -n SimTrade python strategies/grid_multi_asset_v1/optimization/optimize_params.py

断点续传: 直接重新运行，Optuna 自动从 results/optuna_journal.log 恢复
策略模板: ../template.py（优化器读取该文件并注入参数）
直接回测: ../backtest.py（含当前最优参数，可通过 run_backtest.py 直接运行）
"""

from simtradelab.backtest.optimizer_framework import (
    ParameterSpace,
    optimize_strategy,
)


class GridMultiAssetParams(ParameterSpace):
    """可调参数空间。

    参数空间大小: 4×3×2×2×3×3×3×3 = 3,888 组合
    Early-stopping patience: ~972 次无改进后自动停止
    """

    MAX_HOLD             = [10, 20, 30, 50]
    GRID_STEP_VOL_FACTOR = [0.30, 0.45, 0.60]
    GRID_STEP_MIN        = [0.01, 0.02]
    GRID_STEP_MAX        = [0.03, 0.05]
    GRID_MAX_LAYER       = [2, 3, 4]
    LAYER_FRACTION       = [0.08, 0.12, 0.16]
    VOL_WEIGHT           = [0.50, 0.65, 0.80]
    REBALANCE_FREQ       = [5, 10, 20]

    @staticmethod
    def validate(params):
        """拒绝 GRID_STEP_MIN >= GRID_STEP_MAX 的无效组合。"""
        if params['GRID_STEP_MIN'] >= params['GRID_STEP_MAX']:
            raise ValueError(
                'GRID_STEP_MIN={} 必须小于 GRID_STEP_MAX={}'.format(
                    params['GRID_STEP_MIN'], params['GRID_STEP_MAX'],
                )
            )
        return params


if __name__ == '__main__':
    # optimizer_framework 默认用 g.{param_name} 替换，
    # 我们的策略用 context.* 存储参数，需指定 custom_mapping。
    custom_mapping = {
        'MAX_HOLD':             'context.MAX_HOLD',
        'GRID_STEP_VOL_FACTOR': 'context.GRID_STEP_VOL_FACTOR',
        'GRID_STEP_MIN':        'context.GRID_STEP_MIN',
        'GRID_STEP_MAX':        'context.GRID_STEP_MAX',
        'GRID_MAX_LAYER':       'context.GRID_MAX_LAYER',
        'LAYER_FRACTION':       'context.LAYER_FRACTION',
        'VOL_WEIGHT':           'context.VOL_WEIGHT',
        'REBALANCE_FREQ':       'context.REBALANCE_FREQ',
    }

    optimize_strategy(
        parameter_space=GridMultiAssetParams,
        optimization_period=('2019-01-01', '2024-12-31'),
        holdout_period=('2025-01-01', '2026-03-31'),
        initial_capital=500_000.0,
        walk_forward_config={
            'train_months': 24,   # 24个月训练窗口
            'test_months':  6,    # 6个月验证窗口
            'step_months':  6,    # 每次向前滑动6个月
        },
        regularization_weight=0.1,
        stability_weight=0.5,
        custom_mapping=custom_mapping,
        resume=True,
        verbose=False,
        strategy_file='template.py',  # 优化器读取 template.py 作为参数注入模板
    )
