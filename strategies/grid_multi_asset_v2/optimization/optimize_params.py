# strategies/grid_multi_asset_v2/optimization/optimize_params.py
# -*- coding: utf-8 -*-
"""
多标的自适应网格策略 v2 — Walk-Forward 参数优化器

参数空间: 5×3×2×2×3×3×3×3×3×3×3 = 11,664 组合
  v1 基础上新增 BULL_RATIO/NEUTRAL_RATIO/BEAR_RATIO，
  MAX_HOLD 候选值加密为 5/8/10/12/15
优化期: 2019-01-01 ~ 2024-12-31
留存期: 2025-01-01 ~ 2026-03-31（与 v1 相同口径，便于直接对比）

运行方式:
    cd /mnt/c/Quant-Workspace/SimTradeLab
    conda run -n SimTrade python strategies/grid_multi_asset_v2/optimization/optimize_params.py

断点续传: 直接重新运行，Optuna 从 results/optuna_journal.log 恢复
"""

from simtradelab.backtest.optimizer_framework import (
    ParameterSpace,
    optimize_strategy,
)


class GridMultiAssetV2Params(ParameterSpace):
    """v2 可调参数空间。

    参数空间大小: 5×3×2×2×3×3×3×3×3×3×3 = 11,664 组合
    Early-stopping patience: ~2916 次无改进后自动停止
    """

    MAX_HOLD             = [5, 8, 10, 12, 15]
    GRID_STEP_VOL_FACTOR = [0.30, 0.45, 0.60]
    GRID_STEP_MIN        = [0.01, 0.02]
    GRID_STEP_MAX        = [0.03, 0.05]
    GRID_MAX_LAYER       = [2, 3, 4]
    LAYER_FRACTION       = [0.08, 0.12, 0.16]
    VOL_WEIGHT           = [0.50, 0.65, 0.80]
    REBALANCE_FREQ       = [5, 10, 20]
    BULL_RATIO           = [0.70, 0.80, 0.90]
    NEUTRAL_RATIO        = [0.50, 0.60, 0.70]
    BEAR_RATIO           = [0.25, 0.35, 0.45]

    @staticmethod
    def validate(params):
        """拒绝无效参数组合。"""
        if params['GRID_STEP_MIN'] >= params['GRID_STEP_MAX']:
            raise ValueError(
                'GRID_STEP_MIN={} 必须小于 GRID_STEP_MAX={}'.format(
                    params['GRID_STEP_MIN'], params['GRID_STEP_MAX'],
                )
            )
        if not (params['BEAR_RATIO'] < params['NEUTRAL_RATIO'] < params['BULL_RATIO']):
            raise ValueError(
                'BEAR_RATIO < NEUTRAL_RATIO < BULL_RATIO 约束违反: '
                '{} / {} / {}'.format(
                    params['BEAR_RATIO'], params['NEUTRAL_RATIO'], params['BULL_RATIO'],
                )
            )
        return params


if __name__ == '__main__':
    custom_mapping = {
        'MAX_HOLD':             'context.MAX_HOLD',
        'GRID_STEP_VOL_FACTOR': 'context.GRID_STEP_VOL_FACTOR',
        'GRID_STEP_MIN':        'context.GRID_STEP_MIN',
        'GRID_STEP_MAX':        'context.GRID_STEP_MAX',
        'GRID_MAX_LAYER':       'context.GRID_MAX_LAYER',
        'LAYER_FRACTION':       'context.LAYER_FRACTION',
        'VOL_WEIGHT':           'context.VOL_WEIGHT',
        'REBALANCE_FREQ':       'context.REBALANCE_FREQ',
        'BULL_RATIO':           'context.BULL_RATIO',
        'NEUTRAL_RATIO':        'context.NEUTRAL_RATIO',
        'BEAR_RATIO':           'context.BEAR_RATIO',
    }

    optimize_strategy(
        parameter_space=GridMultiAssetV2Params,
        optimization_period=('2019-01-01', '2024-12-31'),
        holdout_period=('2025-01-01', '2026-03-31'),
        initial_capital=500000.0,
        walk_forward_config={
            'train_months': 24,
            'test_months':  6,
            'step_months':  6,
        },
        regularization_weight=0.1,
        stability_weight=0.5,
        custom_mapping=custom_mapping,
        resume=True,
        verbose=False,
        strategy_file='template.py',
    )
