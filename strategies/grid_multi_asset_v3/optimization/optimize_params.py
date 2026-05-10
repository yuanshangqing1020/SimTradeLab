# strategies/grid_multi_asset_v3/optimization/optimize_params.py
# -*- coding: utf-8 -*-
"""
多标的自适应网格策略 v3 — Walk-Forward 参数优化器

在 v2 的 11 维上增加：
  BEAR_UNIVERSE_MODE (2) × BEAR_GRID_MODE (3) × BEAR_GRID_MAX_LAYER_CAP (3)
理论组合数 = v2 组合数 × 18（空间较大，可调低 bear_cap 档位做冒烟）

优化期 / 留存期 / WF 窗口与 v2 对齐，便于与 Trial 29 横比。

运行:
    cd <SimTradeLab 根目录>
    conda run -n SimTrade python strategies/grid_multi_asset_v3/optimization/optimize_params.py
"""

from simtradelab.backtest.optimizer_framework import (
    ParameterSpace,
    optimize_strategy,
)


class GridMultiAssetV3Params(ParameterSpace):
    """v3 = v2 参数空间 + M1/M2 三离散维。"""

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

    BEAR_UNIVERSE_MODE      = ['SAME', 'ETF_DEFENSIVE']
    BEAR_GRID_MODE          = ['NORMAL', 'NO_NET_ADD', 'CAP_LAYER']
    BEAR_GRID_MAX_LAYER_CAP = [0, 1, 2]

    @staticmethod
    def validate(params):
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
        if params['BEAR_GRID_MODE'] == 'CAP_LAYER':
            if int(params['BEAR_GRID_MAX_LAYER_CAP']) > int(params['GRID_MAX_LAYER']):
                raise ValueError(
                    'CAP_LAYER 要求 BEAR_GRID_MAX_LAYER_CAP<=GRID_MAX_LAYER: {} > {}'.format(
                        params['BEAR_GRID_MAX_LAYER_CAP'], params['GRID_MAX_LAYER'],
                    )
                )
        return params


if __name__ == '__main__':
    custom_mapping = {
        'MAX_HOLD':                'context.MAX_HOLD',
        'GRID_STEP_VOL_FACTOR':    'context.GRID_STEP_VOL_FACTOR',
        'GRID_STEP_MIN':           'context.GRID_STEP_MIN',
        'GRID_STEP_MAX':           'context.GRID_STEP_MAX',
        'GRID_MAX_LAYER':          'context.GRID_MAX_LAYER',
        'LAYER_FRACTION':          'context.LAYER_FRACTION',
        'VOL_WEIGHT':              'context.VOL_WEIGHT',
        'REBALANCE_FREQ':          'context.REBALANCE_FREQ',
        'BULL_RATIO':              'context.BULL_RATIO',
        'NEUTRAL_RATIO':           'context.NEUTRAL_RATIO',
        'BEAR_RATIO':              'context.BEAR_RATIO',
        'BEAR_UNIVERSE_MODE':      'context.BEAR_UNIVERSE_MODE',
        'BEAR_GRID_MODE':          'context.BEAR_GRID_MODE',
        'BEAR_GRID_MAX_LAYER_CAP': 'context.BEAR_GRID_MAX_LAYER_CAP',
    }

    optimize_strategy(
        parameter_space=GridMultiAssetV3Params,
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
        patience=500,
    )
