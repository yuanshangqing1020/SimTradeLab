# strategies/grid_multi_asset_v4/optimization/optimize_params.py
# -*- coding: utf-8 -*-
"""
多标的自适应网格策略 v4 — Walk-Forward 参数优化

参数空间在 v2 基础上将 MAX_HOLD 收窄为适合默认窄 ETF 池（最多 6 只）的候选。
三重门禁（FULL/RECENT/II/I）在 WF 结束后由 gate_eval.py / post_select_eligible.py 批处理。

运行:
    cd /path/to/SimTradeLab
    conda run -n SimTrade python strategies/grid_multi_asset_v4/optimization/optimize_params.py
"""

from simtradelab.backtest.optimizer_framework import (
    ParameterSpace,
    optimize_strategy,
)


NARROW_ETF_POOL_SIZE = 6  # 与 template.NARROW_ETF_UNIVERSE 长度一致


class GridMultiAssetV4Params(ParameterSpace):
    """v4 可调参数（默认窄池 MAX_HOLD ≤ 6）。"""

    MAX_HOLD             = [3, 4, 5, 6]
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
        if params['MAX_HOLD'] > NARROW_ETF_POOL_SIZE:
            raise ValueError(
                'MAX_HOLD={} 大于窄 ETF 池规模 {}'.format(
                    params['MAX_HOLD'], NARROW_ETF_POOL_SIZE,
                )
            )
        return params


V4_CUSTOM_MAPPING = {
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


if __name__ == '__main__':
    optimize_strategy(
        parameter_space=GridMultiAssetV4Params,
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
        custom_mapping=V4_CUSTOM_MAPPING,
        resume=True,
        verbose=False,
        strategy_file='template.py',
        #patience=500,
    )
