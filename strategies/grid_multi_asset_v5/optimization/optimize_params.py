# strategies/grid_multi_asset_v5/optimization/optimize_params.py
# -*- coding: utf-8 -*-
"""
多标的自适应网格策略 v5 — Walk-Forward 参数优化

参数空间含 UNIVERSE_MODE（ANCHOR_SATELLITE / WIDE_V2）、合并 ETF 池下的 MAX_HOLD、MIN_ANCHORS_IN_POOL 等。
三重门禁（FULL/RECENT/II/I）在 WF 结束后由 gate_eval.py / post_select_eligible.py 批处理。

运行:
    cd /path/to/SimTradeLab
    conda run -n SimTrade python strategies/grid_multi_asset_v5/optimization/optimize_params.py
"""

from simtradelab.backtest.optimizer_framework import (
    ParameterSpace,
    optimize_strategy,
)

# 与 template.py 保持一致的池规模（修改 template 常量时请同步此处）
CANDIDATE_ETFS = [
    '510300.SS', '510500.SS', '159915.SZ', '512880.SS', '512690.SS',
    '512010.SS', '515050.SS', '512480.SS', '159949.SZ', '588000.SS',
    '512170.SS', '512760.SS', '159792.SZ', '513100.SS', '513050.SS',
]
ANCHOR_ETF_UNIVERSE = ['510300.SS', '510500.SS']
NARROW_ETF_UNIVERSE = [
    '510300.SS',
    '510500.SS',
    '159915.SZ',
    '512010.SS',
    '513100.SS',
    '588000.SS',
]
NARROW_ETF_POOL_SIZE = len(NARROW_ETF_UNIVERSE)
_ANCHOR_SET = frozenset(ANCHOR_ETF_UNIVERSE)

DEFENSIVE_ETF_POOL = [
    '510300.SS',
    '510500.SS',
    '159915.SZ',
    '588000.SS',
    '512880.SS',
]
_DEFENSIVE_SET_FS = frozenset(DEFENSIVE_ETF_POOL)


def _v5_satellite_etfs():
    out = []
    seen = set()
    for x in NARROW_ETF_UNIVERSE + CANDIDATE_ETFS:
        if x in _ANCHOR_SET or x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def _defensive_overlap_for_universe_mode(universe_mode):
    if universe_mode == 'WIDE_V2':
        return [e for e in CANDIDATE_ETFS if e in _DEFENSIVE_SET_FS]
    if universe_mode == 'NARROW_ETF':
        return [e for e in NARROW_ETF_UNIVERSE if e in _DEFENSIVE_SET_FS]
    return [
        e for e in (ANCHOR_ETF_UNIVERSE + _v5_satellite_etfs())
        if e in _DEFENSIVE_SET_FS
    ]


def _v5_combined_anchor_satellite_size():
    sat = _v5_satellite_etfs()
    return len(ANCHOR_ETF_UNIVERSE) + len(sat)


V5_COMB_UNIVERSE_SIZE = _v5_combined_anchor_satellite_size()


class GridMultiAssetV5Params(ParameterSpace):
    """v5 可调参数（ANCHOR_SATELLITE 下 MAX_HOLD ≤ 合并池长）。"""

    UNIVERSE_MODE        = ['ANCHOR_SATELLITE', 'WIDE_V2']
    MAX_HOLD             = [k for k in [3, 4, 5, 6, 8, 10] if k <= V5_COMB_UNIVERSE_SIZE]
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
    MIN_ANCHORS_IN_POOL  = [1, 2]
    BEAR_UNIVERSE_MODE   = ['SAME', 'ETF_DEFENSIVE']
    BEAR_GRID_MODE       = ['NORMAL', 'CAP_LAYER', 'NO_NET_ADD']
    BEAR_GRID_MAX_LAYER_CAP = [0, 1]

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
        mode = params['UNIVERSE_MODE']
        if mode == 'WIDE_V2':
            cap = len(CANDIDATE_ETFS)
        elif mode == 'NARROW_ETF':
            cap = NARROW_ETF_POOL_SIZE
        else:
            cap = V5_COMB_UNIVERSE_SIZE
        if params['MAX_HOLD'] > cap:
            raise ValueError(
                'MAX_HOLD={} 大于 {} 模式下的候选上限 {}'.format(
                    params['MAX_HOLD'], mode, cap,
                )
            )
        min_a = params['MIN_ANCHORS_IN_POOL']
        if min_a > len(ANCHOR_ETF_UNIVERSE):
            raise ValueError(
                'MIN_ANCHORS_IN_POOL={} 大于锚定池只数 {}'.format(
                    min_a, len(ANCHOR_ETF_UNIVERSE),
                )
            )
        if min_a > params['MAX_HOLD']:
            raise ValueError(
                'MIN_ANCHORS_IN_POOL={} 大于 MAX_HOLD={}'.format(
                    min_a, params['MAX_HOLD'],
                )
            )
        if params['BEAR_UNIVERSE_MODE'] == 'ETF_DEFENSIVE':
            if not _defensive_overlap_for_universe_mode(mode):
                raise ValueError(
                    'BEAR ETF_DEFENSIVE 与 UNIVERSE_MODE={} 下 ETF 无交集'.format(mode),
                )
        return params


V5_CUSTOM_MAPPING = {
    'UNIVERSE_MODE':           'context.UNIVERSE_MODE',
    'MIN_ANCHORS_IN_POOL':     'context.MIN_ANCHORS_IN_POOL',
    'BEAR_UNIVERSE_MODE':      'context.BEAR_UNIVERSE_MODE',
    'BEAR_GRID_MODE':          'context.BEAR_GRID_MODE',
    'BEAR_GRID_MAX_LAYER_CAP': 'context.BEAR_GRID_MAX_LAYER_CAP',
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
        parameter_space=GridMultiAssetV5Params,
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
        custom_mapping=V5_CUSTOM_MAPPING,
        resume=True,
        verbose=False,
        strategy_file='template.py',
        patience=500,
    )
