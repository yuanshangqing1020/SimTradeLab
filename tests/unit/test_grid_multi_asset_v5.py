# tests/unit/test_grid_multi_asset_v5.py
# -*- coding: utf-8 -*-
from pathlib import Path
import types

import numpy as np
import pandas as pd
import pytest

_STRATEGY_PATH = Path(__file__).parents[2] / 'strategies' / 'grid_multi_asset_v5' / 'template.py'


def _load_template_ns():
    _log = types.SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
    )
    mock_ns = {
        '__builtins__': __builtins__,
        'np': np,
        'set_benchmark': lambda *a, **kw: None,
        'set_slippage': lambda *a, **kw: None,
        'get_history': lambda *a, **kw: pd.DataFrame(),
        'get_fundamentals': lambda *a, **kw: pd.DataFrame(),
        'get_index_stocks': lambda *a, **kw: [],
        'get_stock_status': lambda *a, **kw: {},
        'order_target': lambda *a, **kw: None,
        'order_target_value': lambda *a, **kw: None,
        'log': _log,
    }
    src = _STRATEGY_PATH.read_text(encoding='utf-8')
    exec(compile(src, str(_STRATEGY_PATH), 'exec'), mock_ns)
    return mock_ns


def test_build_grid_pool_anchor_first_order():
    ns = _load_template_ns()
    fn = ns['build_grid_pool_anchor_first']
    ranked = ['512010.SS', '510300.SS', '159915.SZ', '588000.SS']
    anchors = ['510300.SS', '510500.SS']
    pool = fn(ranked, anchors, max_hold=3)
    assert pool == ['510300.SS', '512010.SS', '159915.SZ']


def test_build_grid_pool_respects_max_hold():
    ns = _load_template_ns()
    fn = ns['build_grid_pool_anchor_first']
    ranked = ['A', 'B', 'C', 'D']
    anchors = ['X', 'Y']
    assert fn(ranked, anchors, max_hold=2) == ['A', 'B']


def test_satellite_universe_at_least_eight():
    ns = _load_template_ns()
    sat = ns['SATELLITE_ETF_UNIVERSE']
    assert len(sat) >= 8


def test_v5_combined_size_matches_optimize_params():
    ns = _load_template_ns()
    opt = _load_optimize_params()
    assert ns['V5_COMBINED_POOL_SIZE'] == opt.V5_COMB_UNIVERSE_SIZE


def test_effective_max_layer_bear_cap_layer():
    ns = _load_template_ns()
    fn = ns['_effective_max_layer_bear']
    assert fn('BEAR', 'CAP_LAYER', 2, 0) == 0
    assert fn('BEAR', 'NORMAL', 2, 0) == 2
    assert fn('NEUTRAL', 'CAP_LAYER', 2, 0) == 2


def test_apply_no_net_add_clips_to_prev():
    ns = _load_template_ns()
    fn = ns['_apply_no_net_add_targets']
    prev = {'510300.SS': 1000.0}
    tgt = {'510300.SS': 2000.0, '159915.SZ': 500.0}
    out = fn(prev, tgt)
    assert out['510300.SS'] == 1000.0
    assert out['159915.SZ'] == 500.0


def _load_gate_eval():
    import importlib.util

    p = (
        Path(__file__).parents[2]
        / 'strategies'
        / 'grid_multi_asset_v5'
        / 'optimization'
        / 'gate_eval.py'
    )
    spec = importlib.util.spec_from_file_location('_v5_gate_unit', str(p))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_gates_equals_i_ii_and_iii():
    ge = _load_gate_eval()
    thr = ge.GateThresholds()
    m_full = {
        'max_drawdown': -0.20,
        'excess_return': 0.0,
        'information_ratio': 0.0,
    }
    m_recent = {
        'annual_return': 0.25,
        'max_drawdown': -0.10,
        'sharpe_ratio': 1.5,
    }
    ok, _ = ge.check_gates(m_full, m_recent, thr)
    ok12, _ = ge.check_gates_i_ii_only(m_full, thr)
    ok3, _ = ge.check_gates_iii_only(m_recent, thr)
    assert ok == (ok12 and ok3)


def test_row_to_params_two_stage():
    import importlib.util

    p = (
        Path(__file__).parents[2]
        / 'strategies'
        / 'grid_multi_asset_v5'
        / 'optimization'
        / 'two_stage_select.py'
    )
    spec = importlib.util.spec_from_file_location('_v5_ts_unit', str(p))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    row = pd.Series({
        'number': 1,
        'params_MAX_HOLD': 6.0,
        'params_UNIVERSE_MODE': 'ANCHOR_SATELLITE',
    })
    d = mod.row_to_params(row)
    assert d['MAX_HOLD'] == 6
    assert d['UNIVERSE_MODE'] == 'ANCHOR_SATELLITE'


def _load_optimize_params():
    import importlib.util

    p = Path(__file__).parents[2] / 'strategies' / 'grid_multi_asset_v5' / 'optimization' / 'optimize_params.py'
    spec = importlib.util.spec_from_file_location('_v5_opt_unit', str(p))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGridMultiAssetV5ParamsValidate:
    def test_max_hold_exceeds_pool_rejected(self):
        opt = _load_optimize_params()
        cap = opt.V5_COMB_UNIVERSE_SIZE

        with pytest.raises(ValueError, match='候选上限'):
            opt.GridMultiAssetV5Params.validate({
                'UNIVERSE_MODE': 'ANCHOR_SATELLITE',
                'MIN_ANCHORS_IN_POOL': 1,
                'BEAR_UNIVERSE_MODE': 'SAME',
                'BEAR_GRID_MODE': 'NORMAL',
                'BEAR_GRID_MAX_LAYER_CAP': 0,
                'MAX_HOLD': cap + 1,
                'GRID_STEP_MIN': 0.01,
                'GRID_STEP_MAX': 0.05,
                'BEAR_RATIO': 0.25,
                'NEUTRAL_RATIO': 0.5,
                'BULL_RATIO': 0.7,
            })

    def test_valid_params(self):
        opt = _load_optimize_params()

        p = opt.GridMultiAssetV5Params.validate({
            'UNIVERSE_MODE': 'ANCHOR_SATELLITE',
            'MIN_ANCHORS_IN_POOL': 1,
            'BEAR_UNIVERSE_MODE': 'SAME',
            'BEAR_GRID_MODE': 'CAP_LAYER',
            'BEAR_GRID_MAX_LAYER_CAP': 0,
            'MAX_HOLD': 6,
            'GRID_STEP_MIN': 0.01,
            'GRID_STEP_MAX': 0.05,
            'BEAR_RATIO': 0.25,
            'NEUTRAL_RATIO': 0.5,
            'BULL_RATIO': 0.7,
        })
        assert p['MAX_HOLD'] == 6
