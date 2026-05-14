# tests/unit/test_grid_multi_asset_v4.py
# -*- coding: utf-8 -*-
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_PATH = Path(__file__).parents[2] / 'strategies' / 'grid_multi_asset_v4' / 'template.py'


def _load_template_ns():
    _log = types.SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
    )
    mock_ns = {
        '__builtins__': __builtins__,
        'np': np,
        'set_benchmark':      lambda *a, **kw: None,
        'set_slippage':       lambda *a, **kw: None,
        'get_history':        lambda *a, **kw: pd.DataFrame(),
        'get_fundamentals':   lambda *a, **kw: pd.DataFrame(),
        'get_index_stocks':   lambda *a, **kw: [],
        'get_stock_status':   lambda *a, **kw: {},
        'order_target':       lambda *a, **kw: None,
        'order_target_value': lambda *a, **kw: None,
        'log': _log,
    }
    src = _STRATEGY_PATH.read_text(encoding='utf-8')
    exec(compile(src, str(_STRATEGY_PATH), 'exec'), mock_ns)
    return mock_ns


_ns = _load_template_ns()
_regime_refresh_day = _ns['_regime_refresh_day']
_etf_list_for_mode = _ns['_etf_list_for_mode']
_max_hold_cap = _ns['_max_hold_cap']


class TestRegimeRefreshDay:
    def test_weekly_includes_trading_week_multiples(self):
        assert _regime_refresh_day(1, 20, 'WEEKLY') is True
        assert _regime_refresh_day(6, 20, 'WEEKLY') is True
        assert _regime_refresh_day(5, 20, 'WEEKLY') is False
        assert _regime_refresh_day(20, 20, 'WEEKLY') is True  # 换仓日

    def test_on_rebalance_only(self):
        assert _regime_refresh_day(10, 10, 'ON_REBALANCE_ONLY') is True
        assert _regime_refresh_day(11, 10, 'ON_REBALANCE_ONLY') is False


class TestNarrowUniverse:
    def test_narrow_list_length(self):
        assert len(_etf_list_for_mode('NARROW_ETF')) == 6
        assert len(_etf_list_for_mode('WIDE_V2')) == 15

    def test_max_hold_cap(self):
        assert _max_hold_cap('NARROW_ETF') == 6
        assert _max_hold_cap('WIDE_V2') == 15


def _load_optimize_params():
    import importlib.util

    p = Path(__file__).parents[2] / 'strategies' / 'grid_multi_asset_v4' / 'optimization' / 'optimize_params.py'
    spec = importlib.util.spec_from_file_location('_v4_opt_unit', str(p))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGridMultiAssetV4ParamsValidate:
    def test_max_hold_exceeds_pool_rejected(self):
        opt = _load_optimize_params()

        with pytest.raises(ValueError, match='窄 ETF'):
            opt.GridMultiAssetV4Params.validate({
                'MAX_HOLD': 7,
                'GRID_STEP_MIN': 0.01,
                'GRID_STEP_MAX': 0.05,
                'BEAR_RATIO': 0.25,
                'NEUTRAL_RATIO': 0.5,
                'BULL_RATIO': 0.7,
            })

    def test_valid_params(self):
        opt = _load_optimize_params()

        p = opt.GridMultiAssetV4Params.validate({
            'MAX_HOLD': 6,
            'GRID_STEP_MIN': 0.01,
            'GRID_STEP_MAX': 0.05,
            'BEAR_RATIO': 0.25,
            'NEUTRAL_RATIO': 0.5,
            'BULL_RATIO': 0.7,
        })
        assert p['MAX_HOLD'] == 6


def _load_gate_eval():
    import importlib.util

    opt_dir = Path(__file__).parents[2] / 'strategies' / 'grid_multi_asset_v4' / 'optimization'
    spec = importlib.util.spec_from_file_location('_ge_unit', str(opt_dir / 'gate_eval.py'))
    assert spec and spec.loader
    ge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ge)
    return ge


class TestGateEvalChecks:
    def test_eligible_both_pass(self):
        ge = _load_gate_eval()

        thr = ge.GateThresholds()
        mf = {'max_drawdown': -0.30, 'excess_return': 0.0, 'information_ratio': 0.1}
        mr = {'annual_return': 0.25, 'max_drawdown': -0.10, 'sharpe_ratio': 1.5}
        ok, fails = ge.check_gates(mf, mr, thr)
        assert ok is True
        assert fails == []

    def test_fail_on_full_drawdown(self):
        ge = _load_gate_eval()
        thr = ge.GateThresholds()
        mf = {'max_drawdown': -0.50, 'excess_return': 0.0, 'information_ratio': 0.1}
        mr = {'annual_return': 0.25, 'max_drawdown': -0.10, 'sharpe_ratio': 1.5}
        ok, fails = ge.check_gates(mf, mr, thr)
        assert ok is False
        assert any('I:' in f for f in fails)
