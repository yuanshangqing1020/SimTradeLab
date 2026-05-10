# tests/unit/test_grid_multi_asset_v3.py
# -*- coding: utf-8 -*-
"""
grid_multi_asset_v3：M1/M2 纯函数单测（exec 加载 template，无 API）
"""
import types
import numpy as np
import pandas as pd
from pathlib import Path

_STRATEGY_PATH = Path(__file__).parents[2] / 'strategies' / 'grid_multi_asset_v3' / 'template.py'


def _load_fns():
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


_ns = _load_fns()
_effective = _ns['_effective_max_layer_bear']
_etf_list = _ns['_etf_list_for_refresh']
_no_net = _ns['_apply_no_net_add_targets']
CAND = ['510300.SS', '513100.SS']
DEF = ['510300.SS']


class TestEffectiveMaxLayerBear:
    def test_non_bear_unchanged(self):
        assert _effective('NEUTRAL', 'CAP_LAYER', 3, 1) == 3
        assert _effective('BEAR', 'NORMAL', 4, 2) == 4

    def test_bear_cap_layers(self):
        assert _effective('BEAR', 'CAP_LAYER', 3, 1) == 1
        assert _effective('BEAR', 'CAP_LAYER', 2, 4) == 2


class TestEtfListForRefresh:
    def test_same_mode_full(self):
        assert _etf_list('BEAR', 'SAME', CAND, DEF) == CAND
        assert _etf_list('BULL', 'ETF_DEFENSIVE', CAND, DEF) == CAND

    def test_defensive_bear_only(self):
        assert _etf_list('BEAR', 'ETF_DEFENSIVE', CAND, DEF) == DEF


class TestNoNetAdd:
    def test_clamp_when_prev_positive(self):
        prev = {'A': 10000.0}
        tgt = {'A': 12000.0, 'B': 5000.0}
        out = _no_net(prev, tgt)
        assert out['A'] == 10000.0
        assert out['B'] == 5000.0

    def test_no_clamp_new_name(self):
        prev = {'A': 10000.0}
        tgt = {'B': 3000.0}
        out = _no_net(prev, tgt)
        assert out['B'] == 3000.0

