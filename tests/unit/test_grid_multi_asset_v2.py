# tests/unit/test_grid_multi_asset_v2.py
# -*- coding: utf-8 -*-
"""
v2 新增功能单元测试：
  - _calc_regime：大盘趋势判断（纯数学，不依赖 API）
  - _apply_weight_cap：单标的权重截断（纯数学）
"""
import types
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# ── 加载策略纯函数 ──────────────────────────────────────────────────────────── #
_STRATEGY_PATH = Path(__file__).parents[2] / 'strategies' / 'grid_multi_asset_v2' / 'template.py'

def _load_fns():
    """用 mock PTrade 全局量执行策略文件，返回其命名空间。"""
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

_fns         = _load_fns()
_calc_regime = _fns['_calc_regime']
_apply_cap   = _fns['_apply_weight_cap']
_normalize   = _fns['_normalize_weights']


# ── _calc_regime ────────────────────────────────────────────────────────────── #
class TestCalcRegime:
    def _make_prices(self, n=260, trend='flat'):
        """生成测试用价格序列。trend: 'up'/'down'/'flat'"""
        np.random.seed(0)
        base = np.ones(n) * 3000.0
        if trend == 'up':
            base += np.linspace(0, 500, n)   # 明确上升趋势
        elif trend == 'down':
            base -= np.linspace(0, 500, n)   # 明确下降趋势
        return base

    def test_bull_when_above_both_ma(self):
        prices = self._make_prices(260, 'up')
        assert _calc_regime(prices) == 'BULL'

    def test_bear_when_below_both_ma(self):
        prices = self._make_prices(260, 'down')
        assert _calc_regime(prices) == 'BEAR'

    def test_neutral_when_between_ma(self):
        # 先下跌后反弹：价格在 MA120 上方但仍低于 MA250
        prices = self._make_prices(260, 'down')
        prices[-60:] += 300  # 近期反弹，越过 MA120 但未越过 MA250
        result = _calc_regime(prices)
        assert result == 'NEUTRAL'

    def test_neutral_on_short_history(self):
        prices = np.ones(100) * 3000.0  # 不足 250 条
        assert _calc_regime(prices) == 'NEUTRAL'

    def test_neutral_on_empty_array(self):
        assert _calc_regime(np.array([])) == 'NEUTRAL'


# ── _apply_weight_cap ───────────────────────────────────────────────────────── #
class TestApplyWeightCap:
    def test_no_overflow_after_cap(self):
        # 4只标的，等权 0.25，cap 设为 0.20
        raw = [0.25, 0.25, 0.25, 0.25]
        result = _apply_cap(raw, max_w=0.20)
        assert all(w <= 0.20 + 1e-9 for w in result), f"weight overflow: {result}"

    def test_sum_equals_one_after_cap(self):
        raw = [0.40, 0.30, 0.20, 0.10]
        result = _apply_cap(raw, max_w=0.25)
        assert abs(sum(result) - 1.0) < 1e-9

    def test_idempotent_after_extra_iteration(self):
        raw = [0.40, 0.30, 0.20, 0.10]
        result3 = _apply_cap(raw, max_w=0.25, iterations=3)
        result6 = _apply_cap(result3, max_w=0.25, iterations=3)
        for a, b in zip(result3, result6):
            assert abs(a - b) < 1e-9, "权重未收敛"

    def test_no_cap_needed_unchanged(self):
        # 权重已低于 cap，不应改变
        raw = [0.10, 0.20, 0.30, 0.40]
        result = _apply_cap(raw, max_w=0.50)
        for a, b in zip(raw, result):
            assert abs(a - b) < 1e-9
