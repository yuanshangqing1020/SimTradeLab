# tests/unit/test_grid_multi_asset.py
# -*- coding: utf-8 -*-
"""
纯数学函数单元测试（_calc_vol_from_prices / _calc_layer /
_normalize_weights / _score_universe）

使用 exec() 在 mock PTrade 命名空间中加载策略文件，
避免 set_benchmark 等运行时 API 未定义的问题。
"""
import types
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# ── 加载策略纯函数 ──────────────────────────────────────────────────────────── #
_STRATEGY_PATH = Path(__file__).parents[2] / 'strategies' / 'grid_multi_asset_v1' / 'backtest.py'

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

_fns = _load_fns()
_calc_vol   = _fns['_calc_vol_from_prices']
_calc_layer = _fns['_calc_layer']
_normalize  = _fns['_normalize_weights']
_score      = _fns['_score_universe']


# ── _calc_vol_from_prices ───────────────────────────────────────────────────── #
class TestCalcVolFromPrices:
    def test_returns_none_for_fewer_than_22_bars(self):
        prices = np.linspace(10, 11, 20)
        assert _calc_vol(prices) is None

    def test_returns_float_for_valid_data(self):
        np.random.seed(42)
        prices = np.cumprod(1 + np.random.normal(0.0005, 0.015, 30)) * 100
        result = _calc_vol(prices)
        assert isinstance(result, float) and result > 0

    def test_constant_prices_return_none(self):
        prices = np.full(30, 10.0)
        assert _calc_vol(prices) is None   # std=0 → vol=0 → returns None

    def test_annualized_range_is_plausible(self):
        np.random.seed(0)
        daily_rets = np.random.normal(0, 0.01, 25)
        prices = np.cumprod(1 + daily_rets) * 100
        vol = _calc_vol(prices)
        assert vol is not None
        assert 0.05 < vol < 0.60

    def test_uses_last_20_returns(self):
        np.random.seed(1)
        prices_noisy = np.cumprod(1 + np.random.normal(0, 0.10, 10)) * 100
        prices_calm  = np.cumprod(1 + np.random.normal(0, 0.005, 22)) * prices_noisy[-1]
        prices = np.concatenate([prices_noisy, prices_calm])
        vol = _calc_vol(prices)
        assert vol is not None and vol < 0.20


# ── _calc_layer ─────────────────────────────────────────────────────────────── #
class TestCalcLayer:
    def test_price_below_ma_positive_layer(self):
        # MA=10, price=9, step=0.05 → raw=(10-9)/(9*0.05)=2.22 → floor(2.22+0.5)=2
        assert _calc_layer(price=9.0, ma20=10.0, step=0.05, max_layer=3) == 2

    def test_price_above_ma_negative_layer(self):
        # MA=10, price=11, step=0.05 → raw=(10-11)/(11*0.05)=-1.82 → floor(-1.82+0.5)=-2
        assert _calc_layer(price=11.0, ma20=10.0, step=0.05, max_layer=3) == -2

    def test_at_ma_zero_layer(self):
        assert _calc_layer(price=10.0, ma20=10.0, step=0.05, max_layer=3) == 0

    def test_clamped_to_positive_max(self):
        assert _calc_layer(price=1.0, ma20=10.0, step=0.01, max_layer=3) == 3

    def test_clamped_to_negative_max(self):
        assert _calc_layer(price=20.0, ma20=10.0, step=0.01, max_layer=3) == -3

    def test_asymmetry_small_step(self):
        # step=0.02, price=9.9, MA=10 → raw=0.1/(9.9*0.02)=0.505 → floor(1.005)=1
        assert _calc_layer(price=9.9, ma20=10.0, step=0.02, max_layer=5) == 1


# ── _normalize_weights ──────────────────────────────────────────────────────── #
class TestNormalizeWeights:
    def test_sums_to_one(self):
        result = _normalize([1.0, 2.0, 3.0])
        assert abs(sum(result) - 1.0) < 1e-9

    def test_proportions_preserved(self):
        result = _normalize([1.0, 3.0])
        assert abs(result[1] / result[0] - 3.0) < 1e-9

    def test_all_zeros_equal_weight(self):
        result = _normalize([0.0, 0.0, 0.0])
        assert result == pytest.approx([1/3, 1/3, 1/3])

    def test_single_element(self):
        assert _normalize([5.0]) == [1.0]

    def test_empty_returns_empty(self):
        assert _normalize([]) == []

    def test_zero_not_amplified(self):
        result = _normalize([2.0, 0.0, 2.0])
        assert result == pytest.approx([0.5, 0.0, 0.5])


# ── _score_universe ─────────────────────────────────────────────────────────── #
class TestScoreUniverse:

    def _fund_df(self, codes):
        return pd.DataFrame({
            'code':       codes,
            'pe_ratio':   [20.0, 15.0, 30.0][:len(codes)],
            'market_cap': [1e11, 5e10, 2e10][:len(codes)],
            'roe':        [0.15, 0.20, 0.10][:len(codes)],
        })

    def test_returns_sorted_descending(self):
        vol = {'A': 0.3, 'B': 0.5, 'C': 0.2}
        result = _score(vol, self._fund_df(['A', 'B', 'C']), [], vol_weight=1.0)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_high_vol_wins_when_vol_weight_1(self):
        vol = {'ETF1': 0.4, 'ETF2': 0.2}
        result = _score(vol, None, ['ETF1', 'ETF2'], vol_weight=1.0)
        assert result[0][0] == 'ETF1'

    def test_empty_vol_dict_returns_empty(self):
        assert _score({}, None, [], vol_weight=0.7) == []

    def test_filters_negative_pe(self):
        vol = {'A': 0.3, 'B': 0.4}
        fd = pd.DataFrame({
            'code': ['A', 'B'],
            'pe_ratio': [-5.0, 20.0],
            'market_cap': [5e10, 5e10],
            'roe': [0.1, 0.15],
        })
        result = _score(vol, fd, [], vol_weight=0.6)
        codes = [c for c, _ in result]
        assert 'A' not in codes
        assert 'B' in codes

    def test_filters_small_cap(self):
        vol = {'A': 0.3, 'B': 0.4}
        fd = pd.DataFrame({
            'code': ['A', 'B'],
            'pe_ratio': [20.0, 20.0],
            'market_cap': [1e9, 5e10],   # A < 30亿门槛
            'roe': [0.1, 0.15],
        })
        result = _score(vol, fd, [], vol_weight=0.6)
        codes = [c for c, _ in result]
        assert 'A' not in codes

    def test_etf_included_without_fund_df(self):
        vol = {'ETF1': 0.35}
        result = _score(vol, None, ['ETF1'], vol_weight=0.7)
        assert len(result) == 1 and result[0][0] == 'ETF1'

    def test_mixed_stocks_and_etfs(self):
        vol = {'S1': 0.3, 'ETF1': 0.6}
        fd = pd.DataFrame({
            'code': ['S1'],
            'pe_ratio': [20.0],
            'market_cap': [5e10],
            'roe': [0.15],
        })
        result = _score(vol, fd, ['ETF1'], vol_weight=0.9)
        codes = [c for c, _ in result]
        assert 'S1' in codes and 'ETF1' in codes
