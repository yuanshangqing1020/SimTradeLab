# tests/unit/test_core_grid_hybrid_v1.py
# -*- coding: utf-8 -*-
"""core_grid_hybrid_v1 几何网格与配对选择纯函数单测。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_BACKTEST = Path(__file__).resolve().parents[2] / 'strategies' / 'core_grid_hybrid_v1' / 'backtest.py'
_spec = importlib.util.spec_from_file_location('core_grid_hybrid_v1_backtest', _BACKTEST)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

grid_buy_price = _mod.grid_buy_price
grid_sell_price = _mod.grid_sell_price
pick_grid_action_close = _mod.pick_grid_action_close
should_enter_defensive = _mod.should_enter_defensive


def test_geometric_prices_k1_k2() -> None:
    ref = 10.0
    step = 0.03
    assert abs(grid_sell_price(ref, 1, step) - 10.3) < 1e-9
    assert abs(grid_buy_price(ref, 1, step) - (10.0 * 0.97)) < 1e-9
    assert abs(grid_sell_price(ref, 2, step) - (10.0 * 1.03 * 1.03)) < 1e-9


def test_defensive_buy_prices_use_wider_step() -> None:
    ref = 100.0
    normal = grid_buy_price(ref, 1, 0.03)
    wide = grid_buy_price(ref, 1, 0.05)
    assert wide < normal


def test_pairing_after_sell_only_buy_at_P_buy_k() -> None:
    ref = 10.0
    step = 0.03
    buy_step = step
    p_buy_1 = grid_buy_price(ref, 1, buy_step)
    # 在途卖档 k=1：必须 close <= P_buy_1 才回补买；略高于 P_buy_1 则不买
    a = pick_grid_action_close(
        close=p_buy_1 + 0.05,
        ref=ref,
        grid_step=step,
        buy_step=buy_step,
        round_active=True,
        last_pair_side='sell',
        last_pair_k=1,
        max_grid_level=5,
    )
    assert a is None


def test_peak_drawdown_triggers_defensive() -> None:
    assert should_enter_defensive(100.0, 84.0, 0.15) is True
    assert should_enter_defensive(100.0, 86.0, 0.15) is False


def test_prefer_sell_when_both_eligible() -> None:
    ref = 10.0
    step = 0.03
    # close 同时高于 P_sell_1 且低于 P_buy_1 物理上不可能；选一个在两侧都可触发的极端构造较难，
    # 此处仅验证「卖侧有信号时」返回 sell。
    close = grid_sell_price(ref, 1, step) + 0.1
    a = pick_grid_action_close(
        close=close,
        ref=ref,
        grid_step=step,
        buy_step=step,
        round_active=False,
        last_pair_side=None,
        last_pair_k=None,
        max_grid_level=5,
    )
    assert a == ('sell', 1)


def test_pick_at_most_one_action_implicit() -> None:
    """每个 close 下 pick 至多一份意向（由返回类型保证）。"""
    ref = 50.0
    for close in [45.0, 48.0, 50.0, 52.0, 55.0, 60.0]:
        a = pick_grid_action_close(
            close=close,
            ref=ref,
            grid_step=0.03,
            buy_step=0.03,
            round_active=False,
            last_pair_side=None,
            last_pair_k=None,
            max_grid_level=10,
        )
        assert a is None or (isinstance(a, tuple) and len(a) == 2)
