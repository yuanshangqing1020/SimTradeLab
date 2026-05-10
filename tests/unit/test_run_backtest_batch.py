# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later

import simtradelab.backtest.run_backtest_batch as batch


def test_run_batch_backtests_runs_strategies_in_order(monkeypatch):
    calls = []
    created_runner_count = 0

    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyRunner:
        def __init__(self):
            nonlocal created_runner_count
            created_runner_count += 1

        def run(self, config):
            strategy_name = config.kwargs["strategy_name"]
            calls.append(strategy_name)
            return {"strategy_name": strategy_name, "ok": True}

    monkeypatch.setattr(batch, "BacktestConfig", DummyConfig)
    monkeypatch.setattr(batch, "BacktestRunner", DummyRunner)

    results = batch.run_batch_backtests(
        strategy_names=["s1", "s2", "s3"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0,
    )

    assert calls == ["s1", "s2", "s3"]
    assert list(results.keys()) == ["s1", "s2", "s3"]
    assert created_runner_count == 3
