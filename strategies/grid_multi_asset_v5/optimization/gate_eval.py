# strategies/grid_multi_asset_v5/optimization/gate_eval.py
# -*- coding: utf-8 -*-
"""
v5 三重门禁：FULL / RECENT / BEAR 分段回测与 I∧II∧III 判定。

阈值默认与 my_docs/grid_multi_asset/v4.0/01-design.md §2.2 首轮建议一致；
校准后仅改 GateThresholds 或 CLI 覆盖。

用法示例:

    python strategies/grid_multi_asset_v5/optimization/gate_eval.py \\
        --params-json strategies/grid_multi_asset_v5/optimization/results/best_params_xxx.json
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from simtradelab.backtest.config import BacktestConfig
from simtradelab.backtest.optimizer_framework import apply_parameter_replacement
from simtradelab.backtest.runner import BacktestRunner

STRATEGY_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = STRATEGY_DIR / 'template.py'
GATE_BACKTEST_REL = 'optimization/gate_injected_backtest.py'

FULL_PERIOD = ('2019-01-01', '2026-04-20')
RECENT_PERIOD = ('2025-01-01', '2026-03-31')
BEAR_PERIOD = ('2021-01-01', '2022-12-31')


@dataclass
class GateThresholds:
    """指标均为回测报告中的数值口径（最大回撤为负数）。"""

    full_max_drawdown: float = -0.38          # I: 不深于 38% → 值须 >= -0.38
    full_excess_return: float = -0.10       # II
    full_information_ratio: float = -0.05   # II
    recent_annual_return: float = 0.20      # III
    recent_max_drawdown: float = -0.18      # III
    recent_sharpe: float = 1.2              # III


def write_injected_strategy(
    params: dict[str, Any],
    custom_mapping: dict[str, str],
    template_path: Optional[Path] = None,
) -> Path:
    tpl = template_path or TEMPLATE_PATH
    code = tpl.read_text(encoding='utf-8')
    modified = apply_parameter_replacement(code, params, custom_mapping)
    out = STRATEGY_DIR / GATE_BACKTEST_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(modified, encoding='utf-8')
    return out


def extract_metrics(report: dict[str, Any]) -> dict[str, float]:
    if not report:
        return {}
    keys = (
        'total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown',
        'information_ratio', 'excess_return', 'win_rate',
    )
    return {k: float(report.get(k, 0.0)) for k in keys}


def check_gates(
    m_full: dict[str, float],
    m_recent: dict[str, float],
    thr: GateThresholds,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    mdd_f = m_full.get('max_drawdown', -1.0)
    if mdd_f < thr.full_max_drawdown:
        failures.append(
            'I: FULL max_drawdown {:.4f} < {:.4f}'.format(mdd_f, thr.full_max_drawdown),
        )
    ex_f = m_full.get('excess_return', -9.0)
    if ex_f < thr.full_excess_return:
        failures.append(
            'II: FULL excess_return {:.4f} < {:.4f}'.format(ex_f, thr.full_excess_return),
        )
    ir_f = m_full.get('information_ratio', -9.0)
    if ir_f < thr.full_information_ratio:
        failures.append(
            'II: FULL information_ratio {:.4f} < {:.4f}'.format(ir_f, thr.full_information_ratio),
        )
    ar_r = m_recent.get('annual_return', 0.0)
    if ar_r < thr.recent_annual_return:
        failures.append(
            'III: RECENT annual_return {:.4f} < {:.4f}'.format(ar_r, thr.recent_annual_return),
        )
    mdd_r = m_recent.get('max_drawdown', -1.0)
    if mdd_r < thr.recent_max_drawdown:
        failures.append(
            'III: RECENT max_drawdown {:.4f} < {:.4f}'.format(mdd_r, thr.recent_max_drawdown),
        )
    sh_r = m_recent.get('sharpe_ratio', 0.0)
    if sh_r < thr.recent_sharpe:
        failures.append(
            'III: RECENT sharpe_ratio {:.4f} < {:.4f}'.format(sh_r, thr.recent_sharpe),
        )
    return (len(failures) == 0, failures)


def run_segment(
    runner: BacktestRunner,
    start: str,
    end: str,
) -> dict[str, Any]:
    cfg = BacktestConfig(
        strategy_name='grid_multi_asset_v5',
        strategy_file=GATE_BACKTEST_REL,
        start_date=start,
        end_date=end,
        initial_capital=500000.0,
        optimization_mode=True,
        enable_charts=False,
        enable_logging=False,
    )
    return runner.run(config=cfg)


def run_triple_gates(
    params: dict[str, Any],
    custom_mapping: dict[str, str],
    runner: Optional[BacktestRunner] = None,
    thr: Optional[GateThresholds] = None,
) -> dict[str, Any]:
    thr = thr or GateThresholds()
    write_injected_strategy(params, custom_mapping)
    if runner is None:
        runner = BacktestRunner()
    r_full = run_segment(runner, FULL_PERIOD[0], FULL_PERIOD[1])
    r_recent = run_segment(runner, RECENT_PERIOD[0], RECENT_PERIOD[1])
    r_bear = run_segment(runner, BEAR_PERIOD[0], BEAR_PERIOD[1])
    mf = extract_metrics(r_full)
    mr = extract_metrics(r_recent)
    mb = extract_metrics(r_bear)
    ok, fails = check_gates(mf, mr, thr)
    return {
        'eligible': ok,
        'failures': fails,
        'metrics_full': mf,
        'metrics_recent': mr,
        'metrics_bear': mb,
        'raw_reports': {'full': r_full, 'recent': r_recent, 'bear': r_bear},
    }


def load_v5_custom_mapping() -> dict[str, str]:
    """与同目录 optimize_params.V5_CUSTOM_MAPPING 一致（文件加载避免包路径依赖）。"""
    import importlib.util

    opt = Path(__file__).resolve().parent / 'optimize_params.py'
    spec = importlib.util.spec_from_file_location('_grid_v5_opt_params', str(opt))
    if spec is None or spec.loader is None:
        raise ImportError('无法加载 optimize_params: {}'.format(opt))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.V5_CUSTOM_MAPPING)


def main() -> None:
    parser = argparse.ArgumentParser(description='v5 triple-gate evaluation')
    parser.add_argument('--params-json', required=True, help='best_params JSON from Optuna')
    parser.add_argument('--thr-full-mdd', type=float, default=-0.38)
    parser.add_argument('--thr-full-excess', type=float, default=-0.10)
    parser.add_argument('--thr-full-ir', type=float, default=-0.05)
    parser.add_argument('--thr-recent-ann', type=float, default=0.20)
    parser.add_argument('--thr-recent-mdd', type=float, default=-0.18)
    parser.add_argument('--thr-recent-sharpe', type=float, default=1.2)
    args = parser.parse_args()

    thr = GateThresholds(
        full_max_drawdown=args.thr_full_mdd,
        full_excess_return=args.thr_full_excess,
        full_information_ratio=args.thr_full_ir,
        recent_annual_return=args.thr_recent_ann,
        recent_max_drawdown=args.thr_recent_mdd,
        recent_sharpe=args.thr_recent_sharpe,
    )

    with open(args.params_json, encoding='utf-8') as f:
        params = json.load(f)

    mapping = load_v5_custom_mapping()
    result = run_triple_gates(params, mapping, thr=thr)

    print('eligible:', result['eligible'])
    for line in result['failures']:
        print('  FAIL', line)
    print('FULL:', result['metrics_full'])
    print('RECENT:', result['metrics_recent'])
    print('BEAR:', result['metrics_bear'])


if __name__ == '__main__':
    main()
