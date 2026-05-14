# strategies/grid_multi_asset_v5/optimization/two_stage_select.py
# -*- coding: utf-8 -*-
"""两阶段选参：FULL 上筛 I+II →（可选）RECENT 粗筛 → 全 III → 按 FULL 年化排序。

读取 Optuna 导出的 trials CSV（含 ``params_*`` 列），对每行还原参数字典后调用与
``gate_eval`` 一致的注入回测链路。

用法（SimTradeLab 根目录）::

    python strategies/grid_multi_asset_v5/optimization/two_stage_select.py \\
        --trials-csv strategies/grid_multi_asset_v5/optimization/results/trials_xxx.csv \\
        --top 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# 同目录 gate_eval（以文件加载避免包路径）
_GATE = None
_OPT = None


def _load_gate():
    global _GATE
    if _GATE is not None:
        return _GATE
    import importlib.util

    p = Path(__file__).resolve().parent / 'gate_eval.py'
    spec = importlib.util.spec_from_file_location('_v5_gate_two_stage', str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError('gate_eval load failed')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _GATE = mod
    return mod


def _load_optimize_params_mod():
    global _OPT
    if _OPT is not None:
        return _OPT
    import importlib.util

    p = Path(__file__).resolve().parent / 'optimize_params.py'
    spec = importlib.util.spec_from_file_location('_v5_opt_two_stage', str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError('optimize_params load failed')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _OPT = mod
    return mod


def row_to_params(row: pd.Series) -> dict[str, Any]:
    """从 trials CSV 一行提取 Optuna 参数字典（列名 ``params_KEY`` → ``KEY``）。"""
    out: dict[str, Any] = {}
    for col, raw in row.items():
        if not isinstance(col, str) or not col.startswith('params_'):
            continue
        key = col[7:]
        if pd.isna(raw):
            continue
        if isinstance(raw, float) and raw == int(raw):
            out[key] = int(raw)
        elif isinstance(raw, str):
            out[key] = raw
        else:
            out[key] = raw
    return out


def _calmar(m: dict[str, float]) -> float:
    ar = float(m.get('annual_return', 0.0))
    mdd = float(m.get('max_drawdown', -1.0))
    if mdd >= -1e-12:
        return 0.0
    return ar / abs(mdd)


def main(
    trials_csv: str,
    top: int = 10,
    thr: Optional[Any] = None,
    use_recent_coarse: bool = True,
    max_rows: Optional[int] = None,
) -> int:
    gate = _load_gate()
    thr = thr or gate.GateThresholds()
    mapping = gate.load_v5_custom_mapping()
    opt_mod = _load_optimize_params_mod()

    path = Path(trials_csv)
    if not path.is_file():
        print('文件不存在:', path, file=sys.stderr)
        return 2

    df = pd.read_csv(path)
    if max_rows is not None:
        df = df.head(int(max_rows))

    eligible_rows: list[tuple[dict[str, Any], dict[str, float], dict[str, float]]] = []

    for _idx, row in df.iterrows():
        params = row_to_params(row)
        if not params:
            continue
        try:
            opt_mod.GridMultiAssetV5Params.validate(params)
        except Exception as exc:
            print('skip validate:', exc, file=sys.stderr)
            continue

        try:
            m_full = gate.run_metrics_for_params_window(
                params, mapping,
                gate.FULL_PERIOD[0], gate.FULL_PERIOD[1],
            )
        except Exception as exc:
            print('skip FULL backtest:', exc, file=sys.stderr)
            continue

        ok12, _f12 = gate.check_gates_i_ii_only(m_full, thr)
        if not ok12:
            continue

        try:
            m_recent = gate.run_metrics_for_params_window(
                params, mapping,
                gate.RECENT_PERIOD[0], gate.RECENT_PERIOD[1],
            )
        except Exception as exc:
            print('skip RECENT backtest:', exc, file=sys.stderr)
            continue

        if use_recent_coarse:
            ok_c, _ = gate.check_recent_sharpe_mdd_only(m_recent, thr)
            if not ok_c:
                continue

        ok3, _ = gate.check_gates_iii_only(m_recent, thr)
        if not ok3:
            continue

        eligible_rows.append((params, m_full, m_recent))

    eligible_rows.sort(key=lambda x: x[1].get('annual_return', 0.0), reverse=True)

    print('eligible_count:', len(eligible_rows))
    for i, (params, m_full, m_recent) in enumerate(eligible_rows[:top], start=1):
        cm = _calmar(m_full)
        print('--- rank', i, '---')
        print('params:', params)
        print('FULL annual_return:', m_full.get('annual_return'), 'calmar:', round(cm, 6))
        print('FULL max_drawdown:', m_full.get('max_drawdown'))
        print('RECENT annual_return:', m_recent.get('annual_return'))
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='v5 two-stage gate filter')
    p.add_argument('--trials-csv', required=True)
    p.add_argument('--top', type=int, default=10)
    p.add_argument('--max-rows', type=int, default=None, help='只处理前 N 行（冒烟）')
    p.add_argument(
        '--skip-recent-coarse',
        action='store_true',
        help='不做 RECENT 夏普/回撤粗筛，直接跑 III 全条件',
    )
    args = p.parse_args()
    raise SystemExit(
        main(
            args.trials_csv,
            top=args.top,
            use_recent_coarse=not args.skip_recent_coarse,
            max_rows=args.max_rows,
        ),
    )
