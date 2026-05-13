# strategies/grid_multi_asset_v5/optimization/post_select_eligible.py
# -*- coding: utf-8 -*-
"""Walk-Forward 结束后对候选参数跑三重门禁的薄封装。

用法（在 SimTradeLab 根目录）::

    python strategies/grid_multi_asset_v5/optimization/post_select_eligible.py \\
        strategies/grid_multi_asset_v5/optimization/results/best_params_xxx.json

等价于带默认阈值的 ``gate_eval.py --params-json ...``。"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: post_select_eligible.py <best_params.json>', file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    params = json.loads(path.read_text(encoding='utf-8'))
    # 同目录模块
    opt_dir = Path(__file__).resolve().parent
    import importlib.util

    spec = importlib.util.spec_from_file_location('_v5_gate', str(opt_dir / 'gate_eval.py'))
    if spec is None or spec.loader is None:
        raise RuntimeError('gate_eval load failed')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    mapping = gate.load_v5_custom_mapping()
    result = gate.run_triple_gates(params, mapping)
    print('eligible:', result['eligible'])
    for line in result['failures']:
        print('  FAIL', line)
    print('FULL:', result['metrics_full'])
    print('RECENT:', result['metrics_recent'])
    print('BEAR:', result['metrics_bear'])


if __name__ == '__main__':
    main()
