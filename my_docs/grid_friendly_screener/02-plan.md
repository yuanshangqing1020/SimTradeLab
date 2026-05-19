# Grid Screener v2 — 实现备忘

> 完整规格见 `01-design.md`。本文仅列目录与验收，供实现/评审对照。

## 包结构

```
src/simtradelab/grid_screener/
  __init__.py
  __main__.py          # CLI
  config.py            # RunConfig, ScreenerParams, preset 展开
  context.py           # FactorContext
  market_data.py       # MarketDataSession, fq 复权
  engine.py            # run_universe → list[dict]
  sort_spec.py         # SortSpec → sort DataFrame
  metrics.py           # 纯 NumPy 统计（不变）
  preprocess.py
  labels.py
  factors/
    base.py
    registry.py
    builtin.py         # 内置因子实现
  explain/
    registry.py
    grid_default.py
  presets/
    grid_friendly_v1.json
  report.py
  data_path.py         # 路径/名称解析
  io_csv.py            # 演示 CSV
```

## 验收

- [ ] 默认 Parquet 全市场：`fq=pre`，不直接裸读未复权价做统计
- [ ] JSON 可只配 `factors` + `sort`，无需改 Python 即可换排序
- [ ] `preset: grid_friendly_v1` 行为与 v1 分项列兼容（含可选 `grid_score`）
- [ ] `pytest tests/unit/test_grid_screener_*.py` 通过

## 删除/废弃（v2）

- `pipeline.py`, `scoring.py`, `io_parquet.py` → 由 engine + factors + market_data 替代
