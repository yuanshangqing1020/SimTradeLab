# 多标的自适应网格策略（JoinQuant 版）

> 对应 SimTradeLab 目录：`strategies/grid_multi_asset/`

## 目录结构

```
grid_multi_asset/
├── README.md       ← 本文件：版本对照表
├── v1/
│   └── strategy.py ← v1 JoinQuant 代码（直接粘贴到聚宽平台）
└── v2/             ← 待开发
    └── strategy.py
```

## 版本对照表

| JQ 版本 | 对应 SimTradeLab 版本 | 参数来源 | 主要变化 | 日期 |
|---------|----------------------|----------|----------|------|
| v1 | `strategies/grid_multi_asset_best/` (v1.0) | Walk-Forward Trial 53<br>score=-0.3665（优化进行中） | 初始版本；修复 PTrade→JQ API 映射、最小手数限制 | 2026-05-08 |

## 使用方法

1. 打开对应版本的 `strategy.py`
2. 全选内容，粘贴到[聚宽回测编辑器](https://www.joinquant.com/algorithm/index/edit)
3. 设置起止日期（建议 2019-01-01 ~ 今日）、初始资金 50 万
4. 点击"运行回测"

## SimTradeLab ↔ JoinQuant 主要差异

| 项目 | SimTradeLab (PTrade) | JoinQuant |
|---|---|---|
| 代码格式 | `.SS` / `.SZ` | `.XSHG` / `.XSHE` |
| 全局变量 | `context.*` | `g.*` |
| 历史行情 | `get_history()` | `history(df=True)` |
| 基本面 | `get_fundamentals(list, table, fields)` | `get_fundamentals(query(...))` |
| 市值单位 | 元（`total_value >= 3e9`） | 亿元（`market_cap >= 30`） |
| PE 字段 | `pe_ttm` | `pe_ratio` |
| 定时执行 | `handle_data` 日频 | `run_daily(func, time='14:50')` |
| 持仓数量 | `p.amount` | `p.total_amount` |
| 总资产 | `portfolio.portfolio_value` | `portfolio.total_value` |
