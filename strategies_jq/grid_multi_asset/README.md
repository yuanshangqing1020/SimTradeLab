# 多标的自适应网格策略（JoinQuant 版）

> 对应 SimTradeLab：`strategies/grid_multi_asset_v1/` … `strategies/grid_multi_asset_v5/`

## 目录结构

```
grid_multi_asset/
├── README.md          ← 本文件：版本对照表
├── v1/
│   └── strategy.py    ← v1 JoinQuant
├── v2/
│   └── strategy.py    ← v2 JoinQuant（Trial 29）
├── v3/
│   └── strategy.py    ← v3 JoinQuant（Trial 357，M1/M2）
├── v4/
│   └── strategy.py    ← v4 JoinQuant（Trial 185，窄池+周频 regime）
└── v5/
    └── strategy.py    ← v5 JoinQuant（Trial 190：锚定 Universe + BEAR 小维 + 周频 regime）
```

## 版本对照表

| JQ 版本 | 对应 SimTradeLab 版本 | 参数来源 | 主要变化 | 日期 |
|---------|----------------------|----------|----------|------|
| v1 | `grid_multi_asset_v1/` | Trial 53，WF ≈ −0.3665 | 初版 API 映射、最小手数 | 2026-05-08 |
| v2 | `grid_multi_asset_v2/` | Trial **29**，WF ≈ −0.3457 | Regime 三总仓 + water-filling | 2026-05-10 |
| v3 | `grid_multi_asset_v3/` | Trial **357**，WF ≈ −0.3463 | **M2** 防御池切换 + **M1** 熊市网格（本参数为 SAME + CAP_LAYER cap=0）；`after_trading_end` 写昨收市值供 NO_NET_ADD | 2026-05-10 |
| v4 | `grid_multi_asset_v4/` | Trial **185**，WF ≈ −0.5180 | **窄 ETF 6 只** + **`REGIME_REFRESH=WEEKLY`**（每 5 交易日或换仓日刷新投入比例）；可选 `WIDE_V2` | 2026-05-13 |
| v5 | `grid_multi_asset_v5/` | Trial **190**，WF ≈ −0.3119 | **`ANCHOR_SATELLITE` / `WIDE_V2` / `NARROW_ETF`**、`MIN_ANCHORS_IN_POOL`、**BEAR 三字段**（`SAME`+`CAP_LAYER` cap=0）、锚定优先入池；默认 **`WIDE_V2`** 与 SimTradeLab 归档 JSON 一致 | 2026-05-14 |

## 使用方法

1. 打开对应版本的 `strategy.py`
2. 全选内容，粘贴到[聚宽回测编辑器](https://www.joinquant.com/algorithm/index/edit)
3. 设置起止日期（建议 **2019-01-01** 起）、初始资金 **50 万**
4. 点击「运行回测」

## SimTradeLab ↔ JoinQuant 主要差异

| 项目 | SimTradeLab (PTrade) | JoinQuant |
|---|---|---|
| 代码格式 | `.SS` / `.SZ` | `.XSHG` / `.XSHE` |
| 全局变量 | `context.*` | `g.*` |
| 历史行情 | `get_history()` | `history(..., df=True)` |
| 基本面 | `get_fundamentals(list, table, fields)` | `get_fundamentals(query(...))` |
| 市值单位 | 元（`total_value >= 3e9`） | 亿元（`market_cap >= 30`） |
| PE 字段 | `pe_ttm` | `pe_ratio` |
| 定时执行 | `handle_data` 日频 | `run_daily(func, time='14:50')` |
| 持仓数量 | `p.amount` | `p.total_amount` |
| 总资产 | `portfolio.portfolio_value` | `portfolio.total_value` |

## 注意事项（v3 / v4 / v5）

- **v3**：`NO_NET_ADD` 模式依赖 `g._prev_eod_position_value`；首日开始交易前快照为空，行为与 SimTradeLab 一致。聚宽 `after_trading_end` 内用 `get_current_data()[code].last_price × total_amount` 近似日终市值。
- **v4**：与 SimTradeLab 一致，默认 **`g.UNIVERSE_MODE='NARROW_ETF'`**；若要在聚宽侧复现「全 ETF+成分股」宽池，可在 `initialize` 中改为 `'WIDE_V2'`（参数空间需与 `MAX_HOLD` 上限一致）。
- **v5**：默认 **`g.UNIVERSE_MODE='WIDE_V2'`**（与 `best_params_20260514_122926.json` 一致）；**`ANCHOR_SATELLITE`** 下换仓使用 `build_grid_pool_anchor_first`，并校验 **`MIN_ANCHORS_IN_POOL`**。全长回测表现与风控结论见 [v5.0 `03-optimization-summary.md`](../../my_docs/grid_multi_asset/v5.0/03-optimization-summary.md)（与聚宽仍可能因数据/规则微差而不完全逐点一致）。**_refresh_pool** 中过滤标的须与 v4 相同：直接写 ``not current_data[s].paused``，**不要使用** ``s in current_data``（聚宽側易导致全体被滤除、无成交）。
- **QDII ETF**（如 513100）在聚宽可能存在涨跌停、汇率或回放差异，与 SimTradeLab 回测数字不必逐点对齐。
