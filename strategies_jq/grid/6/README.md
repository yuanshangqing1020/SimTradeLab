# 多标的分钟网格（聚宽）

设计说明：`docs/superpowers/specs/2026-05-07-joinquant-multi-grid-design.md`  
实现计划：`docs/superpowers/plans/2026-05-07-joinquant-multi-asset-grid.md`

## 聚宽回测（单文件）

聚宽策略编辑器**只能放一个完整策略**，不支持 `import` 你电脑上的其他工程文件。

1. 用本地编辑器打开 **`multi_asset_minute_grid.py`**。
2. **Ctrl+A 全选 → 复制**。
3. 打开 [聚宽](https://www.joinquant.com/) → 投资策略 → 新建策略 → **删除编辑器内默认代码**。
4. **粘贴**整份 `multi_asset_minute_grid.py`。
5. 回测设置：**分钟线**、初始资金（如 50 万）、起止日期。

该文件为单文件策略（无 `dataclasses` 等聚宽环境易缺的依赖）。  
分钟回测：`data[标的].close` 为引擎提供的**上一根已走完分钟 K** 收盘价，与 `g.prev_minute_close` 配对做穿档；勿用 `data.can_trade` / `data.current`。

## 本地开发（可选）

| 文件 | 用途 |
|------|------|
| `multi_asset_minute_grid.py` | 与聚宽**同一份**源码；改完粘贴回聚宽即可 |
| `jq_grid_pure.py` | 仅纯函数，供 `pytest`；**修改策略时请与单文件内「纯逻辑」区块同步** |

运行单元测试（仓库根目录）：

```bash
python3 -m pytest SimTradeLab/strategies_jq/grid/tests/test_jq_grid_pure.py -v
```

## 参数一览

| 变量 | 默认 | 含义 |
|------|------|------|
| N_TOTAL_TARGET | 30 | 目标总只数（含 3 ETF），股票数 = target − 3，且受 N_TOTAL_MAX 限制 |
| GRID_STEP | 0.009 | 统一档距 |
| GRID_LEVELS | 4 | 上下各档位数 |
| VOL_WINDOW / LIQ_WINDOW | 30 / 60 | 波动与流动性窗口 |

## 与 design spec 的实现差异（已知）

- **锚价**：在 `run_daily(9:30)` 内取**调仓当日日线开盘价**；失败则回退当日日线收盘价。
- **冷启动**：`g.securities` 为空时也会在首个 `run_daily` 日换池。
- **分钟收盘价**：design 中的「分钟收盘价」在聚宽分钟 `handle_data` 下即 **`data[标的].close`**（上一完整分钟 K），与官方语义一致；已开 `avoid_future_data`。
- **同 K 多档**：一次穿多档可能产生多笔委托（见策略文件头）。

## Lint（本地）

```bash
python3 useful_skills/joinquant-skill/scripts/strategy_lint.py SimTradeLab/strategies_jq/grid/multi_asset_minute_grid.py
```

## 回测验证（人工，聚宽 Web）

- 震荡段与压力段分别看最大回撤与年化。
- 调整 `GRID_STEP` 做敏感性对比。
- 确认日志中出现 `quarter rebalance` 与每只 `anchor`。
