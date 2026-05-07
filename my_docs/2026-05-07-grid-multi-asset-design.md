# 多标的自适应网格策略设计文档

**日期：** 2026-05-07  
**策略目录：** `strategies/grid_multi_asset/`  
**参考原型：** `strategies_jq/grid/5/高波动优质-网格多标的.py`

---

## 1. 目标与约束

| 项目 | 值 |
|---|---|
| 起始资金 | 50 万元（`initial_capital = 500_000`） |
| 最大持仓标的数 | 10 ～ 50 只（参数 `MAX_HOLD`，默认 20） |
| 标的类型 | 沪深300 + 中证500 成分股 + 主流行业/宽基 ETF |
| 每格步长范围 | 1% ～ 5%（由波动率自适应决定） |
| 参数确定方式 | Optuna Walk-Forward 自动回测调参 |

---

## 2. 总体架构

```
strategies/grid_multi_asset/
├── backtest.py              ← 主策略（PTrade API 格式，可直接在 SimTradeLab 本地回测）
└── optimization/
    └── optimize_params.py   ← Optuna + Walk-Forward 自动调参脚本
```

### 三层逻辑

| 层级 | 执行频率 | 职责 |
|---|---|---|
| **Universe 选股** | 每 `REBALANCE_FREQ` 个交易日 | 从动态指数成分+ETF中，按波动率+基本面综合打分，选 Top-`MAX_HOLD` 只 |
| **Grid 调仓** | 每个交易日收盘前 | 对活跃池每只标的计算网格层数，等权归一化后 `order_target_value` |
| **参数优化** | 离线按需运行 | Optuna TPE + Walk-Forward 滚动验证，防过拟合，输出最优参数 JSON |

---

## 3. Universe 选股逻辑

### 3.1 候选池来源

每次选股时动态拉取：
- `get_index_stocks('000300.SS')` — 沪深300 成分股（约 300 只）
- `get_index_stocks('000905.SS')` — 中证500 成分股（约 500 只）
- 固定 ETF 候选池（约 15 只，含宽基、行业、跨境）：

```python
CANDIDATE_ETFS = [
    '510300.SS',  # 沪深300ETF
    '510500.SS',  # 中证500ETF
    '159915.SZ',  # 创业板ETF
    '512880.SS',  # 证券ETF
    '512690.SS',  # 酒ETF
    '512010.SS',  # 医药ETF
    '515050.SS',  # 5GETF
    '512480.SS',  # 半导体ETF
    '159949.SZ',  # 创业板50ETF
    '588000.SS',  # 科创板50ETF
    '512170.SS',  # 医疗ETF
    '512760.SS',  # 芯片ETF
    '159792.SZ',  # 黄金ETF
    '513100.SS',  # 纳指ETF
    '513050.SS',  # 中概互联ETF
]
```

### 3.2 过滤规则

```python
# 剔除 ST 股
st_status = get_stock_status(codes, 'ST')
# 剔除停牌
halt_status = get_stock_status(codes, 'HALT')
# 股票侧：剔除基本面不合格（市值<30亿、PE<=0或>120）
fundamentals = get_fundamentals(stock_codes, 'valuation', ['pe_ratio', 'market_cap', 'roe'])
```

### 3.3 综合打分

**股票侧（波动率 + 基本面）：**
```
vol_pct  = rank(vol20)  / N          # 波动率百分位
roe_pct  = rank(roe)    / N
inv_pe_pct = rank(1/pe) / N
mcap_pct = rank(mcap)   / N          # 大市值偏稳，作为稳定性因子

qual_pct = roe_pct × 0.45 + inv_pe_pct × 0.35 + mcap_pct × 0.20
score    = vol_pct × VOL_WEIGHT + qual_pct × (1 - VOL_WEIGHT)
```

**ETF 侧（波动率 + 流动性）：**
```
score = vol_pct × VOL_WEIGHT + liquidity_pct × (1 - VOL_WEIGHT)
```

取合并后 Top-`MAX_HOLD` 只作为活跃网格池。调出老标的前先 `order_target(stock, 0)` 清仓。

---

## 4. 自适应网格逻辑（每日）

### 4.1 步长计算

```
vol   = 20日年化已实现波动率（std(ret[-20:]) × √250）
step  = clip(vol × GRID_STEP_VOL_FACTOR, GRID_STEP_MIN, GRID_STEP_MAX)
```

步长范围约束：`1% ≤ step ≤ 5%`（由 `GRID_STEP_MIN`/`GRID_STEP_MAX` 控制）

### 4.2 网格层数

```
MA20  = 近20日收盘均价
layer = clip(floor((MA20 - price) / (price × step) + 0.5), -MAX_LAYER, +MAX_LAYER)
```

- `layer > 0`：价格低于中枢，超配（加仓）
- `layer < 0`：价格高于中枢，欠配（减仓）
- `layer = 0`：在中枢，持等权仓位

### 4.3 权重分配

```
N       = len(active_pool)
w_raw_i = (1/N) × (1 + LAYER_FRACTION × layer_i)
w_i     = w_raw_i / Σw_raw                        # 归一化到 100%
cap     = min(context.portfolio.total_value, 500_000)
target_i = cap × w_i
order_target_value(stock_i, target_i)
```

---

## 5. 可调参数定义

| 参数 | 默认值 | 候选值（调参） | 说明 |
|---|---|---|---|
| `g.MAX_HOLD` | 20 | [10, 20, 30, 50] | 最多同时持仓标的数 |
| `g.GRID_STEP_VOL_FACTOR` | 0.45 | [0.30, 0.45, 0.60] | 波动率乘数 |
| `g.GRID_STEP_MIN` | 0.01 | [0.01, 0.02] | 步长下限 |
| `g.GRID_STEP_MAX` | 0.04 | [0.03, 0.05] | 步长上限 |
| `g.GRID_MAX_LAYER` | 3 | [2, 3, 4] | 最大网格层数 |
| `g.LAYER_FRACTION` | 0.12 | [0.08, 0.12, 0.16] | 每层权重增减幅度 |
| `g.VOL_WEIGHT` | 0.62 | [0.50, 0.65, 0.80] | 波动率评分权重 |
| `g.REBALANCE_FREQ` | 5 | [5, 10, 20] | 重新选股间隔（交易日） |

**参数验证约束：** `GRID_STEP_MIN < GRID_STEP_MAX`

**参数空间总大小：** 4×3×2×2×3×3×3×3 = **3,888 组合**

---

## 6. 自动调参配置

### 6.1 时间划分

```
优化期：2019-01-01 ～ 2024-12-31（6年，覆盖多轮牛熊）
留存期：2025-01-01 ～ 2026-03-31（15个月，样本外验证）

Walk-Forward 滚动窗口：
  train_months = 24  测试窗口前24个月做训练
  test_months  = 6   滑动验证窗口
  step_months  = 6   每次向前滑动6个月
```

### 6.2 防过拟合机制

- **Walk-Forward Analysis**：在滚动时间窗口上取 test score 均值作为目标，而非训练期得分
- **稳定性惩罚**：`final_score -= std(test_scores) × stability_weight`
- **正则化惩罚**：参数极值（边界10%范围内）被惩罚
- **早停**：连续 `patience ≈ 972` 次 trial 无改进即停止（`space_size / 4`）
- **断点续传**：通过 `optuna_journal.log` 支持中断后继续

### 6.3 评分函数

```python
score = sharpe_ratio    × 0.40   # 风险调整收益
      + (-max_drawdown) × 0.30   # 最大回撤控制
      + information_ratio × 0.20 # 超额收益能力
      + win_rate          × 0.10 # 交易质量
```

### 6.4 输出

| 文件 | 位置 | 内容 |
|---|---|---|
| `best_params_<ts>.json` | `optimization/results/` | 最优参数 |
| `optimized_strategy.py` | `optimization/` | 已注入最优参数的策略 |
| `trials_<ts>.csv` | `optimization/results/` | 所有 trial 记录 |
| `optuna_journal.log` | `optimization/results/` | Optuna 断点数据 |

---

## 7. PTrade API 适配说明

| JQ 原始 API | SimTradeLab PTrade API |
|---|---|
| `get_index_stocks('000300.XSHG', date)` | `get_index_stocks('000300.SS')` |
| `get_fundamentals(query(valuation...).filter(...))` | `get_fundamentals(codes, 'valuation', ['pe_ratio','market_cap','roe'])` |
| `get_price(codes, end_date, count, 'daily', ['close'])` | `get_history(count, '1d', 'close', codes)` |
| `attribute_history(s, 30, '1d', ['close'], df=True)` | `get_history(30, '1d', 'close', s)` |
| `d.paused` / `d.is_st` | `get_stock_status(codes, 'HALT')` / `get_stock_status(codes, 'ST')` |
| `run_weekly(fn, weekday=1)` | `handle_data` 中 `g.day_counter % g.REBALANCE_FREQ == 0` |
| `run_daily(fn, time='14:50')` | `handle_data` 末尾统一执行 |
| `order_target_value(stock, value)` | `order_target_value(stock, value)` ✅ 相同 |

---

## 8. 错误处理与边界情况

- 历史数据不足（<20根K线）：跳过该标的，不纳入网格池
- 波动率为零或负数：跳过
- 候选池为空（数据异常）：保留原池，打印警告日志
- 调出标的时先清仓，再更新 pool
- 总资产低于 1000 元时：使用实际总资产而非 500000 作为 cap

---

## 9. 文件命名与目录规范

```
strategies/grid_multi_asset/
├── backtest.py
└── optimization/
    ├── optimize_params.py
    └── results/              ← 自动创建，.gitignore 排除大文件
        ├── best_params_*.json
        ├── trials_*.csv
        ├── optuna_journal.log
        └── backtest_cache/   ← 回测缓存，加速重复调参
```

---

## 10. 实施步骤概览

1. 创建 `strategies/grid_multi_asset/backtest.py`（主策略）
2. 创建 `strategies/grid_multi_asset/optimization/optimize_params.py`（调参脚本）
3. 本地运行单次回测验证策略逻辑无误
4. 运行 `optimize_params.py` 执行 Walk-Forward 调参
5. 使用 `optimization/optimized_strategy.py` 做最终性能验证
