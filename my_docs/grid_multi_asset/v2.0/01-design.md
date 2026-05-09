# 多标的自适应网格策略 v2.0 — 设计文档

**版本：** v2.0  
**日期：** 2026-05-09  
**作者：** AI 协作设计  
**状态：** 已确认，待实施

---

## 一、背景与目标

### 1.1 v1.0 核心成果

- Holdout 2025-2026 样本外：年化 **+60.51%**，夏普 **2.20**，最大回撤 **-16.28%**
- Walk-Forward 综合得分：-0.3665（2021-2022 熊市区间拖累）
- 最优参数：MAX_HOLD=10，VOL_WEIGHT=0.50，REBALANCE_FREQ=10

### 1.2 v1.0 核心缺陷

| 缺陷 | 根本原因 | 影响 |
|---|---|---|
| 第一天满仓 | `cap = min(tv, TARGET_CAPITAL)`，归一化权重之和为 1.0，首日即投入全部资金 | 网格补仓无弹药，逻辑上自相矛盾 |
| 熊市无保护 | 无大盘趋势判断，持续下行时不断补仓加重亏损 | Walk-Forward 熊市区间得分低 |

### 1.3 v2.0 目标

1. **修复仓位管理**：首日只建底仓（~60%），保留现金作为网格补仓弹药
2. **加入趋势保护**：大盘熊市时强制低仓（~35%），减少持续下行时的损失
3. **重新调参验证**：完整 Walk-Forward 优化新参数空间，与 v1 同口径对比
4. **双轨维护**：SimTradeLab 本地开发/调参 + JoinQuant 平台验证，与 v1 保持一致

---

## 二、整体架构

v2 对 v1 做**最小侵入式改造**，只新增/修改两处，其余完全不动：

```
┌─────────────────────────────────────────────────────────────┐
│                      每日 handle_data                        │
├──────────────────┬──────────────────────────────────────────┤
│  _refresh_pool   │  不变（选股、评分、换股清仓）                │
├──────────────────┼──────────────────────────────────────────┤
│  _detect_regime  │  【新增】大盘趋势判断 → BULL/NEUTRAL/BEAR   │
├──────────────────┼──────────────────────────────────────────┤
│  _execute_grid   │  【修改】引入 invested_ratio + 单标的上限    │
└──────────────────┴──────────────────────────────────────────┘
```

**新增全局状态：**
- `context.regime`：当前大盘状态（`'BULL'` / `'NEUTRAL'` / `'BEAR'`）
- `context.invested_ratio`：当前目标总投入比例（由 regime 决定）

**不变部分：**
- 选股逻辑（`_score_universe`、`_refresh_pool`）完全不动
- 网格层数计算（`_calc_layer`）完全不动
- 归一化函数（`_normalize_weights`）完全不动
- SimTradeLab ↔ JoinQuant 双轨结构不变
- v1 的 24 个单元测试继续有效，v2 只追加针对新功能的测试

---

## 三、大盘趋势判断（`_detect_regime`）

### 3.1 判断逻辑

```python
def _detect_regime(context):
    """判断大盘趋势，更新 context.regime 和 context.invested_ratio。
    仅在换股日（_refresh_pool 同一天）调用，避免每日重复拉长历史数据。
    """
    hist = get_history(260, '1d', 'close', ['000300.SS'])
    prices = hist['000300.SS'].dropna().values

    if len(prices) < 250:
        # 历史数据不足时降级为中性，不影响策略启动
        context.regime = 'NEUTRAL'
    else:
        price_now = prices[-1]
        ma120 = prices[-120:].mean()
        ma250 = prices[-250:].mean()
        above_120 = price_now > ma120
        above_250 = price_now > ma250

        if above_120 and above_250:
            context.regime = 'BULL'
        elif (not above_120) and (not above_250):
            context.regime = 'BEAR'
        else:
            context.regime = 'NEUTRAL'

    ratio_map = {
        'BULL':    context.BULL_RATIO,
        'NEUTRAL': context.NEUTRAL_RATIO,
        'BEAR':    context.BEAR_RATIO,
    }
    context.invested_ratio = ratio_map[context.regime]
    log.info('大盘状态: %s | 投入比例: %.0f%%' % (
        context.regime, context.invested_ratio * 100))
```

### 3.2 状态定义

| 状态 | 判断条件 | 含义 |
|---|---|---|
| `BULL` | 沪深300 > MA120 **且** > MA250 | 趋势向上，允许高仓 |
| `BEAR` | 沪深300 < MA120 **且** < MA250 | 趋势向下，强制低仓 |
| `NEUTRAL` | 其他（MA120/MA250 之间，或数据不足） | 震荡，标准仓位 |

### 3.3 调用时机

`_detect_regime` 与 `_refresh_pool` 在同一天调用（即每 `REBALANCE_FREQ` 交易日一次）。非换股日直接使用 `context.invested_ratio` 的缓存值，不重复拉数据。

```python
def handle_data(context, data):
    context.day_counter += 1
    if context.day_counter == 1 or context.day_counter % context.REBALANCE_FREQ == 0:
        _detect_regime(context)   # 先判断大盘
        _refresh_pool(context)    # 再选股
    _execute_grid(context)
```

---

## 四、仓位管理（`_execute_grid` 改造）

### 4.1 单标的权重上限

上限由现有参数自动推导，**不新增额外参数**：

```
max_w_per_stock = (1 / N) × (1 + LAYER_FRACTION × GRID_MAX_LAYER)
```

示例（v1 最优参数 N=10, LAYER_FRACTION=0.08, GRID_MAX_LAYER=2）：
```
max_w = 10% × (1 + 0.08×2) = 10% × 1.16 = 11.6%
```

10 只标的等权为 10%，最多允许超配到 11.6%，语义清晰。调参改变 LAYER_FRACTION 或 GRID_MAX_LAYER 时上限自动跟随，不增加参数空间维度。

### 4.2 权重截断与再归一化

```python
# 计算原始权重（与 v1 相同）
raw_w  = [max((1.0 / N) * (1.0 + context.LAYER_FRACTION * float(lyr)), 1e-9)
          for lyr in layers]
norm_w = _normalize_weights(raw_w)

# 截断 + 再归一化（迭代 3 次确保收敛）
max_w = (1.0 / N) * (1.0 + context.LAYER_FRACTION * context.GRID_MAX_LAYER)
for _ in range(3):
    clipped = [min(w, max_w) for w in norm_w]
    norm_w  = _normalize_weights(clipped)
```

### 4.3 趋势感知的总投入上限

```python
tv  = context.portfolio.portfolio_value
cap = tv * context.invested_ratio          # 动态，随 regime 变化
cap = min(cap, TARGET_CAPITAL)             # 绝对上限，防超规模

for code, w in zip(active, norm_w):
    order_target_value(code, cap * w)
```

### 4.4 效果对比

| 场景 | v1 投入 | v2 投入 | 剩余现金（50万本金） |
|---|---|---|---|
| 第一天（NEUTRAL）| 500,000（100%） | ~300,000（60%） | ~200,000 |
| 牛市稳定期 | 500,000（100%） | ~400,000（80%） | ~100,000 |
| 熊市下行期 | 500,000（100%） | ~175,000（35%） | ~325,000 |

---

## 五、参数空间与 Walk-Forward 调参

### 5.1 完整参数空间

共 11 个参数，候选组合数约 **11,664 种**：

| 参数 | 候选值 | v1 最优值 | 说明 |
|---|---|---|---|
| `MAX_HOLD` | 5, 8, 10, 12, 15 | 10 | 加密区间精调 |
| `GRID_STEP_VOL_FACTOR` | 0.30, 0.45, 0.60 | 0.45 | 步长波动率系数 |
| `GRID_STEP_MIN` | 0.01, 0.02 | 0.01 | 步长下限 |
| `GRID_STEP_MAX` | 0.03, 0.05 | 0.05 | 步长上限 |
| `GRID_MAX_LAYER` | 2, 3, 4 | 2 | 最大偏离层数 |
| `LAYER_FRACTION` | 0.08, 0.12, 0.16 | 0.08 | 层间仓位增量 |
| `VOL_WEIGHT` | 0.50, 0.65, 0.80 | 0.50 | 波动率选股权重 |
| `REBALANCE_FREQ` | 5, 10, 20 | 10 | 换股频率（交易日） |
| `BULL_RATIO` | 0.70, 0.80, 0.90 | — | 牛市投入比例（**新增**） |
| `NEUTRAL_RATIO` | 0.50, 0.60, 0.70 | — | 震荡投入比例（**新增**） |
| `BEAR_RATIO` | 0.25, 0.35, 0.45 | — | 熊市投入比例（**新增**） |

**约束条件（自动拒绝非法 trial）：**
1. `GRID_STEP_MIN < GRID_STEP_MAX`
2. `BEAR_RATIO < NEUTRAL_RATIO < BULL_RATIO`

### 5.2 Walk-Forward 窗口（与 v1 相同）

```
窗口  训练期                    测试期
W1    2019-01 ~ 2021-01        2021-01 ~ 2021-07
W2    2019-07 ~ 2021-07        2021-07 ~ 2022-01
W3    2020-01 ~ 2022-01        2022-01 ~ 2022-07
W4    2020-07 ~ 2022-07        2022-07 ~ 2023-01
W5    2021-01 ~ 2023-01        2023-01 ~ 2023-07
W6    2021-07 ~ 2023-07        2023-07 ~ 2024-01
W7    2022-01 ~ 2024-01        2024-01 ~ 2024-07
Holdout：2025-01-01 ~ 2026-03-31（最终样本外评估）
```

### 5.3 评分公式（与 v1 相同）

```
综合得分 = Sharpe×0.40 + (−MaxDrawdown)×0.30 + IR×0.20 + WinRate×0.10
最终得分 = 测试期均值 − std(测试期得分) × 0.5
```

### 5.4 预估耗时

参数组合 11,664 种（v1 为 3,888 种），早停机制不变（连续 `参数空间÷4` 次无改进）。预估 **30~48 小时**，取决于早停触发时机。

---

## 六、测试计划

### 6.1 v1 原有测试（24 个，不改动）

`tests/unit/test_grid_multi_asset.py` 全部继续有效。

### 6.2 v2 新增测试

文件：`tests/unit/test_grid_multi_asset_v2.py`

| 测试用例 | 验证内容 |
|---|---|
| `test_detect_regime_bull` | 价格在 MA120/MA250 上方 → `'BULL'` |
| `test_detect_regime_bear` | 价格在 MA120/MA250 下方 → `'BEAR'` |
| `test_detect_regime_neutral_cross` | 价格在 MA120 上、MA250 下 → `'NEUTRAL'` |
| `test_detect_regime_short_history` | 历史数据不足 250 条 → 降级为 `'NEUTRAL'` |
| `test_weight_cap_no_overflow` | 截断后任意单标的权重 ≤ max_w |
| `test_weight_cap_converges` | 3 轮迭代后权重幂等（再迭代不变） |
| `test_invested_ratio_bear_lower` | BEAR 下 cap < NEUTRAL 下 cap < BULL 下 cap |
| `test_constraint_ratio_order` | BEAR_RATIO < NEUTRAL_RATIO < BULL_RATIO 约束生效 |

---

## 七、文件结构

```
strategies/
└── grid_multi_asset_v2/          ← 全新目录，v1 完整保留不动
    ├── backtest.py               ← 直接回测（调参完成后填入最优参数）
    ├── template.py               ← Walk-Forward 调参模板
    ├── stats/                    ← 回测输出（log + png）
    └── optimization/
        ├── optimize_params.py    ← 调参入口（更新参数空间）
        └── results/
            └── optuna_journal.log

tests/unit/
├── test_grid_multi_asset.py      ← v1 原有（不改动）
└── test_grid_multi_asset_v2.py   ← v2 新增

strategies_jq/grid_multi_asset/
├── README.md                     ← 版本对照表（新增 v2 行）
├── v1/strategy.py                ← 不动
└── v2/strategy.py                ← JQ 版（调参完成后移植）

my_docs/grid_multi_asset/
├── v1.0/                         ← 不动
└── v2.0/
    ├── 01-design.md              ← 本文件
    ├── 02-plan.md                ← 实施计划（writing-plans 生成）
    └── 03-optimization-summary.md ← 调参完成后填写
```

---

## 八、JoinQuant 移植规范

调参完成、SimTradeLab Holdout 结果满意后，按如下 API 映射移植到 JoinQuant：

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

移植后在 JoinQuant 平台用相同的 Holdout 时段（2025-01-01 ~ 2026-03-31）做独立验证，与 SimTradeLab 结果对比。

---

## 九、成功标准

| 指标 | 目标 |
|---|---|
| Holdout 年化收益 | ≥ v1（+60.51%），或回撤显著改善可接受小幅收益下降 |
| Holdout 最大回撤 | < v1（-16.28%） |
| Walk-Forward 综合得分 | > v1（-0.3665），熊市区间改善是核心目标 |
| 首日仓位 | < 70%（NEUTRAL 状态下） |
| 熊市仓位 | < 50%（BEAR 状态下） |
| 单标的最大权重 | ≤ max_w_per_stock 公式上限 |
