# 多标的自适应网格策略 v3.0 — 调参与回测总结报告

**版本：** v3.0  
**日期：** 2026-05-10  
**对应代码：** `strategies/grid_multi_asset_v3/`  
**分支：** `main`  
**最优 trial：** Trial **357**（同组参数在研究中多次出现等价 trial，以 Optuna **best_params** 为准）  
**最优参数存档：** [`optimization/results/best_params_20260510_232314.json`](../../../strategies/grid_multi_asset_v3/optimization/results/best_params_20260510_232314.json)  
**自动生成策略：** [`optimization/optimized_strategy.py`](../../../strategies/grid_multi_asset_v3/optimization/optimized_strategy.py)（应与 `template.py` / `backtest.py` 中 `initialize` 一致）

---

## 一、相对 v2 的机制增量（简述）

在 v2 基础上仅增加 **M2（BEAR 防御池开关，本最优解为 SAME）** 与 **M1（熊市网格模式）**：

| 维度 | v3 最优（Trial 357） |
|------|----------------------|
| `BEAR_UNIVERSE_MODE` | **SAME**（选股池与 v2 全池一致；未启用 ETF_DEFENSIVE） |
| `BEAR_GRID_MODE` | **CAP_LAYER** |
| `BEAR_GRID_MAX_LAYER_CAP` | **0**（BEAR 时有效层数为 `min(2,0)=0`，等价于横盘层不参与加仓侧） |

其余为联合搜索下的 **网格与仓位旋钮**（`MAX_HOLD=12`、`REBALANCE_FREQ=10`、熊档 **`BEAR_RATIO=0.45`** 等）。

---

## 二、Walk-Forward 与时间轴（与 v2 对齐）

- **优化期：** 2019-01-01 ～ 2024-12-31  
- **Holdout：** 2025-01-01 ～ 2026-03-31  
- **WF 窗口：** 训练 **24** 月 / 测试 **6** 月 / 滑动 **6** 月（7 窗，与 [v2.0 总结](./../v2.0/03-optimization-summary.md) 一致）

综合评分口径与 v1/v2 **相同**（夏普、最大回撤、IR、胜率加权 + 稳定性惩罚）。

---

## 三、调参运行统计

| 项目 | 数值 |
|------|------|
| 结果 CSV | `optimization/results/trials_20260510_232314.csv` |
| Wall clock（CSV 首尾 timestamp） | 2026-05-10 **10:33** ～ **23:23**（约 **12.8 h**） |
| 理论参数组合数（笛卡尔积） | **2,361,960** |
| Trial 取样（journal 总行级） | **858** |
| `COMPLETE` | **341**（其中 **57** 条为占位分 **-9999**：约束拒绝等） |
| **有效 COMPLETE（value > −9000）** | **284** |
| `PRUNED` | **517** |
| **`patience`** | **500**（`optimize_params.py` 显式） |

Walk-Forward **最终最优综合得分：** **−0.3463**（_csv 数值 −0.346336822181109_）。

**与 [v2 Trial 29 WF 得分 −0.3457](../v2.0/03-optimization-summary.md) 对照：** v3 在 **WF 指标上基本持平或略差一点**（同负分区间，不宜解释为「实盘收益优劣」）。

**Trial 357 分项（摘自 CSV）：**

| avg_test | avg_train | test_std | train_test_gap |
|----------|-----------|---------|----------------|
| −0.01879 | +0.14858 | 0.65509 | **0.16737** |

**过拟合比率（按 v2 报告口径：** `train_test_gap / |avg_train|` **×100%）：** ≈ **112.65%**

---

## 四、最优参数（Trial 357 / best_params）

| 参数 | 最优值 |
|------|--------|
| `MAX_HOLD` | **12** |
| `GRID_STEP_VOL_FACTOR` | **0.45** |
| `GRID_STEP_MIN` | **0.01** |
| `GRID_STEP_MAX` | **0.05** |
| `GRID_MAX_LAYER` | **2** |
| `LAYER_FRACTION` | **0.08** |
| `VOL_WEIGHT` | **0.65** |
| `REBALANCE_FREQ` | **10** |
| `BULL_RATIO` | **0.70** |
| `NEUTRAL_RATIO` | **0.50** |
| `BEAR_RATIO` | **0.45** |
| `BEAR_UNIVERSE_MODE` | **SAME** |
| `BEAR_GRID_MODE` | **CAP_LAYER** |
| `BEAR_GRID_MAX_LAYER_CAP` | **0** |

**解读要点：** WF 选了 **更近 v1 风格的「偏高波动权重 + 双周换股 + 多一些持仓」**；熊档给到 **45%**，与 v2 最优 **25%** 侧重不同。**M2 在本次最优仍为 SAME**，防御池未贡献该 trial；**M1 CAP_LAYER+cap 0** 在 BEAR **压缩网格偏移层**，侧重控制逆势加仓路径。

---

## 五、回测结果（SimTradeLab，`backtest.py` 已写入最优参数）

### 5.1 Holdout（2025-01-01 ～ 2026-03-31）

实测日志：`strategies/grid_multi_asset_v3/stats/backtest_250101_260331_260511_071831.log`  
初始资金 **50 万**，策略 `grid_multi_asset_v3`。

| 指标 | v3（Trial 357） |
|------|----------------|
| **总收益率** | **+56.07%** |
| **年化收益率** | **+45.53%** |
| **最大回撤** | **−13.50%** |
| **夏普比率** | **1.943** |
| 信息比率 | **2.123** |
| vs 000300.SS | **超额 +39.59%** \| Alpha **+32.73%** \| Beta **0.933** |
| 胜率 / 盈亏比 | **52.7%** / **1.26** |
| 交易日 | **299** |
| 持仓规模 | **11.6** 只（最大 **13**） |

#### 与 v2 Holdout 对照（同源口径，取自各版总结 / 实测日志）

| 指标 | v2（Trial 29） | v3（Trial 357） |
|------|----------------|-----------------|
| 年化收益 | ~+28.9% | **~+45.5%（更高）** |
| 夏普 | ~1.43 | **~1.94** |
| 最大回撤 | ~−12.4% | ~−13.5% |
| Beta | ~0.96 | ~0.93 |

> **说明：** Holdout 段行情对参数极敏感；v3 在本段 **收益与夏普显著高于 v2**，回撤略深一点，仍属「强区间」表现，**不可单独外推至全长**。

---

### 5.2 全样本长周期（2019-01-01 ～ 2026-04-20）

与 [v2 §5.2](../v2.0/03-optimization-summary.md) **同一截止日**，便于横比。  
日志：`strategies/grid_multi_asset_v3/stats/backtest_190101_260420_260511_071858.log`

| 指标 | v3（Trial 357） |
|------|----------------|
| **总收益率** | **+19.84%** |
| **年化收益率** | **+2.61%** |
| **最大回撤** | **−39.90%** |
| **夏普比率** | **0.238** |
| 信息比率 | **−0.309** |
| vs 000300.SS | **超额 −40.36%** \| Alpha **−1.70%** \| Beta **0.621** |
| 盈利天数 | **901 / 1768（51.0%）** |
| 盈亏比 | **1.00** |
| 持仓规模 | **11.7** 只（最大 **13**） |

**与 v2 同窗对照（v2 数字来自 v2 总结 §5.2）：**

| 指标 | v2（Trial 29） | v3（Trial 357） |
|------|----------------|-----------------|
| 年化 | ~+6.01% | **~+2.61%（更低）** |
| 最大回撤 | ~−39.83% | ~−39.90%（接近） |
| 夏普 | ~0.449 | **~0.238** |
| 超额(300) | ~−9.57% | **~−40.36%** |

> **结论：** 在 **「2019～2026-04-20 全长」** 维度，v3 最优解 **相对 v2 显著跑输净值与相对基准**，与 Holdout 的强势 **同时成立**——属典型 **区间敏感与目标函数未直接优化全长** 现象；部署前须明确资金更重视 **近段 Holdout** 还是 **长跑全长**。

---

### 5.3 熊市专项（2021-01-01 ～ 2022-12-31）

日志：`strategies/grid_multi_asset_v3/stats/backtest_210101_221231_260511_072135.log`  
（日终统计日 **2022-12-30** 为样本内最后交易日属正常。）

| 指标 | v3（Trial 357） |
|------|----------------|
| **总收益率** | **−5.03%** |
| **年化收益率** | **−2.65%** |
| **最大回撤** | **−23.91%** |
| **夏普比率** | **−0.107** |
| vs 000300.SS | **超额 +21.47%** \| Alpha **+3.50%** \| Beta **0.416** |
| 胜率 / 盈亏比 | **52.5%** / **0.89** |

> 该两年窗内 **Beta 明显下降**，相对沪深300 **超额为正**，但策略 **绝对收益仍为小幅负**，符合「风控/相对表现 vs 绝对收益」分化的预期。

---

## 六、文件结构（节选）

```
strategies/grid_multi_asset_v3/
├── backtest.py                    # ✅ Trial 357 最优参数（直接回测）
├── template.py                    # ✅ 同上（便于与优化注入对照）
├── stats/                         # 回测日志（含本报告所列三次）
└── optimization/
    ├── optimize_params.py
    ├── optimized_strategy.py
    └── results/
        ├── best_params_20260510_232314.json
        ├── trials_20260510_232314.csv
        └── optuna_journal.log

my_docs/grid_multi_asset/v3.0/
├── 01-design.md
├── 02-plan.md
└── 03-optimization-summary.md   # 本文件
```

---

## 七、结论与可选后续

1. **WF：** v3 最优 **−0.3463** 与 v2 **−0.3457** **几乎等价**，均属熊市测试窗主导的负分区。  
2. **Holdout：** v3 Trial 357 **远强于** v2 同段表象指标，但 **全长（§5.2）明显弱于 v2**：说明 **不能以单次 Holdout 定版长途资金容量**。  
3. **熊市窗（§5.3）：** 较低 Beta、超额为正，但仍未扭转为 **正绝对收益**。  
4. **后续：** 若在全长上补强，需在 **优化目标或二次筛选 trial** 中显式加权卡玛/最大回撤/全长年化；或对 `ETF_DEFENSIVE` **单独做子研究**（本最优仍为 SAME）。

---

## 八、复核命令

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
conda run -n SimTrade python src/simtradelab/backtest/run_backtest.py   # strategy_name=v3 & 区间见源码注释
conda run -n SimTrade python -m pytest tests/unit/test_grid_multi_asset_v3.py -q
```
