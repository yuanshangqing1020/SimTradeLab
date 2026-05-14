# 多标的自适应网格策略 v4.0 — 调参与回测总结报告

**版本：** v4.0  
**日期：** 2026-05-12 ～ 2026-05-13（WF）；门禁与报告 **2026-05-13**；**全长默认回测** 复核 **2026-05-13**（见 §5.2 日志名）  
**对应代码：** `strategies/grid_multi_asset_v4/`  
**最优 trial：** **185**（Optuna `number`，综合得分最优）  
**最优参数存档：** [`optimization/results/best_params_20260513_120438.json`](../../../strategies/grid_multi_asset_v4/optimization/results/best_params_20260513_120438.json)  
**参数注入快照：** [`optimization/optimized_strategy.py`](../../../strategies/grid_multi_asset_v4/optimization/optimized_strategy.py)  
**已与 `template.py` / `backtest.py` 的 `initialize` 对齐**

---

## 一、相对 v2/v3 的机制（v4）

| 项 | 说明 |
|----|------|
| **Universe** | 默认 **窄 ETF 固定池 6 只**（`NARROW_ETF_UNIVERSE`），不做 300+500 成分股选股 |
| **Regime** | 沪深300 MA120/MA250 三档；**`REGIME_REFRESH=WEEKLY`**：换仓日或每 **5** 个交易日刷新 `invested_ratio` |
| **三重门禁** | WF 结束后对 `best_params` 跑 **FULL / RECENT / BEAR**，按 [01-design.md §2.2](./01-design.md) 校验 **I ∧ II ∧ III**（见 §五） |

---

## 二、Walk-Forward 与时间轴

与 v2/v3 对齐：

- **优化期：** 2019-01-01 ～ 2024-12-31  
- **Holdout（RECENT 门禁窗）：** 2025-01-01 ～ 2026-03-31  
- **WF：** 训练 **24** 月 / 测试 **6** 月 / 步长 **6** 月（7 窗）  
- **评分：** 夏普、最大回撤、信息比率、胜率加权 + 稳定性惩罚（框架默认 `ScoringStrategy`）

---

## 三、调参运行统计

| 项目 | 数值 |
|------|------|
| 结果 CSV | `optimization/results/trials_20260513_120438.csv` |
| Wall clock（CSV 首尾） | 2026-05-12 **13:56** ～ 2026-05-13 **12:04**（约 **22.1 h**） |
| 理论参数组合数（笛卡尔积） | **104,976**（`4×3×2×2×3×3×3×3×3×3×3`） |
| Trial 总行数 | **26,430** |
| `COMPLETE` | **7,408** |
| `PRUNED` | **19,022** |
| `COMPLETE` 且 `value = -9999`（约束占位） | **2,229** |
| **有效 COMPLETE**（`value > −9000`） | **5,179** |
| **最终最优 trial** | **185** |
| **Walk-Forward 综合得分（value）** | **−0.5180**（_CSV 打印值 −0.5179924623587872_） |

**Trial 185 分项（摘自 CSV `user_attrs`）：**

| avg_test | avg_train | test_std | train_test_gap |
|----------|-----------|----------|----------------|
| −0.3911 | −0.0308 | 0.2538 | **0.3603** |

> 训练期均分为弱负、gap 较大，「过拟合比率」不宜简单套用 v2 正值训练口径；以 **分段回测 + 门禁** 为主结论。

---

## 四、最优参数（Trial 185 / `best_params`）

| 参数 | 最优值 |
|------|--------|
| `MAX_HOLD` | **6**（满池） |
| `GRID_STEP_VOL_FACTOR` | **0.60** |
| `GRID_STEP_MIN` | **0.02** |
| `GRID_STEP_MAX` | **0.05** |
| `GRID_MAX_LAYER` | **2** |
| `LAYER_FRACTION` | **0.08** |
| `VOL_WEIGHT` | **0.80** |
| `REBALANCE_FREQ` | **20** |
| `BULL_RATIO` | **0.70** |
| `NEUTRAL_RATIO` | **0.50** |
| `BEAR_RATIO` | **0.45** |

**解读：** WF 在窄池上偏 **高波动权重（0.8）**、**月频换股**、**三档仓位偏宽（熊档 45%）**、步长上下限拉满，与「吃 ETF 波动」设定一致，但对 **全长跑输基准** 仍敏感（见 §5.2）。

---

## 五、三重门禁（`gate_eval.py`，默认阈值见 [01-design.md §2.2](./01-design.md)）

**命令：**

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
conda run -n SimTrade python strategies/grid_multi_asset_v4/optimization/gate_eval.py \
  --params-json strategies/grid_multi_asset_v4/optimization/results/best_params_20260513_120438.json
```

### 5.1 判定结果

**`eligible`：否** — 未同时满足 **II（全长相对 300）** 与 **III（近年窗收益/夏普）**。

| 门禁 | 条件（首轮建议） | 本参数实测结论 |
|------|------------------|----------------|
| **I** | FULL 最大回撤 ≤ **−38%**（不深于 38%） | **通过**（约 **−36.83%**） |
| **II** | FULL 超额 ≥ **−10%**；IR ≥ **−0.05** | **未通过**（超额约 **−34.52%**，IR 约 **−0.355**） |
| **III** | RECENT 年化 ≥ **20%**；回撤 ≤ **18%**；夏普 ≥ **1.2** | **部分通过**（回撤、年化未达夏普/年化门槛） |

**未通过明细（门禁脚本输出）：**

- `II: FULL excess_return -0.3452 < -0.1000`  
- `II: FULL information_ratio -0.3553 < -0.0500`  
- `III: RECENT annual_return 0.1106 < 0.2000`  
- `III: RECENT sharpe_ratio 0.9077 < 1.2000`  

### 5.2 分段回测指标

**FULL（全长）主表** 以 **`run_backtest.py` 默认回测** 为准：`strategy_name=grid_multi_asset_v4`（`backtest.py`，**非** `optimization_mode`），便于与终端/日志逐字对照。**核心数字与同参数下 `gate_eval.py` 的 FULL 段一致**。

#### FULL（2019-01-01 ～ 2026-04-20）

实测 **2026-05-13**（终端摘要：`20190101-20260420`｜**1768** 个交易日｜初始 **50 万** → 期末约 **62.8 万**）。

| 指标 | v4（Trial 185） |
|------|-----------------|
| **总收益率** | **+25.69%** |
| **年化收益率** | **+3.31%** |
| **最大回撤** | **−36.83%** |
| **夏普比率** | **0.325** |
| **信息比率** | **−0.355** |
| **索提诺比率** | **0.413** |
| **卡玛比率** | **0.090** |
| vs **000300.SS** | **超额收益 −34.52%** \| **Alpha −0.64%** \| **Beta 0.569** |
| **盈利天数 / 总交易日** | **890 / 1768**（**50.4%**） |
| **盈亏比** | **1.05** |
| **持仓规模** | 约 **5.7** 只（最大 **6** 只） |

**日志 / 图表（本次全长复核）：**

- `strategies/grid_multi_asset_v4/stats/backtest_190101_260420_260513_142621.log`
- `strategies/grid_multi_asset_v4/stats/backtest_190101_260420_260513_142621.png`

#### RECENT / Holdout（2025-01-01 ～ 2026-03-31）

| 指标 | v4（Trial 185） |
|------|-----------------|
| **总收益率** | **+13.25%** |
| **年化收益率** | **+11.06%** |
| **最大回撤** | **−9.05%** |
| **夏普比率** | **0.908** |
| **信息比率** | **−0.452** |
| **超额收益** | **−3.23%** |
| **胜率** | **53.02%** |

#### BEAR（2021-01-01 ～ 2022-12-31）

| 指标 | v4（Trial 185） |
|------|-----------------|
| **总收益率** | **−25.08%** |
| **年化收益率** | **−13.93%** |
| **最大回撤** | **−30.18%** |
| **夏普比率** | **−1.226** |
| **信息比率** | **+0.065** |
| **超额收益** | **+1.42%** |
| **胜率** | **47.73%** |

> **RECENT / BEAR** 仍来自 **`gate_eval.py`** 分段回测（`optimization_mode`），与 §5.1 门禁一致。**FULL** 上表与 **2026-05-13** `run_backtest.py` 终端/日志一致。  
> 偶发 **「交易量不足 / 无法获取价格」**（如个别 ETF bar）多为数据边界；详见 `stats/*.log`。

---

## 六、与 v2 Trial 29 对照（同 FULL 截止 **2026-04-20**）

| 指标 | v2（Trial 29，摘自 v2 总结 §5.2） | v4（Trial 185） |
|------|-------------------------------------|-----------------|
| 年化 | ~**+6.01%** | **+3.31%** |
| 最大回撤 | ~**−39.83%** | **−36.83%**（略浅） |
| 夏普 | ~**0.449** | **0.325** |
| 相对 300 超额 | ~**−9.57%** | **−34.52%**（显著更差） |

**与 v2 Holdout 对照（RECENT）：** v2 同段年化约 **+28.94%**、夏普 **1.43**；v4 为 **+11.06%** / **0.908**。窄池 + 当前目标下，**WF 最优并未在「全长相对基准 + 近年强势」上同时达标**。

---

## 七、结论与后续（对齐设计 §2.2「无解」分支）

1. **WF：** v4 最优得分 **−0.518**，弱于 v2（**−0.346**）与 v3（**−0.346**）所在档位，属 **熊市测试窗主导下的另一套参数结构**，不宜解读为收益排序。  
2. **三重门禁：** 在 **首轮建议红线** 下 **II、III 未过**；**I**（FULL 回撤）相对 v2 略改善，但 **跑输沪深300 幅度明显扩大**。  
3. **实务含义：** 若坚持 **I+II+III 同时达标**，需按 [01-design.md](./01-design.md)：**放宽顺序（先 II/III 校准阈值）**、改 **优化目标加权全长超额**、或启用计划中 **结构分支**（如双层资金、扩大对照 `WIDE_V2` 子试验），而非仅复用本组参数上实盘。  
4. **工程交付：** `backtest.py` / `template.py` 已写入 Trial **185**；`run_backtest.py` 默认 **`strategy_name=grid_multi_asset_v4`** 时可复现 §5.2 FULL 全长表（参见该次 `stats` 日志时间戳）。

---

## 八、复核命令

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
conda run -n SimTrade python strategies/grid_multi_asset_v4/optimization/gate_eval.py \
  --params-json strategies/grid_multi_asset_v4/optimization/results/best_params_20260513_120438.json
conda run -n SimTrade python src/simtradelab/backtest/run_backtest.py   # strategy_name=grid_multi_asset_v4
conda run -n SimTrade python -m pytest tests/unit/test_grid_multi_asset_v4.py -q
```
