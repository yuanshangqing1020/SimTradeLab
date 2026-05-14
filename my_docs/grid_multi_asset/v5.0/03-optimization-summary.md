# 多标的自适应网格策略 v5.0 — 调参与回测总结报告

**版本：** v5.0  
**日期：** 2026-05-13 ～ **2026-05-14**（WF）；门禁 **2026-05-14**；**全长 `run_backtest.py` 复核** **2026-05-14**（见 §5.2 日志名）  
**对应代码：** `strategies/grid_multi_asset_v5/`  
**归档参数对应 Optuna trial：** **190**（同参重复出现于 CSV 中 **272、561**；见 §三说明）  
**参数文件：** [`optimization/results/best_params_20260514_122926.json`](../../../strategies/grid_multi_asset_v5/optimization/results/best_params_20260514_122926.json)  
**Walk-Forward 结果 CSV：** [`optimization/results/trials_20260514_122926.csv`](../../../strategies/grid_multi_asset_v5/optimization/results/trials_20260514_122926.csv)  
**参数注入快照：** [`optimization/optimized_strategy.py`](../../../strategies/grid_multi_asset_v5/optimization/optimized_strategy.py)（与 `template.py` / `backtest.py` 的 `initialize` 字段一致）  
**聚宽等价实现：** [`strategies_jq/grid_multi_asset/v5/strategy.py`](../../../strategies_jq/grid_multi_asset/v5/strategy.py)（API 映射见 [`strategies_jq/grid_multi_asset/README.md`](../../../strategies_jq/grid_multi_asset/README.md)）

---

## 一、相对 v2/v3/v4 的机制（v5）

| 项 | 说明 |
|----|------|
| **Universe** | 默认搜索维包含 **`ANCHOR_SATELLITE`**（锚定 + 卫星合并池）与 **`WIDE_V2`**（与 v2 一致的 **15 只 ETF 母本**）。归档参数取 **`WIDE_V2`**。 |
| **锚定** | 换仓时在候选集中 **`build_grid_pool_anchor_first`** 优先塞入 **`510300.SS` / `510500.SS`**；**`MIN_ANCHORS_IN_POOL`** 为离散维（本组 **2**）。 |
| **Regime / 网格** | 与 v4 同族：周频刷新 `invested_ratio`、网格步长与层级等；另增 v3 对齐的 **BEAR 小维**（`BEAR_UNIVERSE_MODE` / `BEAR_GRID_MODE` / `BEAR_GRID_MAX_LAYER_CAP`）。 |
| **三重门禁** | WF 结束后对归档参数跑 **FULL / RECENT / BEAR**，按 [v4.0 `01-design.md` §2.2](../v4.0/01-design.md) 校验 **I ∧ II ∧ III**；并可用 [`two_stage_select.py`](../../../strategies/grid_multi_asset_v5/optimization/two_stage_select.py) 对 trials CSV 做 **先 I+II、再 III、再按 FULL 年化排序**（与 [`gate_eval.py`](../../../strategies/grid_multi_asset_v5/optimization/gate_eval.py) 同一套注入回测）。 |

---

## 二、Walk-Forward 与时间轴

与 v2/v3/v4 对齐：

- **优化期：** 2019-01-01 ～ 2024-12-31  
- **Holdout（RECENT 门禁窗）：** 2025-01-01 ～ 2026-03-31  
- **WF：** 训练 **24** 月 / 测试 **6** 月 / 步长 **6** 月（7 窗）  
- **评分：** 框架默认 `ScoringStrategy`（夏普、最大回撤、信息比率、胜率加权 + 稳定性惩罚）

---

## 三、调参运行统计

| 项目 | 数值 |
|------|------|
| 结果 CSV | `optimization/results/trials_20260514_122926.csv` |
| Wall clock（CSV 首尾） | 2026-05-13 **23:09** ～ 2026-05-14 **12:29**（约 **13.3 h**） |
| 理论参数组合数（未剪裁笛卡尔积） | **7,558,272**（`2×6×3×2×2×3×3×3×3×3×3×3×2×2×3×2`，维定义见 `optimize_params.GridMultiAssetV5Params`） |
| Trial 总行数 | **691** |
| `COMPLETE` | **254** |
| `PRUNED` | **437** |
| `COMPLETE` 且 `value = -9999`（约束占位） | **45** |
| **有效 COMPLETE**（`value > −9000`） | **209** |
| **Study 内 WF 目标最小值（有效 COMPLETE）** | **trial 4**，`value ≈ −0.8101`（参数与本文归档不同，见下） |
| **本文归档 trial（同参多行）** | **190**（及 **272、561**），WF `value ≈ −0.3119` |

**说明：** Optuna 的 **`value` 越小越优**（最小化）。**Trial 4** 在 WF 分数上优于 Trial 190，但 **Trial 190** 与目录中 **`best_params_20260514_122926.json`** 一致，代表本次工程选定的「归档候选」（例如经 `two_stage_select` 按 FULL 年化/卡玛重排，或人工导出——**若需与 `study.best_params` 严格一致，请核对本地导出脚本**）。

**Trial 190 分项（摘自 CSV `user_attrs`）：**

| avg_test | avg_train | test_std | train_test_gap |
|----------|-----------|----------|----------------|
| −0.0350 | **+0.0023** | 0.5538 | **0.0373** |

---

## 四、归档最优参数（Trial 190 / `best_params_20260514_122926.json`）

| 参数 | 值 |
|------|-----|
| `UNIVERSE_MODE` | **WIDE_V2** |
| `MAX_HOLD` | **3** |
| `MIN_ANCHORS_IN_POOL` | **2** |
| `GRID_STEP_VOL_FACTOR` | **0.60** |
| `GRID_STEP_MIN` | **0.01** |
| `GRID_STEP_MAX` | **0.05** |
| `GRID_MAX_LAYER` | **2** |
| `LAYER_FRACTION` | **0.08** |
| `VOL_WEIGHT` | **0.50** |
| `REBALANCE_FREQ` | **5** |
| `BULL_RATIO` | **0.70** |
| `NEUTRAL_RATIO` | **0.50** |
| `BEAR_RATIO` | **0.45** |
| `BEAR_UNIVERSE_MODE` | **SAME** |
| `BEAR_GRID_MODE` | **CAP_LAYER** |
| `BEAR_GRID_MAX_LAYER_CAP` | **0** |

**解读：** 在 **宽 ETF 池 + 低换手（5 日）+ 低波动权重（0.5）** 下，配合 **熊市层数封顶（CAP_LAYER，cap=0）**，WF 分段测试均分接近 0、训练测试 gap 较小；但 **FULL 样窗外推仍出现极深回撤**（§五），与「Holdout 极强」并存，需审慎解读数据与实盘可行性。

---

## 五、三重门禁（`gate_eval.py`，默认阈值见 [v4.0 `01-design.md` §2.2](../v4.0/01-design.md)）

**命令：**

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
conda run -n SimTrade python strategies/grid_multi_asset_v5/optimization/gate_eval.py \
  --params-json strategies/grid_multi_asset_v5/optimization/results/best_params_20260514_122926.json
```

**两阶段筛选（示例）：**

```bash
conda run -n SimTrade python strategies/grid_multi_asset_v5/optimization/two_stage_select.py \
  --trials-csv strategies/grid_multi_asset_v5/optimization/results/trials_20260514_122926.csv \
  --top 10
```

### 5.1 判定结果（默认 `GateThresholds`，**2026-05-14** 复现）

**`eligible`：否** — 未同时满足 **I（FULL 回撤）** 与 **II（FULL 超额 / IR）**。

| 门禁 | 条件（首轮建议） | 本参数实测结论 |
|------|------------------|----------------|
| **I** | FULL 最大回撤 ≤ **−38%**（值须 **≥ −0.38**） | **未通过**（约 **−57.16%**） |
| **II** | FULL 超额 ≥ **−10%**；IR ≥ **−0.05** | **未通过**（超额约 **−53.82%**，IR 约 **−0.321**） |
| **III** | RECENT 年化 ≥ **20%**；回撤 ≤ **18%**（值须 **≥ −0.18**）；夏普 ≥ **1.2** | **通过** |

**未通过明细（脚本摘要）：**

- `I: FULL max_drawdown -0.5716 < -0.3800`  
- `II: FULL excess_return -0.5382 < -0.1000`  
- `II: FULL information_ratio -0.3206 < -0.0500`  

> **注意：** 在 **I / II 未过** 的前提下，即便 **III** 通过，按设计 **仍不可标记为 eligible**。若 `two_stage_select` 曾输出本 JSON，请确认阶段 A 的阈值版本是否与当前 `gate_eval` 一致。

### 5.2 分段回测指标

#### FULL（2019-01-01 ～ 2026-04-20）— `gate_eval`（注入临时策略，`optimization_mode`）

与 **§5.1 门禁判据** 同源。

| 指标 | v5（Trial 190 归档） |
|------|------------------------|
| **总收益率** | **+6.39%** |
| **年化收益率** | **+0.89%** |
| **最大回撤** | **−57.16%** |
| **夏普比率** | **0.149** |
| **信息比率** | **−0.321** |
| **卡玛比率**（年化 / \|回撤\|） | **≈ 0.016** |
| **超额收益**（vs **000300.SS**） | **−53.82%** |
| **胜率** | **48.33%** |

#### FULL（同上）— `run_backtest.py` 默认全长（`strategies/grid_multi_asset_v5/backtest.py`，**非** `optimization_mode`）

与 v4 总结口径一致：**用于与终端/日志逐字对照**。修复前 `backtest.py` 曾为未合并的 v4 式副本（如 **仅 `NARROW_ETF` + `MAX_HOLD=6`**），会出现与 **v4 Trial 185** 几乎相同的 **+25.69% / +3.31% / −36.83%** 等结果，**不代表** Trial 190。**2026-05-14** 起 **`backtest.py` 已与 `template.py` 全量对齐** 且 `initialize` 与 **`best_params_20260514_122926.json`** 一致；以下为主机实测摘要（与上表 **数值一致**）。

实测 **2026-05-14**（终端摘要：`20190101-20260420`｜**1768** 个交易日｜初始 **50 万** → 期末约 **53.2 万**；另有一次复核日志 **`…_142108.*`**，**核心数字一致**）。

| 指标 | v5（Trial 190） |
|------|-----------------|
| **总收益率** | **+6.39%** |
| **年化收益率** | **+0.89%** |
| **最大回撤** | **−57.16%** |
| **夏普比率** | **0.149** |
| **信息比率** | **−0.321** |
| **索提诺比率** | **0.230** |
| **卡玛比率** | **0.016** |
| vs **000300.SS** | **超额收益 −53.82%** \| **Alpha −3.74%** \| **Beta 0.666** |
| **盈利天数 / 总交易日** | **854 / 1768**（**48.3%**） |
| **盈亏比** | **1.10** |
| **持仓规模** | 约 **3.0** 只（报告「最大 **4** 只」为框架统计口径） |

**日志 / 图表（全长复核，择一即可对照）：**

- `strategies/grid_multi_asset_v5/stats/backtest_190101_260420_260514_142108.log`
- `strategies/grid_multi_asset_v5/stats/backtest_190101_260420_260514_142108.png`  
- （较早一次）`…_141233.log` / `…_141233.png`

##### 全长表现为何算「很差」——与实务含义

以上数字在 **Triple Gate 口径** 下已触发 **I（回撤过深）** 与 **II（相对 300 极差）**，这里再用白话归纳一次，避免被「总收益仍为勉强正数」误导：

| 观察 | 含义 |
|------|------|
| **7 年+ 累计约 +6.4%** | 名义上略赚，但若扣通胀、摩擦或资金成本，**经济意义接近边缘化**；相对同期 **沪深300** 仍 **巨幅跑输**（超额约 **−54%**）。 |
| **最大回撤约 −57%** | 对于大多数**长尾权益 / 零售**资金约束，属 **不可接受** 级别；**卡玛仅 ~0.016** 说明「收益相对最大损失」极不相称。 |
| **夏普 ~0.15、信息比 ~−0.32** | **单位风险几乎没有补偿**；相对基准的主动管理质量指标为负侧。 |
| **Beta ~0.67 仍大幅跑输** | 说明拖累主要来自 **Alpha / 选股与交易路径**，而非「单纯低贝塔躺平」可解释。 |
| **与 RECENT 窗（§5.2 下表）反差极大** | 全长 **崩盘式回撤 + 线性跑输**，Holdout 却 **极强**，典型 **样本外结构不稳定** 信号：既可能是 **近年行情特例**，也可能含 **过拟合 / 窗口侥幸**；**绝不能**把近 15 个月当作策略「真实能力」的主证据。 |

**结论句：** 在现有结构与参数下，**Trial 190 不是可投的长度期结论**；文档保存这些数字的目的，是如实记录 **「v5 首轮 WF + 宽池试验在全长上的失败形态」**，供后续改目标、缩 Universe、加全长约束或换分支时对照。

#### RECENT / Holdout（2025-01-01 ～ 2026-03-31）— `gate_eval`

| 指标 | v5（Trial 190 归档） |
|------|------------------------|
| **总收益率** | **+77.87%** |
| **年化收益率** | **+62.47%** |
| **最大回撤** | **−12.41%** |
| **夏普比率** | **2.064** |
| **信息比率** | **2.356** |
| **超额收益** | **+61.39%** |
| **胜率** | **49.66%** |

#### BEAR（2021-01-01 ～ 2022-12-31）

| 指标 | v5（Trial 190 归档） |
|------|------------------------|
| **总收益率** | **+7.20%** |
| **年化收益率** | **+3.68%** |
| **最大回撤** | **−27.68%** |
| **夏普比率** | **0.273** |
| **信息比率** | **0.834** |
| **超额收益** | **+33.70%** |
| **胜率** | **49.17%** |

> 回测中偶发 **「交易量不足 / 无法获取价格」**、**科创板最小 200 股** 等提示，与 v4 总结一致，多为数据边界或仿真约束；详见终端完整日志。

---

## 六、与 v4 Trial 185 对照（同 FULL 截止 **2026-04-20**）

| 指标 | v4（Trial 185，摘自 [v4 总结 §5.2](../v4.0/03-optimization-summary.md)） | v5（本文归档） |
|------|----------------------------------|----------------|
| FULL 年化 | **+3.31%** | **+0.89%** |
| FULL 最大回撤 | **−36.83%** | **−57.16%**（更深） |
| FULL 夏普 | **0.325** | **0.149** |
| FULL 相对 300 超额 | **−34.52%** | **−53.82%** |
| RECENT 年化 | **+11.06%** | **+62.47%**（窗口内显著更强） |
| RECENT 夏普 | **0.908** | **2.064** |

**小结：** v5 本组参数在 **RECENT** 上远强于 v4，但在 **FULL** 上 **全面弱于 v4**（收益更低、回撤更深、相对 300 更差），属 **「近年窗极强、全长灾难」** 的割裂形态；**不可**仅凭 Holdout 亮眼即视为可上线版本。从资金属性看，**−57% 量级回撤** 对多数账户已属 **实质不可接受**。

---

## 七、结论与后续（对齐 v5 设计 §2「硬门禁」）

1. **全长结论（首要）：** **Trial 190 在 FULL 上为「极差」档位**——深回撤、巨幅跑输沪深300、风险调整后收益近乎无效（§5.2 末段评述）。在默认三重门禁下 **已被正式判否**；文中保留这些数字，是作为 **负面基线**：说明「仅扩 Universe + BEAR 小维 + WF」**不足以**自动得到可上线参数。  
2. **WF：** 归档 trial **190** 的 `value` 优于多数随机 trial，但 **非** Study 内 `value` 最优（trial **4** 更优）；说明 **单一 WF 标量** 与 **全长风险** 仍可能脱节。  
3. **三重门禁：** 在 **默认首轮红线** 下 **I、II 未过**，**III 通过** → **`eligible` 仍为否**。  
4. **实务含义：** 若坚持 **I ∧ II ∧ III** 同时达标，需继续 **收窄 Universe/换手**、**调整优化目标（例如加大 FULL 超额权重）**、或对 **WIDE_V2 成分在熊市段的暴露** 做结构约束；并建议对 **RECENT 极端优** 做样本外与敏感性复核。  
5. **工程交付：** `template.py` / `backtest.py` / `optimized_strategy.py` 已与 **`best_params_20260514_122926.json`** 对齐；复核命令见 §八。

---

## 八、复核命令

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
conda run -n SimTrade python strategies/grid_multi_asset_v5/optimization/gate_eval.py \
  --params-json strategies/grid_multi_asset_v5/optimization/results/best_params_20260514_122926.json
conda run -n SimTrade python strategies/grid_multi_asset_v5/optimization/two_stage_select.py \
  --trials-csv strategies/grid_multi_asset_v5/optimization/results/trials_20260514_122926.csv --top 10
conda run -n SimTrade python src/simtradelab/backtest/run_backtest.py   # 默认已设为 strategy_name=grid_multi_asset_v5
conda run -n SimTrade python -m pytest tests/unit/test_grid_multi_asset_v5.py -q
```
