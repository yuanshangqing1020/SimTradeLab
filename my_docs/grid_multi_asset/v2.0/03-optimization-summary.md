# 多标的自适应网格策略 v2.0 — 调参与回测总结报告

**版本：** v2.0  
**日期：** 2026-05-09 ～ 2026-05-10  
**对应代码：** `strategies/grid_multi_asset_v2/`  
**分支：** `main`  
**最优 trial：** Trial **29**  
**最优参数存档：** [`optimization/results/best_params_20260510_011341.json`](../../../strategies/grid_multi_asset_v2/optimization/results/best_params_20260510_011341.json)  
**自动生成策略：** [`optimization/optimized_strategy.py`](../../../strategies/grid_multi_asset_v2/optimization/optimized_strategy.py)（与 `template.py` / `backtest.py` 中已对齐的最优参数一致）

---

## 一、调参设计原理

### 1.1 Walk-Forward 滚动优化

与 [v1.0 总结](../v1.0/03-optimization-summary.md) **同一套时间切分**（便于横比）：优化区间不含留存期，留存期为完全样本外。

```
时间轴示意：
2019  2020  2021  2022  2023  2024  2025  2026
├─────────────────────────────────────────────┤
│  优化区间（训练+测试）    │  Holdout（样本外）│
│  2019-01 ~ 2024-12       │  2025-01~2026-03  │
└──────────────────────────┴───────────────────┘

Walk-Forward 7 个滚动窗口（训练 24 月 / 测试 6 月 / 滑动 6 月）：
窗口  训练期                    测试期
W1    2019-01 ~ 2021-01        2021-01 ~ 2021-07
W2    2019-07 ~ 2021-07        2021-07 ~ 2022-01
…（同 v1 图，此处略）…
W7    2022-01 ~ 2024-01        2024-01 ~ 2024-07
```

**每个 trial：** 依次跑完 W1→W7，以 7 个测试期得分经稳定性惩罚（与 v1 相同框架）得到最终综合得分。

### 1.2 综合评分公式（与 v1 一致）

```
综合得分 = Sharpe×0.40 + (−MaxDrawdown)×0.30 + IR×0.20 + WinRate×0.10
最终得分 = 测试期均值 − std(测试期得分) × 0.5
```

### 1.3 Optuna 与 v2 任务设置

- **采样器：** TPESampler  
- **目标：** maximize  
- **剪枝：** 流程中按中间表现提前终止弱 trial（见 journal）  
- **早停：** `optimize_params.py` 中 `patience=500`（与 v1 的「参数空间÷4」口径不同，属显式自定义）  
- **断点续传：** `optimization/results/optuna_journal.log`  
- **策略差异：** v2 在候选参数上增加 `BULL_RATIO` / `NEUTRAL_RATIO` / `BEAR_RATIO` 及约束 `BEAR < NEUTRAL < BULL`

### 1.4 相对 v1 的策略扩展（实现层）

| 模块 | 作用 |
|------|------|
| `_calc_regime` / `_detect_regime` | 沪深300 价格相对 `MA120` / `MA250` → `BULL` / `NEUTRAL` / `BEAR`，并设置总仓 `invested_ratio` |
| `_apply_weight_cap` | 单标的权重 water-filling 截断 |
| `_execute_grid` | 组合资金 = `min(总资产×invested_ratio, TARGET_CAPITAL)`，再按截断后权重分配 |

---

## 二、参数空间

共 **11** 个可调参数，理论组合数 **11,664**（离散网格；无效组合在 `validate` 中拒绝）。

| 参数 | 候选 | **最优值（Trial 29）** | 含义 |
|------|------|------------------------|------|
| `MAX_HOLD` | 5, 8, 10, 12, 15 | **10** | 最大持仓标的数 |
| `GRID_STEP_VOL_FACTOR` | 0.30, 0.45, 0.60 | **0.60** | 步长 = clip(vol×factor, min, max) |
| `GRID_STEP_MIN` | 0.01, 0.02 | **0.02** | 网格步长下限 |
| `GRID_STEP_MAX` | 0.03, 0.05 | **0.03** | 网格步长上限 |
| `GRID_MAX_LAYER` | 2, 3, 4 | **2** | 最大偏离层数 |
| `LAYER_FRACTION` | 0.08, 0.12, 0.16 | **0.08** | 层间权重增量 |
| `VOL_WEIGHT` | 0.50, 0.65, 0.80 | **0.50** | 波动率在综合打分中的权重 |
| `REBALANCE_FREQ` | 5, 10, 20 | **20** | 换股间隔（交易日） |
| `BULL_RATIO` | 0.70, 0.80, 0.90 | **0.70** | 牛市总投入比例 |
| `NEUTRAL_RATIO` | 0.50, 0.60, 0.70 | **0.60** | 震荡总投入比例 |
| `BEAR_RATIO` | 0.25, 0.35, 0.45 | **0.25** | 熊市总投入比例 |

**约束：** `GRID_STEP_MIN < GRID_STEP_MAX`；`BEAR_RATIO < NEUTRAL_RATIO < BULL_RATIO`。

---

## 三、运行方式

### 3.1 启动调参

```bash
cd /mnt/c/QMTReal/SimTrade/SimTradeLab
conda run -n SimTrade python strategies/grid_multi_asset_v2/optimization/optimize_params.py
```

- 支持断点续传；本次批量约 **6.9 小时**墙钟（见结果 CSV 起止时间）。

### 3.2 本地回测（修改 `run_backtest.py` 后）

```bash
# 顶部配置示例：strategy_name / start_date / end_date / initial_capital
conda run -n SimTrade python src/simtradelab/backtest/run_backtest.py
```

**注意：** 策略设计容量 **50 万**；资金过小易导致单手无法成交。

### 3.3 监控调参进度

```bash
python3 - << 'EOF'
import json
trials = {}
with open('strategies/grid_multi_asset_v2/optimization/results/optuna_journal.log') as f:
    for line in f:
        d = json.loads(line)
        if d.get('op_code') == 6 and d.get('state') == 2:
            trials[d['trial_id']] = d['values'][0]
if trials:
    best_tid = max(trials, key=trials.get)
    print(f"已完成: {len(trials)} trials，最佳 Trial {best_tid}: {trials[best_tid]:.4f}")
else:
    print("尚无完成的 trial")
EOF
```

### 3.4 单元测试

```bash
conda run -n SimTrade python -m pytest tests/unit/test_grid_multi_asset_v2.py -q
```

---

## 四、调参运行结果

### 4.1 运行统计

| 项目 | 数值 |
|------|------|
| 结果 CSV | `optimization/results/trials_20260510_011341.csv` |
| 墙钟跨度（CSV 首尾） | 2026-05-09 18:22 ～ 2026-05-10 01:13（约 **6.9 小时**） |
| Trial 总数 | **530** |
| 状态 `COMPLETE` | **151**（其中 **47** 条为约束/无效组合占位分 **-9999**） |
| 有效完整 trial（走完 WF） | **104** |
| `PRUNED` | **379** |
| 最终最优 trial | **Trial 29** |

### 4.2 最优参数（Trial 29）

```python
context.MAX_HOLD             = 10    # 与 v1 一致：集中持仓
context.GRID_STEP_VOL_FACTOR = 0.60
context.GRID_STEP_MIN        = 0.02  # 相对 v1 更窄的步长区间
context.GRID_STEP_MAX        = 0.03
context.GRID_MAX_LAYER       = 2
context.LAYER_FRACTION       = 0.08
context.VOL_WEIGHT           = 0.50
context.REBALANCE_FREQ       = 20     # 月频换股（v1 最优常为 10）
context.BULL_RATIO           = 0.70
context.NEUTRAL_RATIO        = 0.60
context.BEAR_RATIO           = 0.25
```

**解读要点：**

- **步长收窄 + 层数 2：** 控制极端逆势加仓；配合熊档 **25%** 总仓。
- **`REBALANCE_FREQ=20`：** 与 v1 最优「双周」不同，来自联合搜索下的稳定折中。

### 4.3 Walk-Forward 优化期综合得分

| 指标 | 数值 | 说明 |
|------|------|------|
| 综合得分（最终） | **-0.3457** | 仍为负，主因测试窗覆盖 2021-2022 等弱势段（与 v1 同型） |
| 训练期均分 | +0.1707 | |
| 测试期均分 | -0.0234 | |
| 测试期标准差 | 0.6447 | |
| 训练/测试差距 | 0.1941 | |
| 过拟合比率 | **113.68%** | 行情结构差 > 单纯过拟合时需对照 Holdout / 全样本 |

> **对照 v1（Trial 53）WF：** v1 报告综合得分 **-0.3665**。v2（Trial 29）**-0.3457** 略优（负得更少），但二者同为熊市窗主导下的负分区间，不宜单独解读为「预测收益」。

---

## 五、回测结果

### 5.1 Holdout 样本外（2025-01-01 ～ 2026-03-31）

> **未参与训练与调参**，与优化脚本 `holdout_period` 一致；用于与 v1 同口径对比。

| 指标 | v2（Trial 29） |
|------|----------------|
| **总收益率** | **+35.20%** |
| **年化收益率** | **+28.94%** |
| **夏普比率** | **1.426** |
| **最大回撤** | **-12.35%** |
| 信息比率 | 1.205 |
| Alpha | +15.73% |
| Beta | 0.963 |
| 胜率 | 53.36% |
| 盈亏比 | 1.116 |
| 优化器给出的样本外综合得分 | **0.9019** |

#### 与 v1 Holdout 对照（同区间，数据来自各版本总结报告）

| 指标 | v1（Trial 53） | v2（Trial 29） |
|------|----------------|----------------|
| 年化收益 | ~+60.5% | ~+28.9% |
| 夏普 | ~2.20 | ~1.43 |
| 最大回撤 | ~-16.3% | **~-12.4%（更缓和）** |
| Beta | ~1.07 | **~0.96** |

> **背景：** 该 Holdout 段偏有利于趋势/动量类结构的行情；v2 **主动降仓**，在同段上通常 **牺牲收益换回撤与市场暴露**，与上表一致。

---

### 5.2 全样本长周期回测（2019-01-01 ～ 2026-03-31）

> **含优化区间内全部牛熊结构**，用于观察「参数在整段历史上的可实现路径」；**与 §5.1 结论可同时阅读**——短段大涨不必然等于长段高年化。

SimTradeLab 实测（`run_backtest.py`：`grid_multi_asset_v2`，初始资金 **50 万**，2026-05-10 运行日志）：

| 指标 | 数值 |
|------|------|
| **区间** | 2019-01-01 ～ 2026-03-31（**1755** 个交易日） |
| **总收益率** | **+48.95%** |
| **年化收益率** | **+5.89%** |
| **最大回撤** | **-39.83%** |
| **夏普比率** | **0.441** |
| 信息比率 | -0.006 |
| 索提诺比率 | 0.597 |
| 卡玛比率 | 0.148 |
| vs 000300.SS | 超额收益 **-0.91%**；Alpha **+2.62%**；Beta **0.547** |
| 盈利天数 | 883 / 1755（**50.3%**） |
| 盈亏比 | 1.07 |
| 持仓规模 | 约 **9.8** 只（最大约 **11** 只） |
| 期末资产 | 约 **50 万 → 74.5 万** |

**与 §5.1 的对照说明：**

| 维度 | Holdout（约 15 个月） | 全样本（约 7.2 年） |
|------|------------------------|----------------------|
| 年化 | **高**（+28.94%） | **低**（+5.89%） |
| 最大回撤 | **-12.35%** | **-39.83%**（长段含深度熊市与漫长 recovery） |
| 信息比 | 正 | 接近 0 |

因此：**v2 在「未参与优化的近年段」仍具正 Holdout 表现，但全历史收益与夏普偏温和、回撤仍深**——说明 regime 与降仓能缓解部分风险，却**不能替代**对长周期极值风险的预期管理。

---

## 六、文件结构

```
strategies/grid_multi_asset_v2/
├── backtest.py                 # 直接回测（Trial 29 参数）
├── template.py                 # Walk-Forward 优化器读取的模板
├── stats/                      # 回测 log / png
└── optimization/
    ├── optimize_params.py
    ├── optimized_strategy.py
    └── results/
        ├── optuna_journal.log
        ├── trials_20260510_011341.csv
        └── best_params_20260510_011341.json

tests/unit/
└── test_grid_multi_asset_v2.py

strategies_jq/grid_multi_asset/
├── README.md
├── v1/strategy.py
└── v2/strategy.py              # JoinQuant 版（参数与 Trial 29 对齐）

my_docs/grid_multi_asset/v2.0/
├── 01-design.md
├── 02-plan.md
└── 03-optimization-summary.md   # 本文件
```

---

## 七、已知问题与注意事项

| 问题/现象 | 说明 |
|-----------|------|
| 科创板最小 200 股 | 与 v1 相同，小资金网格可能产生无法成交的委托，回测日志中可见取消/失败 |
| Holdout 与全样本结论反差 | 短窗结构性行情 vs 长段复利与极大回撤并存，报告需同时披露（§5.1 + §5.2） |
| 优化目标为 WF 综合得分 | 非直接最大化 Holdout 或全长年化；若重点为某一子区间，需单独设定优化目标或约束 |

---

## 八、关键结论

1. **WF 层面对比 v1：** Trial 29 综合得分 **-0.3457** 略优于 v1 报告的 **-0.3665**，二者仍处于熊市测试窗主导的负分区间。
2. **Holdout 仍为正：** 样本外综合得分 **0.9019**，同段年化与夏普低于 v1，但 **回撤与 Beta 更温和**（§5.1 对照表）。
3. **全长回测更克制：** 2019～2026 实测年化约 **+5.89%**，最大回撤 **-39.83%**，夏普 **0.441**；说明策略在 **完整牛熊** 下的体验与「近段牛市 Holdout」不可混为一谈。
4. **工程定位：** v2 适合作为「降 Beta、控回撤尝试」的网格变体；是否替代 v1 取决于资金对收益与回撤的偏好。
5. **后续可做：** 2021～2022 熊市区间专项报表、聚宽 `v2/strategy.py` 与 SimTradeLab 对齐验证、参数对「全长回撤」的敏感度分析。

---

## 九、后续可选方向

| 优先级 | 方向 | 具体内容 |
|--------|------|----------|
| 高 | 熊市截面 | 对 2021-01～2022-12 等区间单独出表，与 v1 同区间对比回撤与收益 |
| 中 | 目标对齐 | 若最关心全长卡玛/最大回撤，可讨论将子目标写入优化或二次筛选 trial |
| 中 | 外部验证 | JoinQuant 回放与 SimTradeLab 分段对齐（注意复权、停牌、最小手差异） |
| 低 | 运营 | 模拟盘、仓位再平衡日历、与组合其余策略的相关性 |
