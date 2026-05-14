# core_grid_hybrid（核心仓 + 网格混合）研究文档索引

> 与 `grid_multi_asset` 相同：按版本号分目录，设计 / 计划 / 调参总结三包；代码在 `strategies/core_grid_hybrid_v{}/`。

---

## 仓库说明

版本控制根目录为 **`SimTradeLab/`**（与 [`my_docs/grid_multi_asset/README.md`](../grid_multi_asset/README.md) 一致）。

---

## 版本目录

| 版本 | 状态 | 摘要 | 文档 |
|------|------|------|------|
| **v1.0** | 设计已定稿 | 单标的、死仓+活仓+现金；**网格基准 = 固定锚定价（A）**；整体止盈；SimTradeLab 单轨回测 | [设计 →](v1.0/01-design.md) · `02-plan` / `03-optimization-summary` 待编写 |

---

## 文档命名规范（与 grid_multi_asset 对齐）

```
my_docs/
└── core_grid_hybrid/
    └── {版本号}/
        ├── 01-design.md                 # 策略设计（背景、架构、逻辑、目录）
        ├── 02-plan.md                   # 实施计划（任务分解、单测与验收）
        └── 03-optimization-summary.md   # 调参与回测总结
```

**代码目录（规划）：**

```
strategies/
└── core_grid_hybrid_v1/
    ├── backtest.py              # 回测入口（BacktestConfig.strategy_name 指向本目录名）
    ├── template.py              # 可选：与 backtest 同步的逻辑草稿
    ├── stats/                  # 回测日志
    └── optimization/           # 可选 v1.1+：Optuna / 离散搜索
        ├── optimize_params.py
        └── results/
```

**聚宽对照：** 不强制；若需要可增设 `strategies_jq/core_grid_hybrid/v1/strategy.py`（与 v5 策略族约定一致：**SimTradeLab 为规格真源**）。

---

## 理念说明（原文）

[`../core_grid_hybrid_strategy /README.MD`](../core_grid_hybrid_strategy%20/README.MD)（目录名含空格，建议日后迁入本目录 `v1.0/00-rationale.md` 或合并入 `01-design.md` §0）。
