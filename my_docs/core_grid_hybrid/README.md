# core_grid_hybrid（核心仓 + 网格混合）研究文档索引

> 与 `grid_multi_asset` 相同：按版本号分目录；**当前实现以 `strategies/core_grid_hybrid_v1/backtest.py` 为准**。

---

## 仓库说明

版本控制根目录为 **`SimTradeLab/`**。

---

## 当前实现摘要（v1.1 实际默认）

| 项 | 说明 |
|----|------|
| **默认模式** | `regime_mode='trend_sizing'`：单标的、按**慢均线偏离** \(z\) 映射目标权益权重 `w`，`order_target_value` 调仓 |
| **辅助风控** | 收盘创新高以来的**峰值回撤**超阈值则压低 `w`；`|w - 当前仓占比|` 小于**再平衡死区**则不调仓 |
| **原 v1.0 网格** | 几何网格 + `pick_grid_action_close` 等**纯函数仍保留**（单测），`handle_data` 默认**不再走**网格主路径 |
| **聚宽对照** | [`strategies_jq/core_grid_hybrid/v1/strategy.py`](../../strategies_jq/core_grid_hybrid/v1/strategy.py)（[`README`](../../strategies_jq/core_grid_hybrid/README.md)） |

设计文档 `v1.0/01-design.md` 仍以 **固定 ref 网格方案 A** 为历史规格；与代码的差异见同目录 **「附录 A」** 及 [`03-optimization-summary.md`](v1.0/03-optimization-summary.md)。

---

## 版本目录

| 版本 | 状态 | 摘要 | 文档 |
|------|------|------|------|
| **v1.0** | 设计已定稿；**实现已演进** | 单标的；原设计为死仓+活仓+几何网格；**当前 SimTradeLab 默认 = trend_sizing** | [理念 →](v1.0/00-rationale.md) · [设计 →](v1.0/01-design.md) · [实施计划 →](v1.0/02-plan.md) · [回测纪要 →](v1.0/03-optimization-summary.md) |

---

## 文档命名规范（与 grid_multi_asset 对齐）

```
my_docs/
└── core_grid_hybrid/
    └── v1.0/
        ├── 00-rationale.md          # 投资理念（原独立目录已合并）
        ├── 01-design.md
        ├── 02-plan.md
        └── 03-optimization-summary.md
```

**代码目录：**

```
strategies/
└── core_grid_hybrid_v1/
    ├── backtest.py
    ├── stats/
    └── …
```

**聚宽：**

```
strategies_jq/
└── core_grid_hybrid/
    ├── README.md
    └── v1/
        └── strategy.py
```

---

## 投资理念（文字稿）

详细直觉与纪律说明见 **[`v1.0/00-rationale.md`](v1.0/00-rationale.md)**（由原 `core_grid_hybrid_strategy` 目录合并而来）。
