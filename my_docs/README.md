# SimTradeLab 策略研究文档索引

> 本目录按策略名称和版本号分级管理，每个版本独立成目录，方便追溯。

---

## 仓库说明

版本控制仅存在于 **`SimTradeLab/`** 目录（本仓库 Git 根目录）；上级目录 **`SimTrade/` 下无 `.git`**。克隆、分支、标签等命令均在 `SimTradeLab` 路径下执行即可。

---

## 策略目录

### grid_multi_asset — 多标的自适应网格

| 版本 | 状态 | 核心成果 | 文档 |
|---|---|---|---|
| **v1.0** | ✅ 已完成 | Holdout 年化 +60.51%，夏普 2.20，最大回撤 -16.28%；全长见 `03-optimization-summary` §5 | [查看 →](grid_multi_asset/v1.0/) |
| **v2.0** | ✅ WF/Holdout 已完成 | Trial 29；与 v1 相同 Holdout 区间：**+28.94%** 年化 / 夏普 **1.43** / 回撤 **-12.35%**（risk-off 结构，收益低于 v1 同段但回撤更紧） | [索引 →](grid_multi_asset/v2.0/README.md) · [报告 →](grid_multi_asset/v2.0/) |
| **v3.0** | 📝 设计已定稿 · 有计划 | M1+M2 · 仅用现货多头 · 与 v2 同口径 WF（见文档） | [设计 →](grid_multi_asset/v3.0/01-design.md) · [实施计划 →](grid_multi_asset/v3.0/02-plan.md) |

**代码对照：** v1 → `strategies/grid_multi_asset/`；v2 → `strategies/grid_multi_asset_v2/`（沪深300 regime 三档总仓 + 单标的水注权重上限）；v3（规划）→ `strategies/grid_multi_asset_v3/`。

---

## 文档命名规范

```
my_docs/
└── {策略名}/
    └── {版本号}/
        ├── 01-design.md              # 策略设计（背景、架构、逻辑）
        ├── 02-plan.md                # 实施计划（任务分解、TDD）
        └── 03-optimization-summary.md # 调参与回测总结报告
```

## Git 版本标签约定（SimTradeLab 仓库）

在 **`SimTradeLab/`** 下打标签时，建议使用下列语义与本策略里程碑对齐：

| Tag | 含义 |
|---|---|
| `v1.0` | grid_multi_asset 第一版：**Walk-Forward 调参与 Holdout** 已定稿；代码在 `grid_multi_asset/` |
| `v2.0` | grid_multi_asset 第二版：**regime + 总仓比例 + water-filling 上限**，WF/Holdout 已定稿（Trial 29）；代码在 `grid_multi_asset_v2/` |
| `v3.0` | 第三版：**M1/M2（防御池 + 熊市网格语义）**，WF 完成后打标；代码在 `grid_multi_asset_v3/` |

若在合并进 `main` 的里程碑提交上还未打语义化标签，可在该仓库根目录按需执行：`git tag -a <tag> <commit>`（与已有的其他版本标签并存即可）。
