# SimTradeLab 策略研究文档索引

> 本目录按策略名称和版本号分级管理，每个版本独立成目录，方便追溯。

---

## 策略目录

### grid_multi_asset — 多标的自适应网格

| 版本 | 状态 | 核心成果 | 文档 |
|---|---|---|---|
| **v1.0** | ✅ 已发布 | Holdout 年化 +11.99%，夏普 0.70 | [查看 →](grid_multi_asset/v1.0/) |
| v2.0 | 🚧 规划中 | 熊市保护 + 修复持仓不足 | [查看 →](grid_multi_asset/v2.0/) |

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

## Git 版本标签规范

| Tag | 含义 |
|---|---|
| `v1.0` | grid_multi_asset 第一版，Walk-Forward 调参完成 |
| `v2.0`（待创建）| grid_multi_asset 第二版，加入熊市保护机制 |
