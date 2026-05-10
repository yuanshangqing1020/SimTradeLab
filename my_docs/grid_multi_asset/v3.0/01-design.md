# 多标的自适应网格策略 v3.0 — 设计文档

**版本：** v3.0  
**日期：** 2026-05-10  
**策略目录（规划）：** `strategies/grid_multi_asset_v3/`（fork 自 v2，实施后以此为准）  
**参考基线：** v2.0 设计文档 [../v2.0/01-design.md](../v2.0/01-design.md)、代码 `strategies/grid_multi_asset_v2/`  
**状态：** 已确认，待实施  

---

## 一、目标与非目标

### 1.1 目标

在 **仅持有现货多头（股票与 ETF）** 的前提下，相对 v2 再引入一层 **机制级** 变化：

- **M1（熊市网格行为）**：在大盘处于下行 regime 时，限制网格的 **净加仓** 或 **最大层数**，避免在持续下行中被动把弹药打满。
- **M2（防御型 Universe）**：在相应 regime 下，将选股候选池 **收窄** 为预定义的 **低风险宽基/红利等 ETF 子集**（仍全部为多头标的），与 v2 的三档总仓比例、单标 water-filling **正交**，并连同 M1 **一并进入 Walk-Forward + Optuna** 下一轮优化。

**验收：** 与 v2 **同一时间切分、同一 Holdout** 并列披露三张表：**Holdout**、**2019 年起至约定截止日的全长**、**2021～2022 熊市专项**：便于与 v2 Trial 29 对照。

### 1.2 非目标（硬约束）

以下 **不得** 纳入 v3.0 设计及回测假设：

| 类别 | 说明 |
|------|------|
| 衍生品/做空 | 股指期货、期权、反向 ETF、融券卖空等 |
| 对冲 | 任何形式的 **Beta 对冲腿** 或「用现金模拟对冲」的虚拟账户 |
| 双轨时效 | 不要求 v3.0 与 JoinQuant 同步上线；**SimTradeLab 为规格与实现首选真源** |

---

## 二、与 v2 的关系

- **基线：** 从 v2 策略 **fork** 为新目录 `strategies/grid_multi_asset_v3/`（命名以仓库约定为准）。
- **保留：** `_detect_regime`（沪深300 相对 MA120/MA250 → BULL / NEUTRAL / BEAR）、`context.invested_ratio` 与三档 ratio、单标 water-filling、Universe 打分与网格步长/layer 主体框架。
- **新增：** 仅 **M1**（`_execute_grid` 语义）、**M2**（`_refresh_pool` 候选池分流），资产类别仍为 **全多头现货**。

---

## 三、执行顺序（`handle_data`）

与 v2 一致，并标明 M1/M2 触点：

```
若 首日 或 day_counter % REBALANCE_FREQ == 0：
    _detect_regime(context)     # 更新 regime、invested_ratio（仅在此处刷新）
    _refresh_pool(context)      # 【M2】按 regime 选全量合并池或防御 ETF 子池

每日：
    _execute_grid(context)      # 【M1】在约定 regime 下应用网格加仓限制
```

**约定：**

- `regime` 与 `invested_ratio` **仅在换股日**更新；非换股日使用缓存。
- **M2 只在 `_refresh_pool` 生效**，避免日内切换 Universe 语义。
- **M1 在 `_execute_grid` 生效**，在 `invested_ratio` 与资金 cap 之内限制相对前一日的净加仓或有效层数上限。

---

## 四、M2：防御型 Universe

### 4.1 行为

- **NEUTRAL / BULL：** 候选池与 **v2 完全一致**（见 v2 `backtest.py` / `template`）。
- **BEAR：** 候选池切换为 **`DEFENSIVE_ETF_POOL`** —— 项目内 **单一常量表** 维护的宽基/红利等 ETF；**不得**含期货、融券、反向品种。

### 4.2 Walk-Forward 离散参数（建议）

| 符号名 | 含义 |
|--------|------|
| `BEAR_UNIVERSE_MODE` | `SAME`：BEAR 仍用合并全池（关 M2 对照） / `ETF_DEFENSIVE`：BEAR 仅用防御 ETF 池 |

**可选扩展：** `ETF_PLUS_LOW_VOL_STOCK` 等第三档不在 v3.0 必达；若引入须单独立项评估 trial 规模与过滤逻辑。

---

## 五、M1：熊市网格行为

### 5.1 适用 regime

**默认仅 BEAR**；是否扩展到 NEUTRAL 由实现计划收口为离散参数（避免本设计阶段无谓膨胀）。

### 5.2 离散参数（建议）

| 符号名 | 含义 |
|--------|------|
| `BEAR_GRID_MODE` | `NORMAL`：与 v2 相同 / `NO_NET_ADD`：相对 **前一交易日**不允许净加仓（可减仓）/ `CAP_LAYER`：BEAR 下有效最大层数为 `min(GRID_MAX_LAYER, BEAR_GRID_MAX_LAYER_CAP)` |
| `BEAR_GRID_MAX_LAYER_CAP` | `CAP_LAYER` 时使用；WF 候选建议 `{0, 1, 2}` |

**NO_NET_ADD：** 参照必须使用回测可用的 **无前视** 定义（如前收后名义持仓/市值），在具体实现计划中 **唯一固定**，不得混用易引入歧义的日内价格。

### 5.3 约束

有效层数实现上取 `min` 即可；M1 **不改变** v2 总资金上限公式，仅约束加仓方向或层上限。

---

## 六、Walk-Forward 与 Optuna

- **时间轴、Holdout、WF 窗口** 与 v2 [03-optimization-summary.md](../v2.0/03-optimization-summary.md) **对齐**。
- **参数空间：** v2 已有维度 + 第四节 + 第五节离散维；`optimize_params.py` 中延续无效组合拒绝（如步长边界）。
- **工程：** 早停、剪枝、journal 断点续传对齐 v2；维度增加后可调整 `patience` 等，见 `02-plan.md`。
- **报告：** 除 Holdout、全长外，须含 **2021-01～2022-12**（或与代码锁定的一致熊市窗）专项表。

---

## 七、测试与文档

- **单元测试：** 继承 v2 测试结构；覆盖 `BEAR` + `ETF_DEFENSIVE` 下池仅为防御子集；`NO_NET_ADD` 下合成行情不增加名义仓位。
- **回归：** v3 独立路径，不得改写 v2 行为。
- **本目录文档：**  
  - `01-design.md` — 本文件  
  - `02-plan.md` — 实施计划（待编写）  
  - `03-optimization-summary.md` — 调参与回测总结（优化完成后编写）  

实施计划若以 superpowers 工作流书写，可复制摘要或链接至此目录，避免与 v1/v2 文档树分裂。

---

## 八、附录：可选后继项（非 v3.0 必达）

- `BEAR_UNIVERSE_MODE` 第三档与股票过滤联合 WF  
- JoinQuant `v3/strategy.py`  
- WF 目标函数显式加权全长最大回撤等  

---

## 规格自检

| 项 | 结果 |
|----|------|
| 占位 | 无未关闭 TBD |
| 一致性 | M2 仅换仓日；M1 每日网格；无衍生品 |
| 范围 | M1+M2 + WF；对冲排除 |
| 歧义 | NO_NET_ADD 前一日口径在实现计划中固定 |
