# 日线标的筛选器 — 设计规格 v2.0

> **定位**：可插拔的 **日频统计筛选框架**。默认附带一套「网格友好度」参考因子与解释规则，但 **因子列表、排序、解释模板均可配置**，便于后续替换或扩展。  
> **与策略关系**：统计与 `grid_multi_asset` 等策略 **解耦**；v2 **不**输出机读 Universe 文件。

---

## 1. 目标与非目标

### 1.1 目标

- 对 Universe 内每个标的，在统一日频窗口上计算 **可配置因子集**（原始数值，可审计）。
- 行情经 SimTradeLab **与回测一致的数据根 + 复权管线** 取得（默认前复权 `fq=pre`）。
- 导出 **单张 CSV**（列 = 元数据 + 各因子列 + 可选规则解释列）。
- **排序完全由配置指定**（多键、升/降序），不硬编码「网格友好」排序逻辑。
- 新增因子：实现 `Factor` 协议并注册（或放入 `factors/custom/` 后在配置中引用）。

### 1.2 非目标（v2）

- 分钟/tick、逐标的完整网格回测、ML 黑箱主排序。
- 自动生成策略 Universe JSON。
- 运行时从网络拉行情（仅用本地 `data/` Parquet，与回测相同）。

---

## 2. 架构

```
RunConfig (JSON)
    │
    ├─► MarketDataSession ──► storage.load_stock + adj 缓存 (fq)
    │
    ├─► FactorRegistry ──► [factor₁, factor₂, …] 按序 compute
    │         │
    │         └─► 合并为 row dict
    │
    ├─► ExplainRegistry (可选) ──► explanations 列
    │
    └─► sort_spec ──► report.write_csv
```

### 2.1 核心抽象

| 组件 | 职责 |
|------|------|
| `MarketDataSession` | 解析 `data_path`/`market`，加载 OHLCV，按 `fq` 复权 |
| `FactorContext` | 单标的：窗口化 `DataFrame`、numpy 序列、`params`、`outputs` 累积 |
| `Factor` | `name: str` + `compute(ctx) -> dict[str, object]` |
| `FactorRegistry` | 内置名 → 实例；`resolve(names) -> list[Factor]` |
| `SortSpec` | `[{field, ascending}, …]`，传给 `pandas.sort_values` |
| `ExplainRuleSet` | 命名规则集（如 `grid_default`），读 `row` 生成中文短句 |

### 2.2 内置因子（preset `grid_friendly_v1`）

| 因子名 | 输出列（节选） | 说明 |
|--------|----------------|------|
| `meta` | symbol, name, asset_type | 元数据 |
| `sample_quality` | effective_days, history_short, insufficient_data | 样本门槛 |
| `trend` | trend_t, trend_r2 | 对数价 OLS 趋势 |
| `variance_ratio` | variance_ratio | 方差比率 VR(2) |
| `acf1` | acf1_ret | 收益 lag-1 自相关 |
| `volatility` | rv_ann, vol_comfort_score, vol_band | 实现波动与舒适区 |
| `gap` | mean_abs_gap, gap_tail_ratio, intraday_extreme_ratio | 跳空与极端日 |
| `range_regime` | range_time_ratio | 区间震荡占比 |
| `grid_score` | grid_friendly_score | **可选** 固定权重综合分 |

用户可在配置中 **删掉 `grid_score`**、**调换因子顺序**、或 **只保留子集**。

### 2.3 扩展新因子（开发者）

1. 在 `simtradelab/grid_screener/factors/` 下新增模块，实现：

```python
class MyFactor:
    name = "my_factor"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        if ctx.insufficient:
            return {"my_metric": float("nan")}
        return {"my_metric": ...}
```

2. 在 `factors/registry.py` 的 `register_builtin_factors()` 中注册，或调用 `FactorRegistry.register(MyFactor())`。
3. 在 JSON 配置的 `factors` 数组中加入 `"my_factor"`。

**约定**：若 `sample_quality` 判定 `insufficient_data`，后续因子应返回 NaN 或跳过（引擎在 `insufficient` 时仍调用各因子，由因子自行处理）。

### 2.4 排序

配置示例：

```json
"sort": [
  {"field": "range_time_ratio", "ascending": false},
  {"field": "trend_t", "ascending": true}
]
```

- 未配置 `sort` 时：仅按 `symbol` 字母序，保证输出稳定可 diff。
- 字段不存在时：`sort_values` 行为与 Pandas 一致（列缺失则报错，配置需自检）。

### 2.5 行情与复权

- **入口**：`DataServer` + `PtradeAPI`（与 `strategies/grid_mining/miner.py` 中 `init_api` 相同写法，在 `grid_screener/api_data.py` 内初始化）。
- **取数**：`PtradeAPI.get_price(..., fq=...)`，**不**在 screener 内重复实现 `storage.load_stock` / 复权公式。
- **Universe / 名称**：`get_Ashares`、`get_stock_name`。
- **窗口**：`count=window_trading_days` + `end_date=as_of`（`context.current_dt` 同步为 `as_of`）。

---

## 3. 数据与窗口（与 v1 统计口径一致）

- **字段**：OHLCV；收益用对数收益 \(\ln(C_t/C_{t-1})\)。
- **默认窗口**：W = 1250 交易日；`n_min_valid` 默认 500。
- **短历史**：保留行，打 `history_short` / `insufficient_data`。
- **可调参数**：集中在 `params`（`ScreenerParams`），各因子只读 `ctx.params`。

---

## 4. 配置

### 4.1 方式一：显式因子列表

```json
{
  "market": "CN",
  "fq": "pre",
  "factors": ["meta", "sample_quality", "trend", "variance_ratio", "acf1", "volatility", "gap", "range_regime"],
  "sort": [{"field": "range_time_ratio", "ascending": false}],
  "explain": "grid_default",
  "params": { "window_trading_days": 1250 },
  "output_csv": "report.csv"
}
```

### 4.2 方式二：预设 + 覆盖

```json
{
  "preset": "grid_friendly_v1",
  "sort": [{"field": "grid_friendly_score", "ascending": false}],
  "fq": "pre"
}
```

`preset` 展开为默认 `factors` / `explain`；顶层字段覆盖预设。

### 4.3 CSV 演示模式

设置 `ohlcv_glob` 时从外部 CSV 读行情（**不复权**，用于单测/演示）；生产全市场扫描 **不设置** `ohlcv_glob`。

---

## 5. 产物

- **主表 CSV**：UTF-8 BOM；浮点默认 4 位小数；`explanations` 为规则拼接（可关闭 `explain: null`）。
- **固定风险提示**（CLI  stdout）：跨 asset_type 比较需谨慎；分项非收益承诺。

---

## 6. CLI

```bash
python -m simtradelab.grid_screener --config examples/grid_screener/run_config.json
```

---

## 7. 测试

- 指标函数：合成序列对账（保留 `test_grid_screener_metrics.py`）。
- 引擎：合成 OHLCV + 因子子集键存在。
- 排序：`SortSpec` 多键顺序。
- 复权：mock adj 因子，验证除权日前后 close 连续（可选，有 fixture 时）。

---

## 8. 版本

| 项 | 内容 |
|----|------|
| 规格版本 | v2.0 |
| 路径 | `my_docs/grid_friendly_screener/01-design.md` |
| 实现包 | `src/simtradelab/grid_screener/` |

**v1 → v2 变更**：可插拔因子/排序/解释；行情走复权管线；删除硬编码 `rows_to_sorted_frame` 默认排序。
