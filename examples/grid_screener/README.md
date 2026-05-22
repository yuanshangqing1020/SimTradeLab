# grid_screener 示例

## 用哪个配置？

只需 **`run_config.json`**：全市场 Parquet + 前复权 + 默认网格友好因子预设。

```bash
cd SimTradeLab
python -m simtradelab.grid_screener --config examples/grid_screener/run_config.json
```

复制该文件到你的工作目录后，改 `output_csv` 即可。若要换因子或排序，在同一 JSON 里覆盖 `factors` / `sort`（见 `my_docs/grid_friendly_screener/01-design.md` §4）。

### 网格适宜度模型 (GSS)

另一套五维打分思路见 `my_docs/grid_friendly_screener/03-另一个思路.md`，已做成独立因子 + 预设：

```bash
python -m simtradelab.grid_screener --config examples/grid_screener/run_config_gss.json
```

预设 `grid_suitability_v1`：因子 `gss_volatility` … `gss_score`，按 `gss_score` 降序。

### 纯做T存钱罐网格回测 (Grid-T)

聚宽「收银台模式」思路见 `my_docs/grid_friendly_screener/04-另一个思路2/思路.md`：对每只股票/ETF 独立跑 3% 等额网格，统计窗口内落袋现金利润并排序：

```bash
python -m simtradelab.grid_screener --config examples/grid_screener/run_config_grid_t.json
```

预设 `grid_t_profit_v1`：因子 `grid_t_profit`，按 `grid_t_profit_yuan` 降序。使用 **未复权** 日线收盘价（`fq: none`，对齐聚宽 `use_real_price=True`），并自动按 `adj_a` 在除权日缩放网格基准价。参数与 jq.py 默认一致（底仓 10 万、步长 3%、每格 1 万）。

注意：全市场扫描为 **日线** 粒度，绝对利润通常低于聚宽 **分钟** 回测；结果更适合横向排序，不宜直接对比 jq 绝对金额。

内置预设定义在 `src/simtradelab/grid_screener/presets/`，一般不必单独拷贝。
