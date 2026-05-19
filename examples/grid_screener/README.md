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

内置预设定义在 `src/simtradelab/grid_screener/presets/`，一般不必单独拷贝。
