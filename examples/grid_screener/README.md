# grid_screener 示例

## 用哪个配置？

只需 **`run_config.json`**：全市场 Parquet + 前复权 + 默认网格友好因子预设。

```bash
cd SimTradeLab
python -m simtradelab.grid_screener --config examples/grid_screener/run_config.json
```

复制该文件到你的工作目录后，改 `output_csv` 即可。若要换因子或排序，在同一 JSON 里覆盖 `factors` / `sort`（见 `my_docs/grid_friendly_screener/01-design.md` §4）。

内置预设定义在 `src/simtradelab/grid_screener/presets/grid_friendly_v1.json`，一般不必单独拷贝。
