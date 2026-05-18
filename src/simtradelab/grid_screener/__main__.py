from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import pandas as pd

from simtradelab.grid_screener.config import (
    RunConfig,
    UniverseItem,
    load_etf_symbol_set,
    load_run_config,
)
from simtradelab.grid_screener.data_path import load_stock_name_map, resolve_stock_data_root
from simtradelab.grid_screener.explain import explain_row
from simtradelab.grid_screener.io_csv import read_ohlcv_csv
from simtradelab.grid_screener.io_parquet import ohlcv_from_stock_parquet_df
from simtradelab.grid_screener.pipeline import compute_screener_row
from simtradelab.grid_screener.report import (
    rows_to_sorted_frame,
    write_csv,
    write_csv_chunked,
    write_markdown,
    write_parquet,
)
from simtradelab.ptrade import storage

_DISCLAIMER = (
    "风险提示：分项仅描述历史统计特征，不构成收益承诺；股票与 ETF 同表并列时跨类绝对值比较需谨慎。"
    " trend_t 为经典 OLS t 统计量（非同方差稳健）。"
    " Parquet 模式与回测相同：data_path + market 解析后使用 storage.list_stocks / storage.load_stock。"
)


def _apply_as_of(df: pd.DataFrame, as_of: str | None) -> pd.DataFrame:
    if as_of is None:
        return df
    cutoff = pd.Timestamp(as_of)
    return df.loc[df.index <= cutoff]


def _finalize_row(row: dict) -> dict:
    expl = explain_row(row)
    out = dict(row)
    out["explanations"] = json.dumps(expl, ensure_ascii=False)
    return out


def _nan_row_missing_file(meta: UniverseItem) -> dict:
    nan = float("nan")
    return {
        "symbol": meta.symbol,
        "name": meta.name,
        "asset_type": meta.asset_type,
        "effective_days": 0,
        "history_short": False,
        "insufficient_data": True,
        "trend_t": nan,
        "trend_r2": nan,
        "variance_ratio": nan,
        "acf1_ret": nan,
        "rv_ann": nan,
        "vol_comfort_score": nan,
        "mean_abs_gap": nan,
        "gap_tail_ratio": nan,
        "intraday_extreme_ratio": nan,
        "range_time_ratio": nan,
        "vol_band": "unknown",
        "grid_friendly_score": nan,
        "explanations": json.dumps(["未找到匹配的行情文件。"], ensure_ascii=False),
    }


def _run_parquet_mode(cfg: RunConfig, etf_set: set[str], progress: bool) -> list[dict]:
    """全市场：与 `DataServer` / 回测相同的 Parquet 根目录与 storage API。"""
    root = resolve_stock_data_root(cfg.data_path, cfg.market)
    symbols = sorted(storage.list_stocks(root))
    name_map = load_stock_name_map(root)
    rows: list[dict] = []

    it = symbols
    if progress:
        try:
            from tqdm import tqdm

            it = tqdm(symbols, desc="grid_screener(parquet)", unit="sym")
        except Exception:
            pass

    for sym in it:
        atype = "etf" if sym in etf_set else cfg.default_asset_type
        meta = UniverseItem(symbol=sym, name=name_map.get(sym, ""), asset_type=atype)
        raw = storage.load_stock(root, sym)
        if raw is None or raw.empty:
            empty_ohlcv = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            row = compute_screener_row(empty_ohlcv, meta, cfg.params)
        else:
            raw = ohlcv_from_stock_parquet_df(raw)
            raw = _apply_as_of(raw, cfg.as_of)
            row = compute_screener_row(raw, meta, cfg.params)
        rows.append(_finalize_row(row))
    return rows


def _run_csv_mode(cfg: RunConfig, etf_set: set[str]) -> list[dict]:
    assert cfg.ohlcv_glob is not None
    paths = sorted(glob(cfg.ohlcv_glob))
    if not paths:
        raise SystemExit("no OHLCV files matched: {0}".format(cfg.ohlcv_glob))

    sym_to_path = {Path(p).stem: p for p in paths}

    if cfg.discover_glob:
        items = []
        for p in paths:
            stem = Path(p).stem
            at = "etf" if stem in etf_set else cfg.default_asset_type
            items.append(UniverseItem(symbol=stem, name="", asset_type=at))
    else:
        items = cfg.universe

    rows: list[dict] = []
    for item in items:
        pth = sym_to_path.get(item.symbol)
        if pth is None:
            rows.append(_nan_row_missing_file(item))
            continue
        df = read_ohlcv_csv(pth)
        df = _apply_as_of(df, cfg.as_of)
        row = compute_screener_row(df, item, cfg.params)
        rows.append(_finalize_row(row))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid-friendly daily screener")
    ap.add_argument("--config", required=True, help="Path to RunConfig JSON")
    ap.add_argument("-q", "--quiet", action="store_true", help="不写完成摘要到标准输出")
    ap.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条（Parquet 全市场模式）")
    args = ap.parse_args()
    cfg = load_run_config(args.config)
    etf_set = load_etf_symbol_set(cfg.etf_symbols_path)

    use_csv = cfg.ohlcv_glob is not None and str(cfg.ohlcv_glob).strip() != ""
    if use_csv:
        rows = _run_csv_mode(cfg, etf_set)
    else:
        rows = _run_parquet_mode(cfg, etf_set, progress=not args.quiet and not args.no_progress)

    out = rows_to_sorted_frame(rows)
    n_full = len(out)

    if cfg.full_parquet_path and str(cfg.full_parquet_path).strip() != "":
        write_parquet(out, cfg.full_parquet_path)

    view = out
    if cfg.csv_max_rows is not None:
        view = out.head(cfg.csv_max_rows)

    chunk_sz = cfg.csv_chunk_rows
    if chunk_sz is not None:
        csv_written = write_csv_chunked(view, cfg.output_csv, chunk_sz)
    else:
        write_csv(view, cfg.output_csv)
        csv_written = [str(Path(cfg.output_csv).resolve())]

    if cfg.output_md and str(cfg.output_md).strip() != "":
        md_cap = cfg.markdown_max_rows
        if md_cap is not None:
            md_df = view.head(md_cap)
            tail_note = ""
            if len(view) > md_cap:
                tail_note = "\n\n（Markdown 仅前 {n} 行，避免文件过大；完整请用 Parquet 或 CSV。）".format(n=md_cap)
        else:
            md_df = view
            tail_note = ""
        write_markdown(md_df, cfg.output_md, _DISCLAIMER + tail_note)

    if not args.quiet:
        root_info = ""
        if not use_csv:
            root_info = "，数据根目录={root!r}".format(root=resolve_stock_data_root(cfg.data_path, cfg.market))
        md_disp = cfg.output_md if (cfg.output_md and str(cfg.output_md).strip()) else "（未配置，已跳过）"
        pq_set = cfg.full_parquet_path and str(cfg.full_parquet_path).strip() != ""
        print(
            "grid_screener: 完成。模式={mode}，全量行数={nf}，CSV 视图行数={nv}{root}".format(
                mode="csv" if use_csv else "parquet(同回测)",
                nf=n_full,
                nv=len(view),
                root=root_info,
            )
        )
        if pq_set:
            print("  全量 Parquet: {0!r}".format(str(Path(cfg.full_parquet_path).resolve())))
        if len(csv_written) == 1:
            print("  CSV: {0!r}".format(csv_written[0]))
        else:
            print("  CSV 分片共 {0} 个，首文件: {1!r}".format(len(csv_written), csv_written[0]))
        if md_disp != "（未配置，已跳过）":
            print("  Markdown: {0!r}".format(str(Path(md_disp).resolve())))
        else:
            print("  Markdown: （未配置，已跳过）")


if __name__ == "__main__":
    main()
