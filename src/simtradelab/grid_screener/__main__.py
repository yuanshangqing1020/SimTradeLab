from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

from simtradelab.grid_screener.config import RunConfig, UniverseItem, load_etf_symbol_set, load_run_config
from simtradelab.grid_screener.api_data import ScreenerDataAPI
from simtradelab.grid_screener.engine import compute_row, run_symbol_csv, run_symbol_parquet
from simtradelab.grid_screener.explain import format_explanations_for_export
from simtradelab.grid_screener.explain.registry import get_explain
from simtradelab.grid_screener.factors.registry import default_registry
from simtradelab.grid_screener.report import format_export_table, rows_to_sorted_frame, write_csv
_DISCLAIMER = (
    "风险提示：分项仅描述历史统计特征，不构成收益承诺；股票与 ETF 同表并列时跨类绝对值比较需谨慎。"
    " trend_t 为经典 OLS t 统计量（非同方差稳健）。"
)


def _attach_explanations(row: dict, cfg: RunConfig) -> dict:
    fn = get_explain(cfg.explain)
    out = dict(row)
    if fn is None:
        out["explanations"] = ""
    else:
        out["explanations"] = format_explanations_for_export(fn(row))
    return out


def _nan_row_missing_file(meta: UniverseItem, cfg: RunConfig) -> dict:
    import pandas as pd

    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    row = compute_row(empty, meta, cfg)
    row["explanations"] = format_explanations_for_export(["未找到匹配的行情文件。"])
    return row


def _run_parquet_mode(cfg: RunConfig, etf_set: set[str], progress: bool, *, quiet: bool) -> list[dict]:
    data_api = ScreenerDataAPI(cfg, quiet=quiet)
    symbols = data_api.list_symbols()
    registry = default_registry()
    rows: list[dict] = []

    it = symbols
    if progress:
        try:
            from tqdm import tqdm

            it = tqdm(symbols, desc="grid_screener", unit="sym")
        except Exception:
            pass

    for sym in it:
        atype = "etf" if sym in etf_set else cfg.default_asset_type
        meta = UniverseItem(symbol=sym, name=data_api.get_stock_name(sym), asset_type=atype)
        row = run_symbol_parquet(data_api, meta, cfg, registry)
        rows.append(_attach_explanations(row, cfg))
    return rows


def _run_csv_mode(cfg: RunConfig, etf_set: set[str]) -> list[dict]:
    assert cfg.ohlcv_glob is not None
    paths = sorted(glob(cfg.ohlcv_glob))
    if not paths:
        raise SystemExit("no OHLCV files matched: {0}".format(cfg.ohlcv_glob))

    sym_to_path = {Path(p).stem: p for p in paths}
    registry = default_registry()

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
            rows.append(_nan_row_missing_file(item, cfg))
            continue
        row = run_symbol_csv(pth, item, cfg, registry)
        rows.append(_attach_explanations(row, cfg))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily screener with pluggable factors and sort")
    ap.add_argument("--config", required=True, help="Path to RunConfig JSON")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    cfg = load_run_config(args.config)
    etf_set = load_etf_symbol_set(cfg.etf_symbols_path)

    use_csv = cfg.ohlcv_glob is not None and str(cfg.ohlcv_glob).strip() != ""
    if use_csv:
        rows = _run_csv_mode(cfg, etf_set)
    else:
        rows = _run_parquet_mode(
            cfg,
            etf_set,
            progress=not args.quiet and not args.no_progress,
            quiet=args.quiet,
        )

    out = rows_to_sorted_frame(rows, cfg.sort_spec())
    export_df = format_export_table(out)
    write_csv(export_df, cfg.output_csv)

    if not args.quiet:
        print(_DISCLAIMER)
        extra = ""
        if not use_csv:
            extra = " api=get_price fq={0!r}".format(cfg.resolved_fq())
        print(
            "grid_screener: rows={0} factors={1}{2}".format(
                len(out), cfg.resolved_factors(), extra
            )
        )
        print("  CSV: {0!r}".format(str(Path(cfg.output_csv).resolve())))


if __name__ == "__main__":
    main()
