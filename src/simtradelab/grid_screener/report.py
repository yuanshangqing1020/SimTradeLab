from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def rows_to_sorted_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "grid_friendly_score" in df.columns:
        df = df.sort_values(
            by=["grid_friendly_score", "range_time_ratio"],
            ascending=[False, False],
            na_position="last",
        )
    elif "range_time_ratio" in df.columns and "trend_t" in df.columns:
        df = df.sort_values(by=["range_time_ratio", "trend_t"], ascending=[False, True], na_position="last")
    return df.reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")


def write_csv_chunked(df: pd.DataFrame, path: str | Path, chunk_rows: int) -> list[str]:
    """行数 <= chunk_rows 时仍写入 path；否则写入 path 同级 ``{stem}_part0001.csv`` …"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(df)
    if n == 0:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return [str(path.resolve())]
    if n <= chunk_rows:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return [str(path.resolve())]
    stem, suffix = path.stem, path.suffix
    written: list[str] = []
    part = 1
    for start in range(0, n, chunk_rows):
        chunk = df.iloc[start : start + chunk_rows]
        out = path.parent / ("{0}_part{1:04d}{2}".format(stem, part, suffix))
        chunk.to_csv(out, index=False, encoding="utf-8-sig")
        written.append(str(out.resolve()))
        part += 1
    return written


def write_markdown(df: pd.DataFrame, path: str | Path, disclaimer_zh: str) -> None:
    lines = [disclaimer_zh, "", df.to_markdown(index=False)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
