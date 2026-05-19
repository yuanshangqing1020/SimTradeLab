from __future__ import annotations

from simtradelab.grid_screener.explain.grid_default import explain_grid_default
from simtradelab.grid_screener.explain.registry import get_explain

__all__ = ["explain_grid_default", "get_explain", "format_explanations_for_export", "explain_row"]


def format_explanations_for_export(lines: list[str]) -> str:
    s = " | ".join(lines)
    s = (
        s.replace("\uff1a", ": ")
        .replace("\uff0c", ", ")
        .replace("\u3001", ", ")
        .replace("\u3002", ". ")
        .replace("\uff08", "(")
        .replace("\uff09", ")")
        .replace("\uff1b", "; ")
    )
    return " ".join(s.split()).strip()


def explain_row(row: dict) -> list[str]:
    return explain_grid_default(row)
