from __future__ import annotations


def history_insufficient_flags(effective_days: int, window_w: int, n_min: int) -> tuple[bool, bool]:
    insufficient = effective_days < n_min
    history_short = (not insufficient) and (effective_days < window_w)
    return history_short, insufficient
