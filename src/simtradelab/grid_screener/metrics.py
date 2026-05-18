from __future__ import annotations

import numpy as np


def ols_log_close_trend(log_close: np.ndarray) -> tuple[float, float]:
    """OLS: log_close ~ a + b*t; return (t_stat on b, R^2). Homoskedastic SE."""
    y = np.asarray(log_close, dtype=float)
    n = y.size
    if n < 3:
        return float("nan"), float("nan")
    x = np.arange(n, dtype=float)
    X = np.column_stack((np.ones(n), x))
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    if dof <= 0:
        return float("nan"), float("nan")
    rss = float(np.sum(resid**2))
    s2 = rss / dof
    try:
        inv_xtx = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    var_beta = s2 * inv_xtx
    se_b = float(np.sqrt(var_beta[1, 1]))
    if se_b > 0 and np.isfinite(se_b):
        t_stat = float(beta[1] / se_b)
    elif abs(float(beta[1])) < 1e-15:
        t_stat = 0.0
    else:
        t_stat = float("nan")
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - rss / tss) if tss > 0 else 0.0
    return t_stat, r2


def variance_ratio_lm(log_close: np.ndarray, q: int = 2) -> float:
    """Lo–MacKinlay style VR(q)=Var(r^{(q)})/(q*Var(r^{(1)})) with overlap."""
    lc = np.asarray(log_close, dtype=float)
    if lc.size <= q + 1:
        return float("nan")
    r1 = np.diff(lc)
    rq = lc[q:] - lc[:-q]
    r1_aligned = r1[q - 1 :]
    n = min(r1_aligned.size, rq.size)
    r1_aligned = r1_aligned[:n]
    rq = rq[:n]
    v1 = float(np.var(r1_aligned, ddof=1))
    if v1 == 0.0:
        return float("nan")
    vq = float(np.var(rq, ddof=1))
    return vq / (q * v1)


def acf1_returns(r: np.ndarray) -> float:
    x = np.asarray(r, dtype=float)
    if x.size < 3:
        return float("nan")
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:] - x[1:].mean()
    denom = float(np.sqrt(np.dot(x0, x0) * np.dot(x1, x1)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x0, x1) / denom)


def realized_vol_ann(r: np.ndarray, trading_days: int = 252) -> float:
    x = np.asarray(r, dtype=float)
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1) * np.sqrt(float(trading_days)))


def mean_abs_overnight_gap(open_: np.ndarray, close: np.ndarray) -> float:
    o = np.asarray(open_, dtype=float)
    c = np.asarray(close, dtype=float)
    if o.size < 2 or c.size < 2:
        return float("nan")
    prev_c = c[:-1]
    cur_o = o[1:]
    mask = np.isfinite(prev_c) & np.isfinite(cur_o) & (prev_c > 0)
    if not np.any(mask):
        return float("nan")
    g = np.log(cur_o[mask] / prev_c[mask])
    return float(np.mean(np.abs(g)))


def gap_tail_ratio(open_: np.ndarray, close: np.ndarray, delta: float) -> float:
    o = np.asarray(open_, dtype=float)
    c = np.asarray(close, dtype=float)
    if o.size < 2:
        return float("nan")
    prev_c = c[:-1]
    cur_o = o[1:]
    mask = np.isfinite(prev_c) & np.isfinite(cur_o) & (prev_c > 0)
    if not np.any(mask):
        return float("nan")
    g = np.log(cur_o[mask] / prev_c[mask])
    return float(np.mean(np.abs(g) > float(delta)))


def intraday_extreme_ratio(open_: np.ndarray, high: np.ndarray, low: np.ndarray, delta: float) -> float:
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    if o.size < 1:
        return float("nan")
    rng = np.log(h / l)
    inside = np.isfinite(rng) & np.isfinite(o) & (l > 0)
    if not np.any(inside):
        return float("nan")
    amp = np.exp(rng[inside]) - 1.0
    return float(np.mean(amp > float(delta)))


def range_time_ratio(close: np.ndarray, ma_short: np.ndarray, ma_long: np.ndarray, b: float, b2: float) -> float:
    c = np.asarray(close, dtype=float)
    ms = np.asarray(ma_short, dtype=float)
    ml = np.asarray(ma_long, dtype=float)
    if c.size == 0:
        return float("nan")
    ok = np.isfinite(c) & np.isfinite(ms) & np.isfinite(ml) & (ml > 0)
    if not np.any(ok):
        return float("nan")
    cond = (np.abs(c - ml) / ml < b) & (np.abs(ms - ml) / ml < b2)
    return float(np.mean(cond[ok]))


def vol_comfort_score(rv_ann: float, sigma_low: float, sigma_high: float) -> float:
    if not np.isfinite(rv_ann):
        return float("nan")
    if rv_ann < sigma_low:
        return 0.4
    if rv_ann > sigma_high:
        return 0.2
    return 1.0
