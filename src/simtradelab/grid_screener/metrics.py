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


def _wilder_smooth(x: np.ndarray, period: int) -> np.ndarray:
    out = np.full(x.size, np.nan, dtype=float)
    if x.size < period:
        return out
    out[period - 1] = float(np.nanmean(x[:period]))
    alpha = 1.0 / float(period)
    for i in range(period, x.size):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def atr_ratio(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    if c.size < period + 1:
        return float("nan")
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = _wilder_smooth(tr, period)
    last_atr = atr[-1]
    last_c = c[-1]
    if not np.isfinite(last_atr) or not np.isfinite(last_c) or last_c <= 0:
        return float("nan")
    return float(last_atr / last_c)


def adx_last(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = c.size
    if n < period + 2:
        return float("nan")
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    prev_c = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
    atr = _wilder_smooth(tr, period)
    plus_di = 100.0 * _wilder_smooth(plus_dm, period) / atr
    minus_di = 100.0 * _wilder_smooth(minus_dm, period) / atr
    denom = plus_di + minus_di
    dx = np.where(np.isfinite(denom) & (denom > 0), 100.0 * np.abs(plus_di - minus_di) / denom, np.nan)
    adx = _wilder_smooth(dx, period)
    val = adx[-1]
    return float(val) if np.isfinite(val) else float("nan")


def hurst_exponent(close: np.ndarray, max_lag: int = 20) -> float:
    """R/S 估计赫斯特指数；H<0.5 偏均值回归。"""
    x = np.asarray(close, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < max_lag * 3:
        return float("nan")
    log_ret = np.diff(np.log(x))
    if log_ret.size < max_lag * 2:
        return float("nan")
    tau_list: list[float] = []
    rs_list: list[float] = []
    for lag in range(2, min(max_lag + 1, log_ret.size // 2)):
        n_chunks = log_ret.size // lag
        if n_chunks < 2:
            continue
        rs_vals: list[float] = []
        for i in range(n_chunks):
            seg = log_ret[i * lag : (i + 1) * lag]
            if seg.size < 2:
                continue
            z = seg - seg.mean()
            cum = np.cumsum(z)
            r = float(cum.max() - cum.min())
            s = float(seg.std(ddof=1))
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            tau_list.append(float(lag))
            rs_list.append(float(np.mean(rs_vals)))
    if len(tau_list) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(tau_list), np.log(rs_list), 1)
    return float(slope)


def price_percentile(close: np.ndarray) -> float:
    c = np.asarray(close, dtype=float)
    c = c[np.isfinite(c)]
    if c.size < 2:
        return float("nan")
    last = c[-1]
    return float(np.mean(c <= last))


def average_daily_turnover(close: np.ndarray, volume: np.ndarray, lookback: int) -> float:
    c = np.asarray(close, dtype=float)
    v = np.asarray(volume, dtype=float)
    n = min(lookback, c.size, v.size)
    if n < 1:
        return float("nan")
    tv = c[-n:] * v[-n:]
    tv = tv[np.isfinite(tv)]
    if tv.size == 0:
        return float("nan")
    return float(np.mean(tv))


def clip01(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(np.clip(x, 0.0, 1.0))
