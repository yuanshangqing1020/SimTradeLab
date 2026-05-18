# Grid Friendly Screener Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a daily OHLCV-based, explainable “grid friendliness” screener that outputs one merged human-readable table (stocks + ETFs), matching `01-design.md`.

**Architecture:** Add a small first-party package `simtradelab.grid_screener` with pure NumPy metric functions (unit-tested), a thin Pandas preprocessing/pipeline layer, rule-based Chinese explanations, and a JSON-configured CLI. v1 includes an **optional CSV OHLCV loader** for batch runs (one file per symbol or long tables); integration with live QMT/PTrade data can wrap the same `DataFrame` contract later without changing metrics.

**Tech Stack:** Python 3.9+ (per `pyproject.toml`), NumPy, Pandas, Pydantic v2, pytest. No new runtime dependencies (no statsmodels; OLS uses `numpy.linalg.lstsq`). Optional `composite` uses fixed weights from config only.

**Spec:** `SimTradeLab/my_docs/grid_friendly_screener/01-design.md`

---

## File map (v1)

| File | Responsibility |
|------|------------------|
| `src/simtradelab/grid_screener/__init__.py` | Public exports (optional thin) |
| `src/simtradelab/grid_screener/config.py` | `ScreenerParams` / `RunConfig` (Pydantic), JSON load |
| `src/simtradelab/grid_screener/preprocess.py` | Drop invalid rows, sort index, slice last `W` valid rows, build `log_close`, `r1` |
| `src/simtradelab/grid_screener/metrics.py` | Pure metric functions (NumPy arrays) |
| `src/simtradelab/grid_screener/labels.py` | `history_short`, `insufficient_data`, optional `vol_band` strings |
| `src/simtradelab/grid_screener/explain.py` | Rule-based explanation lines (zh) |
| `src/simtradelab/grid_screener/pipeline.py` | `compute_row(symbol, df, meta, params) -> dict` |
| `src/simtradelab/grid_screener/report.py` | Assemble `DataFrame`, sort, `to_csv`, simple Markdown |
| `src/simtradelab/grid_screener/io_csv.py` | Load OHLCV CSV → normalized `DataFrame` |
| `src/simtradelab/grid_screener/__main__.py` | CLI entry (`python -m simtradelab.grid_screener`) |
| `tests/unit/test_grid_screener_metrics.py` | Metric + preprocess tests |
| `tests/unit/test_grid_screener_pipeline.py` | End-to-end row on synthetic data |

---

### Task 1: Package skeleton + Pydantic config

**Files:**
- Create: `src/simtradelab/grid_screener/__init__.py`
- Create: `src/simtradelab/grid_screener/config.py`
- Test: (config exercised in Task 7; optional instant test in REPL)

- [ ] **Step 1: Create package init**

Create `src/simtradelab/grid_screener/__init__.py`:

```python
"""Grid-friendly daily screener (statistics-only, strategy-agnostic)."""

__all__: list[str] = []
```

- [ ] **Step 2: Add Pydantic models**

Create `src/simtradelab/grid_screener/config.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ScreenerParams(BaseModel):
    """Defaults aligned with `01-design.md` §3–§4."""

    window_trading_days: int = Field(default=1250, ge=50)
    n_min_valid: int = Field(default=500, ge=10)
    sigma_low: float = Field(default=0.10, gt=0)
    sigma_high: float = Field(default=0.40, gt=0)
    gap_tail_delta: float = Field(default=0.01, gt=0)
    range_ma_long: int = Field(default=60, ge=5)
    range_ma_short: int = Field(default=20, ge=2)
    range_band_price_vs_long: float = Field(default=0.05, gt=0)
    range_band_spread_vs_long: float = Field(default=0.03, gt=0)
    intraday_extreme_delta: float = Field(default=0.02, gt=0)
    enable_composite: bool = False


class UniverseItem(BaseModel):
    symbol: str
    name: str = ""
    asset_type: Literal["stock", "etf"]


class RunConfig(BaseModel):
    as_of: str | None = None  # ISO date; None = use last row per file
    params: ScreenerParams = Field(default_factory=ScreenerParams)
    universe: list[UniverseItem]
    ohlcv_glob: str | None = None  # e.g. data/daily/{symbol}.csv
    output_csv: str = "grid_screener_report.csv"
    output_md: str | None = "grid_screener_report.md"
    composite_weights: dict[str, float] = Field(default_factory=dict)


def load_run_config(path: str | Path) -> RunConfig:
    p = Path(path)
    return RunConfig.model_validate_json(p.read_text(encoding="utf-8"))
```

- [ ] **Step 3: Commit** (skip if your policy is “no commits unless asked”)

```bash
cd SimTradeLab && git add src/simtradelab/grid_screener/__init__.py src/simtradelab/grid_screener/config.py && git commit -m "feat(grid_screener): add package skeleton and RunConfig"
```

---

### Task 2: Preprocess — clean OHLCV and window slice

**Files:**
- Create: `src/simtradelab/grid_screener/preprocess.py`
- Create: `tests/unit/test_grid_screener_metrics.py` (start file; preprocess tests first)
- Modify: `tests/unit/test_grid_screener_metrics.py`

- [ ] **Step 1: Write failing tests for preprocess**

Append to `tests/unit/test_grid_screener_metrics.py`:

```python
# tests/unit/test_grid_screener_metrics.py
import numpy as np
import pandas as pd

from simtradelab.grid_screener.preprocess import normalize_ohlcv, slice_window


def test_normalize_ohlcv_drops_nan_close_and_sorts():
    df = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [np.nan, 10.0, 11.0],
            "volume": [100, 100, 100],
        },
        index=pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
    )
    got = normalize_ohlcv(df)
    assert list(got.index.date) == [pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-01-02").date()]
    assert got["close"].tolist() == [10.0, 11.0]


def test_slice_window_truncates_last_w_rows():
    df = pd.DataFrame(
        {"close": np.arange(10.0, 20.0)},
        index=pd.date_range("2020-01-01", periods=10, freq="B"),
    )
    got = slice_window(df, 5)
    assert len(got) == 5
    assert got["close"].iloc[-1] == 19.0
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd SimTradeLab && poetry run pytest tests/unit/test_grid_screener_metrics.py::test_normalize_ohlcv_drops_nan_close_and_sorts -v
```

Expected: `ModuleNotFoundError` or import error for `preprocess`.

- [ ] **Step 3: Implement preprocess**

Create `src/simtradelab/grid_screener/preprocess.py`:

```python
from __future__ import annotations

import pandas as pd


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without close; sort ascending by index."""
    need = ["open", "high", "low", "close", "volume"]
    for c in need:
        if c not in df.columns:
            raise KeyError("missing column: {0}".format(c))
    out = df.loc[df["close"].notna(), need].copy()
    out.sort_index(inplace=True)
    return out


def slice_window(df: pd.DataFrame, w: int) -> pd.DataFrame:
    if w < 1:
        raise ValueError("w must be >= 1")
    return df.iloc[-w:].copy()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd SimTradeLab && poetry run pytest tests/unit/test_grid_screener_metrics.py::test_normalize_ohlcv_drops_nan_close_and_sorts tests/unit/test_grid_screener_metrics.py::test_slice_window_truncates_last_w_rows -v
```

- [ ] **Step 5: Commit**

```bash
cd SimTradeLab && git add src/simtradelab/grid_screener/preprocess.py tests/unit/test_grid_screener_metrics.py && git commit -m "feat(grid_screener): OHLCV normalize and window slice"
```

---

### Task 3: Core metrics — trend, variance ratio, ACF1, realized vol

**Files:**
- Create: `src/simtradelab/grid_screener/metrics.py`
- Modify: `tests/unit/test_grid_screener_metrics.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_grid_screener_metrics.py`:

```python
import math

import numpy as np

from simtradelab.grid_screener.metrics import acf1_returns, ols_log_close_trend, realized_vol_ann, variance_ratio_lm


def test_ols_flat_series_near_zero_t():
    lc = np.log(np.ones(30))
    t_stat, r2 = ols_log_close_trend(lc)
    assert abs(t_stat) < 0.5
    assert r2 < 1e-6


def test_variance_ratio_random_walk_near_one():
    rng = np.random.default_rng(0)
    steps = rng.normal(0.0, 0.01, size=500)
    lc = np.cumsum(steps)
    vr = variance_ratio_lm(lc, q=2)
    assert 0.85 <= vr <= 1.15


def test_acf1_alternating_negative():
    r = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01], dtype=float)
    rho = acf1_returns(r)
    assert rho < -0.3


def test_realized_vol_positive():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.015, size=200)
    rv = realized_vol_ann(r)
    assert rv > 0.1
    assert math.isfinite(rv)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd SimTradeLab && poetry run pytest tests/unit/test_grid_screener_metrics.py -k "ols_flat or variance_ratio or acf1_altern or realized_vol" -v
```

- [ ] **Step 3: Implement metrics (NumPy-only)**

Create `src/simtradelab/grid_screener/metrics.py`:

```python
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
    t_stat = float(beta[1] / se_b) if se_b > 0 else float("nan")
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd SimTradeLab && poetry run pytest tests/unit/test_grid_screener_metrics.py -k "ols_flat or variance_ratio or acf1_altern or realized_vol" -v
```

- [ ] **Step 5: Commit**

```bash
cd SimTradeLab && git add src/simtradelab/grid_screener/metrics.py tests/unit/test_grid_screener_metrics.py && git commit -m "feat(grid_screener): trend t-stat, variance ratio, acf1, realized vol"
```

---

### Task 4: Metrics — gaps, range time, vol comfort

**Files:**
- Modify: `src/simtradelab/grid_screener/metrics.py`
- Modify: `tests/unit/test_grid_screener_metrics.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
from simtradelab.grid_screener.metrics import (
    gap_tail_ratio,
    intraday_extreme_ratio,
    mean_abs_overnight_gap,
    range_time_ratio,
    vol_comfort_score,
)


def test_mean_abs_gap_positive():
    o = np.array([10.0, 10.5, 10.2], dtype=float)
    c = np.array([10.0, 10.0, 10.4], dtype=float)
    g = mean_abs_overnight_gap(o, c)
    assert g > 0


def test_vol_comfort_mid_band_one():
    assert vol_comfort_score(0.20, sigma_low=0.10, sigma_high=0.40) == 1.0


def test_range_time_ratio_normalizes():
    close = np.linspace(100.0, 100.01, 80)
    ma_long = pd.Series(close).rolling(60, min_periods=60).mean().to_numpy()
    ma_short = pd.Series(close).rolling(20, min_periods=20).mean().to_numpy()
    r = range_time_ratio(close, ma_short, ma_long, b=0.10, b2=0.10)
    assert 0.0 <= r <= 1.0
```

Note: import `pandas as pd` already at top if not — add if needed.

- [ ] **Step 2: Run tests — expect FAIL** (functions missing)

```bash
cd SimTradeLab && poetry run pytest tests/unit/test_grid_screener_metrics.py -k "mean_abs_gap or vol_comfort or range_time" -v
```

- [ ] **Step 3: Implement functions**

Append to `src/simtradelab/grid_screener/metrics.py`:

```python
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd SimTradeLab && poetry run pytest tests/unit/test_grid_screener_metrics.py -k "mean_abs_gap or vol_comfort or range_time" -v
```

- [ ] **Step 5: Commit**

```bash
cd SimTradeLab && git add src/simtradelab/grid_screener/metrics.py tests/unit/test_grid_screener_metrics.py && git commit -m "feat(grid_screener): gap, range-time, vol comfort metrics"
```

---

### Task 5: Labels + single-symbol pipeline

**Files:**
- Create: `src/simtradelab/grid_screener/labels.py`
- Create: `src/simtradelab/grid_screener/pipeline.py`
- Create: `tests/unit/test_grid_screener_pipeline.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/unit/test_grid_screener_pipeline.py`:

```python
import numpy as np
import pandas as pd

from simtradelab.grid_screener.config import ScreenerParams, UniverseItem
from simtradelab.grid_screener.pipeline import compute_screener_row


def _synth(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    vol = np.full(n, 1e6)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_compute_screener_row_has_expected_keys():
    df = _synth(600)
    meta = UniverseItem(symbol="000001.SZ", name="Ping An Bank", asset_type="stock")
    params = ScreenerParams(window_trading_days=500, n_min_valid=200)
    row = compute_screener_row(df, meta, params)
    for k in (
        "symbol",
        "name",
        "asset_type",
        "effective_days",
        "history_short",
        "insufficient_data",
        "trend_t",
        "trend_r2",
        "variance_ratio",
        "acf1_ret",
        "rv_ann",
        "vol_comfort_score",
        "mean_abs_gap",
        "gap_tail_ratio",
        "intraday_extreme_ratio",
        "range_time_ratio",
    ):
        assert k in row
    assert row["insufficient_data"] in (True, False)
    assert row["effective_days"] <= 500
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
cd SimTradeLab && poetry run pytest tests/unit/test_grid_screener_pipeline.py::test_compute_screener_row_has_expected_keys -v
```

- [ ] **Step 3: Implement labels + pipeline**

Create `src/simtradelab/grid_screener/labels.py`:

```python
from __future__ import annotations


def history_insufficient_flags(effective_days: int, window_w: int, n_min: int) -> tuple[bool, bool]:
    insufficient = effective_days < n_min
    history_short = (not insufficient) and (effective_days < window_w)
    return history_short, insufficient
```

Create `src/simtradelab/grid_screener/pipeline.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from simtradelab.grid_screener.config import ScreenerParams, UniverseItem
from simtradelab.grid_screener.labels import history_insufficient_flags
from simtradelab.grid_screener.metrics import (
    acf1_returns,
    gap_tail_ratio,
    intraday_extreme_ratio,
    mean_abs_overnight_gap,
    ols_log_close_trend,
    range_time_ratio,
    realized_vol_ann,
    variance_ratio_lm,
    vol_comfort_score,
)
from simtradelab.grid_screener.preprocess import normalize_ohlcv, slice_window


def compute_screener_row(raw: pd.DataFrame, meta: UniverseItem, params: ScreenerParams) -> dict[str, object]:
    df0 = normalize_ohlcv(raw)
    if df0.empty:
        return _empty_row(meta, params, effective_days=0)

    df = slice_window(df0, min(params.window_trading_days, len(df0)))
    c = df["close"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    log_c = np.log(c)
    r1 = np.diff(log_c)

    eff = int(c.size)
    hist_short, insuff = history_insufficient_flags(eff, params.window_trading_days, params.n_min_valid)
    if insuff:
        base = _empty_row(meta, params, effective_days=eff)
        base.update({"history_short": False, "insufficient_data": True})
        return base

    trend_t, trend_r2 = ols_log_close_trend(log_c)
    vr = variance_ratio_lm(log_c, q=2)
    rho1 = acf1_returns(r1)
    rv = realized_vol_ann(r1)
    vcomf = vol_comfort_score(rv, params.sigma_low, params.sigma_high)

    mag = mean_abs_overnight_gap(o, c)
    gtr = gap_tail_ratio(o, c, params.gap_tail_delta)

    ms = pd.Series(c).rolling(params.range_ma_short, min_periods=params.range_ma_short).mean().to_numpy()
    ml = pd.Series(c).rolling(params.range_ma_long, min_periods=params.range_ma_long).mean().to_numpy()
    rtr = range_time_ratio(c, ms, ml, params.range_band_price_vs_long, params.range_band_spread_vs_long)
    ier = intraday_extreme_ratio(o, h, l, params.intraday_extreme_delta)

    row: dict[str, object] = {
        "symbol": meta.symbol,
        "name": meta.name,
        "asset_type": meta.asset_type,
        "effective_days": eff,
        "history_short": hist_short,
        "insufficient_data": False,
        "trend_t": trend_t,
        "trend_r2": trend_r2,
        "variance_ratio": vr,
        "acf1_ret": rho1,
        "rv_ann": rv,
        "vol_comfort_score": vcomf,
        "mean_abs_gap": mag,
        "gap_tail_ratio": gtr,
        "intraday_extreme_ratio": ier,
        "range_time_ratio": rtr,
    }
    _attach_vol_band(row, rv, params)
    return row


def _vol_band(rv: float, params: ScreenerParams) -> str:
    if not np.isfinite(rv):
        return "unknown"
    if rv < params.sigma_low:
        return "vol_low"
    if rv > params.sigma_high:
        return "vol_high"
    return "vol_mid"


def _attach_vol_band(row: dict[str, object], rv: float, params: ScreenerParams) -> None:
    row["vol_band"] = _vol_band(rv, params)


def _empty_row(meta: UniverseItem, params: ScreenerParams, effective_days: int) -> dict[str, object]:
    nan = float("nan")
    return {
        "symbol": meta.symbol,
        "name": meta.name,
        "asset_type": meta.asset_type,
        "effective_days": effective_days,
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
    }
```

- [ ] **Step 4: Run test — expect PASS**

```bash
cd SimTradeLab && poetry run pytest tests/unit/test_grid_screener_pipeline.py::test_compute_screener_row_has_expected_keys -v
```

- [ ] **Step 5: Commit**

```bash
cd SimTradeLab && git add src/simtradelab/grid_screener/labels.py src/simtradelab/grid_screener/pipeline.py tests/unit/test_grid_screener_pipeline.py && git commit -m "feat(grid_screener): labels and single-symbol pipeline"
```

---

### Task 6: Explanations + report assembly + CSV IO

**Files:**
- Create: `src/simtradelab/grid_screener/explain.py`
- Create: `src/simtradelab/grid_screener/report.py`
- Create: `src/simtradelab/grid_screener/io_csv.py`
- Modify: `tests/unit/test_grid_screener_pipeline.py`

- [ ] **Step 1: Write tests for explain**

Append to `tests/unit/test_grid_screener_pipeline.py`:

```python
from simtradelab.grid_screener.explain import explain_row
from simtradelab.grid_screener.report import rows_to_sorted_frame


def test_explain_emits_zh_strings():
    row = {
        "trend_t": 5.0,
        "variance_ratio": 1.3,
        "acf1_ret": 0.2,
        "vol_band": "vol_low",
        "gap_tail_ratio": 0.4,
        "range_time_ratio": 0.8,
        "insufficient_data": False,
    }
    lines = explain_row(row)
    assert isinstance(lines, list) and len(lines) >= 1
    assert all(isinstance(s, str) for s in lines)


def test_rows_to_sorted_frame_sorts():
    rows = [
        {"symbol": "A", "range_time_ratio": 0.1, "trend_t": 0.5},
        {"symbol": "B", "range_time_ratio": 0.9, "trend_t": 0.1},
    ]
    df = rows_to_sorted_frame(rows)
    assert df.iloc[0]["symbol"] == "B"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cd SimTradeLab && poetry run pytest tests/unit/test_grid_screener_pipeline.py -k "explain_emits or rows_to_sorted" -v
```

- [ ] **Step 3: Implement explain, report, io_csv**

Create `src/simtradelab/grid_screener/explain.py`:

```python
from __future__ import annotations

from typing import Any


def explain_row(row: dict[str, Any]) -> list[str]:
    if row.get("insufficient_data"):
        return ["有效交易日不足：分项未计算或不可靠。"]
    out: list[str] = []
    tt = float(row.get("trend_t") or 0.0)
    if abs(tt) > 2.5:
        out.append("趋势检验较强：窗口内对数价格可能存在显著漂移，网格假设偏弱。")

    vr = float(row.get("variance_ratio") or 1.0)
    rho = float(row.get("acf1_ret") or 0.0)
    if vr > 1.05 and rho > 0.05:
        out.append("收益持续性偏高：方差比率与一阶自相关同向不利震荡网格。")

    band = str(row.get("vol_band") or "")
    if band == "vol_low":
        out.append("实现波动偏低：理论格子空间可能不足。")
    elif band == "vol_high":
        out.append("实现波动偏高：执行与假突破风险上升。")

    gtr = float(row.get("gap_tail_ratio") or 0.0)
    if gtr > 0.1:
        out.append("大跳空占比偏高：隔夜跳开可能放大滑点与挂单风险。")

    rtr = float(row.get("range_time_ratio") or 0.0)
    if rtr > 0.55 and abs(tt) < 2.0:
        out.append("区间震荡时间占比较高：与网格友好方向更一致（仍需结合趋势项）。")

    if not out:
        out.append("未触发强提示：请结合分项与基本面/流动性人工复核。")
    return out
```

Create `src/simtradelab/grid_screener/report.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def rows_to_sorted_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "range_time_ratio" in df.columns and "trend_t" in df.columns:
        df = df.sort_values(by=["range_time_ratio", "trend_t"], ascending=[False, True], na_position="last")
    return df.reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_markdown(df: pd.DataFrame, path: str | Path, disclaimer_zh: str) -> None:
    lines = [disclaimer_zh, "", df.to_markdown(index=False)]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
```

Create `src/simtradelab/grid_screener/io_csv.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_ohlcv_csv(path: str | Path, date_col: str = "date") -> pd.DataFrame:
    """Expect columns: date, open, high, low, close, volume (case-insensitive ok)."""
    p = Path(path)
    df = pd.read_csv(p)
    lower = {c.lower(): c for c in df.columns}
    for need in ("open", "high", "low", "close", "volume"):
        if need not in lower:
            raise KeyError("missing column: {0}".format(need))
    dc = lower.get(date_col) or lower.get("datetime") or lower.get("time")
    if dc is None:
        raise KeyError("need date column: date/datetime/time")
    rename = {lower["open"]: "open", lower["high"]: "high", lower["low"]: "low", lower["close"]: "close", lower["volume"]: "volume"}
    out = df.rename(columns=rename).copy()
    out.index = pd.to_datetime(df[dc])
    return out[["open", "high", "low", "close", "volume"]]
```

- [ ] **Step 4: Run tests — expect PASS**

Note: `to_markdown` requires `tabulate` for pandas; if missing, either `poetry add --group dev tabulate` or change `write_markdown` to skip / simple format. Prefer adding **`tabulate`** as dev dependency to satisfy `to_markdown`.

If test does not call `write_markdown`, pipeline tests pass; add dev dep when implementing CLI.

```bash
cd SimTradeLab && poetry add --group dev tabulate
cd SimTradeLab && poetry run pytest tests/unit/test_grid_screener_pipeline.py -k "explain_emits or rows_to_sorted" -v
```

- [ ] **Step 5: Commit**

```bash
cd SimTradeLab && git add pyproject.toml poetry.lock src/simtradelab/grid_screener/explain.py src/simtradelab/grid_screener/report.py src/simtradelab/grid_screener/io_csv.py tests/unit/test_grid_screener_pipeline.py && git commit -m "feat(grid_screener): explanations, report export, CSV IO"
```

---

### Task 7: CLI `__main__.py` + example JSON

**Files:**
- Create: `src/simtradelab/grid_screener/__main__.py`
- Create: `SimTradeLab/examples/grid_screener/sample_run_config.json` (or `my_docs/grid_friendly_screener/examples/sample_run_config.json` — prefer `examples/` under repo root)
- Modify: `tests/unit/test_grid_screener_pipeline.py` (optional CLI smoke via subprocess — can skip to keep plan short)

- [ ] **Step 1: Add example config**

Create `examples/grid_screener/sample_run_config.json`:

```json
{
  "params": {
    "window_trading_days": 1250,
    "n_min_valid": 500
  },
  "universe": [
    {"symbol": "DEMO001", "name": "Demo Stock", "asset_type": "stock"}
  ],
  "ohlcv_glob": "examples/grid_screener/data/DEMO001.csv",
  "output_csv": "examples/grid_screener/out/report_demo.csv",
  "output_md": "examples/grid_screener/out/report_demo.md"
}
```

生成 `examples/grid_screener/data/DEMO001.csv`（≥520 行以满足 `n_min_valid=500` 下的窗口截取后仍有效）。可在 Task 7 实施时运行一次性脚本（不要依赖网络），例如：

```bash
cd SimTradeLab && poetry run python -c "
import numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(0)
n = 600
idx = pd.date_range('2018-01-01', periods=n, freq='B')
close = 100 * np.cumprod(1.0 + rng.normal(0, 0.012, n))
open_ = np.concatenate([[close[0]], close[:-1]])
high = np.maximum(open_, close) * 1.002
low = np.minimum(open_, close) * 0.998
vol = np.full(n, 1_000_000)
Path('examples/grid_screener/data').mkdir(parents=True, exist_ok=True)
pd.DataFrame({'date': idx, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': vol}).to_csv('examples/grid_screener/data/DEMO001.csv', index=False)
"
```

- [ ] **Step 2: Implement `__main__.py`**

Create `src/simtradelab/grid_screener/__main__.py`:

```python
from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import pandas as pd

from simtradelab.grid_screener.config import RunConfig, load_run_config
from simtradelab.grid_screener.explain import explain_row
from simtradelab.grid_screener.io_csv import read_ohlcv_csv
from simtradelab.grid_screener.pipeline import compute_screener_row
from simtradelab.grid_screener.report import rows_to_sorted_frame, write_csv, write_markdown

_DISCLAIMER = (
    "风险提示：分项仅描述历史统计特征，不构成收益承诺；股票与 ETF 同表并列时跨类绝对值比较需谨慎。"
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid-friendly daily screener")
    ap.add_argument("--config", required=True, help="Path to RunConfig JSON")
    args = ap.parse_args()
    cfg = load_run_config(args.config)
    if cfg.ohlcv_glob is None:
        raise SystemExit("ohlcv_glob is required for v1 CLI")

    rows: list[dict] = []
    paths = sorted(glob(cfg.ohlcv_glob))
    if not paths:
        raise SystemExit("no OHLCV files matched: {0}".format(cfg.ohlcv_glob))

    sym_to_path = {Path(p).stem: p for p in paths}

    for item in cfg.universe:
        pth = sym_to_path.get(item.symbol)
        if pth is None:
            nan = float("nan")
            rows.append(
                {
                    "symbol": item.symbol,
                    "name": item.name,
                    "asset_type": item.asset_type,
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
                    "explanations": json.dumps(["未找到匹配的行情文件。"], ensure_ascii=False),
                }
            )
            continue
        df = read_ohlcv_csv(pth)
        row = compute_screener_row(df, item, cfg.params)
        expl = explain_row(row)
        row["explanations"] = json.dumps(expl, ensure_ascii=False)
        rows.append(row)

    out = rows_to_sorted_frame(rows)
    write_csv(out, cfg.output_csv)
    if cfg.output_md:
        write_markdown(out, cfg.output_md, _DISCLAIMER)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke run**

```bash
cd SimTradeLab && poetry run python -m simtradelab.grid_screener --config examples/grid_screener/sample_run_config.json
```

Expected: CSV and MD written under `examples/grid_screener/out/`.

- [ ] **Step 4: Run full unit suite**

```bash
cd SimTradeLab && poetry run pytest tests/unit/test_grid_screener_metrics.py tests/unit/test_grid_screener_pipeline.py -v
```

- [ ] **Step 5: Commit**

```bash
cd SimTradeLab && git add src/simtradelab/grid_screener/__main__.py examples/grid_screener && git commit -m "feat(grid_screener): CLI and sample config"
```

---

### Task 8 (Optional): Fixed-weight composite score

**Files:**
- Modify: `src/simtradelab/grid_screener/pipeline.py` or `report.py`
- Modify: `src/simtradelab/grid_screener/config.py` (document default weights in field description)

Only implement if `enable_composite` is True: map `range_time_ratio` (+), `vol_comfort_score` (+), `trend_t` (penalize `abs`), `gap_tail_ratio` (-), `variance_ratio` penalize if `>1`, into a weighted sum; write column `composite_optional`. Add one unit test with hand-picked numbers.

---

## Spec coverage checklist (self-review)

| Spec section | Tasks |
|--------------|-------|
| §3 window / N_min / tags | Task 2, 5 |
| §4.1 trend | Task 3, 5 |
| §4.2 VR | Task 3, 5 |
| §4.3 ACF1 | Task 3, 5 |
| §4.4 rv + comfort | Task 3–5 |
| §4.5 gaps + intraday extreme | Task 4–5 |
| §4.6 range_time_ratio | Task 4–5 |
| §6 human report + disclaimer | Task 6–7 |
| §8 tests | All Tasks |
| 综合分 optional | Task 8 (optional) |

**Placeholder scan:** No `TBD` / `TODO` steps.

**Consistency:** Column names match `01-design.md` where applicable; `trend_t` / `variance_ratio` / `acf1_ret` / `rv_ann` / `gap_tail_ratio` / `range_time_ratio` aligned.

**Known deviation:** Trend SE uses **homoskedastic OLS** (footnote in report disclaimer: “t 统计量为经典 OLS，非 HAC。”).

---

## Execution handoff

计划已保存到 `SimTradeLab/my_docs/grid_friendly_screener/02-plan.md`。执行方式二选一：

**1. Subagent-Driven（推荐）** — 每个 Task 派生子代理，Task 间人工/quick review，迭代快。需配合 **superpowers:subagent-driven-development**。

**2. Inline Execution** — 本会话内按 Task 顺序实施，配合 **superpowers:executing-plans** 与检查点。

你想用哪一种？
