"""网格适宜度模型 (GSS) 因子 — 见 my_docs/grid_friendly_screener/03-另一个思路.md"""

from __future__ import annotations

import re

import numpy as np

from simtradelab.grid_screener.config import GssParams
from simtradelab.grid_screener.context import FactorContext
from simtradelab.grid_screener.metrics import (
    adx_last,
    atr_ratio,
    average_daily_turnover,
    clip01,
    hurst_exponent,
    ols_log_close_trend,
    price_percentile,
    realized_vol_ann,
)


def _gss(ctx: FactorContext) -> GssParams:
    return ctx.params.gss


def _is_st(symbol: str, name: str) -> bool:
    text = "{0} {1}".format(symbol, name).upper()
    if re.search(r"(?:^|\W)ST(?:\W|$)", text):
        return True
    if "*ST" in text or "ST*" in text:
        return True
    return False


def _nan_map(keys: tuple[str, ...]) -> dict[str, float]:
    v = float("nan")
    return dict.fromkeys(keys, v)


class GssVolatilityFactor:
    name = "gss_volatility"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        if ctx.insufficient:
            return {"hv_ann": float("nan"), "atr_ratio": float("nan")}
        g = _gss(ctx)
        hv = realized_vol_ann(ctx.r1)
        atr_r = atr_ratio(ctx.high, ctx.low, ctx.close, period=g.atr_period)
        return {"hv_ann": hv, "atr_ratio": atr_r}


class GssMeanReversionFactor:
    name = "gss_mean_reversion"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        if ctx.insufficient or ctx.close.size < 30:
            return _nan_map(("hurst", "adx", "trend_slope", "trend_r2"))
        g = _gss(ctx)
        hurst = hurst_exponent(ctx.close)
        adx = adx_last(ctx.high, ctx.low, ctx.close, period=g.adx_period)
        _, r2 = ols_log_close_trend(ctx.log_close)
        n = ctx.log_close.size
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, ctx.log_close, 1)[0]) if n >= 2 else float("nan")
        return {"hurst": hurst, "adx": adx, "trend_slope": slope, "trend_r2": r2}


class GssLiquidityFactor:
    name = "gss_liquidity"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        if ctx.insufficient:
            return {"adtv_yuan": float("nan"), "liquidity_ok": False}
        g = _gss(ctx)
        vol = ctx.window["volume"].to_numpy(dtype=float) if "volume" in ctx.window.columns else np.array([])
        adtv = average_daily_turnover(ctx.close, vol, g.adtv_lookback_days)
        ok = bool(np.isfinite(adtv) and adtv >= g.adtv_min_yuan)
        return {"adtv_yuan": adtv, "liquidity_ok": ok}


class GssSafetyFactor:
    name = "gss_safety"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        if ctx.insufficient:
            return {
                "price_percentile": float("nan"),
                "st_flag": False,
                "safety_score": float("nan"),
                "price_high_veto": False,
            }
        g = _gss(ctx)
        pct = price_percentile(ctx.close)
        st = _is_st(ctx.meta.symbol, ctx.meta.name)
        high_veto = bool(np.isfinite(pct) and pct > g.price_percentile_veto)

        if ctx.meta.asset_type == "etf":
            safety = 1.0
        elif st:
            safety = 0.0
        else:
            # 无基本面数据时：非 ST 股票给中等分，高位再扣
            safety = 0.6
            if np.isfinite(pct):
                safety += 0.4 * (1.0 - pct)

        return {
            "price_percentile": pct,
            "st_flag": st,
            "safety_score": float(safety),
            "price_high_veto": high_veto,
        }


class GssFrictionFactor:
    name = "gss_friction"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        if ctx.insufficient:
            return {"friction_score": float("nan")}
        if ctx.meta.asset_type == "etf":
            return {"friction_score": 1.0}
        return {"friction_score": 0.5}


class GssScoreFactor:
    """综合 GSS 分 + 一票否决标记（应放在 gss_* 因子之后）。"""

    name = "gss_score"

    def compute(self, ctx: FactorContext) -> dict[str, object]:
        if ctx.insufficient:
            return {
                "gss_veto": True,
                "gss_veto_reason": "insufficient_data",
                "gss_m": float("nan"),
                "gss_v": float("nan"),
                "gss_s": float("nan"),
                "gss_c": float("nan"),
                "gss_score": float("nan"),
            }

        g = _gss(ctx)
        row = ctx.outputs
        reasons: list[str] = []

        if not row.get("liquidity_ok", False):
            reasons.append("low_liquidity")
        if row.get("st_flag"):
            reasons.append("st")
        if row.get("price_high_veto"):
            reasons.append("price_percentile_high")

        if reasons:
            return {
                "gss_veto": True,
                "gss_veto_reason": "|".join(reasons),
                "gss_m": float("nan"),
                "gss_v": float("nan"),
                "gss_s": float("nan"),
                "gss_c": float("nan"),
                "gss_score": 0.0,
            }

        hurst = float(row.get("hurst") or float("nan"))
        adx = float(row.get("adx") or float("nan"))
        hv = float(row.get("hv_ann") or float("nan"))
        atr_r = float(row.get("atr_ratio") or float("nan"))
        safety = float(row.get("safety_score") or float("nan"))
        friction = float(row.get("friction_score") or float("nan"))

        m_hurst = clip01((g.hurst_mean_revert_below - hurst) / g.hurst_mean_revert_below) if np.isfinite(hurst) else 0.0
        m_adx = clip01((g.adx_max - adx) / g.adx_max) if np.isfinite(adx) else 0.0
        gss_m = 0.5 * m_hurst + 0.5 * m_adx

        v_hv = clip01((hv - g.hv_min) / 0.5) if np.isfinite(hv) else 0.0
        v_atr = clip01(atr_r / 0.05) if np.isfinite(atr_r) else 0.0
        gss_v = 0.5 * v_hv + 0.5 * v_atr

        gss_s = clip01(safety) if np.isfinite(safety) else 0.0
        gss_c = clip01(friction) if np.isfinite(friction) else 0.0

        w_sum = g.w_mean_reversion + g.w_volatility + g.w_safety + g.w_friction
        if w_sum <= 0:
            score = float("nan")
        else:
            score = (
                g.w_mean_reversion * gss_m
                + g.w_volatility * gss_v
                + g.w_safety * gss_s
                + g.w_friction * gss_c
            ) / w_sum

        return {
            "gss_veto": False,
            "gss_veto_reason": "",
            "gss_m": gss_m,
            "gss_v": gss_v,
            "gss_s": gss_s,
            "gss_c": gss_c,
            "gss_score": float(score) if np.isfinite(score) else float("nan"),
        }


GSS_FACTORS = (
    GssVolatilityFactor(),
    GssMeanReversionFactor(),
    GssLiquidityFactor(),
    GssSafetyFactor(),
    GssFrictionFactor(),
    GssScoreFactor(),
)
