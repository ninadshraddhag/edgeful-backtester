"""
indicators.py — Technical & structural indicators for the Advanced Backtester
=============================================================================
Pure, vectorized, Streamlit-free. Operates on the project's standard cleaned
minute frame (see build_facts.clean_min): columns
    date (datetime64), open, high, low, close, date_only (date), t_min (int)

What's here
-----------
• resample_ohlc(df, tf, open_t)  — aggregate 1-min bars to any timeframe.
    Each output candle carries `open_time` (its index) AND `close_time` (the
    timestamp of its *last* constituent minute). close_time is the anchor used
    downstream for no-lookahead HTF alignment — a candle is only "known" once
    its close_time has passed.
• ema / add_emas                 — exponential moving averages (customizable spans).
• rsi                            — Wilder's RSI (customizable lookback).
• central_pivot_range            — Daily/Weekly CPR (P, BC, TC) from the PREVIOUS
    period's H/L/C, classified NARROW / MEDIUM / WIDE by width-vs-price %.
• key_levels                     — Previous-Day High / Low (PDH / PDL) per date.

Everything that references a "previous" period is shifted so a row only ever
sees data that closed before that row's session — zero forward-looking bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Intraday timeframes expressed in minutes; "Daily"/"Weekly" handled separately.
TF_MINUTES = {
    "1-min":  1,
    "3-min":  3,
    "5-min":  5,
    "15-min": 15,
    "30-min": 30,
    "1-hour": 60,
}
TIMEFRAMES = list(TF_MINUTES.keys()) + ["Daily", "Weekly"]


# ─── resampling ───────────────────────────────────────────────────────────────

def resample_ohlc(df: pd.DataFrame, tf: str, open_t: int) -> pd.DataFrame:
    """
    Aggregate a cleaned 1-min frame to timeframe `tf`.

    Intraday buckets are anchored to the session open (`open_t`, minutes-since-
    midnight) and never cross a day boundary, so the first bucket of every day
    starts exactly at the open regardless of the chosen interval.

    Returns a frame indexed by each candle's open timestamp with columns:
        open, high, low, close, date_only, close_time
    `close_time` = timestamp of the candle's last 1-min bar (its confirmation
    reference for leak-free higher-timeframe alignment).
    """
    if tf not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {tf!r}")

    if tf == "1-min":
        out = df[["date", "open", "high", "low", "close", "date_only"]].copy()
        out["close_time"] = out["date"]
        return out.set_index("date", drop=False).rename(columns={"date": "open_time"}) \
                  .rename_axis(None)

    if tf in ("Daily", "Weekly"):
        key = df["date_only"] if tf == "Daily" else df["date"].dt.to_period("W")
        g = df.groupby(key, sort=True)
        out = pd.DataFrame({
            "open":       g["open"].first(),
            "high":       g["high"].max(),
            "low":        g["low"].min(),
            "close":      g["close"].last(),
            "open_time":  g["date"].first(),
            "close_time": g["date"].last(),
        })
        out["date_only"] = out["open_time"].dt.date
        return out.set_index("open_time", drop=False).rename_axis(None)

    # generic intraday bucketing, anchored at the session open, never crossing days
    minutes = TF_MINUTES[tf]
    bucket = (df["t_min"].to_numpy() - open_t) // minutes
    g = df.groupby([df["date_only"].to_numpy(), bucket], sort=True)
    out = pd.DataFrame({
        "open":       g["open"].first(),
        "high":       g["high"].max(),
        "low":        g["low"].min(),
        "close":      g["close"].last(),
        "open_time":  g["date"].first(),
        "close_time": g["date"].last(),
    })
    out["date_only"] = out["open_time"].dt.date
    return out.sort_values("open_time").set_index("open_time", drop=False).rename_axis(None)


# ─── moving averages & momentum ───────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (adjust=False → standard trading EMA)."""
    return series.ewm(span=int(period), adjust=False).mean()


def add_emas(frame: pd.DataFrame, periods, src: str = "close") -> pd.DataFrame:
    """Attach EMA columns (`ema_<p>`) for each period in `periods`, in place-ish."""
    frame = frame.copy()
    for p in periods:
        frame[f"ema_{int(p)}"] = ema(frame[src], int(p))
    return frame


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI. Uses an exponentially-weighted (alpha = 1/period) average of
    gains/losses — the standard charting definition. Leading values are NaN
    until `period` bars exist.
    """
    period = int(period)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out[avg_loss == 0.0] = 100.0          # all-gains window → RSI 100
    return out


# ─── central pivot range (CPR) ────────────────────────────────────────────────

def classify_cpr_width(width_pct: pd.Series, narrow: float, wide: float) -> pd.Series:
    """
    Map CPR width (% of pivot price) to NARROW / MEDIUM / WIDE.
        width% < narrow         → NARROW   (trend / breakout day)
        narrow ≤ width% ≤ wide  → MEDIUM
        width% > wide           → WIDE     (range / mean-reversion day)
    """
    out = pd.Series(np.where(width_pct < narrow, "NARROW",
                    np.where(width_pct > wide, "WIDE", "MEDIUM")),
                    index=width_pct.index, dtype=object)
    out[width_pct.isna()] = None
    return out


def _cpr_from_hlc(high, low, close):
    """Pivot P, Bottom-Central BC and Top-Central TC from one period's H/L/C."""
    p  = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = 2.0 * p - bc                      # mirror of BC about the pivot
    top    = np.maximum(tc, bc)            # geometric top/bottom of the band
    bottom = np.minimum(tc, bc)
    return p, bc, tc, top, bottom


def central_pivot_range(df: pd.DataFrame, period: str = "Daily",
                        narrow: float = 0.30, wide: float = 0.75) -> pd.DataFrame:
    """
    Per-period CPR computed from the PREVIOUS period's H/L/C (so today's CPR is
    fully known at today's open — no lookahead). `period` is "Daily" or "Weekly".

    Returns a frame indexed by the *date the CPR applies to* (date_only for
    Daily; every session date within a week for Weekly) with columns:
        cpr_p, cpr_bc, cpr_tc, cpr_top, cpr_bottom, cpr_width_pct, cpr_type
    Joinable onto an intraday frame by `date_only`.
    """
    if period == "Daily":
        g = df.groupby(df["date_only"], sort=True)
    else:
        g = df.groupby(df["date"].dt.to_period("W"), sort=True)
    agg = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last()})

    prev = agg.shift(1)                    # previous period drives the levels
    p, bc, tc, top, bottom = _cpr_from_hlc(prev["high"], prev["low"], prev["close"])
    width_pct = (top - bottom).abs() / p * 100.0

    cpr = pd.DataFrame({
        "cpr_p": p.values, "cpr_bc": bc.values, "cpr_tc": tc.values,
        "cpr_top": top.values, "cpr_bottom": bottom.values,
        "cpr_width_pct": width_pct.values,
    }, index=agg.index)
    cpr["cpr_type"] = classify_cpr_width(width_pct, narrow, wide).values

    if period == "Daily":
        cpr.index.name = "date_only"
        return cpr

    # Weekly: map each session date → its week period → that week's CPR row.
    day_week = df.groupby("date_only")["date"].first().dt.to_period("W")
    out = pd.DataFrame({c: day_week.map(cpr[c]) for c in cpr.columns},
                       index=day_week.index)
    out.index.name = "date_only"
    return out


# ─── session VWAP ─────────────────────────────────────────────────────────────

def session_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session-anchored VWAP on a cleaned 1-min frame: cumulative Σ(typical×vol) /
    Σ(vol), reset every trading day. Typical price = (H+L+C)/3.

    Volume handling: uses the `volume` column when the day actually has volume;
    days with zero/absent volume (e.g. NIFTY index data) fall back to equal
    weights — i.e. the running average of typical price (TWAP), the standard
    proxy for instruments without traded volume.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    if "volume" in df.columns:
        v = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        day_total = v.groupby(df["date_only"]).transform("sum")
        w = pd.Series(np.where(day_total.to_numpy() > 0, v.to_numpy(), 1.0),
                      index=df.index)
    else:
        w = pd.Series(1.0, index=df.index)
    num = (tp * w).groupby(df["date_only"]).cumsum()
    den = w.groupby(df["date_only"]).cumsum()
    return num / den


def vwap5_steps(sess: pd.DataFrame, open_t: int, tf: int = 5):
    """
    For ONE day's session, return (pv5, is5) aligned to its 1-min rows:
      pv5 — the tf-minute session VWAP value of the tf-min bucket each 1-min bar
            belongs to (volume-weighted; TWAP fallback). At a bucket's CLOSE this
            equals the completed tf-min VWAP, so it carries no lookahead there.
      is5 — True on the last 1-min bar of each tf-min bucket (a tf-min close).
    Lets a 1-min sim run a *tf-min* VWAP-close exit: check `is5 & close-vs-pv5`.
    """
    t = sess["t_min"].to_numpy()
    n = len(t)
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    b = (t - open_t) // tf
    typ = ((sess["high"] + sess["low"] + sess["close"]) / 3.0).to_numpy()
    if "volume" in sess.columns:
        v = pd.to_numeric(sess["volume"], errors="coerce").fillna(0.0).to_numpy()
        if v.sum() <= 0:
            v = np.ones(n)
    else:
        v = np.ones(n)
    agg = pd.DataFrame({"b": b, "tv": typ * v, "v": v}).groupby("b", sort=True)
    bucket_vwap = agg["tv"].sum().cumsum() / agg["v"].sum().cumsum()
    pv5 = bucket_vwap.reindex(b).to_numpy()
    # a tf-min close is the last minute of a complete bucket on the grid anchored
    # at open_t — deterministic, so a partial bucket at the session end is NOT a
    # tf-min close (the engine's square-off handles that final bar instead)
    is5 = ((t - open_t) % tf == (tf - 1))
    return pv5, is5


# ─── key levels: PDH / PDL ────────────────────────────────────────────────────

def key_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Previous-Day High / Low / Close per session date (the classic PDH / PDL).
    Indexed by date_only, shifted so each date sees only the prior day → no leak.
    """
    daily = resample_ohlc(df, "Daily", 0)
    out = pd.DataFrame({
        "pdh":        daily["high"].shift(1).values,
        "pdl":        daily["low"].shift(1).values,
        "prev_close": daily["close"].shift(1).values,
    }, index=daily["date_only"].values)
    out.index.name = "date_only"
    return out
