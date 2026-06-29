"""Vectorised technical & ICT indicators.

Every function takes a price ``DataFrame`` (columns ``open, high, low, close,
volume``) and returns either a ``Series`` or a small ``DataFrame`` aligned to the
input index.  No Python-level row loops - everything is numpy / pandas vectorised
so it stays fast on million-row minute data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Classic indicators
# --------------------------------------------------------------------------- #
def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Exponential moving average (recursive, no look-ahead)."""
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing (RMA)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


# --------------------------------------------------------------------------- #
# ICT primitives
# --------------------------------------------------------------------------- #
def fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """3-candle Fair Value Gaps (imbalances).

    Bullish FVG at bar ``i`` when ``low[i] > high[i-2]`` (a gap the market left
    between candle i-2's high and candle i's low). Bearish when
    ``high[i] < low[i-2]``.  Returns boolean flags plus the gap boundaries.
    """
    high, low = df["high"], df["low"]
    high2 = high.shift(2)
    low2 = low.shift(2)

    bull = low > high2
    bear = high < low2

    out = pd.DataFrame(index=df.index)
    out["fvg_bull"] = bull.fillna(False)
    out["fvg_bear"] = bear.fillna(False)
    # Gap zone (only meaningful where the flag is True).
    out["fvg_top"] = np.where(bull, low, np.where(bear, low2, np.nan))
    out["fvg_bottom"] = np.where(bull, high2, np.where(bear, high, np.nan))
    out["fvg_size"] = (out["fvg_top"] - out["fvg_bottom"]).abs()
    return out


def order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Displacement-confirmed order blocks.

    A bullish order block is the last *down* candle immediately before a bullish
    displacement candle that closes above that down candle's high.  The flag is
    set on the confirmation bar ``i`` (so it is tradable without look-ahead); the
    OB zone is the prior candle's body.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    prev_bear = (c.shift(1) < o.shift(1))
    prev_bull = (c.shift(1) > o.shift(1))

    ob_bull = prev_bear & (c > h.shift(1))           # bullish breakout of a down candle
    ob_bear = prev_bull & (c < l.shift(1))           # bearish breakdown of an up candle

    out = pd.DataFrame(index=df.index)
    out["ob_bull"] = ob_bull.fillna(False)
    out["ob_bear"] = ob_bear.fillna(False)
    # Zone = the prior (origin) candle's open/close range.
    out["ob_top"] = np.where(ob_bull | ob_bear, np.maximum(o.shift(1), c.shift(1)), np.nan)
    out["ob_bottom"] = np.where(ob_bull | ob_bear, np.minimum(o.shift(1), c.shift(1)), np.nan)
    return out


def liquidity_sweeps(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Liquidity sweeps (stop hunts) of recent swing highs/lows.

    ``sweep_high``: bar pierces the prior ``lookback`` high but closes back below
    it -> sell-side reversal signal.  ``sweep_low``: bar pierces the prior
    ``lookback`` low but closes back above it -> buy-side reversal signal.
    The rolling extreme is shifted by 1 so the current bar is excluded.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prior_high = high.rolling(lookback).max().shift(1)
    prior_low = low.rolling(lookback).min().shift(1)

    sweep_high = (high > prior_high) & (close < prior_high)
    sweep_low = (low < prior_low) & (close > prior_low)

    out = pd.DataFrame(index=df.index)
    out["sweep_high"] = sweep_high.fillna(False)
    out["sweep_low"] = sweep_low.fillna(False)
    out["swept_high_level"] = np.where(sweep_high, prior_high, np.nan)
    out["swept_low_level"] = np.where(sweep_low, prior_low, np.nan)
    return out


# --------------------------------------------------------------------------- #
# Convenience: bundle every signal a strategy might reference.
# --------------------------------------------------------------------------- #
def compute_all(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14,
                sweep_lookback: int = 20) -> pd.DataFrame:
    """Attach every indicator/signal column to a copy of ``df``."""
    out = df.copy()
    out["ema"] = ema(out["close"], ema_period)
    out["atr"] = atr(out, atr_period)
    out = pd.concat([out, fair_value_gaps(out), order_blocks(out),
                     liquidity_sweeps(out, sweep_lookback)], axis=1)

    # Derived directional context used by strategies.
    out["price_above_ema"] = out["close"] > out["ema"]
    out["price_below_ema"] = out["close"] < out["ema"]
    cross_up = (out["close"] > out["ema"]) & (out["close"].shift(1) <= out["ema"].shift(1))
    cross_dn = (out["close"] < out["ema"]) & (out["close"].shift(1) >= out["ema"].shift(1))
    out["ema_cross_up"] = cross_up.fillna(False)
    out["ema_cross_down"] = cross_dn.fillna(False)
    return out
