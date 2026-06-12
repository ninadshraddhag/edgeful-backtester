"""
ict.py — ICT structural concepts for the Advanced Backtester
============================================================
Algorithmic detection of the three structural primitives the strategy engine
consumes. All functions are pure and vectorized where practical, and operate on
an OHLC frame produced by indicators.resample_ohlc (any timeframe), i.e. columns
    open, high, low, close, close_time   (index = candle open time)

Each detector returns PER-BAR signal columns aligned to the input frame, so the
engine can attach them directly and reference them bar-by-bar with no lookahead:
the signal for a 3-candle pattern lands on the *confirming* candle (index i),
which is the first bar at which the pattern is fully known.

Concepts
--------
• Fair Value Gap (FVG)  — 3-candle imbalance. Bullish: low[i] > high[i-2]
  (a gap the market skipped). The gap edges double as risk levels: a bullish FVG
  is invalidated if price closes back below high[i-2] (the gap bottom).
• Order Block (OB)      — the last opposite-color candle before a displacement
  leg (here: the displacement that opens an FVG). Mitigation = price trading
  back into the block's body range later.
• Liquidity Sweep       — a wick that pierces PDH/PDL and closes back inside,
  i.e. stop-hunt-then-reject. Sweeping PDL (low<PDL, close>PDL) is bullish;
  sweeping PDH (high>PDH, close<PDH) is bearish.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Fair Value Gaps ──────────────────────────────────────────────────────────

def fair_value_gaps(frame: pd.DataFrame, min_gap_pct: float = 0.0) -> pd.DataFrame:
    """
    Per-bar FVG signals on the confirming candle (index i of the i-2,i-1,i trio).

    Columns added (aligned to `frame`):
        bull_fvg, bear_fvg                 : bool — a gap was confirmed on this bar
        fvg_gap_bottom, fvg_gap_top        : float — the imbalance edges
            bullish: bottom = high[i-2], top = low[i]   (price gapped up through it)
            bearish: bottom = high[i],   top = low[i-2]
        fvg_dir                            : +1 bullish / -1 bearish / 0 none

    `min_gap_pct` filters out micro-gaps smaller than that % of price.
    """
    n = len(frame)
    hi = frame["high"].to_numpy(dtype=float)
    lo = frame["low"].to_numpy(dtype=float)
    cl = frame["close"].to_numpy(dtype=float)

    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    gap_bottom = np.full(n, np.nan)
    gap_top = np.full(n, np.nan)

    if n >= 3:
        i = np.arange(2, n)
        # bullish: candle i's low sits above candle i-2's high → unfilled gap
        b_up = lo[2:] > hi[:-2]
        # bearish: candle i's high sits below candle i-2's low
        b_dn = hi[2:] < lo[:-2]

        if min_gap_pct > 0:
            up_sz = (lo[2:] - hi[:-2]) / np.where(cl[2:] != 0, cl[2:], np.nan) * 100
            dn_sz = (lo[:-2] - hi[2:]) / np.where(cl[2:] != 0, cl[2:], np.nan) * 100
            b_up &= up_sz >= min_gap_pct
            b_dn &= dn_sz >= min_gap_pct

        bull[i[b_up]] = True
        bear[i[b_dn]] = True
        gap_bottom[i[b_up]] = hi[:-2][b_up]
        gap_top[i[b_up]] = lo[2:][b_up]
        gap_bottom[i[b_dn]] = hi[2:][b_dn]
        gap_top[i[b_dn]] = lo[:-2][b_dn]

    out = pd.DataFrame(index=frame.index)
    out["bull_fvg"] = bull
    out["bear_fvg"] = bear
    out["fvg_gap_bottom"] = gap_bottom
    out["fvg_gap_top"] = gap_top
    out["fvg_dir"] = np.where(bull, 1, np.where(bear, -1, 0))
    return out


# ─── Order Blocks (displacement-anchored) ─────────────────────────────────────

def order_blocks(frame: pd.DataFrame, lookback: int = 5,
                 track_mitigation: bool = False) -> pd.DataFrame:
    """
    Detect order blocks anchored to FVG displacement.

    A bullish OB is the most recent bearish (close<open) candle within `lookback`
    bars before a bullish FVG's confirming candle; bearish OB mirrors this. The
    block zone is that candle's body [min(open,close), max(open,close)].

    Columns added (aligned to `frame`):
        bull_ob, bear_ob               : bool — an OB was identified at this bar
        ob_zone_low, ob_zone_high      : float — the block's body range
        ob_mitigated                   : bool — this OB was later re-entered
                                          (set on the ORIGIN bar once mitigation occurs)

    `track_mitigation` is OFF by default: the scan is O(blocks × future bars),
    which on multi-year 1-min/5-min data (e.g. XAUUSD) takes minutes and is not
    consumed by the strategy engine. Enable it only for targeted analysis.
    """
    n = len(frame)
    op = frame["open"].to_numpy(dtype=float)
    hi = frame["high"].to_numpy(dtype=float)
    lo = frame["low"].to_numpy(dtype=float)
    cl = frame["close"].to_numpy(dtype=float)
    bearish_candle = cl < op
    bullish_candle = cl > op

    fvg = fair_value_gaps(frame)
    bull_fvg = fvg["bull_fvg"].to_numpy()
    bear_fvg = fvg["bear_fvg"].to_numpy()

    bull_ob = np.zeros(n, dtype=bool)
    bear_ob = np.zeros(n, dtype=bool)
    zlow = np.full(n, np.nan)
    zhigh = np.full(n, np.nan)
    mitig = np.zeros(n, dtype=bool)

    for i in range(2, n):
        if bull_fvg[i]:
            for j in range(i - 1, max(i - 1 - lookback, -1), -1):
                if bearish_candle[j]:
                    bull_ob[j] = True
                    zlow[j] = min(op[j], cl[j])
                    zhigh[j] = max(op[j], cl[j])
                    break
        if bear_fvg[i]:
            for j in range(i - 1, max(i - 1 - lookback, -1), -1):
                if bullish_candle[j]:
                    bear_ob[j] = True
                    zlow[j] = min(op[j], cl[j])
                    zhigh[j] = max(op[j], cl[j])
                    break

    # mitigation: a later bar trades back into the block's body range.
    # O(blocks × future) — opt-in only (see docstring).
    if track_mitigation:
        ob_idx = np.where(bull_ob | bear_ob)[0]
        for j in ob_idx:
            z0, z1 = zlow[j], zhigh[j]
            future_lo = lo[j + 1:]
            future_hi = hi[j + 1:]
            if np.any((future_hi >= z0) & (future_lo <= z1)):
                mitig[j] = True

    out = pd.DataFrame(index=frame.index)
    out["bull_ob"] = bull_ob
    out["bear_ob"] = bear_ob
    out["ob_zone_low"] = zlow
    out["ob_zone_high"] = zhigh
    out["ob_mitigated"] = mitig
    return out


# ─── Liquidity Sweeps of PDH / PDL ────────────────────────────────────────────

def liquidity_sweeps(frame: pd.DataFrame, pdh: np.ndarray, pdl: np.ndarray) -> pd.DataFrame:
    """
    Per-bar liquidity-sweep flags against the day's PDH / PDL (passed aligned to
    each bar). A sweep is a wick beyond the level that closes back inside it.

    Columns added (aligned to `frame`):
        swept_pdl : bool — low pierced PDL, close back above  → bullish reversal
        swept_pdh : bool — high pierced PDH, close back below  → bearish reversal
    """
    hi = frame["high"].to_numpy(dtype=float)
    lo = frame["low"].to_numpy(dtype=float)
    cl = frame["close"].to_numpy(dtype=float)
    pdh = np.asarray(pdh, dtype=float)
    pdl = np.asarray(pdl, dtype=float)

    swept_pdl = (lo < pdl) & (cl > pdl)
    swept_pdh = (hi > pdh) & (cl < pdh)
    swept_pdl = np.where(np.isnan(pdl), False, swept_pdl)
    swept_pdh = np.where(np.isnan(pdh), False, swept_pdh)

    out = pd.DataFrame(index=frame.index)
    out["swept_pdl"] = swept_pdl
    out["swept_pdh"] = swept_pdh
    return out
