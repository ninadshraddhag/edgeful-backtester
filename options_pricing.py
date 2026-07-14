"""
options_pricing.py — synthetic ATM/by-delta option pricing for the NIFTY Option
Strategy mode. Pure functions, no Streamlit. Vectorized where it matters.

Model: Black-Scholes European, flat IV (the "synthetic premiums (flat IV)"
convention). Spot-only data → we PRICE options off the index, we do not use
real premiums. Position: long the ATM (~0.5Δ) option in the trade direction.

Conventions
  • Strikes on a 50-point grid (NIFTY). ATM = nearest 50 to spot.
  • Nearest weekly expiry = the coming Thursday (NIFTY weekly). DTE floored at
    1 day so intraday theta on expiry day doesn't blow up. Pre-2019 weeklies
    didn't exist in reality; this is a synthetic-consistency choice.
  • r = 6.5% annual, flat IV (caller-supplied, default 13%).
  • Premiums are per-unit (index points); ₹ P&L = points × LOT_SIZE × lots.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

LOT_SIZE = 75           # NIFTY lot (fixed, current convention)
STRIKE_STEP = 50        # NIFTY strike grid
R_ANNUAL = 0.065        # risk-free (minor intraday)
YEAR_MIN = 365 * 24 * 60.0
SESSION_CLOSE_MIN = 15 * 60 + 30    # 15:30 — options value to the cash close

_SQRT2 = math.sqrt(2.0)


# ─── normal cdf / pdf (scalar + vector) ───────────────────────────────────────

def _ncdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, float) / _SQRT2)) \
        if np.ndim(x) else 0.5 * (1.0 + math.erf(x / _SQRT2))


def _npdf(x):
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


# ─── expiry / time-to-expiry ──────────────────────────────────────────────────

def next_weekly_expiry(d: date) -> date:
    """Coming Thursday (weekday 3), same day if it IS Thursday."""
    return d + timedelta(days=(3 - d.weekday()) % 7)


def tte_years(dt, floor_days: float = 1.0) -> float:
    """
    Time to next weekly expiry in YEARS, counting the fractional intraday part
    (expiry assumed at 15:30 on expiry day). Floored at `floor_days` days so
    expiry-day gamma/theta stay finite.
    """
    d = dt.date() if hasattr(dt, "date") else dt
    exp = next_weekly_expiry(d)
    now_min = dt.hour * 60 + dt.minute if hasattr(dt, "hour") else 9 * 60 + 15
    minutes = (exp - d).days * 1440 + (SESSION_CLOSE_MIN - now_min)
    minutes = max(minutes, floor_days * 1440.0)
    return minutes / YEAR_MIN


# ─── Black-Scholes ────────────────────────────────────────────────────────────

def _d1(S, K, T, iv, r=R_ANNUAL):
    S = np.asarray(S, float); K = np.asarray(K, float)
    T = np.maximum(np.asarray(T, float), 1e-9)
    return (np.log(S / K) + (r + 0.5 * iv * iv) * T) / (iv * np.sqrt(T))


def bs_price(S, K, T, iv, is_call, r=R_ANNUAL):
    """Black-Scholes premium (index points). Scalar or vector S/T."""
    T = np.maximum(np.asarray(T, float), 1e-9)
    d1 = _d1(S, K, T, iv, r)
    d2 = d1 - iv * np.sqrt(T)
    disc = np.exp(-r * T)
    if is_call:
        return _ncdf(d1) * np.asarray(S, float) - _ncdf(d2) * np.asarray(K, float) * disc
    return _ncdf(-d2) * np.asarray(K, float) * disc - _ncdf(-d1) * np.asarray(S, float)


# ─── strike selection ─────────────────────────────────────────────────────────

def atm_strike(spot: float) -> float:
    return round(spot / STRIKE_STEP) * STRIKE_STEP


# ─── cost model ───────────────────────────────────────────────────────────────

def round_trip_cost(entry_prem: float, exit_prem: float, lots: int,
                    slippage_pts: float = 0.5, brokerage: float = 20.0,
                    cost_mult: float = 1.0, slip_mult: float = 1.0) -> float:
    """
    All-in round-trip cost in ₹ for one option leg (buy+sell or sell+buy).
      • STT 0.0625% of premium on the SELL side (options).
      • Exchange+SEBI+stamp ≈ 0.05% of turnover (both sides), approximated.
      • Brokerage flat per side.
      • Slippage: slippage_pts index points per side (× lot × lots).
    cost_mult / slip_mult are the stress-test multipliers (×1.5, ×2).
    """
    qty = LOT_SIZE * lots
    sell_prem = max(entry_prem, exit_prem)          # whichever side is the sell
    stt = 0.000625 * sell_prem * qty
    txn = 0.0005 * (entry_prem + exit_prem) * qty
    brk = 2 * brokerage
    slip = 2 * slippage_pts * qty * slip_mult
    return (stt + txn + brk) * cost_mult + slip
