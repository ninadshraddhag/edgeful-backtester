"""
option_data.py — read access to the local options parquet store (data_options/).

The store is produced by build_options_store.py and is LOCAL ONLY (gitignored,
~1 GB). Everything here degrades gracefully when the store is absent (cloud):
`available()` returns [] and the option backtester shows a setup hint.

Layout consumed:
    data_options/<UND>/<YYYY-MM>.parquet   dt,expiry,strike,right,o,h,l,c,oi,volume
    data_options/<UND>_spot_minute.parquet dt,open,high,low,close
    data_options/<UND>_fut_minute.parquet  dt,open,high,low,close,volume

Access pattern is day-centric (a backtest walks one trading day at a time):
    ch  = day_chain(u, date)                     # whole chain for the day
    exp = nearest_expiry(ch, date, min_dte=0)    # weekly expiry to trade
    px  = underlying_price(u, date, t_min)        # spot at entry minute
    k   = atm_strike(ch, exp, px)                 # nearest available strike
    leg = leg_series(ch, exp, k, "CE")            # that contract's minute OHLC
"""
from __future__ import annotations

import datetime as _dt
import functools
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "data_options")

# defaults (user-overridable in the UI); strike step is also inferred from data
STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100}
LOT_SIZE = {"NIFTY": 75, "BANKNIFTY": 15}   # current NSE lots; historical values differ


# ─── discovery ────────────────────────────────────────────────────────────────

def available() -> list[str]:
    """Underlyings that have a built chain store, e.g. ['BANKNIFTY', 'NIFTY']."""
    if not os.path.isdir(STORE):
        return []
    out = []
    for u in sorted(os.listdir(STORE)):
        d = os.path.join(STORE, u)
        if os.path.isdir(d) and any(f.endswith(".parquet") for f in os.listdir(d)):
            out.append(u)
    return out


def _months(u: str) -> list[str]:
    d = os.path.join(STORE, u)
    if not os.path.isdir(d):
        return []
    return sorted(f[:-8] for f in os.listdir(d) if f.endswith(".parquet"))


def date_range(u: str) -> tuple[_dt.date | None, _dt.date | None]:
    ms = _months(u)
    if not ms:
        return None, None
    lo = _month(u, ms[0])["dt"].min().date()
    hi = _month(u, ms[-1])["dt"].max().date()
    return lo, hi


# ─── cached loaders ───────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=8)
def _month(u: str, ym: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(STORE, u, f"{ym}.parquet"))
    df["t_min"] = (df["dt"].dt.hour * 60 + df["dt"].dt.minute).astype("int16")
    df["d"] = df["dt"].dt.normalize()          # midnight timestamp per row (fast filter)
    return df


@functools.lru_cache(maxsize=8)
def day_chain(u: str, date) -> pd.DataFrame:
    """Whole option chain for one trading day (all expiries/strikes/rights)."""
    date = pd.Timestamp(date).normalize()
    ym = f"{date.year:04d}-{date.month:02d}"
    if ym not in _months(u):
        return pd.DataFrame()
    m = _month(u, ym)
    return m[m["d"] == date]


@functools.lru_cache(maxsize=2)
def _spot(u: str) -> pd.DataFrame:
    p = os.path.join(STORE, f"{u}_spot_minute.parquet")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["t_min"] = (df["dt"].dt.hour * 60 + df["dt"].dt.minute).astype("int16")
    df["d"] = df["dt"].dt.normalize()
    return df


def trading_days(u: str) -> list:
    """Sorted list of dates that have chain data."""
    out = []
    for ym in _months(u):
        out.extend(pd.to_datetime(_month(u, ym)["d"].unique()))
    return sorted(pd.Timestamp(x).date() for x in set(out))


# ─── per-day helpers (operate on a day_chain frame) ───────────────────────────

def expiries(chain: pd.DataFrame) -> list:
    return sorted(pd.Timestamp(x).date() for x in chain["expiry"].unique()) if len(chain) else []


def nearest_expiry(chain: pd.DataFrame, date, min_dte: int = 0, which: int = 0):
    """The `which`-th expiry (0 = nearest) with days-to-expiry >= min_dte."""
    date = pd.Timestamp(date).normalize()
    exps = [e for e in expiries(chain) if (pd.Timestamp(e) - date).days >= min_dte]
    if not exps:
        return None
    return exps[min(which, len(exps) - 1)]


def infer_step(chain: pd.DataFrame, expiry) -> int:
    ks = np.sort(chain.loc[chain["expiry"] == pd.Timestamp(expiry), "strike"].unique())
    if len(ks) < 2:
        return 50
    return int(np.min(np.diff(ks)))


def underlying_price(u: str, date, t_min: int) -> float | None:
    """Spot close at/just-before t_min (falls back to the chain's fut/underlying)."""
    sp = _spot(u)
    if len(sp):
        day = sp[sp["d"] == pd.Timestamp(date).normalize()]
        day = day[day["t_min"] <= t_min]
        if len(day):
            return float(day.iloc[-1]["close"])
    return None


def atm_strike(chain: pd.DataFrame, expiry, price: float) -> int | None:
    """Nearest AVAILABLE strike to `price` for the given expiry."""
    ks = chain.loc[chain["expiry"] == pd.Timestamp(expiry), "strike"].unique()
    if len(ks) == 0 or price is None:
        return None
    return int(ks[np.argmin(np.abs(ks.astype(float) - price))])


def pick_strike(chain: pd.DataFrame, expiry, right: str, target: float) -> int | None:
    """Nearest available strike to `target` for expiry+right (used for offset legs)."""
    sel = chain[(chain["expiry"] == pd.Timestamp(expiry)) & (chain["right"] == right)]
    ks = sel["strike"].unique()
    if len(ks) == 0:
        return None
    return int(ks[np.argmin(np.abs(ks.astype(float) - target))])


def leg_series(chain: pd.DataFrame, expiry, strike: int, right: str) -> pd.DataFrame:
    """Minute OHLC for one contract on the day, indexed columns dt,t_min,o,h,l,c.
    Empty frame if the contract didn't trade that day."""
    sel = chain[(chain["expiry"] == pd.Timestamp(expiry))
                & (chain["strike"] == int(strike))
                & (chain["right"] == right)]
    return sel[["dt", "t_min", "open", "high", "low", "close"]].sort_values("t_min").reset_index(drop=True)
