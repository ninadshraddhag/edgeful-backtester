"""Accuracy of the dual-Supertrend systems in the LONDON first 2 hours
(03:00-05:00 ET) vs the NY open (09:30-11:30 ET), on real NQ 1-min.
VWAP is anchored to each session's own start.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import loader
from core import indicators

PV = 20.0
TICK = 0.25


def st_line(high, low, close, period, mult):
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    atr = indicators.atr(df, period).to_numpy(float)
    hl2 = (high + low) / 2.0
    up = hl2 + mult * atr; dn = hl2 - mult * atr
    n = len(close); fu = np.full(n, np.nan); fl = np.full(n, np.nan)
    d = np.ones(n, int); line = np.full(n, np.nan)
    for i in range(n):
        if i == 0 or np.isnan(atr[i]):
            fu[i] = up[i]; fl[i] = dn[i]; d[i] = 1; line[i] = fl[i]; continue
        fu[i] = up[i] if (up[i] < fu[i-1] or close[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = dn[i] if (dn[i] > fl[i-1] or close[i-1] < fl[i-1]) else fl[i-1]
        d[i] = 1 if close[i] > fu[i-1] else (-1 if close[i] < fl[i-1] else d[i-1])
        line[i] = fl[i] if d[i] == 1 else fu[i]
    return d, line


def anchored_vwap(data, start, end):
    mins = pd.to_datetime(data["date"]).dt.strftime("%H:%M")
    win = (mins >= start) & (mins < end)
    tp = (data["high"] + data["low"] + data["close"]) / 3.0
    vol = data["volume"].astype(float).clip(lower=0)
    pv = (tp * vol).where(win, 0.0); vv = vol.where(win, 0.0)
    g = data["session_date"]
    return (pv.groupby(g).cumsum() / vv.groupby(g).cumsum().replace(0, np.nan)).to_numpy(float)


print("Loading real NQ 1-min...")
data = loader.load_data("NQ", timeframe="1min", rth_only=False)
close = data["close"].to_numpy(float); high = data["high"].to_numpy(float); low = data["low"].to_numpy(float)
o = data["open"].to_numpy(float)
mjd, _ = st_line(high, low, close, 9, 4.0)
mnd, mnl = st_line(high, low, close, 4, 1.7)
atr14 = indicators.atr(data, 14).to_numpy(float)
mins = pd.to_datetime(data["date"]).dt.strftime("%H:%M").to_numpy(); day = data["session_date"].to_numpy()
n = len(data)
pmnd = np.roll(mnd, 1); pmnd[0] = 0
pmjd = np.roll(mjd, 1); pmjd[0] = 0
pclose = np.roll(close, 1); pclose[0] = np.nan
pl = np.roll(mnl, 1); pl[0] = np.nan
patr = np.roll(atr14, 1); patr[0] = np.nan


def bias_ok(i, d, mode, pvw):
    if mode == "raw":
        return True
    if np.isnan(pvw[i]):
        return False
    return (pclose[i] > pvw[i]) if d == 1 else (pclose[i] < pvw[i])


def run_flip(start, end, pvw, mode, stop_atr=1.5, max_trades=5):
    """Market-on-flip entry, ST-trail exit."""
    long_sig = (mnd == 1) & (np.roll(mnd, 1) == -1) & (mjd == 1)
    short_sig = (mnd == -1) & (np.roll(mnd, 1) == 1) & (mjd == -1)
    R = []; cur = day[0]; tdc = 0; i = 1
    while i < n - 1:
        if day[i] != cur:
            cur = day[i]; tdc = 0
        if not (start <= mins[i] < end) or np.isnan(patr[i]) or patr[i] <= 0 or tdc >= max_trades:
            i += 1; continue
        d = 1 if long_sig[i] else (-1 if short_sig[i] else 0)
        if d == 0 or not bias_ok(i, d, mode, pvw):
            i += 1; continue
        entry = o[i + 1]; risk = stop_atr * atr14[i]
        stop = entry - risk if d == 1 else entry + risk
        j = i + 1; xp = close[j]; last = i + 1
        while j < n and (start <= mins[j] < end) and day[j] == cur:
            if d == 1 and low[j] <= stop:
                xp = stop; break
            if d == -1 and high[j] >= stop:
                xp = stop; break
            if mnd[j] == -d and j > i + 1:
                xp = close[j]; break
            last = j; j += 1
        else:
            xp = close[last]
        pnl = (xp - entry) if d == 1 else (entry - xp)
        R.append(pnl / risk); tdc += 1; i = j + 1
    return np.array(R)


def run_limit(start, end, pvw, mode, stop_atr, tgt_atr, pessimistic, tol=2.0, rearm=0.4, max_trades=5):
    R = []; armed = False; cur = day[0]; tdc = 0; i = 1
    while i < n - 1:
        if day[i] != cur:
            cur = day[i]; tdc = 0; armed = False
        if not (start <= mins[i] < end) or np.isnan(pl[i]) or np.isnan(patr[i]) or patr[i] <= 0:
            i += 1; continue
        d = pmnd[i]
        if d == 0 or (mode == "raw" and False):
            pass
        if d == 0:
            i += 1; continue
        if mode != "raw":
            if d == 1 and not (pmjd[i] == 1 and not np.isnan(pvw[i]) and pclose[i] > pvw[i]):
                i += 1; continue
            if d == -1 and not (pmjd[i] == -1 and not np.isnan(pvw[i]) and pclose[i] < pvw[i]):
                i += 1; continue
        else:
            if d == 1 and pmjd[i] != 1:
                i += 1; continue
            if d == -1 and pmjd[i] != -1:
                i += 1; continue
        a = patr[i]; line = pl[i]; thru = TICK if pessimistic else 0.0
        if d == 1:
            fill = line + tol; pulled = (pclose[i] - line) >= rearm * a; hit = low[i] <= fill - thru
        else:
            fill = line - tol; pulled = (line - pclose[i]) >= rearm * a; hit = high[i] >= fill + thru
        if not armed and pulled:
            armed = True
        if tdc >= max_trades or not (armed and hit):
            i += 1; continue
        risk = stop_atr * a; slip = TICK if pessimistic else 0.0; tthru = TICK if pessimistic else 0.0
        tgt_from = i + 1 if pessimistic else i
        if d == 1:
            stop = fill - risk; tgt = fill + tgt_atr * a
        else:
            stop = fill + risk; tgt = fill - tgt_atr * a
        j = i; xp = None; last = i
        while j < n and (start <= mins[j] < end) and day[j] == cur:
            if d == 1:
                if low[j] <= stop:
                    xp = stop - slip; break
                if j >= tgt_from and high[j] >= tgt + tthru:
                    xp = tgt; break
            else:
                if high[j] >= stop:
                    xp = stop + slip; break
                if j >= tgt_from and low[j] <= tgt - tthru:
                    xp = tgt; break
            last = j; j += 1
        if xp is None:
            xp = close[last]; j = last
        pnl = (xp - fill) if d == 1 else (fill - xp)
        R.append(pnl / risk); tdc += 1; armed = False; i = j + 1
    return np.array(R)


def acc(R):
    if len(R) == 0:
        return (0, 0.0, 0.0)
    return (len(R), 100 * np.mean(R > 0), R.mean())


sessions = {
    "LONDON 03:00-05:00": ("03:00", "05:00"),
    "NY     09:30-11:30": ("09:30", "11:30"),
}
vwaps = {
    "LONDON 03:00-05:00": anchored_vwap(data, "03:00", "05:00"),
    "NY     09:30-11:30": anchored_vwap(data, "09:30", "16:00"),
}
for name, (s, e) in sessions.items():
    pvw = np.roll(vwaps[name], 1); pvw[0] = np.nan
    nd = pd.Series(day[(mins >= s) & (mins < e)]).nunique()
    print(f"\n===== {name}   ({nd} days) =====")
    print(f"{'system':<34}{'trades':>8}{'accuracy':>10}{'exp_R':>9}")
    for mode in ("raw", "filt"):
        R = run_flip(s, e, pvw, mode)
        nn, w, ex = acc(R)
        print(f"{'flip + ST-trail (' + mode + ')':<34}{nn:>8}{w:>9.1f}%{ex:>+9.3f}")
    for mode in ("raw", "filt"):
        R = run_limit(s, e, pvw, mode, 0.5, 1.0, pessimistic=False)
        nn, w, ex = acc(R)
        print(f"{'limit 0.5/1ATR OPTIMISTIC (' + mode + ')':<34}{nn:>8}{w:>9.1f}%{ex:>+9.3f}")
    for mode in ("raw", "filt"):
        R = run_limit(s, e, pvw, mode, 0.5, 1.0, pessimistic=True)
        nn, w, ex = acc(R)
        print(f"{'limit 0.5/1ATR REALISTIC (' + mode + ')':<34}{nn:>8}{w:>9.1f}%{ex:>+9.3f}")
