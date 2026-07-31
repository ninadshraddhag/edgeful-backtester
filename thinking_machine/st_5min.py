"""1-min vs 5-min execution: same settings, does the timeframe change the edge?
Config: minor ST(1.7,4) pullback + major ST(4,9) + VWAP, first hour 09:30-10:30,
1.5 ATR stop / 1.5 ATR target. Limit entry (ideal vs realistic) + market-flip.
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
    nn = len(close); fu = np.full(nn, np.nan); fl = np.full(nn, np.nan)
    d = np.ones(nn, int); line = np.full(nn, np.nan)
    for i in range(nn):
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


def build(tf):
    data = loader.load_data("NQ", timeframe=tf, rth_only=False)
    A = {}
    A["o"] = data["open"].to_numpy(float); A["c"] = data["close"].to_numpy(float)
    A["h"] = data["high"].to_numpy(float); A["l"] = data["low"].to_numpy(float)
    mjd, _ = st_line(A["h"], A["l"], A["c"], 9, 4.0)
    mnd, mnl = st_line(A["h"], A["l"], A["c"], 4, 1.7)
    A["atr"] = indicators.atr(data, 14).to_numpy(float)
    vw = anchored_vwap(data, "09:30", "16:00")
    A["mins"] = pd.to_datetime(data["date"]).dt.strftime("%H:%M").to_numpy()
    A["day"] = data["session_date"].to_numpy()
    A["mnd"] = mnd; A["mnl"] = mnl; A["mjd"] = mjd
    A["pmnd"] = np.roll(mnd, 1); A["pmnd"][0] = 0
    A["pmjd"] = np.roll(mjd, 1); A["pmjd"][0] = 0
    A["pclose"] = np.roll(A["c"], 1); A["pclose"][0] = np.nan
    A["pl"] = np.roll(mnl, 1); A["pl"][0] = np.nan
    A["patr"] = np.roll(A["atr"], 1); A["patr"][0] = np.nan
    A["pvw"] = np.roll(vw, 1); A["pvw"][0] = np.nan
    A["n"] = len(data)
    A["ndays"] = pd.Series(A["day"][(A["mins"] >= "09:30") & (A["mins"] < "10:30")]).nunique()
    return A


def run_limit(A, stop_atr, tgt_atr, pess, rearm=0.4, start="09:30", end="10:30", tol=2.0):
    n = A["n"]; mins = A["mins"]; day = A["day"]; pl = A["pl"]; patr = A["patr"]
    pmnd = A["pmnd"]; pmjd = A["pmjd"]; pclose = A["pclose"]; pvw = A["pvw"]
    low = A["l"]; high = A["h"]; close = A["c"]
    R = []; RC = []; armed = False; cur = day[0]; i = 1
    while i < n - 1:
        if day[i] != cur:
            cur = day[i]; armed = False
        if not (start <= mins[i] < end) or np.isnan(pl[i]) or np.isnan(patr[i]) or patr[i] <= 0:
            i += 1; continue
        d = pmnd[i]
        if (d == 1 and pmjd[i] != 1) or (d == -1 and pmjd[i] != -1) or d == 0:
            i += 1; continue
        if d == 1 and not (not np.isnan(pvw[i]) and pclose[i] > pvw[i]):
            i += 1; continue
        if d == -1 and not (not np.isnan(pvw[i]) and pclose[i] < pvw[i]):
            i += 1; continue
        a = patr[i]; line = pl[i]; thru = TICK if pess else 0.0
        if d == 1:
            fill = line + tol; pulled = (pclose[i] - line) >= rearm * a; hit = low[i] <= fill - thru
        else:
            fill = line - tol; pulled = (line - pclose[i]) >= rearm * a; hit = high[i] >= fill + thru
        if not armed and pulled:
            armed = True
        if not (armed and hit):
            i += 1; continue
        risk = stop_atr * a; slip = TICK if pess else 0.0; tthru = TICK if pess else 0.0
        tgt_from = i + 1 if pess else i
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
        R.append(pnl / risk); RC.append(risk * PV); armed = False; i = j + 1
    return np.array(R), np.array(RC)


def run_flip(A, stop_atr=1.5, start="09:30", end="10:30", slip_ticks=1):
    n = A["n"]; mins = A["mins"]; day = A["day"]; patr = A["patr"]; atr = A["atr"]
    mnd = A["mnd"]; mjd = A["mjd"]; pvw = A["pvw"]; pclose = A["pclose"]
    o = A["o"]; low = A["l"]; high = A["h"]; close = A["c"]
    long_sig = (mnd == 1) & (np.roll(mnd, 1) == -1) & (mjd == 1)
    short_sig = (mnd == -1) & (np.roll(mnd, 1) == 1) & (mjd == -1)
    R = []; RC = []; cur = day[0]; i = 1
    sl = slip_ticks * TICK
    while i < n - 1:
        if day[i] != cur:
            cur = day[i]
        if not (start <= mins[i] < end) or np.isnan(atr[i]) or atr[i] <= 0:
            i += 1; continue
        d = 1 if long_sig[i] else (-1 if short_sig[i] else 0)
        if d == 0:
            i += 1; continue
        if d == 1 and not (not np.isnan(pvw[i]) and pclose[i] > pvw[i]):
            i += 1; continue
        if d == -1 and not (not np.isnan(pvw[i]) and pclose[i] < pvw[i]):
            i += 1; continue
        entry = o[i + 1] + (sl if d == 1 else -sl)   # market entry w/ slippage
        risk = stop_atr * atr[i]
        stop = entry - risk if d == 1 else entry + risk
        j = i + 1; xp = None; last = i + 1
        while j < n and (start <= mins[j] < end) and day[j] == cur:
            if d == 1 and low[j] <= stop:
                xp = stop - sl; break
            if d == -1 and high[j] >= stop:
                xp = stop + sl; break
            if mnd[j] == -d and j > i + 1:
                xp = close[j]; break
            last = j; j += 1
        if xp is None:
            xp = close[last]; j = last
        pnl = (xp - entry) if d == 1 else (entry - xp)
        R.append(pnl / risk); RC.append(risk * PV); i = j + 1
    return np.array(R), np.array(RC)


def stat(name, R, RC, ndays, cost=10.0):
    if len(R) == 0:
        return f"{name:<28}{0:>7}"
    win = 100 * np.mean(R > 0); exp = R.mean()
    cost_r = (cost / RC).mean()
    return (f"{name:<28}{len(R):>7}{len(R)/ndays:>7.2f}{win:>9.1f}%{exp:>+8.3f}"
            f"{exp-cost_r:>+8.3f}{RC.mean():>7.0f}")


print("Building 1-min and 5-min datasets...")
A1 = build("1min")
A5 = build("5min")
hdr = f"{'config':<28}{'trds':>7}{'t/day':>7}{'accuracy':>10}{'exp_R':>8}{'netExp':>8}{'$risk':>7}"

print(f"\nLIMIT 1.5/1.5, first hour, VWAP  (costs @ $10/RT)")
print(hdr); print("-" * len(hdr))
for tf, A in (("1min", A1), ("5min", A5)):
    Ro, RCo = run_limit(A, 1.5, 1.5, pess=False)
    Rp, RCp = run_limit(A, 1.5, 1.5, pess=True)
    print(stat(f"{tf} 1.5/1.5 OPTIMISTIC", Ro, RCo, A["ndays"]))
    print(stat(f"{tf} 1.5/1.5 REALISTIC", Rp, RCp, A["ndays"]))
    print("-" * len(hdr))

print(f"\nMARKET-FLIP + ST-trail (no fill artifact), first hour, VWAP, 1-tick slip")
print(hdr); print("-" * len(hdr))
for tf, A in (("1min", A1), ("5min", A5)):
    R, RC = run_flip(A)
    print(stat(f"{tf} flip ST-trail", R, RC, A["ndays"]))
