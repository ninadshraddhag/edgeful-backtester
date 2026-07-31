"""Validate: does re-arm distance = 3 ATR give ~66% accuracy?
Matches Pine defaults: 1.5/1.5 stop-target, first hour 09:30-10:30, VWAP filter.
Sweeps re-arm distance, both fill modes, on the full parquet.
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
mjd, _ = st_line(high, low, close, 9, 4.0)
mnd, mnl = st_line(high, low, close, 4, 1.7)
atr14 = indicators.atr(data, 14).to_numpy(float)
vw = anchored_vwap(data, "09:30", "16:00")
mins = pd.to_datetime(data["date"]).dt.strftime("%H:%M").to_numpy(); day = data["session_date"].to_numpy()
yr = pd.to_datetime(data["date"]).dt.year.to_numpy()
n = len(data)
pmnd = np.roll(mnd, 1); pmnd[0] = 0
pmjd = np.roll(mjd, 1); pmjd[0] = 0
pclose = np.roll(close, 1); pclose[0] = np.nan
pl = np.roll(mnl, 1); pl[0] = np.nan
patr = np.roll(atr14, 1); patr[0] = np.nan
pvw = np.roll(vw, 1); pvw[0] = np.nan


def run(stop_atr, tgt_atr, rearm, pessimistic, use_vwap=True,
        start="09:30", end="10:30", tol=2.0, max_trades=50):
    rows = []; armed = False; cur = day[0]; tdc = 0; i = 1
    while i < n - 1:
        if day[i] != cur:
            cur = day[i]; tdc = 0; armed = False
        if not (start <= mins[i] < end) or np.isnan(pl[i]) or np.isnan(patr[i]) or patr[i] <= 0:
            i += 1; continue
        d = pmnd[i]
        if d == 1 and pmjd[i] != 1:
            i += 1; continue
        if d == -1 and pmjd[i] != -1:
            i += 1; continue
        if d == 0:
            i += 1; continue
        if use_vwap:
            if d == 1 and not (not np.isnan(pvw[i]) and pclose[i] > pvw[i]):
                i += 1; continue
            if d == -1 and not (not np.isnan(pvw[i]) and pclose[i] < pvw[i]):
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
        rows.append((yr[i], pnl / risk, risk * PV))
        tdc += 1; armed = False; i = j + 1
    return pd.DataFrame(rows, columns=["year", "r", "risk_cash"])


def line_out(name, df, cost=10.0):
    nn = len(df)
    if nn == 0:
        return f"{name:<30}{0:>7}"
    r = df["r"].to_numpy()
    win = 100 * np.mean(r > 0)
    cost_r = (cost / df["risk_cash"]).mean()
    exp = r.mean()
    return (f"{name:<30}{nn:>7}{nn/765:>7.2f}{win:>8.1f}%{exp:>+8.3f}"
            f"{exp-cost_r:>+8.3f}{r.sum():>+9.1f}")


hdr = f"{'config (1.5/1.5, 1st hr, VWAP)':<30}{'trds':>7}{'t/day':>7}{'accuracy':>9}{'exp_R':>8}{'netExp':>8}{'totR':>9}"
print("\n" + hdr); print("-" * len(hdr))
for rearm in (0.4, 1.0, 2.0, 3.0, 4.0):
    opt = run(1.5, 1.5, rearm, pessimistic=False)
    print(line_out(f"rearm {rearm}  OPTIMISTIC(touch)", opt))
    pes = run(1.5, 1.5, rearm, pessimistic=True)
    print(line_out(f"rearm {rearm}  REALISTIC(through)", pes))
    print("-" * len(hdr))

print("\nrearm=3 detail — per year (accuracy | trades):")
for tag, pess in (("OPTIMISTIC", False), ("REALISTIC", True)):
    df = run(1.5, 1.5, 3.0, pessimistic=pess)
    parts = []
    for y in (2023, 2024, 2025):
        g = df[df["year"] == y]
        wr = 100 * np.mean(g["r"].to_numpy() > 0) if len(g) else 0
        parts.append(f"{y}: {wr:.0f}% (n={len(g)})")
    print(f"  {tag:<11} " + "   ".join(parts))
