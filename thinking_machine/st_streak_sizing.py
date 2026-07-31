"""Does streak-based position sizing help the 80%-accuracy config?
Extracts the real ordered trade sequence, tests for win autocorrelation, and
simulates anti-martingale (press-after-win / bank-after-buffer) vs flat sizing.
Config: stop 2.0 ATR / target 0.3 ATR / rearm 3.0, first hour, VWAP, realistic fills.
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
rng = np.random.default_rng(42)


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


print("Loading real NQ 1-min...")
data = loader.load_data("NQ", timeframe="1min", rth_only=False)
close = data["close"].to_numpy(float); high = data["high"].to_numpy(float); low = data["low"].to_numpy(float)
mjd, _ = st_line(high, low, close, 9, 4.0)
mnd, mnl = st_line(high, low, close, 4, 1.7)
atr14 = indicators.atr(data, 14).to_numpy(float)
vw = anchored_vwap(data, "09:30", "16:00")
mins = pd.to_datetime(data["date"]).dt.strftime("%H:%M").to_numpy(); day = data["session_date"].to_numpy()
n = len(data)
pmnd = np.roll(mnd, 1); pmnd[0] = 0
pmjd = np.roll(mjd, 1); pmjd[0] = 0
pclose = np.roll(close, 1); pclose[0] = np.nan
pl = np.roll(mnl, 1); pl[0] = np.nan
patr = np.roll(atr14, 1); patr[0] = np.nan
pvw = np.roll(vw, 1); pvw[0] = np.nan


def trade_sequence(stop_atr=2.0, tgt_atr=0.3, rearm=3.0, start="09:30", end="10:30", tol=2.0):
    R = []; armed = False; cur = day[0]; i = 1
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
        a = patr[i]; line = pl[i]
        if d == 1:
            fill = line + tol; pulled = (pclose[i] - line) >= rearm * a; hit = low[i] <= fill - TICK
        else:
            fill = line - tol; pulled = (line - pclose[i]) >= rearm * a; hit = high[i] >= fill + TICK
        if not armed and pulled:
            armed = True
        if not (armed and hit):
            i += 1; continue
        risk = stop_atr * a
        if d == 1:
            stop = fill - risk; tgt = fill + tgt_atr * a
        else:
            stop = fill + risk; tgt = fill - tgt_atr * a
        j = i; xp = None; last = i
        while j < n and (start <= mins[j] < end) and day[j] == cur:
            if d == 1:
                if low[j] <= stop:
                    xp = stop - TICK; break
                if j > i and high[j] >= tgt + TICK:
                    xp = tgt; break
            else:
                if high[j] >= stop:
                    xp = stop + TICK; break
                if j > i and low[j] <= tgt - TICK:
                    xp = tgt; break
            last = j; j += 1
        if xp is None:
            xp = close[last]; j = last
        pnl = (xp - fill) if d == 1 else (fill - xp)
        R.append(pnl / risk)
        armed = False; i = j + 1
    return np.array(R)


R = trade_sequence()
wins = R > 0
print(f"\nMax-accuracy config trade sequence: {len(R)} trades, {100*wins.mean():.1f}% win, flat total = {R.sum():+.1f}R")

# --- win autocorrelation: is a streak more than luck? ---
prev = wins[:-1]; nxt = wins[1:]
p_ww = nxt[prev].mean() * 100
p_wl = nxt[~prev].mean() * 100
base = wins.mean() * 100
print(f"\nWin autocorrelation:")
print(f"  base win rate          : {base:.1f}%")
print(f"  P(win | previous WIN)  : {p_ww:.1f}%")
print(f"  P(win | previous LOSS) : {p_wl:.1f}%")
print(f"  -> persistence edge    : {p_ww - base:+.1f} pts (0 = streaks are pure luck)")

# --- sizing simulation ---
def simulate(Rseq, up=1.0, cap=1.0, bank=0):
    size = 1.0; eq = 0.0; peak = 0.0; maxdd = 0.0; ws = 0
    for r in Rseq:
        eq += size * r
        peak = max(peak, eq); maxdd = min(maxdd, eq - peak)
        if r > 0:
            ws += 1
            size = min(size * up, cap)
            if bank and ws >= bank:
                size = 1.0; ws = 0
        else:
            size = 1.0; ws = 0
    return eq, maxdd

print(f"\nSizing schemes on the REAL sequence (units = base risk R):")
print(f"  {'scheme':<34}{'final R':>10}{'maxDD R':>10}")
schemes = [
    ("flat (1 unit)", 1.0, 1.0, 0),
    ("double after win, cap 4", 2.0, 4.0, 0),
    ("double after win, cap 8", 2.0, 8.0, 0),
    ("double, cap 8, bank after 3", 2.0, 8.0, 3),
    ("triple after win, cap 9", 3.0, 9.0, 0),
]
for name, up, cap, bank in schemes:
    eq, dd = simulate(R, up, cap, bank)
    print(f"  {name:<34}{eq:>+10.1f}{dd:>+10.1f}")

# --- Monte Carlo: shuffle order, does anti-martingale beat flat by skill or luck? ---
flat_final = R.sum()
M = 5000
beat = 0; finals = np.empty(M)
for k in range(M):
    sh = rng.permutation(R)
    eq, _ = simulate(sh, up=2.0, cap=8.0, bank=0)
    finals[k] = eq
    if eq > flat_final:
        beat += 1
print(f"\nMonte Carlo ({M} shuffles) of 'double-cap8' anti-martingale:")
print(f"  mean final           : {finals.mean():+.1f}R   (flat = {flat_final:+.1f}R)")
print(f"  std of final          : {finals.std():.1f}R   <- variance explosion")
print(f"  % of shuffles beating flat: {100*beat/M:.1f}%")
