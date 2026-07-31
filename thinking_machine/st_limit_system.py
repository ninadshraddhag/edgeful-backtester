"""Limit entry at minor ST line, TIGHT 0.2 ATR stop, far ATR target.
Tests the '40% hold x big RR = extraordinary' hypothesis as a real backtest.
R is defined as the risk = stop_atr * ATR, so a 2 ATR target on a 0.2 ATR stop = 10R.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import loader
from core import indicators

PV = 20.0


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


def vwap_of(data):
    mins = pd.to_datetime(data["date"]).dt.strftime("%H:%M")
    rth = (mins >= "09:30") & (mins < "16:00")
    tp = (data["high"] + data["low"] + data["close"]) / 3.0
    vol = data["volume"].astype(float).clip(lower=0)
    pv = (tp * vol).where(rth, 0.0); vv = vol.where(rth, 0.0)
    g = data["session_date"]
    return (pv.groupby(g).cumsum() / vv.groupby(g).cumsum().replace(0, np.nan)).to_numpy(float)


print("Loading real NQ 1-min...")
data = loader.load_data("NQ", timeframe="1min", rth_only=False)
close = data["close"].to_numpy(float); high = data["high"].to_numpy(float); low = data["low"].to_numpy(float)
mjd, _ = st_line(high, low, close, 9, 4.0)
mnd, mnl = st_line(high, low, close, 4, 1.7)
vw = vwap_of(data); atr14 = indicators.atr(data, 14).to_numpy(float)
mins = pd.to_datetime(data["date"]).dt.strftime("%H:%M").to_numpy(); day = data["session_date"].to_numpy()
n = len(data)
pl = np.roll(mnl, 1); pl[0] = np.nan
pmnd = np.roll(mnd, 1); pmnd[0] = 0
pmjd = np.roll(mjd, 1); pmjd[0] = 0
pclose = np.roll(close, 1); pclose[0] = np.nan
pvw = np.roll(vw, 1); pvw[0] = np.nan
ndays = pd.Series(day[(mins >= "09:30") & (mins < "11:30")]).nunique()


def run(mode, stop_atr, target_atr, exit_mode="fixed", tol=2.0,
        start="09:30", end="11:30", rearm=0.4, max_trades=5):
    R = []; RC = []
    armed = False; cur = day[0]; tdc = 0; i = 1
    while i < n - 1:
        if day[i] != cur:
            cur = day[i]; tdc = 0; armed = False
        if not (start <= mins[i] < end) or np.isnan(pl[i]) or np.isnan(atr14[i]) or atr14[i] <= 0:
            i += 1; continue
        d = pmnd[i]
        if d == 0:
            i += 1; continue
        if mode == "r4":
            if d == 1 and not (pmjd[i] == 1 and pclose[i] > pvw[i]):
                i += 1; continue
            if d == -1 and not (pmjd[i] == -1 and pclose[i] < pvw[i]):
                i += 1; continue
        a = atr14[i]; line = pl[i]
        if d == 1:
            fill = line + tol; pulled = (pclose[i] - line) >= rearm * a; hit = low[i] <= fill
        else:
            fill = line - tol; pulled = (line - pclose[i]) >= rearm * a; hit = high[i] >= fill
        if not armed and pulled:
            armed = True
        if tdc >= max_trades or not (armed and hit):
            i += 1; continue

        risk = stop_atr * a
        if d == 1:
            stop = fill - risk; tgt = fill + target_atr * a
        else:
            stop = fill + risk; tgt = fill - target_atr * a
        j = i; reason = "eod"; xprice = close[i]
        while j < n and (start <= mins[j] < end) and day[j] == cur:
            if d == 1:
                if low[j] <= stop: xprice, reason = stop, "stop"; break
                if exit_mode == "fixed" and high[j] >= tgt: xprice, reason = tgt, "target"; break
            else:
                if high[j] >= stop: xprice, reason = stop, "stop"; break
                if exit_mode == "fixed" and low[j] <= tgt: xprice, reason = tgt, "target"; break
            if exit_mode == "trail" and pmnd[j] == -d and j > i:
                xprice, reason = close[j], "st_flip"; break
            j += 1
        else:
            xprice, reason = close[min(j, n-1)-0] if j < n else close[n-1], "window"
        pnl = (xprice - fill) if d == 1 else (fill - xprice)
        R.append(pnl / risk); RC.append(risk * PV)
        tdc += 1; armed = False; i = j + 1
    return np.array(R), np.array(RC)


def line_out(name, R, RC, cost=10.0):
    n_ = len(R)
    if n_ == 0:
        return f"{name:<26} {0:>6}"
    win = 100 * np.mean(R > 0)
    wins = R[R > 0]; losses = R[R < 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    exp = R.mean()
    cost_r = (cost / RC).mean()
    net = exp - cost_r
    avg_risk = RC.mean()
    pf_s = "inf" if not np.isfinite(pf) else f"{pf:.2f}"
    return (f"{name:<26} {n_:>6} {n_/ndays:>5.1f} {win:>6.1f}% {exp:>+7.3f} {pf_s:>5} "
            f"{R.sum():>+8.1f} ${avg_risk:>5.0f} {cost_r:>6.3f} {net:>+7.3f} {net*n_:>+8.1f}")


hdr = (f"{'variant':<26} {'trds':>6} {'t/d':>5} {'win%':>7} {'exp_R':>7} {'PF':>5} "
       f"{'totR':>8} {'$risk':>6} {'cost_R':>6} {'netExp':>7} {'netTot':>8}")
print("\nEntry = LIMIT at minor ST line (+2pt). Stop 0.2 ATR. R = 0.2 ATR risk. Costs @ $10/RT.\n")
print(hdr); print("-" * len(hdr))
for mode in ("raw", "r4"):
    for tatr in (1.0, 2.0, 3.0):
        R, RC = run(mode, 0.2, tatr)
        print(line_out(f"{mode} stop0.2 tgt{tatr:.0f}ATR({tatr/0.2:.0f}R)", R, RC))
    R, RC = run(mode, 0.2, 99, exit_mode="trail")
    print(line_out(f"{mode} stop0.2 ST-trail", R, RC))
    print("-" * len(hdr))

print("\nContrast: wider 0.5 ATR stop (R = 0.5 ATR)\n")
print(hdr); print("-" * len(hdr))
for mode in ("raw", "r4"):
    for tatr in (1.0, 2.0):
        R, RC = run(mode, 0.5, tatr)
        print(line_out(f"{mode} stop0.5 tgt{tatr:.0f}ATR({tatr/0.5:.0f}R)", R, RC))
    R, RC = run(mode, 0.5, 99, exit_mode="trail")
    print(line_out(f"{mode} stop0.5 ST-trail", R, RC))
    print("-" * len(hdr))
