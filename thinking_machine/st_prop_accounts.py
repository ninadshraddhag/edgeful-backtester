"""Prop-account pass rate: reach +$3000 before -$2000, $400 risk/trade, 1:1.
Models the account as a biased random walk (trades are independent -> the
'stop on loss / continue on win' session rule doesn't change pass/blow odds).
Uses the REAL ideal-fill AND realistic-fill 1:1 trade distributions.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import loader
from core import indicators

TICK = 0.25
RISK_USD = 400.0
TARGET = 3000.0
MAXLOSS = -2000.0
N_ACC = 5
rng = np.random.default_rng(7)


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


def seq(pessimistic, stop_atr=1.5, tgt_atr=1.5, rearm=0.4, start="09:30", end="10:30", tol=2.0):
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
        a = patr[i]; line = pl[i]; thru = TICK if pessimistic else 0.0
        if d == 1:
            fill = line + tol; pulled = (pclose[i] - line) >= rearm * a; hit = low[i] <= fill - thru
        else:
            fill = line - tol; pulled = (line - pclose[i]) >= rearm * a; hit = high[i] >= fill + thru
        if not armed and pulled:
            armed = True
        if not (armed and hit):
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
        R.append(pnl / risk); armed = False; i = j + 1
    return np.array(R)


def simulate_accounts(R, n_mc=50000):
    pnl = R * RISK_USD           # $ per trade (win ~ +400, loss ~ -400)
    passes = 0; term = np.empty(n_mc); ntr = np.empty(n_mc)
    for k in range(n_mc):
        bal = 0.0; t = 0
        while bal < TARGET and bal > MAXLOSS:
            bal += pnl[rng.integers(len(pnl))]
            t += 1
            if t > 2000:
                break
        term[k] = bal; ntr[k] = t
        if bal >= TARGET:
            passes += 1
    return passes / n_mc, term, ntr


for tag, pess in (("IDEAL fills (touch)", False), ("REALISTIC fills (trade-through)", True)):
    R = seq(pess)
    wr = 100 * np.mean(R > 0)
    p_pass, term, ntr = simulate_accounts(R)
    exp_pass = N_ACC * p_pass
    # distribution of how many of 5 accounts pass (binomial)
    from math import comb
    dist = [comb(N_ACC, k) * p_pass**k * (1 - p_pass)**(N_ACC - k) for k in range(N_ACC + 1)]
    exp_dollars_1 = term.mean()
    print(f"\n===== {tag} =====")
    print(f"  per-trade win rate        : {wr:.1f}%   (trades in sample: {len(R)})")
    print(f"  P(account PASSES +$3000)  : {p_pass*100:.1f}%")
    print(f"  P(account BLOWS -$2000)   : {(1-p_pass)*100:.1f}%")
    print(f"  avg trades to resolution  : {ntr.mean():.0f}")
    print(f"  expected passes out of 5  : {exp_pass:.2f} accounts")
    print(f"  E[$ per account]          : {exp_dollars_1:+,.0f}")
    print(f"  E[$ across 5 accounts]     : {5*exp_dollars_1:+,.0f}")
    print(f"  P(k of 5 pass):  " + "  ".join(f"{k}:{dist[k]*100:.0f}%" for k in range(6)))
    print(f"  P(at least 1 passes)      : {(1-dist[0])*100:.1f}%")
