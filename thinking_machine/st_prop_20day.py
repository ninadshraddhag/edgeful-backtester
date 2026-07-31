"""Prop pass rate with a 20-TRADING-DAY deadline.
Rules: $400 risk/trade, 1:1, target +$3000, blow at -$2000.
Per day (first hour): take trades in order, continue on a win, STOP on the first
loss. Accumulate across up to 20 active trading days. Pass = reach +$3000 in time.
Bootstraps whole real trading days to keep trades-per-day realistic.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import loader
from core import indicators

TICK = 0.25
RISK_USD = 400.0
TARGET = 3000.0
MAXLOSS = -2000.0
DAYS_LIMIT = 20
N_ACC = 5
rng = np.random.default_rng(11)


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


def seq_days(pessimistic, stop_atr=1.5, tgt_atr=1.5, rearm=0.4, start="09:30", end="10:30", tol=2.0):
    R = []; DAY = []; armed = False; cur = day[0]; i = 1
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
        R.append(pnl / risk); DAY.append(cur); armed = False; i = j + 1
    # group into per-day ordered sequences (only days with >=1 trade)
    groups = {}
    for r, dd in zip(R, DAY):
        groups.setdefault(dd, []).append(r * RISK_USD)
    return [np.array(v) for v in groups.values()], np.array(R)


def run_account(day_seqs, n_days):
    idx = rng.integers(len(day_seqs), size=n_days)
    bal = 0.0; taken = 0
    for di in idx:
        for pnl in day_seqs[di]:
            bal += pnl; taken += 1
            if bal >= TARGET:
                return "pass", bal, taken
            if bal <= MAXLOSS:
                return "blow", bal, taken
            if pnl < 0:      # stop on the first loss of the day
                break
    return "timeout", bal, taken


def mc(day_seqs, n_mc=50000):
    res = {"pass": 0, "blow": 0, "timeout": 0}; bals = []
    for _ in range(n_mc):
        r, bal, _ = run_account(day_seqs, DAYS_LIMIT)
        res[r] += 1; bals.append(bal)
    return res, np.array(bals)


for tag, pess in (("IDEAL fills (touch)", False), ("REALISTIC fills (trade-through)", True)):
    day_seqs, R = seq_days(pess)
    wr = 100 * np.mean(R > 0)
    setpd = np.mean([len(s) for s in day_seqs])
    res, bals = mc(day_seqs)
    tot = sum(res.values())
    p_pass = res["pass"] / tot
    dist = [comb(N_ACC, k) * p_pass**k * (1 - p_pass)**(N_ACC - k) for k in range(N_ACC + 1)]
    print(f"\n===== {tag}  ({DAYS_LIMIT}-day deadline) =====")
    print(f"  per-trade win rate         : {wr:.1f}%   |  active days: {len(day_seqs)}  |  avg setups/day: {setpd:.2f}")
    print(f"  PASS (+$3000 in <=20 days) : {p_pass*100:.1f}%")
    print(f"  BLOW (-$2000)              : {res['blow']/tot*100:.1f}%")
    print(f"  TIMEOUT (ran out of days)  : {res['timeout']/tot*100:.1f}%")
    print(f"  expected passes of 5       : {N_ACC*p_pass:.2f} accounts")
    print(f"  E[$ per account @ day 20]  : {bals.mean():+,.0f}")
    print(f"  E[$ across 5 accounts]      : {5*bals.mean():+,.0f}")
    print(f"  P(k of 5 pass): " + "  ".join(f"{k}:{dist[k]*100:.0f}%" for k in range(6)))
