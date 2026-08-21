"""Full trade-level stats for the 1.5 ATR / 1.5 ATR config (market-touch fills)."""
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


def sim(stop_mult, tgt_mult, slip_pts, comm=5.0, rearm=0.4, start="09:30", end="10:30", tol=2.0, max_trades=5):
    rows = []; armed = False; cur = day[0]; tdc = 0; i = 1
    while i < n - 1:
        if day[i] != cur:
            cur = day[i]; armed = False; tdc = 0
        if not (start <= mins[i] < end) or np.isnan(pl[i]) or np.isnan(patr[i]) or patr[i] <= 0:
            i += 1; continue
        d = pmnd[i]
        if (d == 1 and pmjd[i] != 1) or (d == -1 and pmjd[i] != -1) or d == 0:
            i += 1; continue
        if d == 1 and not (not np.isnan(pvw[i]) and pclose[i] > pvw[i]):
            i += 1; continue
        if d == -1 and not (not np.isnan(pvw[i]) and pclose[i] < pvw[i]):
            i += 1; continue
        a = patr[i]; line = pl[i]; risk = stop_mult * a; tgtd = tgt_mult * a
        if d == 1:
            fill = line + tol; pulled = (pclose[i] - line) >= rearm * a; hit = low[i] <= fill
        else:
            fill = line - tol; pulled = (line - pclose[i]) >= rearm * a; hit = high[i] >= fill
        if not armed and pulled:
            armed = True
        if tdc >= max_trades or not (armed and hit):
            i += 1; continue
        if d == 1:
            stop = fill - risk; tgt = fill + tgtd
        else:
            stop = fill + risk; tgt = fill - tgtd
        j = i; reason = "wclose"; exitp = close[i]; last = i
        while j < n and (start <= mins[j] < end) and day[j] == cur:
            if d == 1:
                if low[j] <= stop:
                    exitp, reason = stop, "stop"; break
                if high[j] >= tgt:
                    exitp, reason = tgt, "target"; break
            else:
                if high[j] >= stop:
                    exitp, reason = stop, "stop"; break
                if low[j] <= tgt:
                    exitp, reason = tgt, "target"; break
            last = j; j += 1
        else:
            exitp = close[last]; j = last
        ent = fill + (slip_pts if d == 1 else -slip_pts)
        ex = exitp if reason == "target" else exitp - (slip_pts if d == 1 else -slip_pts)
        pnl_pts = (ex - ent) if d == 1 else (ent - ex)
        usd = pnl_pts * PV - comm
        rows.append((cur, usd, usd / (risk * PV)))
        tdc += 1; armed = False; i = j + 1
    return pd.DataFrame(rows, columns=["day", "usd", "r"])


def streaks(win_bool):
    mw = ml = cw = cl = 0
    for w in win_bool:
        if w:
            cw += 1; cl = 0
        else:
            cl += 1; cw = 0
        mw = max(mw, cw); ml = max(ml, cl)
    return mw, ml


def full_stats(slip):
    t = sim(1.5, 1.5, slip)
    win = t["usd"] > 0
    ndays = t["day"].nunique()
    per_day = t.groupby("day").size()
    mw, ml = streaks(win.to_numpy())
    eq_usd = t["usd"].cumsum().to_numpy()
    dd_usd = (eq_usd - np.maximum.accumulate(eq_usd)).min()
    eq_r = t["r"].cumsum().to_numpy()
    dd_r = (eq_r - np.maximum.accumulate(eq_r)).min()
    wins = t.loc[win, "usd"]; losses = t.loc[~win, "usd"]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    print(f"\n================  1.5 ATR / 1.5 ATR (1:1)  |  slippage {slip} pt/side  ================")
    print(f"  Total trades        : {len(t)}   (wins {win.sum()}, losses {(~win).sum()})")
    print(f"  Trade win rate      : {100*win.mean():.1f}%")
    print(f"  Trading days        : {ndays}")
    print(f"  Avg trades / day    : {len(t)/ndays:.2f}   (median {int(per_day.median())}, max {per_day.max()} in a day)")
    print(f"  Max WINNING streak  : {mw} trades")
    print(f"  Max LOSING streak   : {ml} trades")
    print(f"  Avg win / avg loss  : +${wins.mean():,.0f} / -${abs(losses.mean()):,.0f}")
    print(f"  Profit factor       : {pf:.2f}")
    print(f"  Total P&L           : +${t['usd'].sum():,.0f}   ({t['r'].sum():+.1f} R)")
    print(f"  Max DRAWDOWN        : -${abs(dd_usd):,.0f}   ({dd_r:.1f} R)")
    print(f"  Expectancy / trade  : +${t['usd'].mean():.1f}   ({t['r'].mean():+.3f} R)")


for slip in (0.0, 1.0, 2.0):
    full_stats(slip)
