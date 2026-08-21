"""Days profitable vs total days, with MARKET-on-touch fills + point slippage.
Ideal (touch) fills — you fill on every touch of the line — then dock realistic
point slippage (entry + stop) and $ commission. Reports the green/red DAY split.
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

avg_atr_1h = np.nanmean(atr14[(mins >= "09:30") & (mins < "10:30")])
print(f"Avg ATR(14) in 09:30-10:30 window: {avg_atr_1h:.1f} pts  ->  0.2 ATR = {0.2*avg_atr_1h:.1f} pt, 1 ATR = {avg_atr_1h:.1f} pt\n")


def sim(stop_mode, stop_val, tgt_mode, tgt_val, slip_pts, comm=5.0,
        rearm=0.4, start="09:30", end="10:30", tol=2.0, max_trades=5):
    rows = []  # (day, pnl_usd, win)
    armed = False; cur = day[0]; tdc = 0; i = 1
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
        a = patr[i]; line = pl[i]
        risk = stop_val * a if stop_mode == "atr" else stop_val
        tgtd = tgt_val * a if tgt_mode == "atr" else tgt_val
        if d == 1:
            fill = line + tol; pulled = (pclose[i] - line) >= rearm * a; hit = low[i] <= fill  # ideal touch
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
        j = i; reason = "win_close"; exitp = close[i]
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
            exitp = close[last] if 'last' in dir() else close[j-1]; j = last if 'last' in dir() else j-1
        # apply point slippage: entry worse by slip; stop exit worse by slip; target = limit (no slip)
        ent = fill + (slip_pts if d == 1 else -slip_pts)
        if reason == "target":
            ex = exitp
        else:
            ex = exitp - (slip_pts if d == 1 else -slip_pts)
        pnl_pts = (ex - ent) if d == 1 else (ent - ex)
        pnl_usd = pnl_pts * PV - comm
        rows.append((cur, pnl_usd, pnl_usd > 0))
        tdc += 1; armed = False; i = j + 1
    return pd.DataFrame(rows, columns=["day", "usd", "win"])


def report(name, tdf):
    if tdf.empty:
        print(f"{name}: no trades"); return
    ntr = len(tdf); acc = 100 * tdf["win"].mean()
    daily = tdf.groupby("day")["usd"].sum()
    green = (daily > 0).sum(); red = (daily < 0).sum(); flat = (daily == 0).sum()
    tot = len(daily)
    print(f"\n{name}")
    print(f"  trades {ntr}  |  trade win {acc:.1f}%  |  total $ {tdf['usd'].sum():+,.0f}  |  $/trade {tdf['usd'].mean():+.1f}")
    print(f"  DAYS: {tot} traded  |  GREEN {green} ({100*green/tot:.1f}%)  |  RED {red} ({100*red/tot:.1f}%)  |  flat {flat}")
    print(f"  avg green day {daily[daily>0].mean():+,.0f}  |  avg red day {daily[daily<0].mean():+,.0f}  |  avg day {daily.mean():+,.0f}")


for slip in (0.0, 1.0, 2.0):
    print("=" * 78)
    print(f"SLIPPAGE = {slip} pt/side,  commission $5/RT,  ideal touch fills")
    print("=" * 78)
    report(f"A) 1.5 ATR stop / 1.5 ATR target (1:1)", sim("atr", 1.5, "atr", 1.5, slip))
    report(f"B) 0.2 ATR stop / 1.0 ATR target (~5:1)", sim("atr", 0.2, "atr", 1.0, slip))
    report(f"C) FIXED 12 pt stop / 100 pt target (~8:1)", sim("pt", 12.0, "pt", 100.0, slip))
