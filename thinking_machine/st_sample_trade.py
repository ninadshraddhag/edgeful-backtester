"""Extract ONE real R4 sample trade with full indicator context for plotting."""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import loader
from core import indicators

POINT_VALUE = 20.0


def supertrend_line(high, low, close, period, mult):
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    atr = indicators.atr(df, period).to_numpy(float)
    hl2 = (high + low) / 2.0
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    n = len(close)
    fu = np.full(n, np.nan); fl = np.full(n, np.nan)
    dir_ = np.ones(n, dtype=int); line = np.full(n, np.nan)
    for i in range(n):
        if i == 0 or np.isnan(atr[i]):
            fu[i] = upper[i]; fl[i] = lower[i]; dir_[i] = 1; line[i] = fl[i]; continue
        fu[i] = upper[i] if (upper[i] < fu[i-1] or close[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = lower[i] if (lower[i] > fl[i-1] or close[i-1] < fl[i-1]) else fl[i-1]
        if close[i] > fu[i-1]:
            dir_[i] = 1
        elif close[i] < fl[i-1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i-1]
        line[i] = fl[i] if dir_[i] == 1 else fu[i]
    return dir_, line


def session_vwap(data):
    mins = pd.to_datetime(data["date"]).dt.strftime("%H:%M")
    rth = (mins >= "09:30") & (mins < "16:00")
    tp = (data["high"] + data["low"] + data["close"]) / 3.0
    vol = data["volume"].astype(float).clip(lower=0)
    pv = (tp * vol).where(rth, 0.0); vv = vol.where(rth, 0.0)
    g = data["session_date"]
    return (pv.groupby(g).cumsum() / vv.groupby(g).cumsum().replace(0, np.nan)).to_numpy(float)


data = loader.load_data("NQ", timeframe="1min", rth_only=False)
close = data["close"].to_numpy(float); high = data["high"].to_numpy(float); low = data["low"].to_numpy(float)
mjd, mjl = supertrend_line(high, low, close, 9, 4.0)
mnd, mnl = supertrend_line(high, low, close, 4, 1.7)
vwap = session_vwap(data)
risk_atr = indicators.atr(data, 14).to_numpy(float)
mins = pd.to_datetime(data["date"]).dt.strftime("%H:%M").to_numpy()
day = data["session_date"].to_numpy()
o = data["open"].to_numpy(float); ts = data["date"]
n = len(data)

long_sig = (mnd == 1) & (np.roll(mnd, 1) == -1) & (mjd == 1) & (close > vwap)
short_sig = (mnd == -1) & (np.roll(mnd, 1) == 1) & (mjd == -1) & (close < vwap)
long_sig[0] = short_sig[0] = False

# find R4 trades, keep clean winners of moderate length that exit on ST flip
cands = []
i = 1; trades_today = 0; cur = day[0]
while i < n - 1:
    if day[i] != cur:
        cur = day[i]; trades_today = 0
    if not (mins[i] >= "09:30" and mins[i] < "10:30") or trades_today >= 4 or np.isnan(risk_atr[i]) or risk_atr[i] <= 0:
        i += 1; continue
    side = "long" if long_sig[i] else ("short" if short_sig[i] else None)
    if side is None:
        i += 1; continue
    ei = i + 1; entry = o[ei]; risk = 1.5 * risk_atr[i]
    stop = entry - risk if side == "long" else entry + risk
    j = ei; reason = "eod"; exit_price = close[ei]; exit_idx = ei
    while j < n:
        endw = (mins[j] >= "10:30") or (j == n-1) or (day[j+1] != day[j])
        if side == "long":
            if low[j] <= stop: exit_idx, exit_price, reason = j, stop, "stop"; break
            if mnd[j] == -1 and j > ei: exit_idx, exit_price, reason = j, close[j], "st_flip"; break
        else:
            if high[j] >= stop: exit_idx, exit_price, reason = j, stop, "stop"; break
            if mnd[j] == 1 and j > ei: exit_idx, exit_price, reason = j, close[j], "st_flip"; break
        if endw: exit_idx, exit_price, reason = j, close[j], "window_close"; break
        j += 1
    pnl = (exit_price - entry) if side == "long" else (entry - exit_price)
    rmult = pnl / risk
    held = exit_idx - ei
    cands.append((i, ei, exit_idx, side, entry, stop, exit_price, rmult, reason, held))
    trades_today += 1
    i = exit_idx + 1

# pick a readable winner: st_flip exit, +1.8..3.5R, 12..35 bars held
best = None
for c in cands:
    _, _, _, side, _, _, _, rmult, reason, held = c
    if reason == "st_flip" and 1.8 <= rmult <= 3.5 and 12 <= held <= 35:
        best = c; break
if best is None:
    best = max(cands, key=lambda c: c[7])

sig_i, ei, xi, side, entry, stop, exitp, rmult, reason, held = best
lo_b = max(0, sig_i - 6); hi_b = min(n - 1, xi + 4)
bars = []
for k in range(lo_b, hi_b + 1):
    bars.append({
        "t": pd.Timestamp(ts.iloc[k]).strftime("%H:%M"),
        "o": round(o[k], 2), "h": round(high[k], 2), "l": round(low[k], 2), "c": round(close[k], 2),
        "mjl": None if np.isnan(mjl[k]) else round(mjl[k], 2),
        "mnl": None if np.isnan(mnl[k]) else round(mnl[k], 2),
        "vwap": None if np.isnan(vwap[k]) else round(vwap[k], 2),
        "mjd": int(mjd[k]), "mnd": int(mnd[k]),
        "sig": k == sig_i, "entry": k == ei, "exit": k == xi,
    })
out = {
    "date": pd.Timestamp(ts.iloc[ei]).strftime("%Y-%m-%d"),
    "side": side, "entry": round(entry, 2), "stop": round(stop, 2), "exit": round(exitp, 2),
    "target_note": "no fixed target",
    "risk_pts": round(1.5 * risk_atr[sig_i], 2), "rmult": round(rmult, 2), "reason": reason,
    "held": held, "sig_time": bars[sig_i - lo_b]["t"],
    "entry_time": pd.Timestamp(ts.iloc[ei]).strftime("%H:%M"),
    "exit_time": pd.Timestamp(ts.iloc[xi]).strftime("%H:%M"),
    "bars": bars,
}
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_trade.json")
with open(p, "w") as f:
    json.dump(out, f, indent=2)
print(f"{side.upper()} {out['date']}  entry {entry:.2f} @ {out['entry_time']}  "
      f"stop {stop:.2f}  exit {exitp:.2f} @ {out['exit_time']} ({reason})  "
      f"{rmult:+.2f}R  held {held} bars  risk {out['risk_pts']}pts")
print("wrote", p)
