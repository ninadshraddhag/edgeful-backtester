"""Minor-Supertrend(4,1.7) as a LIMIT entry level.

Question: rest a limit at the minor ST line (+/- tolerance). When price pulls
back and fills, how often does it 'bounce' cleanly = adverse excursion beyond
the fill stays <= 0.2 ATR (price reaches +0.2 ATR in-trend before -0.2 ATR)?

Non-anticipative: the limit level for bar i is the PRIOR bar's ST value.
"""
from __future__ import annotations
import os, sys
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


print("Loading real NQ 1-min...")
data = loader.load_data("NQ", timeframe="1min", rth_only=False)
close = data["close"].to_numpy(float); high = data["high"].to_numpy(float); low = data["low"].to_numpy(float)
mjd, _ = supertrend_line(high, low, close, 9, 4.0)
mnd, mnl = supertrend_line(high, low, close, 4, 1.7)
vwap = session_vwap(data)
atr14 = indicators.atr(data, 14).to_numpy(float)
mins = pd.to_datetime(data["date"]).dt.strftime("%H:%M").to_numpy()
day = data["session_date"].to_numpy()
n = len(data)

# prior-bar values (what you know when resting the order for bar i)
pl = np.roll(mnl, 1); pl[0] = np.nan
pmnd = np.roll(mnd, 1); pmnd[0] = 0
pmjd = np.roll(mjd, 1); pmjd[0] = 0
pclose = np.roll(close, 1); pclose[0] = np.nan
pvwap = np.roll(vwap, 1); pvwap[0] = np.nan


def study(mode, tol, favm=0.2, advm=0.2, start="09:30", end="11:30", rearm=0.4):
    touches = wins = fails = flips = timeouts = 0
    mae_atr = []
    reach = {0.5: 0, 1.0: 0, 2.0: 0}       # follow-through of winners (max fav to flip, in ATR)
    armed = False; cur = day[0]; i = 1
    while i < n - 1:
        if day[i] != cur:
            cur = day[i]; armed = False
        if not (start <= mins[i] < end) or np.isnan(pl[i]) or np.isnan(atr14[i]) or atr14[i] <= 0:
            i += 1; continue
        d = pmnd[i]
        if d == 0:
            i += 1; continue
        if mode == "r4":
            if d == 1 and not (pmjd[i] == 1 and pclose[i] > pvwap[i]):
                i += 1; continue
            if d == -1 and not (pmjd[i] == -1 and pclose[i] < pvwap[i]):
                i += 1; continue
        a = atr14[i]; line = pl[i]
        if d == 1:
            fill = line + tol
            pulled = (pclose[i] - line) >= rearm * a
            hit = low[i] <= fill
        else:
            fill = line - tol
            pulled = (line - pclose[i]) >= rearm * a
            hit = high[i] >= fill
        if not armed and pulled:
            armed = True
        if not (armed and hit):
            i += 1; continue

        # FILL -> resolve first passage
        if d == 1:
            adv, fav = fill - advm * a, fill + favm * a
        else:
            adv, fav = fill + advm * a, fill - favm * a
        res = "timeout"; mae = 0.0; j = i
        while j < n and (start <= mins[j] < end) and day[j] == cur:
            if d == 1:
                mae = max(mae, fill - low[j])
                hit_adv, hit_fav = low[j] <= adv, high[j] >= fav
            else:
                mae = max(mae, high[j] - fill)
                hit_adv, hit_fav = high[j] >= adv, low[j] <= fav
            if hit_adv:
                res = "fail"; break
            if hit_fav:
                res = "win"; break
            if pmnd[j] == -d and j > i:
                res = "flip"; break
            j += 1

        touches += 1
        mae_atr.append(mae / a)
        if res == "win":
            wins += 1
            mx = 0.0; k = j
            while k < n and (start <= mins[k] < end) and day[k] == cur:
                mx = max(mx, (high[k] - fill) if d == 1 else (fill - low[k]))
                if pmnd[k] == -d and k > j:
                    break
                k += 1
            for thr in reach:
                if mx / a >= thr:
                    reach[thr] += 1
        elif res == "fail":
            fails += 1
        elif res == "flip":
            flips += 1
        else:
            timeouts += 1
        armed = False
        i = (j + 1) if res in ("fail", "win", "flip") else (i + 1)

    return dict(touches=touches, wins=wins, fails=fails, flips=flips, timeouts=timeouts,
                mae_atr=np.array(mae_atr), reach=reach)


ndays = pd.Series(day[(mins >= "09:30") & (mins < "11:30")]).nunique()
print(f"days in 09:30-11:30 window: {ndays}\n")

hdr = f"{'mode':<6} {'tol':>4} {'touches':>8} {'t/day':>6} {'clean%':>7} {'fail%':>6} {'flip%':>6} {'MAE<=0.2%':>10}"
print(hdr); print("-" * len(hdr))
for mode in ("raw", "r4"):
    for tol in (0.0, 2.0, 5.0):
        r = study(mode, tol)
        t = r["touches"]
        if t == 0:
            print(f"{mode:<6} {tol:>4.0f} {0:>8}"); continue
        cw = 100 * r["wins"] / t
        cf = 100 * r["fails"] / t
        cfl = 100 * r["flips"] / t
        mae_ok = 100 * np.mean(r["mae_atr"] <= 0.2)
        print(f"{mode:<6} {tol:>4.0f} {t:>8} {t/ndays:>6.2f} {cw:>6.1f}% {cf:>5.1f}% {cfl:>5.1f}% {mae_ok:>9.1f}%")

print("\nFollow-through of CLEAN bounces (raw, tol=2): how far winners run before minor ST flips")
r = study("raw", 2.0)
w = r["wins"]
print(f"  clean bounces: {w}")
for thr, cnt in r["reach"].items():
    print(f"    reached +{thr:>3} ATR in-trend: {cnt:>5}  ({100*cnt/w:.1f}% of winners)")
print(f"\n  MAE distribution across ALL fills (raw tol=2):  median {np.median(r['mae_atr']):.3f} ATR"
      f"  |  75th pct {np.percentile(r['mae_atr'],75):.3f}  |  90th {np.percentile(r['mae_atr'],90):.3f} ATR")

print("\nSURVIVAL CURVE: % of limit fills that hold (MAE <= stop) vs stop distance")
r4r = study("r4", 2.0)
print(f"  {'stop(ATR)':>9} {'raw hold%':>10} {'r4 hold%':>10}")
for thr in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
    raw_h = 100 * np.mean(r["mae_atr"] <= thr)
    r4_h = 100 * np.mean(r4r["mae_atr"] <= thr)
    print(f"  {thr:>9.2f} {raw_h:>9.1f}% {r4_h:>9.1f}%")
