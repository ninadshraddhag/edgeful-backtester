"""RE-VERIFY the 0.5 ATR stop / 1 ATR target limit system.

Adds realism the first pass lacked:
  * pessimistic fills: price must trade THROUGH the limit by 1 tick to fill;
    target must trade through by 1 tick; stops slip 1 tick; no same-bar target.
  * non-anticipative sizing: stop/target sized off PRIOR bar's ATR.
  * per-year split (2023/2024/2025) to check regime robustness.
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
yr = pd.to_datetime(data["date"]).dt.year.to_numpy()
n = len(data)
pl = np.roll(mnl, 1); pl[0] = np.nan
pmnd = np.roll(mnd, 1); pmnd[0] = 0
pmjd = np.roll(mjd, 1); pmjd[0] = 0
pclose = np.roll(close, 1); pclose[0] = np.nan
pvw = np.roll(vw, 1); pvw[0] = np.nan
patr = np.roll(atr14, 1); patr[0] = np.nan


def run(mode, stop_atr, target_atr, pessimistic, tol=2.0,
        start="09:30", end="11:30", rearm=0.4, max_trades=5):
    rows = []  # (year, r_multiple, risk_cash)
    armed = False; cur = day[0]; tdc = 0; i = 1
    while i < n - 1:
        if day[i] != cur:
            cur = day[i]; tdc = 0; armed = False
        if not (start <= mins[i] < end) or np.isnan(pl[i]) or np.isnan(patr[i]) or patr[i] <= 0:
            i += 1; continue
        d = pmnd[i]
        if d == 0:
            i += 1; continue
        if mode == "r4":
            if d == 1 and not (pmjd[i] == 1 and pclose[i] > pvw[i]):
                i += 1; continue
            if d == -1 and not (pmjd[i] == -1 and pclose[i] < pvw[i]):
                i += 1; continue
        a = patr[i]; line = pl[i]
        thru = TICK if pessimistic else 0.0
        if d == 1:
            fill = line + tol
            pulled = (pclose[i] - line) >= rearm * a
            hit = low[i] <= fill - thru
        else:
            fill = line - tol
            pulled = (line - pclose[i]) >= rearm * a
            hit = high[i] >= fill + thru
        if not armed and pulled:
            armed = True
        if tdc >= max_trades or not (armed and hit):
            i += 1; continue

        risk = stop_atr * a
        slip = TICK if pessimistic else 0.0
        tthru = TICK if pessimistic else 0.0
        tgt_from = i + 1 if pessimistic else i     # no same-bar target when pessimistic
        if d == 1:
            stop = fill - risk; tgt = fill + target_atr * a
        else:
            stop = fill + risk; tgt = fill - target_atr * a
        j = i; reason = "window"; xprice = None
        while j < n and (start <= mins[j] < end) and day[j] == cur:
            if d == 1:
                if low[j] <= stop:
                    xprice, reason = stop - slip, "stop"; break
                if j >= tgt_from and high[j] >= tgt + tthru:
                    xprice, reason = tgt, "target"; break
            else:
                if high[j] >= stop:
                    xprice, reason = stop + slip, "stop"; break
                if j >= tgt_from and low[j] <= tgt - tthru:
                    xprice, reason = tgt, "target"; break
            last_valid = j
            j += 1
        if xprice is None:   # ran out of window/day -> flat at last valid close
            xprice = close[last_valid]
            j = last_valid
        pnl = (xprice - fill) if d == 1 else (fill - xprice)
        rows.append((yr[i], pnl / risk, risk * PV))
        tdc += 1; armed = False; i = j + 1
    return pd.DataFrame(rows, columns=["year", "r", "risk_cash"])


def stats(df, cost=10.0):
    n_ = len(df)
    if n_ == 0:
        return dict(n=0)
    r = df["r"].to_numpy()
    win = 100 * np.mean(r > 0)
    wsum = r[r > 0].sum(); lsum = -r[r < 0].sum()
    pf = wsum / lsum if lsum > 0 else float("inf")
    exp = r.mean()
    cost_r = (cost / df["risk_cash"]).mean()
    return dict(n=n_, win=win, exp=exp, pf=pf, cost_r=cost_r,
               net=exp - cost_r, nettot=(exp - cost_r) * n_, totr=r.sum(),
               risk=df["risk_cash"].mean())


def prow(name, s):
    if s.get("n", 0) == 0:
        return f"{name:<30} {0:>6}"
    pf = "inf" if not np.isfinite(s["pf"]) else f"{s['pf']:.2f}"
    return (f"{name:<30} {s['n']:>6} {s['win']:>6.1f}% {s['exp']:>+7.3f} {pf:>5} "
            f"${s['risk']:>5.0f} {s['cost_r']:>6.3f} {s['net']:>+7.3f} {s['nettot']:>+8.1f}")


ndays = pd.Series(day[(mins >= "09:30") & (mins < "11:30")]).nunique()
hdr = f"{'variant':<30} {'trds':>6} {'win%':>7} {'exp_R':>7} {'PF':>5} {'$risk':>6} {'cost_R':>6} {'netExp':>7} {'netTot':>8}"

print(f"\ndays: {ndays}   |   0.5 ATR stop, 1 ATR target (2R).  Costs @ $10/RT.\n")
print(hdr); print("-" * len(hdr))
for mode in ("raw", "r4"):
    opt = run(mode, 0.5, 1.0, pessimistic=False)
    pes = run(mode, 0.5, 1.0, pessimistic=True)
    print(prow(f"{mode} 0.5/1ATR  OPTIMISTIC", stats(opt)))
    print(prow(f"{mode} 0.5/1ATR  PESSIMISTIC", stats(pes)))
    print("-" * len(hdr))

print("\nPER-YEAR (pessimistic fills):\n")
print(hdr); print("-" * len(hdr))
for mode in ("raw", "r4"):
    pes = run(mode, 0.5, 1.0, pessimistic=True)
    for y in (2023, 2024, 2025):
        print(prow(f"{mode} 0.5/1ATR  {y}", stats(pes[pes['year'] == y])))
    print("-" * len(hdr))

print("\nTarget sensitivity (r4, pessimistic):\n")
print(hdr); print("-" * len(hdr))
for tatr, lbl in ((0.75, "1.5R"), (1.0, "2R"), (1.5, "3R")):
    pes = run("r4", 0.5, tatr, pessimistic=True)
    print(prow(f"r4 0.5/{tatr}ATR ({lbl})", stats(pes)))

print("\n" + "=" * 90)
print("1:1 TEST  —  1.5 ATR stop / 1.5 ATR target (NY 09:30-11:30)")
print("=" * 90)
print(hdr); print("-" * len(hdr))
for mode in ("raw", "r4"):
    opt = run(mode, 1.5, 1.5, pessimistic=False)
    pes = run(mode, 1.5, 1.5, pessimistic=True)
    print(prow(f"{mode} 1.5/1.5  OPTIMISTIC", stats(opt)))
    print(prow(f"{mode} 1.5/1.5  PESSIMISTIC", stats(pes)))
    print("-" * len(hdr))
print("\nPer-year (r4 1.5/1.5, pessimistic):")
pes = run("r4", 1.5, 1.5, pessimistic=True)
for y in (2023, 2024, 2025):
    print(prow(f"r4 1.5/1.5  {y}", stats(pes[pes['year'] == y])))
