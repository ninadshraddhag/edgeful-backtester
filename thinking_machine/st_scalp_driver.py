"""Dual-Supertrend 9:30-11:30 NQ scalping driver.

Major ST (9,4) = trend bias; minor ST (4,1.7) = entry trigger.
Trend-following pullback: enter on a minor-ST flip in the direction of major bias.
Runs on REAL NQ 1-min parquet. Reports R-normalized metrics (GROSS of costs).
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import loader
from core import indicators, metrics as metrics_mod

POINT_VALUE = 20.0  # NQ


# ---------------------------------------------------------------- Supertrend
def supertrend(high, low, close, period, mult):
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    atr = indicators.atr(df, period).to_numpy(float)
    hl2 = (high + low) / 2.0
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    n = len(close)
    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    dir_ = np.ones(n, dtype=int)
    for i in range(n):
        if i == 0 or np.isnan(atr[i]):
            fu[i] = upper[i]; fl[i] = lower[i]; dir_[i] = 1
            continue
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if close[i] > fu[i - 1]:
            dir_[i] = 1
        elif close[i] < fl[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
    return dir_


# ---------------------------------------------------------------- Backtest
def session_vwap(data):
    """RTH-anchored VWAP: resets each day at 09:30, typical-price weighted."""
    mins = pd.to_datetime(data["date"]).dt.strftime("%H:%M")
    rth = (mins >= "09:30") & (mins < "16:00")
    tp = (data["high"] + data["low"] + data["close"]) / 3.0
    vol = data["volume"].astype(float).clip(lower=0)
    pv = (tp * vol).where(rth, 0.0)
    vv = vol.where(rth, 0.0)
    g = data["session_date"]
    cum_pv = pv.groupby(g).cumsum()
    cum_vv = vv.groupby(g).cumsum()
    vwap = cum_pv / cum_vv.replace(0, np.nan)
    return vwap.to_numpy(float)


def run_variant(data, major_dir, minor_dir, risk_atr, exit_mode, stop_atr, rr,
                start="09:30", end="11:30", max_trades=4,
                trail_dir=None, extra_long=None, extra_short=None):
    o = data["open"].to_numpy(float)
    h = data["high"].to_numpy(float)
    l = data["low"].to_numpy(float)
    c = data["close"].to_numpy(float)
    ts = data["date"].to_numpy()
    day = data["session_date"].to_numpy()
    mins = pd.to_datetime(data["date"]).dt.strftime("%H:%M").to_numpy()
    n = len(data)

    if trail_dir is None:
        trail_dir = minor_dir
    if extra_long is None:
        extra_long = np.ones(n, dtype=bool)
    if extra_short is None:
        extra_short = np.ones(n, dtype=bool)

    # minor-ST flips in direction of major bias
    long_sig = (minor_dir == 1) & (np.roll(minor_dir, 1) == -1) & (major_dir == 1) & extra_long
    short_sig = (minor_dir == -1) & (np.roll(minor_dir, 1) == 1) & (major_dir == -1) & extra_short
    long_sig[0] = short_sig[0] = False

    trades = []
    i = 1
    trades_today = 0
    cur_day = day[0]
    while i < n - 1:
        if day[i] != cur_day:
            cur_day = day[i]; trades_today = 0
        in_win = (mins[i] >= start) and (mins[i] < end)
        if not in_win or trades_today >= max_trades or np.isnan(risk_atr[i]) or risk_atr[i] <= 0:
            i += 1; continue

        side = "long" if long_sig[i] else ("short" if short_sig[i] else None)
        if side is None:
            i += 1; continue

        entry_idx = i + 1
        entry = o[entry_idx]
        risk = stop_atr * risk_atr[i]
        if side == "long":
            stop = entry - risk; target = entry + rr * risk
        else:
            stop = entry + risk; target = entry - rr * risk

        exit_idx, exit_price, reason = _exit(
            side, entry_idx, stop, target, h, l, c, day, mins, n, end,
            exit_mode, trail_dir)

        pnl = (exit_price - entry) if side == "long" else (entry - exit_price)
        trades.append({
            "entry_time": pd.Timestamp(ts[entry_idx]),
            "exit_time": pd.Timestamp(ts[exit_idx]),
            "side": side, "entry": entry, "exit": exit_price,
            "pnl_points": pnl, "pnl_cash": pnl * POINT_VALUE,
            "r_multiple": pnl / risk if risk else 0.0, "reason": reason,
            "risk_pts": risk, "risk_cash": risk * POINT_VALUE,
        })
        trades_today += 1
        i = exit_idx + 1
    return pd.DataFrame(trades)


def _exit(side, entry_idx, stop, target, h, l, c, day, mins, n, end, exit_mode, trail_dir):
    j = entry_idx
    while j < n:
        end_win = (mins[j] >= end) or (j == n - 1) or (day[j + 1] != day[j])
        hi, lo = h[j], l[j]
        if side == "long":
            hit_stop, hit_tgt = lo <= stop, hi >= target
        else:
            hit_stop, hit_tgt = hi >= stop, lo <= target
        if hit_stop:                       # conservative: stop before target
            return j, stop, "stop"
        if exit_mode == "fixed" and hit_tgt:
            return j, target, "target"
        if exit_mode == "st_trail":        # exit when trail ST flips against us
            flipped = (trail_dir[j] == -1) if side == "long" else (trail_dir[j] == 1)
            if flipped and j > entry_idx:
                return j, c[j], "st_flip"
        if end_win:
            return j, c[j], "window_close"
        j += 1
    return n - 1, c[n - 1], "eod"


def summarize(name, tdf, n_days):
    m = metrics_mod.compute_metrics(tdf)
    tpd = m["trades"] / n_days if n_days else 0
    pf = m["profit_factor"]; pf = "inf" if pf == float("inf") else f"{pf:.2f}"
    rr = m["avg_rr"]; rr = "inf" if rr == float("inf") else f"{rr:.2f}"
    return (f"{name:<22} {m['trades']:>5} {tpd:>5.1f} {m['win_rate']:>6.1f}% "
            f"{rr:>6} {m['expectancy_r']:>+7.3f} {pf:>6} {m['total_r']:>+8.1f} "
            f"{m['max_drawdown_r']:>+8.1f}")


def main():
    print("Loading real NQ 1-min parquet (continuous, for ST warm-up)...")
    data = loader.load_data("NQ", timeframe="1min", rth_only=False)
    print(f"  rows={len(data):,}  range={data['date'].min()} -> {data['date'].max()}")

    close = data["close"].to_numpy(float)
    high = data["high"].to_numpy(float)
    low = data["low"].to_numpy(float)

    print("Computing Supertrends (major 9x4, minor 4x1.7) + risk ATR(14)...")
    major_dir = supertrend(high, low, close, 9, 4.0)
    minor_dir = supertrend(high, low, close, 4, 1.7)
    risk_atr = indicators.atr(data, 14).to_numpy(float)

    # trading-day count inside window for trades/day
    mins = pd.to_datetime(data["date"]).dt.strftime("%H:%M")
    win_mask = (mins >= "09:30") & (mins < "11:30")
    n_days = data.loc[win_mask.to_numpy(), "session_date"].nunique()
    print(f"  trading days in 09:30-11:30 window: {n_days}\n")

    hdr = (f"{'variant':<22} {'trds':>5} {'t/d':>5} {'win%':>7} {'rr':>6} "
           f"{'exp_R':>7} {'PF':>6} {'totR':>8} {'maxDD':>8}")
    print(hdr); print("-" * len(hdr))

    # Accuracy <-> RR frontier: fixed targets, stop = 1.0x ATR14
    for rr in (0.75, 1.0, 1.5, 2.0, 3.0):
        tdf = run_variant(data, major_dir, minor_dir, risk_atr,
                          exit_mode="fixed", stop_atr=1.0, rr=rr)
        print(summarize(f"fixed 1.0stop {rr}R", tdf, n_days))

    # Tighter stop (0.6x ATR) for scalpy high-accuracy
    for rr in (0.75, 1.0, 1.5):
        tdf = run_variant(data, major_dir, minor_dir, risk_atr,
                          exit_mode="fixed", stop_atr=0.6, rr=rr)
        print(summarize(f"fixed 0.6stop {rr}R", tdf, n_days))

    # Supertrend-trailing exit (let minor ST manage), stop 1.0x ATR
    tdf = run_variant(data, major_dir, minor_dir, risk_atr,
                      exit_mode="st_trail", stop_atr=1.0, rr=99)
    print(summarize("st_trail 1.0stop", tdf, n_days))
    tdf = run_variant(data, major_dir, minor_dir, risk_atr,
                      exit_mode="st_trail", stop_atr=1.5, rr=99)
    print(summarize("st_trail 1.5stop", tdf, n_days))

    # ---- REFINEMENTS: the edge is trend-riding, so filter for real trends ----
    print("\n" + "REFINEMENTS".center(len(hdr), "."))
    print(hdr); print("-" * len(hdr))
    vwap = session_vwap(data)
    above_vwap = close > vwap
    below_vwap = close < vwap

    # R1: st_trail(minor) + require price on correct side of session VWAP
    tdf = run_variant(data, major_dir, minor_dir, risk_atr, "st_trail", 1.5, 99,
                      extra_long=above_vwap, extra_short=below_vwap)
    print(summarize("R1 st_trail+VWAP", tdf, n_days))

    # R2: trail on MAJOR ST instead of minor (ride legs longer)
    tdf = run_variant(data, major_dir, minor_dir, risk_atr, "st_trail", 1.5, 99,
                      trail_dir=major_dir)
    print(summarize("R2 trail=major", tdf, n_days))

    # R3: major-trail + VWAP filter combined
    tdf = run_variant(data, major_dir, minor_dir, risk_atr, "st_trail", 1.5, 99,
                      trail_dir=major_dir, extra_long=above_vwap, extra_short=below_vwap)
    print(summarize("R3 major+VWAP", tdf, n_days))

    # R4: first-hour only (09:30-10:30), minor trail + VWAP
    tdf = run_variant(data, major_dir, minor_dir, risk_atr, "st_trail", 1.5, 99,
                      start="09:30", end="10:30",
                      extra_long=above_vwap, extra_short=below_vwap)
    print(summarize("R4 first-hr+VWAP", tdf, n_days))

    # R5: first-hour + major-trail + VWAP (cleanest trend, longest ride)
    tdf = run_variant(data, major_dir, minor_dir, risk_atr, "st_trail", 1.5, 99,
                      start="09:30", end="10:30", trail_dir=major_dir,
                      extra_long=above_vwap, extra_short=below_vwap)
    print(summarize("R5 first-hr+maj+VWAP", tdf, n_days))

    # ---- NET-OF-COST check on the two finalists ----
    print("\n" + "NET OF COSTS".center(len(hdr), "."))
    finalists = {
        "st_trail 1.5 (full win)": run_variant(data, major_dir, minor_dir, risk_atr,
                                                "st_trail", 1.5, 99),
        "R4 first-hr+VWAP":        run_variant(data, major_dir, minor_dir, risk_atr,
                                               "st_trail", 1.5, 99, start="09:30",
                                               end="10:30", extra_long=above_vwap,
                                               extra_short=below_vwap),
    }
    for name, tdf in finalists.items():
        avg_risk_pts = tdf["risk_pts"].mean()
        avg_risk_cash = tdf["risk_cash"].mean()
        gross_r = tdf["r_multiple"].sum()
        print(f"\n{name}:  {len(tdf)} trades | avg risk {avg_risk_pts:.1f} pts "
              f"(${avg_risk_cash:,.0f}/R) | gross {gross_r:+.1f}R")
        for cost in (5, 10, 15, 20):
            cost_r = (cost / tdf["risk_cash"]).mean()
            net_exp = tdf["r_multiple"].mean() - cost_r
            net_tot = gross_r - cost_r * len(tdf)
            print(f"    @ ${cost:>2}/RT:  cost {cost_r:.3f}R/trade  "
                  f"net exp {net_exp:+.3f}R  net total {net_tot:+.1f}R")

    # ---- Robustness breakdown of the finalist (R4) ----
    print("\n" + "R4 ROBUSTNESS (gross R)".center(len(hdr), "."))
    r4 = finalists["R4 first-hr+VWAP"].copy()
    for side in ("long", "short"):
        s = r4[r4["side"] == side]
        wr = (s["r_multiple"] > 0).mean() * 100 if len(s) else 0
        print(f"  {side:<6} {len(s):>4} trades  win {wr:4.1f}%  total {s['r_multiple'].sum():+7.1f}R  "
              f"exp {s['r_multiple'].mean():+.3f}R")
    r4["yr"] = pd.to_datetime(r4["exit_time"]).dt.year
    print("  by year:")
    for yr, g in r4.groupby("yr"):
        wr = (g["r_multiple"] > 0).mean() * 100
        print(f"    {yr}  {len(g):>4} trades  win {wr:4.1f}%  total {g['r_multiple'].sum():+7.1f}R")


if __name__ == "__main__":
    main()
