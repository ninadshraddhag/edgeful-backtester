"""ORB Fakeout (false breakdown / breakout) with FVG / IFVG reclaim confirmation.

Rules (per day, on 1-minute candles):
  * ORB = high/low of the first 15 one-minute candles of the session.
  * LONG (false breakdown):
        1. a 1-min candle CLOSES below ORB-low                     (the fakeout)
        2. a later 1-min candle CLOSES back above ORB-low          (the reclaim)
        3. confirmation on the reclaim bar, EITHER:
             - FVG  : a bullish 3-candle fair-value gap on/at the reclaim, OR
             - IFVG : a bearish FVG formed during the breakdown whose top the
                      reclaim now closes back above (the gap "inverts" to support)
        -> enter LONG at the next bar's open.
        -> STOP = the fakeout extreme (deepest low of the false break)
        -> TARGET = opposite side of the ORB (ORB-high)
  * SHORT = exact mirror (false breakout above ORB-high).
  * One trade per day (first valid setup, either side); no new entries after the
    cutoff; flat at session close. A "fakeout" must not extend more than
    `max_depth` x ORB-range beyond the level (else it's a genuine break, skipped).

Designed for NQ but works on any instrument the loader knows (uses its session).
Produces a trades DataFrame compatible with core.metrics / core.visualizer.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indicators import fair_value_gaps
from core import metrics as metrics_mod, visualizer, calendar_view
from data import loader

REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def _hhmm(ts) -> str:
    return pd.Timestamp(ts).strftime("%H:%M")


def _add_minutes(hhmm: str, mins: int) -> str:
    t = datetime.strptime(hhmm, "%H:%M") + timedelta(minutes=mins)
    return t.strftime("%H:%M")


def _gate(confirm: str, fvg_conf: bool, ifvg_conf: bool) -> bool:
    """Decide if a reclaim confirmation passes, given the chosen mode."""
    if confirm == "fvg":
        return fvg_conf
    if confirm == "ifvg":
        return ifvg_conf
    if confirm == "both":
        return fvg_conf and ifvg_conf
    return fvg_conf or ifvg_conf            # "either" (default)


def _label(fvg_conf: bool, ifvg_conf: bool) -> str:
    if fvg_conf and ifvg_conf:
        return "FVG+IFVG"
    return "IFVG" if ifvg_conf else "FVG"


def backtest_orb_fakeout(df: pd.DataFrame, instrument: str, point_value: float,
                         orb_minutes: int = 15, max_depth: float = 1.0,
                         sides: str = "both", confirm: str = "either",
                         cost_rt: float = 0.0) -> pd.DataFrame:
    """Run the ORB fakeout backtest. `df` must be 1-minute, RTH-filtered.

    sides     : 'long' | 'short' | 'both'
    confirm   : 'either' | 'fvg' | 'ifvg' | 'both'
    cost_rt   : round-trip cost in cash (e.g. $5 on NQ); deducted from each trade
    """
    sess_open, sess_close = loader.session_for(instrument)
    orb_end = _add_minutes(sess_open, orb_minutes)        # e.g. 09:45
    cutoff = _add_minutes(sess_close, -30)                # last entry time
    trades = []

    for _, day in df.groupby("session_date", sort=True):
        day = day.reset_index(drop=True)
        fvg = fair_value_gaps(day)

        o = day["open"].to_numpy(float)
        h = day["high"].to_numpy(float)
        l = day["low"].to_numpy(float)
        c = day["close"].to_numpy(float)
        mins = pd.to_datetime(day["date"]).dt.strftime("%H:%M").to_numpy()

        fvg_bull = fvg["fvg_bull"].to_numpy(bool)
        fvg_bear = fvg["fvg_bear"].to_numpy(bool)
        fvg_top = fvg["fvg_top"].to_numpy(float)
        fvg_bottom = fvg["fvg_bottom"].to_numpy(float)

        orb_mask = (mins >= sess_open) & (mins < orb_end)
        if orb_mask.sum() < orb_minutes - 2:              # need a near-complete ORB
            continue
        orb_high = h[orb_mask].max()
        orb_low = l[orb_mask].min()
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            continue

        start = int(np.argmax(mins >= orb_end))
        if mins[start] < orb_end:
            continue

        broke_dn = broke_up = False
        fake_low, fake_high = np.inf, -np.inf
        last_bear_top = last_bull_bot = None
        trade = None

        for i in range(start, len(day) - 1):
            if mins[i] >= cutoff:
                break

            if fvg_bear[i]:
                last_bear_top = fvg_top[i]
            if fvg_bull[i]:
                last_bull_bot = fvg_bottom[i]

            if c[i] < orb_low:
                broke_dn = True
            if broke_dn:
                fake_low = min(fake_low, l[i])
            if c[i] > orb_high:
                broke_up = True
            if broke_up:
                fake_high = max(fake_high, h[i])

            # ---- LONG: false breakdown reclaimed ----
            if sides in ("long", "both") and broke_dn and c[i] > orb_low and \
                    (orb_low - fake_low) <= max_depth * orb_range:
                fvg_conf = bool(fvg_bull[i] or fvg_bull[i - 1])
                ifvg_conf = last_bear_top is not None and c[i] > last_bear_top
                if _gate(confirm, fvg_conf, ifvg_conf):
                    entry = o[i + 1]
                    if entry > fake_low:
                        trade = dict(side="long", entry_idx=i + 1, entry=entry,
                                     stop=fake_low, target=orb_high,
                                     setup=_label(fvg_conf, ifvg_conf))
                        break

            # ---- SHORT: false breakout reclaimed ----
            if sides in ("short", "both") and broke_up and c[i] < orb_high and \
                    (fake_high - orb_high) <= max_depth * orb_range:
                fvg_conf = bool(fvg_bear[i] or fvg_bear[i - 1])
                ifvg_conf = last_bull_bot is not None and c[i] < last_bull_bot
                if _gate(confirm, fvg_conf, ifvg_conf):
                    entry = o[i + 1]
                    if entry < fake_high:
                        trade = dict(side="short", entry_idx=i + 1, entry=entry,
                                     stop=fake_high, target=orb_low,
                                     setup=_label(fvg_conf, ifvg_conf))
                        break

        if trade is None:
            continue

        # ---- forward exit simulation within the day ----
        side, ei = trade["side"], trade["entry_idx"]
        stop, target, entry = trade["stop"], trade["target"], trade["entry"]
        exit_price, reason, xi = c[-1], "session_close", len(day) - 1
        for j in range(ei, len(day)):
            if side == "long":
                hit_stop, hit_tgt = l[j] <= stop, h[j] >= target
            else:
                hit_stop, hit_tgt = h[j] >= stop, l[j] <= target
            if hit_stop:
                exit_price, reason, xi = stop, "stop", j
                break
            if hit_tgt:
                exit_price, reason, xi = target, "target", j
                break
            if j == len(day) - 1:
                exit_price, reason, xi = c[j], "session_close", j

        risk = abs(entry - stop)
        gross = (exit_price - entry) if side == "long" else (entry - exit_price)
        pnl = gross - (cost_rt / point_value if point_value else 0.0)   # net of costs
        trades.append({
            "entry_time": pd.Timestamp(day["date"].iloc[ei]),
            "exit_time": pd.Timestamp(day["date"].iloc[xi]),
            "side": side, "setup": trade["setup"],
            "entry": round(entry, 2), "stop": round(stop, 2), "target": round(target, 2),
            "exit": round(exit_price, 2),
            "pnl_points": round(pnl, 2), "pnl_cash": round(pnl * point_value, 2),
            "r_multiple": round(pnl / risk, 3) if risk else 0.0,
            "reason": reason, "bars_held": xi - ei,
        })

    return pd.DataFrame(trades)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="NQ")
    ap.add_argument("--max-days", type=int, default=300)
    ap.add_argument("--orb-minutes", type=int, default=15)
    ap.add_argument("--max-depth", type=float, default=1.0)
    ap.add_argument("--sides", default="both", choices=["long", "short", "both"])
    ap.add_argument("--confirmation", default="either", choices=["either", "fvg", "ifvg", "both"])
    ap.add_argument("--cost-rt", type=float, default=0.0, help="round-trip cost in cash")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.instrument} 1-min ({args.max_days} days)...")
    df = loader.load_data(args.instrument, timeframe="1min", max_days=args.max_days)
    pv = loader.point_value(args.instrument)
    print(f"  {len(df):,} bars, {df['date'].min()} -> {df['date'].max()}")

    trades = backtest_orb_fakeout(df, args.instrument, pv,
                                  orb_minutes=args.orb_minutes, max_depth=args.max_depth,
                                  sides=args.sides, confirm=args.confirmation,
                                  cost_rt=args.cost_rt)
    m = metrics_mod.compute_metrics(trades)
    desc = (f"ORB({args.orb_minutes}m) fakeout, {args.sides} side, {args.confirmation} confirm, "
            f"target opposite ORB | stop=fakeout extreme | cost=${args.cost_rt:g}/rt")
    print("\n" + metrics_mod.format_report(m, desc, args.instrument))

    if not trades.empty:
        by = trades.groupby(["side", "setup"])["r_multiple"].agg(["count", "sum", "mean"])
        print("\n  Breakdown by side / confirmation:")
        print(by.round(3).to_string())
        rc = trades["reason"].value_counts().to_dict()
        print(f"\n  Exits: {rc}")

        if not args.no_charts:
            c = visualizer.plot_trades(df, trades, args.instrument,
                                       os.path.join(REPORTS, f"{args.instrument}_orb_fakeout_trades.html"))
            h = calendar_view.monthly_pnl_heatmap(trades, args.instrument,
                                                  os.path.join(REPORTS, f"{args.instrument}_orb_fakeout_calendar.html"))
            print(f"\n  chart    -> {os.path.relpath(c)}")
            print(f"  calendar -> {os.path.relpath(h)}")


if __name__ == "__main__":
    main()
