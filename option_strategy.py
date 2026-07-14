"""
option_strategy.py — NIFTY Option Strategy engine (three legs + portfolio).

Legs (all: long the ATM option; 1 lot; ≤1 trade/day; flat 15:15; stops/targets
in % of PDC on the SPOT, translated to option P&L via options_pricing):

  T1  Trend-Ride  — first 3-min close beyond PDH/PDL before 13:00, EMA + VWAP
                    filters; stop 0.5% PDC, trail 1.2% PDC.
  GF  Gap-Fade    — gap 0.35–1.0% vs PDC, enter 09:21, fade to PDC±0.04%,
                    stop 0.8% PDC.
  ORB IB-breakout — first 3-min close beyond the 60-min IB before 13:30,
                    VWAP-aligned; stop = opposite IB side capped 0.8% PDC,
                    trail 1.2% PDC.

Entries/filters on 3-min closes; exits resolved on 1-min highs/lows.
Pure functions, no Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import build_facts
import options_pricing as op

OPEN_T = 9 * 60 + 15
FLAT_T = 15 * 60 + 15            # square-off 15:15
IB_END = OPEN_T + 60 - 1        # 10:14
LEGS = ("T1", "GF", "ORB")

DEFAULTS = {
    "iv": 0.13, "lots": 1,                            # long the ATM option (buy only)
    "slippage_pts": 0.5, "brokerage": 20.0,
    "cost_mult": 1.0, "slip_mult": 1.0,
    "T1":  {"on": True, "ema_gap_pct": 0.05, "stop_pct": 0.5, "trail_pct": 1.2,
            "cutoff": 13 * 60, "use_vwap": True},
    "GF":  {"on": True, "gap_min": 0.35, "gap_max": 1.0, "enter_t": 9 * 60 + 21,
            "target_buf_pct": 0.04, "stop_pct": 0.8},
    "ORB": {"on": True, "stop_cap_pct": 0.8, "trail_pct": 1.2,
            "cutoff": 13 * 60 + 30, "use_vwap": True},
}


# ─── data prep ────────────────────────────────────────────────────────────────

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def prep_days(mdf: pd.DataFrame):
    """
    One record per trading day with everything the legs need:
    prev-day HLC (PDH/PDL/PDC), the 1-min session arrays (for exits), the 3-min
    signal frame (close, ema9, ema21, vwap-proxy), IB high/low, gap %.
    """
    days = []
    prev = None
    for d, g0 in mdf.groupby("date_only", sort=True):
        sess = g0[(g0["t_min"] >= OPEN_T) & (g0["t_min"] <= FLAT_T)].sort_values("t_min")
        if len(sess) < 60:
            prev = None
            continue
        pdh = sess["high"].max(); pdl = sess["low"].min(); pdc = sess.iloc[-1]["close"]
        if prev is not None:
            b3 = build_facts.to_timeframe(sess, 3, open_t=OPEN_T)
            tp = (b3["high"] + b3["low"] + b3["close"]) / 3.0
            rec = {
                "date": pd.Timestamp(d), "dow": pd.Timestamp(d).day_name(),
                "PDC": prev["c"], "PDH": prev["h"], "PDL": prev["l"],
                "day_open": float(sess.iloc[0]["open"]),
                # 1-min arrays for exit resolution
                "t": sess["t_min"].to_numpy(), "hi": sess["high"].to_numpy(float),
                "lo": sess["low"].to_numpy(float), "cl": sess["close"].to_numpy(float),
                # 3-min signal frame
                "b3_t": b3["t_min"].to_numpy(),
                "b3_c": b3["close"].to_numpy(float),
                "b3_ema9": _ema(b3["close"], 9).to_numpy(),
                "b3_ema21": _ema(b3["close"], 21).to_numpy(),
                "b3_vwap": tp.expanding().mean().to_numpy(),
                "ib_hi": float(sess[sess["t_min"] <= IB_END]["high"].max()),
                "ib_lo": float(sess[sess["t_min"] <= IB_END]["low"].min()),
            }
            rec["gap_pct"] = (rec["day_open"] - rec["PDC"]) / rec["PDC"] * 100
            days.append(rec)
        prev = {"c": float(pdc), "h": float(pdh), "l": float(pdl)}
    return days


# ─── option P&L for one resolved trade ────────────────────────────────────────

def _price_trade(rec, bullish, entry_t, entry_spot, exit_t, exit_spot, cfg):
    """Turn a resolved spot trade into a LONG ATM option P&L (₹, all costs)."""
    iv = cfg["iv"]
    dt_e = rec["date"] + pd.Timedelta(minutes=int(entry_t))
    dt_x = rec["date"] + pd.Timedelta(minutes=int(exit_t))
    Te, Tx = op.tte_years(dt_e), op.tte_years(dt_x)

    is_call = bullish                            # long the option in trade direction
    K = op.atm_strike(entry_spot)
    pe = float(op.bs_price(entry_spot, K, Te, iv, is_call))
    px = float(op.bs_price(exit_spot, K, Tx, iv, is_call))
    pts = px - pe                                # long

    cost = op.round_trip_cost(pe, px, cfg["lots"], cfg["slippage_pts"],
                              cfg["brokerage"], cfg["cost_mult"], cfg["slip_mult"])
    pnl = pts * op.LOT_SIZE * cfg["lots"] - cost
    return dict(strike=K, opt="+" + ("CE" if is_call else "PE"),
                entry_prem=round(pe, 2), exit_prem=round(px, 2),
                points=round(pts, 2), cost=round(cost, 1), pnl=round(pnl, 1))


def _walk_exit(rec, i0, bullish, init_stop_lvl, trail_dist, target_lvl=None):
    """
    Resolve the exit on 1-min bars from index i0 onward. Returns (exit_t,
    exit_spot, reason). `init_stop_lvl` is the fixed initial stop price;
    `trail_dist` (price units, or None) trails from the running extreme;
    `target_lvl` optional fixed target price.
    """
    t, hi, lo, cl = rec["t"], rec["hi"], rec["lo"], rec["cl"]
    n = len(t)
    ext = cl[i0]
    for j in range(i0, n):
        if t[j] >= FLAT_T:
            return int(t[j]), float(cl[j]), "flat 15:15"
        if bullish:
            ext = max(ext, hi[j])
            stop = init_stop_lvl if trail_dist is None else max(init_stop_lvl, ext - trail_dist)
            if target_lvl is not None and hi[j] >= target_lvl:
                return int(t[j]), float(target_lvl), "target"
            if lo[j] <= stop:
                return int(t[j]), float(stop), "stop/trail"
        else:
            ext = min(ext, lo[j])
            stop = init_stop_lvl if trail_dist is None else min(init_stop_lvl, ext + trail_dist)
            if target_lvl is not None and lo[j] <= target_lvl:
                return int(t[j]), float(target_lvl), "target"
            if hi[j] >= stop:
                return int(t[j]), float(stop), "stop/trail"
    return int(t[-1]), float(cl[-1]), "flat 15:15"


def _idx_at(rec, t_min):
    """First 1-min index at/after t_min."""
    i = np.searchsorted(rec["t"], t_min)
    return i if i < len(rec["t"]) else None


# ─── the three legs ───────────────────────────────────────────────────────────

def leg_T1(rec, cfg):
    c = cfg["T1"]; pdc = rec["PDC"]
    thr = c["ema_gap_pct"] / 100.0
    for k, tb in enumerate(rec["b3_t"]):
        if tb >= c["cutoff"]:
            break
        px = rec["b3_c"][k]
        e9, e21, vw = rec["b3_ema9"][k], rec["b3_ema21"][k], rec["b3_vwap"][k]
        bull = px > rec["PDH"] and (e9 - e21) > thr * px and (not c["use_vwap"] or px > vw)
        bear = px < rec["PDL"] and (e21 - e9) > thr * px and (not c["use_vwap"] or px < vw)
        if not (bull or bear):
            continue
        i0 = _idx_at(rec, tb)
        if i0 is None:
            return None
        stop = (px - c["stop_pct"] / 100 * pdc) if bull else (px + c["stop_pct"] / 100 * pdc)
        xt, xs, why = _walk_exit(rec, i0, bull, stop, c["trail_pct"] / 100 * pdc)
        tr = _price_trade(rec, bull, tb, px, xt, xs, cfg)
        tr.update(date=rec["date"], dow=rec["dow"], leg="T1",
                  dir="bull" if bull else "bear", entry_t=int(tb),
                  entry_spot=round(px, 1), exit_t=xt, exit_spot=round(xs, 1),
                  reason=why)
        return tr
    return None


def leg_GF(rec, cfg):
    c = cfg["GF"]; pdc = rec["PDC"]; g = abs(rec["gap_pct"])
    if not (c["gap_min"] <= g <= c["gap_max"]):
        return None
    i0 = _idx_at(rec, c["enter_t"])
    if i0 is None:
        return None
    spot = rec["cl"][i0]
    gap_up = rec["gap_pct"] > 0
    bull = not gap_up                             # gap-up → fade down (bearish); gap-down → bullish
    buf = c["target_buf_pct"] / 100 * pdc
    if gap_up:                                    # fade down toward PDC
        target = pdc + buf; stop = spot + c["stop_pct"] / 100 * pdc
    else:                                         # fade up toward PDC
        target = pdc - buf; stop = spot - c["stop_pct"] / 100 * pdc
    xt, xs, why = _walk_exit(rec, i0, bull, stop, None, target_lvl=target)
    tr = _price_trade(rec, bull, c["enter_t"], spot, xt, xs, cfg)
    tr.update(date=rec["date"], dow=rec["dow"], leg="GF",
              dir="bull" if bull else "bear", entry_t=int(c["enter_t"]),
              entry_spot=round(spot, 1), exit_t=xt, exit_spot=round(xs, 1),
              reason=why)
    return tr


def leg_ORB(rec, cfg):
    c = cfg["ORB"]; pdc = rec["PDC"]
    for k, tb in enumerate(rec["b3_t"]):
        if tb <= IB_END:
            continue
        if tb >= c["cutoff"]:
            break
        px = rec["b3_c"][k]; vw = rec["b3_vwap"][k]
        bull = px > rec["ib_hi"] and (not c["use_vwap"] or px > vw)
        bear = px < rec["ib_lo"] and (not c["use_vwap"] or px < vw)
        if not (bull or bear):
            continue
        i0 = _idx_at(rec, tb)
        if i0 is None:
            return None
        cap = c["stop_cap_pct"] / 100 * pdc
        if bull:
            stop = max(rec["ib_lo"], px - cap)
        else:
            stop = min(rec["ib_hi"], px + cap)
        xt, xs, why = _walk_exit(rec, i0, bull, stop, c["trail_pct"] / 100 * pdc)
        tr = _price_trade(rec, bull, tb, px, xt, xs, cfg)
        tr.update(date=rec["date"], dow=rec["dow"], leg="ORB",
                  dir="bull" if bull else "bear", entry_t=int(tb),
                  entry_spot=round(px, 1), exit_t=xt, exit_spot=round(xs, 1),
                  reason=why)
        return tr
    return None


_LEG_FN = {"T1": leg_T1, "GF": leg_GF, "ORB": leg_ORB}
_COLS = ["date", "dow", "leg", "dir", "opt", "strike", "entry_t", "entry_spot",
         "exit_t", "exit_spot", "entry_prem", "exit_prem", "points", "cost",
         "pnl", "reason"]


def run_leg(days, cfg, leg) -> pd.DataFrame:
    fn = _LEG_FN[leg]
    rows = [tr for rec in days if (tr := fn(rec, cfg)) is not None]
    if not rows:
        return pd.DataFrame(columns=_COLS)
    return pd.DataFrame(rows)[_COLS].sort_values("date").reset_index(drop=True)


def run_portfolio(days, cfg) -> dict:
    """Run every enabled leg; return {leg: trades_df} plus 'ALL' concatenated."""
    out = {}
    for leg in LEGS:
        if cfg[leg]["on"]:
            out[leg] = run_leg(days, cfg, leg)
    if out:
        out["ALL"] = (pd.concat(out.values(), ignore_index=True)
                      .sort_values("date").reset_index(drop=True))
    return out


# ─── metrics, correlation, splits ─────────────────────────────────────────────

SPLITS = {"Train 2015–21": ("2015-01-01", "2021-12-31"),
          "Val 2022–24": ("2022-01-01", "2024-12-31"),
          "Test 2025–26 🔒": ("2025-01-01", "2026-12-31"),
          "Full 2015–26": ("2015-01-01", "2026-12-31")}


def filter_days(days, d0, d1):
    lo, hi = pd.Timestamp(d0), pd.Timestamp(d1)
    return [r for r in days if lo <= r["date"] <= hi]


def _month_streaks(dates, pnl):
    """(max consecutive GREEN months, max consecutive RED months)."""
    idx = pd.to_datetime(pd.Series(list(dates)))
    msum = pd.Series(np.asarray(pnl, float)).groupby(
        (idx.dt.year * 12 + idx.dt.month).values).sum().sort_index()
    best_g = best_r = cur_g = cur_r = 0
    for v in msum.values:
        if v > 0:
            cur_g += 1; cur_r = 0
        else:
            cur_r += 1; cur_g = 0
        best_g = max(best_g, cur_g); best_r = max(best_r, cur_r)
    return best_g, best_r


def metrics(trades: pd.DataFrame) -> dict:
    """Portfolio-style metrics on a trades frame (needs date + pnl)."""
    import daywise
    if trades.empty:
        return {"n": 0}
    p = trades["pnl"].to_numpy(float)
    gp, gl = p[p > 0].sum(), -p[p < 0].sum()
    # daily P&L → drawdown + Sharpe
    daily = trades.groupby(pd.to_datetime(trades["date"]).dt.normalize())["pnl"].sum()
    cum = daily.cumsum()
    dd = (cum - cum.cummax()).min()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    gmp, gmg, gmn = daywise.green_months(trades["date"], p)
    max_g, max_r = _month_streaks(trades["date"], p)
    return {"n": int(len(p)), "exp": p.mean(), "win": (p > 0).mean() * 100,
            "pf": (gp / gl) if gl > 0 else np.inf, "net": p.sum(),
            "max_dd": float(dd), "sharpe": float(sharpe),
            "green_pct": gmp, "green": f"{gmg}/{gmn}", "months": gmn,
            "max_green_streak": max_g, "max_red_streak": max_r,
            "avg_month": (cum.iloc[-1] / gmn) if gmn else 0.0}


def daily_by_leg(port: dict) -> pd.DataFrame:
    """Wide frame: one column of daily P&L per leg (0-filled), for correlation."""
    cols = {}
    for leg, t in port.items():
        if leg == "ALL" or t.empty:
            continue
        cols[leg] = t.groupby(pd.to_datetime(t["date"]).dt.normalize())["pnl"].sum()
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).fillna(0.0)


def correlations(port: dict):
    """(daily_corr, monthly_corr) between the leg P&L streams."""
    d = daily_by_leg(port)
    if d.shape[1] < 2:
        return None, None
    m = d.resample("ME").sum()
    return d.corr().round(2), m.corr().round(2)


# ─── per-leg optimizer ────────────────────────────────────────────────────────

OPT_GRIDS = {
    "T1":  {"ema_gap_pct": [0.02, 0.05, 0.10], "stop_pct": [0.4, 0.5, 0.7],
            "trail_pct": [1.0, 1.2, 1.5], "cutoff": [12 * 60, 13 * 60]},
    "GF":  {"gap_min": [0.25, 0.35, 0.5], "gap_max": [0.8, 1.0, 1.25],
            "target_buf_pct": [0.02, 0.04, 0.08], "stop_pct": [0.6, 0.8, 1.0]},
    "ORB": {"stop_cap_pct": [0.6, 0.8, 1.0], "trail_pct": [1.0, 1.2, 1.5],
            "cutoff": [13 * 60, 13 * 60 + 30, 14 * 60]},
}


def optimize_leg(days, base_cfg, leg, rank="exp", min_trades=100,
                 min_green=0, progress=None):
    """
    Sweep the leg's parameter grid; one row per config with full metrics.
    rank ∈ {exp, pf, green_months, sharpe, net}. All rows returned so parameter
    cliffs are visible. Uses the current cost settings.
    """
    import copy, itertools
    grid = OPT_GRIDS[leg]
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    rows = []
    for i, vals in enumerate(combos):
        cfg = copy.deepcopy(base_cfg)
        for k, v in zip(keys, vals):
            cfg[leg][k] = v
        if leg == "GF" and cfg["GF"]["gap_min"] >= cfg["GF"]["gap_max"]:
            if progress:
                progress(i + 1, len(combos))
            continue
        t = run_leg(days, cfg, leg)
        if len(t) >= min_trades:
            m = metrics(t)
            if m["green_pct"] >= min_green:
                row = {k: vals[j] for j, k in enumerate(keys)}
                row.update(trades=m["n"], exp=round(m["exp"], 0),
                           win_pct=round(m["win"], 1),
                           pf=round(m["pf"], 2) if np.isfinite(m["pf"]) else 99,
                           green_pct=round(m["green_pct"], 1), months=m["green"],
                           red_streak=m["max_red_streak"],
                           sharpe=round(m["sharpe"], 2), net=round(m["net"], 0),
                           max_dd=round(m["max_dd"], 0))
                rows.append(row)
        if progress:
            progress(i + 1, len(combos))
    if not rows:
        return pd.DataFrame()
    col = {"exp": "exp", "pf": "pf", "net": "net", "sharpe": "sharpe",
           "green_months": "green_pct"}[rank]
    return pd.DataFrame(rows).sort_values(col, ascending=False).reset_index(drop=True)


# ─── trade-browser support ────────────────────────────────────────────────────

def find_day(days, d):
    d = pd.Timestamp(d)
    for r in days:
        if r["date"] == d:
            return r
    return None


def premium_series(rec, trade, cfg):
    """
    Re-price the trade's option on every 1-min bar of its life (entry→exit) so
    the trade browser can plot the premium curve. Returns a DataFrame with
    t_min, spot, premium.
    """
    is_call = (trade["opt"][-2:] == "CE")
    K = trade["strike"]
    e_i = _idx_at(rec, trade["entry_t"])
    x_i = _idx_at(rec, trade["exit_t"])
    if e_i is None:
        return pd.DataFrame()
    x_i = (x_i if x_i is not None else len(rec["t"]) - 1) + 1
    t = rec["t"][e_i:x_i]; s = rec["cl"][e_i:x_i]
    prem = [float(op.bs_price(sp, K, op.tte_years(rec["date"] + pd.Timedelta(minutes=int(tm))),
                              cfg["iv"], is_call)) for tm, sp in zip(t, s)]
    return pd.DataFrame({"t_min": t, "spot": s, "premium": np.round(prem, 2)})
