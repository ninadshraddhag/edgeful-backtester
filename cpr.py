"""
cpr.py — Central Pivot Range trading-systems engine  (NIFTY)
===========================================================
A faithful, data-driven implementation of the three CPR systems from
Frank Ochoa's *Secrets of a Pivot Boss*, adapted for NIFTY:

  System 1 — Reversal   : fade a test of the CPR back toward value, with the
                          two-day relationship as a bias filter and a Wick/Doji
                          reversal candle as the trigger. Scale out  P → outer pivot.
  System 2 — Trend      : Mode A pull-back to the near central in a trend, or
                          Mode B break of the first-15-min range on an open
                          beyond yesterday's range (1.5×ATR trailing stop).
  System 3 — Scalp      : on a wide-CPR / range regime, fade the fences
                          (S1 · P · R1 or the CPR edges) with tight stops, fast
                          targets, re-entry, and a range-break kill switch.

Everything executes on the configured execution timeframe (5-min in the app).
The module is pure (no Streamlit). Shapes mirror ib50.py so the app can reuse
the same optimizer / trade-browser plumbing.

CPR formula (Ch. 6), from YESTERDAY's session High / Low / Close:
    P  = (H + L + C) / 3
    BC = (H + L) / 2
    TC = (P - BC) + P            # = 2P - BC
    TC_line = max(TC, BC) ,  BC_line = min(TC, BC) ,  width = TC_line - BC_line
Floor pivots (same prior-day H/L/C):
    R1 = 2P - L ,  S1 = 2P - H
    R2 = P + (H - L) ,  S2 = P - (H - L)
    R3 = H + 2(P - L) ,  S3 = L - 2(H - P)
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
import pandas as pd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TRADING_DAYS = 252

SYSTEMS = ["reversal", "trend", "scalp"]
SYSTEM_LABELS = {"reversal": "System 1 · Reversal (fade)",
                 "trend": "System 2 · Trend following",
                 "scalp": "System 3 · Scalp (range)"}

# Two-day relationship buckets (today's CPR vs yesterday's CPR)
RELATIONSHIPS = ["Higher Value", "Overlapping Higher", "Lower Value",
                 "Overlapping Lower", "Unchanged", "Outside Value", "Inside Value"]


# ─── CPR maths ────────────────────────────────────────────────────────────────

def compute_cpr(h: float, l: float, c: float) -> dict:
    """All CPR centrals + floor pivots from a prior session's H/L/C."""
    p = (h + l + c) / 3.0
    bc = (h + l) / 2.0
    tc = 2.0 * p - bc
    tc_line, bc_line = (tc, bc) if tc >= bc else (bc, tc)
    return {
        "pp": p, "tc": tc_line, "bc": bc_line, "width": tc_line - bc_line,
        "r1": 2 * p - l, "s1": 2 * p - h,
        "r2": p + (h - l), "s2": p - (h - l),
        "r3": h + 2 * (p - l), "s3": l - 2 * (h - p),
    }


def classify_relationship(today: dict, yest: dict, eps_frac: float = 0.0008) -> str:
    """
    The seven two-day CPR configurations. `eps` (a fraction of price) sets the
    'virtually identical' band for the Unchanged case. Priority order matters.
    """
    tTC, tBC = today["tc"], today["bc"]
    yTC, yBC = yest["tc"], yest["bc"]
    eps = eps_frac * max(today["pp"], 1.0)
    if abs(tTC - yTC) <= eps and abs(tBC - yBC) <= eps:
        return "Unchanged"
    if tTC > yTC + eps and tBC < yBC - eps:
        return "Outside Value"
    if tTC < yTC - eps and tBC > yBC + eps:
        return "Inside Value"
    if tBC > yTC:                       # entirely above, no overlap
        return "Higher Value"
    if tTC < yBC:                       # entirely below, no overlap
        return "Lower Value"
    return "Overlapping Higher" if today["pp"] >= yest["pp"] else "Overlapping Lower"


def width_regime(width: float, price: float, narrow_pct: float, wide_pct: float) -> str:
    """narrow / normal / wide, by CPR width as a % of price (book Ch. 6)."""
    wp = (width / price * 100.0) if price else 0.0
    if wp < narrow_pct:
        return "narrow"
    if wp > wide_pct:
        return "wide"
    return "normal"


def _regime(rec: dict, cfg: dict) -> str:
    """Width regime from the record's stored width-% and the LIVE cfg thresholds,
    so narrow/wide can be tuned in the UI without re-preparing the data."""
    wp = rec["width_pct"]
    if wp < cfg["narrow_pct"]:
        return "narrow"
    if wp > cfg["wide_pct"]:
        return "wide"
    return "normal"


# ─── candle primitives (reversal / doji triggers, Ch. 2) ──────────────────────

def _cdl(o, h, l, c):
    rng = h - l
    body = abs(c - o)
    up_wick = h - max(o, c)
    lo_wick = min(o, c) - l
    loc = (c - l) / rng if rng > 0 else 0.5      # close location: 0=low, 1=high
    return rng, body, up_wick, lo_wick, loc


def _bear_trigger(o, h, l, c, mult, loc_max, doji_on, doji_frac):
    """Rejection from above: long upper wick + close in the lower part, or a doji."""
    rng, body, up_wick, lo_wick, loc = _cdl(o, h, l, c)
    if rng <= 0:
        return False
    if doji_on and body <= doji_frac * rng:
        return True
    return up_wick >= mult * max(body, 1e-9) and loc <= loc_max


def _bull_trigger(o, h, l, c, mult, loc_min, doji_on, doji_frac):
    """Rejection from below: long lower wick + close in the upper part, or a doji."""
    rng, body, up_wick, lo_wick, loc = _cdl(o, h, l, c)
    if rng <= 0:
        return False
    if doji_on and body <= doji_frac * rng:
        return True
    return lo_wick >= mult * max(body, 1e-9) and loc >= loc_min


# ─── per-day preparation ──────────────────────────────────────────────────────

def prep_days(min_df: pd.DataFrame, open_t: int, close_t: int, ib_min: int,
              tf: int = 5, narrow_pct: float = 0.12, wide_pct: float = 0.25,
              atr_n: int = 14) -> list:
    """
    Build per-day records on `tf`-minute bars. Today's CPR is built from
    YESTERDAY's session H/L/C; the two-day relationship compares it to the CPR
    that was in force the previous day. Each record carries the CPR/pivots, the
    regime, the IB & first-15-min stats, and the full post-open bar arrays
    (open/high/low/close/t_min/atr) within [open_t, close_t].
    """
    ib_end = open_t + ib_min - 1
    f15_end = open_t + 15 - 1
    out = []
    prev_cpr = None                 # CPR that was active on the previous day
    prev_hlc = None                 # previous day's session (H, L, C) → today's CPR
    widths = []                     # rolling history of CPR widths (for 20-day avg)

    for day, g0 in min_df.groupby("date_only", sort=True):
        sess = g0[(g0["t_min"] >= open_t) & (g0["t_min"] <= close_t)].sort_values("t_min")
        if sess.empty:
            continue
        s_h, s_l, s_c = float(sess["high"].max()), float(sess["low"].min()), \
            float(sess.iloc[-1]["close"])
        day_open = float(sess.iloc[0]["open"])

        # need a prior day to define today's CPR, and a day before that for the
        # two-day relationship
        today_cpr = compute_cpr(*prev_hlc) if prev_hlc else None
        if today_cpr is None or prev_cpr is None or len(sess) < 6:
            prev_cpr = today_cpr if today_cpr is not None else prev_cpr
            prev_hlc = (s_h, s_l, s_c)
            continue

        rel = classify_relationship(today_cpr, prev_cpr)
        avg20 = float(np.mean(widths[-20:])) if widths else today_cpr["width"]
        regime = width_regime(today_cpr["width"], today_cpr["pp"], narrow_pct, wide_pct)

        ibwin = sess[sess["t_min"] <= ib_end]
        f15 = sess[sess["t_min"] <= f15_end]
        ib_high = float(ibwin["high"].max()) if len(ibwin) else np.nan
        ib_low = float(ibwin["low"].min()) if len(ibwin) else np.nan
        f15_high = float(f15["high"].max()) if len(f15) else np.nan
        f15_low = float(f15["low"].min()) if len(f15) else np.nan

        # previous day's high / low (= today's PDH / PDL)
        pdh, pdl, _ = prev_hlc

        # ATR on the session bars (Wilder-style simple rolling on true range)
        hi = sess["high"].to_numpy(float)
        lo = sess["low"].to_numpy(float)
        cl = sess["close"].to_numpy(float)
        prev_c = np.concatenate([[cl[0]], cl[:-1]])
        tr = np.maximum.reduce([hi - lo, np.abs(hi - prev_c), np.abs(lo - prev_c)])
        atr = pd.Series(tr).rolling(atr_n, min_periods=1).mean().to_numpy()

        out.append({
            "date": pd.Timestamp(day), "dow": pd.Timestamp(day).day_name(),
            "day_open": day_open, "pdh": pdh, "pdl": pdl,
            "open_beyond": day_open > pdh or day_open < pdl,
            "open_above_pdh": day_open > pdh, "open_below_pdl": day_open < pdl,
            **today_cpr, "relationship": rel, "regime": regime,
            "width_pct": today_cpr["width"] / today_cpr["pp"] * 100.0,
            "avg20_width": avg20,
            "ib_high": ib_high, "ib_low": ib_low,
            "f15_high": f15_high, "f15_low": f15_low,
            "po": sess["open"].to_numpy(float), "ph": hi, "pl": lo, "pc": cl,
            "pt": sess["t_min"].to_numpy(float), "atr": atr,
        })
        widths.append(today_cpr["width"])
        prev_cpr = today_cpr
        prev_hlc = (s_h, s_l, s_c)
    return out


# ─── trade construction + bar-by-bar manager ──────────────────────────────────

def _mk_trade(rec, system, direction, entry, init_stop, tp1, tp2,
              exit_px, outcome, e_t, x_t, pnl, scaled):
    risk = max(abs(entry - init_stop), 1e-9)
    r = pnl / risk
    return {
        "date": rec["date"], "dow": rec["dow"], "system": system,
        "direction": direction, "entry_t": float(e_t), "exit_t": float(x_t),
        "entry": round(entry, 2), "stop": round(init_stop, 2),
        "tp1": round(tp1, 2) if tp1 is not None else np.nan,
        "tp2": round(tp2, 2), "exit": round(exit_px, 2),
        "outcome": outcome, "pnl": round(pnl, 2), "r": round(r, 3),
        "win": pnl > 0, "scaled": scaled,
        # levels for the trade-browser chart
        "tc": rec["tc"], "pp": rec["pp"], "bc": rec["bc"],
        "r1": rec["r1"], "s1": rec["s1"], "r2": rec["r2"], "s2": rec["s2"],
        "r3": rec["r3"], "s3": rec["s3"],
        "ib_high": rec["ib_high"], "ib_low": rec["ib_low"],
        "relationship": rec["relationship"], "width_pct": round(rec["width_pct"], 3),
    }


def _manage(rec, cfg, system, direction, entry_i, entry_px, init_stop,
            tp1, tp2, use_trail):
    """
    Bar-by-bar exit machine from `entry_i` onward. Supports optional 50/50
    scale-out at tp1 (then breakeven on the runner), a 1.5×ATR trailing stop,
    and the square-off. Returns a finished trade dict. Conservative tie rule:
    a stop touched in the same bar as a target counts as the stop.
    """
    ph, pl, pc, pt, atr = rec["ph"], rec["pl"], rec["pc"], rec["pt"], rec["atr"]
    n = len(pt)
    longp = direction == "long"
    sgn = 1.0 if longp else -1.0
    # only scale out if tp1 sits in the profit direction beyond the entry
    tp1_valid = tp1 is not None and ((tp1 > entry_px) if longp else (tp1 < entry_px))
    scale = cfg["scale_out"] and tp1_valid
    be = cfg["be_after_tp1"]
    trail_mult = cfg["atr_mult"]
    eff_min = cfg["eff_min"]

    stop = init_stop
    qty = 1.0
    realized = 0.0          # pnl from the booked half (points, per 1.0 unit)
    half_done = False

    for j in range(entry_i, n):
        hi, lo, cl, tm = ph[j], pl[j], pc[j], int(pt[j])

        # trailing stop (tighten only)
        if use_trail and not np.isnan(atr[j]):
            ts = (hi - trail_mult * atr[j]) if longp else (lo + trail_mult * atr[j])
            stop = max(stop, ts) if longp else min(stop, ts)

        hit_stop = (lo <= stop) if longp else (hi >= stop)
        if hit_stop:
            pnl = realized + qty * sgn * (stop - entry_px)
            outcome = ("TP1+BE" if (half_done and abs(stop - entry_px) < 1e-6)
                       else ("Trail" if use_trail and not hit_initial(stop, init_stop, longp)
                             else "SL"))
            return _finish(rec, cfg, system, direction, entry_px, init_stop, tp1, tp2,
                           stop, outcome, pt[entry_i], pt[j], pnl, half_done), j

        # scale out half at tp1, move runner to breakeven
        if scale and not half_done:
            hit_tp1 = (hi >= tp1) if longp else (lo <= tp1)
            if hit_tp1:
                realized += 0.5 * sgn * (tp1 - entry_px)
                qty = 0.5
                half_done = True
                if be:
                    stop = entry_px

        hit_tp2 = (hi >= tp2) if longp else (lo <= tp2)
        if hit_tp2:
            pnl = realized + qty * sgn * (tp2 - entry_px)
            outcome = "TP1+TP2" if half_done else "TP"
            return _finish(rec, cfg, system, direction, entry_px, init_stop, tp1, tp2,
                           tp2, outcome, pt[entry_i], pt[j], pnl, half_done), j

        if tm >= eff_min:
            pnl = realized + qty * sgn * (cl - entry_px)
            outcome = "TP1+EOD" if half_done else "EOD"
            return _finish(rec, cfg, system, direction, entry_px, init_stop, tp1, tp2,
                           cl, outcome, pt[entry_i], pt[j], pnl, half_done), j

    # ran out of bars
    pnl = realized + qty * sgn * (pc[-1] - entry_px)
    return _finish(rec, cfg, system, direction, entry_px, init_stop, tp1, tp2,
                   pc[-1], "EOD", pt[entry_i], pt[-1], pnl, half_done), n - 1


def hit_initial(stop, init_stop, longp):
    """True if the resting stop is still the original (no trail improvement)."""
    return abs(stop - init_stop) < 1e-6


def _finish(rec, cfg, system, direction, entry_px, init_stop, tp1, tp2,
            exit_px, outcome, e_t, x_t, pnl, scaled):
    return _mk_trade(rec, system, direction, entry_px, init_stop, tp1, tp2,
                     exit_px, outcome, e_t, x_t, pnl, scaled)


# ─── target geometry per system ───────────────────────────────────────────────

def _reversal_targets(rec, cfg, direction):
    """TP1 = Pivot; TP2 = outer pivot, modulated by the width regime (Ch. 6)."""
    pp, tc, bc = rec["pp"], rec["tc"], rec["bc"]
    regime = _regime(rec, cfg)
    if direction == "short":
        if regime == "wide":
            return pp, bc                       # wide day → near edge only
        if regime == "narrow":
            return pp, rec["s2"]                # narrow day → run to S2
        return pp, rec["s1"]
    else:
        if regime == "wide":
            return pp, tc
        if regime == "narrow":
            return pp, rec["r2"]
        return pp, rec["r1"]


# ─── the three systems (entry logic) ──────────────────────────────────────────

def _sim_reversal(rec, cfg):
    po, ph, pl, pc, pt = rec["po"], rec["ph"], rec["pl"], rec["pc"], rec["pt"]
    n = len(pt)
    tc, pp, bc = rec["tc"], rec["pp"], rec["bc"]
    tol = cfg["touch_pct"] / 100.0 * rec["pp"]
    buf = cfg["buf_pct"] / 100.0 * rec["pp"]
    e0, e1, eff = cfg["entry_start_t"], cfg["entry_end_t"], cfg["eff_min"]
    short_set, long_set = cfg["rev_short_rel"], cfg["rev_long_rel"]

    # trap-day guard: an Inside-Value open beyond yesterday's range is a breakout
    if cfg["skip_trap"] and rec["relationship"] == "Inside Value" and rec["open_beyond"]:
        return []

    max_trades = cfg.get("max_per_day", 1) if cfg.get("allow_reentry") else 1
    trades = []
    pending = None
    next_i = 0
    for i in range(n):
        if pending is not None and i == pending["fill_i"]:
            d = pending["dir"]
            entry_px = po[i]
            tp1, tp2 = _reversal_targets(rec, cfg, d)
            if d == "short":
                stop = pending["sig_hi"] + buf
                ok = stop > entry_px and tp2 < entry_px
            else:
                stop = pending["sig_lo"] - buf
                ok = stop < entry_px and tp2 > entry_px
            pending = None
            if ok:
                tr, exit_i = _manage(rec, cfg, "reversal", d, i, entry_px, stop,
                                     tp1, tp2, use_trail=False)
                trades.append(tr)
                if len(trades) >= max_trades:
                    break
                next_i = exit_i + 1            # no overlapping positions
            continue

        o, h, l, c, tm = po[i], ph[i], pl[i], pc[i], int(pt[i])
        if i < next_i or tm < e0 or tm > e1 or tm >= eff or i + 1 >= n:
            continue

        # SHORT: opened below the CPR, rallied into the zone, bearish rejection
        if rec["day_open"] < bc and rec["relationship"] in short_set:
            into_zone = h >= pp and h <= tc + tol
            if into_zone and _bear_trigger(o, h, l, c, cfg["wick_mult"], cfg["close_loc"],
                                           cfg["doji_on"], cfg["doji_frac"]):
                pending = {"dir": "short", "fill_i": i + 1, "sig_hi": h, "sig_lo": l}
                continue
        # LONG: opened above the CPR, pulled into the zone, bullish rejection
        if rec["day_open"] > tc and rec["relationship"] in long_set:
            loc_min = (1.0 - (1.0 - cfg["close_loc"]) * 0.5) if cfg["strong_long"] else cfg["close_loc"]
            mult = cfg["wick_mult"] * (1.2 if cfg["strong_long"] else 1.0)
            into_zone = l <= pp and l >= bc - tol
            if into_zone and _bull_trigger(o, h, l, c, mult, loc_min,
                                           cfg["doji_on"], cfg["doji_frac"]):
                pending = {"dir": "long", "fill_i": i + 1, "sig_hi": h, "sig_lo": l}
                continue
    return trades


def _sim_trend(rec, cfg):
    mode = cfg["trend_mode"]
    trades = []
    if mode in ("A", "both"):
        trades = _trend_pullback(rec, cfg)
    if not trades and mode in ("B", "both"):
        trades = _trend_breakout(rec, cfg)
    return trades


def _trend_pullback(rec, cfg):
    po, ph, pl, pc, pt = rec["po"], rec["ph"], rec["pl"], rec["pc"], rec["pt"]
    n = len(pt)
    tc, pp, bc = rec["tc"], rec["pp"], rec["bc"]
    tol = cfg["touch_pct"] / 100.0 * rec["pp"]
    buf = cfg["buf_pct"] / 100.0 * rec["pp"]
    e0, e1, eff = cfg["entry_start_t"], cfg["entry_end_t"], cfg["eff_min"]
    up_set, dn_set = cfg["trend_up_rel"], cfg["trend_dn_rel"]

    max_trades = cfg.get("max_per_day", 1) if cfg.get("allow_reentry") else 1
    trades = []
    pending = None
    next_i = 0
    for i in range(n):
        if pending is not None and i == pending["fill_i"]:
            d = pending["dir"]
            entry_px = po[i]
            if d == "long":
                stop = min(pending["sig_lo"] - buf, bc - buf)
                tp1, tp2 = rec["r1"], rec["r2"]
                ok = stop < entry_px and tp2 > entry_px
            else:
                stop = max(pending["sig_hi"] + buf, tc + buf)
                tp1, tp2 = rec["s1"], rec["s2"]
                ok = stop > entry_px and tp2 < entry_px
            pending = None
            if ok:
                tr, exit_i = _manage(rec, cfg, "trend", d, i, entry_px, stop,
                                     tp1, tp2, use_trail=False)
                trades.append(tr)
                if len(trades) >= max_trades:
                    break
                next_i = exit_i + 1
            continue

        o, h, l, c, tm = po[i], ph[i], pl[i], pc[i], int(pt[i])
        if i < next_i or tm < e0 or tm > e1 or tm >= eff or i + 1 >= n:
            continue

        # uptrend: above CPR, bullish bias, pull back to TC, bullish wick
        if rec["day_open"] > tc and rec["relationship"] in up_set:
            at_tc = abs(l - tc) <= tol or (l <= tc <= h)
            if at_tc and _bull_trigger(o, h, l, c, cfg["wick_mult"], cfg["close_loc"],
                                       cfg["doji_on"], cfg["doji_frac"]):
                pending = {"dir": "long", "fill_i": i + 1, "sig_hi": h, "sig_lo": l}
                continue
        # downtrend: below CPR, bearish bias, pull back to BC, bearish wick
        if rec["day_open"] < bc and rec["relationship"] in dn_set:
            at_bc = abs(h - bc) <= tol or (l <= bc <= h)
            if at_bc and _bear_trigger(o, h, l, c, cfg["wick_mult"], cfg["close_loc"],
                                       cfg["doji_on"], cfg["doji_frac"]):
                pending = {"dir": "short", "fill_i": i + 1, "sig_hi": h, "sig_lo": l}
                continue
    return trades


def _trend_breakout(rec, cfg):
    """Mode B: open beyond yesterday's range + Inside/Unchanged → break the
    first-15-min range in the open's direction; ride with a 1.5×ATR trail."""
    if not cfg["use_breakout"]:
        return []
    if rec["relationship"] not in cfg["breakout_rel"] or not rec["open_beyond"]:
        return []
    ph, pl, pt = rec["ph"], rec["pl"], rec["pt"]
    n = len(pt)
    f15_h, f15_l = rec["f15_high"], rec["f15_low"]
    buf = cfg["buf_pct"] / 100.0 * rec["pp"]
    f15_end = cfg["open_t"] + 15
    eff = cfg["eff_min"]
    longb = rec["open_above_pdh"]
    if not longb and not rec["open_below_pdl"]:
        return []

    for i in range(n):
        tm = int(pt[i])
        if tm < f15_end or tm >= eff:
            continue
        if longb and ph[i] >= f15_h:
            entry_px = f15_h
            stop = f15_l - buf
            tp2 = rec["r3"] if _regime(rec, cfg) == "narrow" else rec["r2"]
            if stop < entry_px:
                tr, _ = _manage(rec, cfg, "trend", "long", i, entry_px, stop,
                                None, tp2, use_trail=True)
                return [tr]
            return []
        if (not longb) and pl[i] <= f15_l:
            entry_px = f15_l
            stop = f15_h + buf
            tp2 = rec["s3"] if _regime(rec, cfg) == "narrow" else rec["s2"]
            if stop > entry_px:
                tr, _ = _manage(rec, cfg, "trend", "short", i, entry_px, stop,
                                None, tp2, use_trail=True)
                return [tr]
            return []
    return []


def _sim_scalp(rec, cfg):
    """Range regime only: fade the fences (R1/TC top, S1/BC bottom) with tight
    stops, target the middle Pivot then the opposite fence, re-entry allowed,
    kill switch on a decisive close beyond a fence."""
    # regime gate
    gate = (_regime(rec, cfg) == "wide") or (rec["relationship"] in cfg["scalp_rel"])
    if cfg["scalp_block_breakout"] and rec["open_beyond"]:
        gate = False
    if not gate:
        return []

    po, ph, pl, pc, pt = rec["po"], rec["ph"], rec["pl"], rec["pc"], rec["pt"]
    n = len(pt)
    pp = rec["pp"]
    top = rec["r1"] if cfg["scalp_fence"] == "S1/R1" else rec["tc"]
    bot = rec["s1"] if cfg["scalp_fence"] == "S1/R1" else rec["bc"]
    tol = cfg["touch_pct"] / 100.0 * rec["pp"]
    buf = cfg["buf_pct"] / 100.0 * rec["pp"]
    e0, e1, eff = cfg["scalp_start_t"], cfg["entry_end_t"], cfg["eff_min"]

    trades = []
    pending = None
    dead = False
    next_i = 0
    for i in range(n):
        if pending is not None and i == pending["fill_i"]:
            d = pending["dir"]
            entry_px = po[i]
            # book's scalp target is CLOSE & certain — the middle Pivot — not the
            # far opposite fence. TP1 banks half halfway to P; runner targets P.
            if d == "short":
                stop = pending["sig_hi"] + buf
                tp2 = pp if pp < entry_px else bot
                tp1 = entry_px - 0.5 * (entry_px - tp2)
                ok = stop > entry_px and tp2 < entry_px
            else:
                stop = pending["sig_lo"] - buf
                tp2 = pp if pp > entry_px else top
                tp1 = entry_px + 0.5 * (tp2 - entry_px)
                ok = stop < entry_px and tp2 > entry_px
            pending = None
            if ok:
                tr, exit_i = _manage(rec, cfg, "scalp", d, i, entry_px, stop,
                                     tp1, tp2, use_trail=False)
                trades.append(tr)
                next_i = exit_i + 1            # no overlapping scalps
            continue

        o, h, l, c, tm = po[i], ph[i], pl[i], pc[i], int(pt[i])
        # kill switch: decisive close beyond a fence → stop scalping
        if c > top + tol or c < bot - tol:
            dead = True
        if dead or i < next_i or tm < e0 or tm > e1 or tm >= eff or i + 1 >= n:
            continue
        if not (cfg["allow_reentry"] or not trades):
            continue
        # fade upper fence (short)
        if (abs(h - top) <= tol or (l <= top <= h)) and \
                _bear_trigger(o, h, l, c, cfg["wick_mult"], cfg["close_loc"],
                              cfg["doji_on"], cfg["doji_frac"]):
            pending = {"dir": "short", "fill_i": i + 1, "sig_hi": h, "sig_lo": l}
            continue
        # fade lower fence (long)
        if (abs(l - bot) <= tol or (l <= bot <= h)) and \
                _bull_trigger(o, h, l, c, cfg["wick_mult"], cfg["close_loc"],
                              cfg["doji_on"], cfg["doji_frac"]):
            pending = {"dir": "long", "fill_i": i + 1, "sig_hi": h, "sig_lo": l}
            continue
    return trades


def sim_day(rec, cfg) -> list:
    sysname = cfg["system"]
    if sysname == "reversal":
        return _sim_reversal(rec, cfg)
    if sysname == "trend":
        return _sim_trend(rec, cfg)
    if sysname == "scalp":
        return _sim_scalp(rec, cfg)
    return []


def run(prepped: list, cfg: dict) -> pd.DataFrame:
    rows = []
    for rec in prepped:
        rows.extend(sim_day(rec, cfg))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["date", "entry_t"]).reset_index(drop=True)


# ─── metrics + equity ─────────────────────────────────────────────────────────

def metrics(trades: pd.DataFrame, n_days: int) -> dict | None:
    if trades is None or trades.empty:
        return None
    pnl = trades["pnl"].to_numpy(float)
    r = trades["r"].to_numpy(float)
    wins = pnl > 0
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    cum = np.cumsum(pnl)
    dd = cum - np.maximum.accumulate(cum)
    by_day = trades.groupby(trades["date"].dt.normalize())["pnl"].sum().to_numpy()
    sharpe = (by_day.mean() / by_day.std() * np.sqrt(TRADING_DAYS)
              if len(by_day) > 1 and by_day.std() > 0 else float("nan"))
    return {
        "n_days": int(n_days), "trades": int(len(pnl)),
        "wins": int(wins.sum()), "losses": int((~wins).sum()),
        "win_rate": float(wins.mean()),
        "net": float(pnl.sum()), "net_r": float(r.sum()),
        "expectancy": float(pnl.mean()), "avg_r": float(r.mean()),
        "pf": float(gp / gl) if gl > 0 else float("inf"),
        "max_dd": float(-dd.min()) if len(dd) else 0.0,
        "avg_win": float(pnl[wins].mean()) if wins.any() else 0.0,
        "avg_loss": float(pnl[~wins].mean()) if (~wins).any() else 0.0,
        "sharpe": float(sharpe),
    }


def equity_curve(trades: pd.DataFrame, prepped: list) -> pd.DataFrame:
    all_days = pd.to_datetime(pd.Series([r["date"] for r in prepped])).dt.normalize()
    all_days = pd.Series(sorted(all_days.unique()))
    if trades is None or trades.empty:
        return pd.DataFrame({"date": all_days, "equity": 0.0})
    by_day = trades.groupby(trades["date"].dt.normalize())["pnl"].sum()
    eq = pd.Series(0.0, index=all_days.values)
    eq.loc[by_day.index] = by_day.values
    return pd.DataFrame({"date": eq.index, "equity": eq.cumsum().values})


# ─── defaults + optimizer ─────────────────────────────────────────────────────

def default_cfg(open_t: int, close_t: int, ib_min: int) -> dict:
    """Sensible NIFTY starting config. Thresholds in % of price (Ch. 6)."""
    return dict(
        system="reversal", open_t=open_t, close_t=close_t, ib_min=ib_min,
        eff_min=close_t,
        entry_start_t=open_t + 15, entry_end_t=14 * 60 + 30,
        scalp_start_t=open_t + ib_min,           # scalp only after IB confirms
        # triggers
        wick_mult=2.5, close_loc=0.65, doji_on=True, doji_frac=0.10,
        strong_long=True,
        # geometry (all % of price). buf_pct is the stop distance BEYOND the
        # signal candle: on an intraday index a too-tight stop whipsaws the fade,
        # so the CPR edge (price closing back inside the range) needs ~0.15%.
        touch_pct=0.05, buf_pct=0.15,
        narrow_pct=0.12, wide_pct=0.25,
        scale_out=True, be_after_tp1=True, atr_mult=1.5,
        # bias-filter sets. On NIFTY the validated edge is the SHORT fade (~65%);
        # the long fade (~59%) tested as a net drag, so longs default OFF —
        # add relationships to the long set to enable them.
        rev_short_rel=["Lower Value", "Overlapping Lower", "Outside Value"],
        rev_long_rel=[],
        skip_trap=True,
        trend_mode="A", use_breakout=True,
        trend_up_rel=["Higher Value", "Overlapping Higher"],
        trend_dn_rel=["Lower Value", "Overlapping Lower"],
        breakout_rel=["Inside Value", "Unchanged"],
        scalp_rel=["Outside Value", "Unchanged"], scalp_fence="S1/R1",
        scalp_block_breakout=True, allow_reentry=True, max_per_day=3,
    )


def optimize(prepped: list, cfg: dict, grids: dict, min_trades: int, rank_by: str,
             progress=None) -> pd.DataFrame:
    """
    Sweep the trigger / geometry knobs that matter for the selected system:
        wick_mult × close_loc × buf_pct × scale_out × be_after_tp1
    Everything else stays as configured. Min-trades guard; ranked.
    """
    combos = list(itertools.product(
        grids["wick_mult"], grids["close_loc"], grids["buf_pct"],
        grids["scale_out"], grids["be"]))
    n_days = len(prepped)
    rows = []
    for k, (wm, cl, buf, so, be) in enumerate(combos):
        c = copy.deepcopy(cfg)
        c.update(wick_mult=wm, close_loc=cl, buf_pct=buf, scale_out=so, be_after_tp1=be)
        trades = run(prepped, c)
        if len(trades) >= min_trades:
            m = metrics(trades, n_days)
            rows.append({
                "wick×": wm, "close-loc": cl, "buf %": buf,
                "scale-out": "on" if so else "off", "BE": "on" if be else "off",
                "trades": m["trades"], "win %": round(m["win_rate"] * 100, 1),
                "net pts": round(m["net"], 0), "net R": round(m["net_r"], 1),
                "PF": round(m["pf"], 2) if np.isfinite(m["pf"]) else 99,
                "expectancy": round(m["expectancy"], 2),
                "avg R": round(m["avg_r"], 3), "max DD": round(m["max_dd"], 0),
            })
        if progress:
            progress(k + 1, len(combos))
    if not rows:
        return pd.DataFrame()
    col = {"net": "net pts", "net_r": "net R", "win_rate": "win %",
           "pf": "PF", "expectancy": "expectancy"}[rank_by]
    return pd.DataFrame(rows).sort_values(col, ascending=False).reset_index(drop=True)


LOG_COLS = ["date", "dow", "system", "direction", "entry time", "exit time",
            "entry", "stop", "tp1", "tp2", "exit", "outcome", "pnl", "r",
            "relationship", "width_pct"]
