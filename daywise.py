"""
daywise.py — Day-wise IB Retracement strategy engine (pure, no Streamlit)
========================================================================
Each weekday has its own config:

    trade        : take trades this weekday at all
    entry        : retracement entry level, % of IB range (0 = breakout at boundary)
    stop         : retracement stop level, % of IB range  (must be > entry)
    tp1, tp2     : extension targets, % of IB range
    min_size     : min IB size as % of price  (range / day_open * 100)
    max_size     : max IB size as % of price
    allow_long   : take long setups   (reference = IB High)
    allow_short  : take short setups   (reference = IB Low)
    cutoff       : latest entry time, minutes-since-midnight, or None

Prices (LONG, points):
    H, L = IB high/low,  Rg = H - L
    entry = H - entry/100*Rg
    stop  = H - stop /100*Rg          (deeper into the box)
    tp1   = H + tp1 /100*Rg ,  tp2 = H + tp2/100*Rg     (extensions, above the box)
SHORT is the mirror off L.

Exit (scale-out): half the position at TP1, then stop → breakeven, remaining
half at TP2.  Per-trade P&L is in points for 1 unit (so a full TP1+TP2 winner
books 0.5*(TP1-entry) + 0.5*(TP2-entry)).  Conservative tie rule: if stop and a
target are both touched in the same candle, the stop is assumed hit first.
"""
import numpy as np
import pandas as pd

import indicators as ind

DEFAULT_OPEN_T  = 9 * 60 + 15   # 09:15 (NSE); auto-detected per instrument in the app
DEFAULT_CLOSE_T = 15 * 60 + 15  # 15:15 square-off — positions force-exit, no overnight carry
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def detect_open_t(min_df):
    """Session open = most common first-candle minute across days."""
    return int(min_df.groupby("date_only")["t_min"].min().mode().iloc[0])

# Defaults mirror the user's screenshots
DEFAULT_CONFIG = {
    "Monday":    dict(trade=True, entry=10, stop=50,  tp1=100, tp2=125, min_size=0.0, max_size=1.0, allow_long=True,  allow_short=False, cutoff=None),
    "Tuesday":   dict(trade=True, entry=10, stop=50,  tp1=25,  tp2=150, min_size=0.0, max_size=2.0, allow_long=True,  allow_short=False, cutoff=None),
    "Wednesday": dict(trade=True, entry=0,  stop=100, tp1=25,  tp2=50,  min_size=0.0, max_size=1.0, allow_long=False, allow_short=True,  cutoff=None),
    "Thursday":  dict(trade=True, entry=50, stop=100, tp1=50,  tp2=75,  min_size=0.0, max_size=2.0, allow_long=False, allow_short=True,  cutoff=None),
    "Friday":    dict(trade=True, entry=30, stop=100, tp1=50,  tp2=75,  min_size=0.0, max_size=2.0, allow_long=True,  allow_short=True,  cutoff=None),
}


def prep_days(min_df: pd.DataFrame, open_t=DEFAULT_OPEN_T, close_t=DEFAULT_CLOSE_T,
              ib_min=60) -> dict:
    """
    Group minute data into per-day records keyed by weekday.
    Each record: H, L, Rg, day_open, close, ph/pl/pt (post-IB arrays), date, dow.
    open_t  : session open (IB starts here).  close_t : square-off (no carry past this).
    ib_min  : Initial-Balance window length in minutes.
    """
    ib_end = open_t + ib_min - 1
    min_bars = min(20, max(5, int(ib_min * 0.66)))
    # session VWAP for the optional "exit on close beyond VWAP" rule
    if "vwap" not in min_df.columns:
        min_df = min_df.copy()
        min_df["vwap"] = ind.session_vwap(min_df).to_numpy()
    out = {d: [] for d in WEEKDAYS}
    for day, g0 in min_df.groupby("date_only", sort=True):
        dow = pd.Timestamp(day).day_name()
        if dow not in out:
            continue
        # session window only → enforces the square-off
        sess = g0[(g0["t_min"] >= open_t) & (g0["t_min"] <= close_t)].sort_values("t_min")
        win  = sess[sess["t_min"] <= ib_end]
        post = sess[sess["t_min"] > ib_end]
        if len(win) < min_bars or post.empty:
            continue
        H, L = win["high"].max(), win["low"].min()
        Rg = H - L
        if Rg <= 0:
            continue
        # IB context for the optional directional filters: which extreme of the
        # IB formed first, and whether the IB's last candle closed above its mid
        t_hi = win.loc[win["high"] >= H, "t_min"].min()
        t_lo = win.loc[win["low"] <= L, "t_min"].min()
        if t_hi < t_lo:
            first_side = "high"
        elif t_lo < t_hi:
            first_side = "low"
        else:                       # both in the same candle → use its colour
            c0 = win.loc[win["t_min"] == t_hi].iloc[0]
            first_side = "low" if c0["close"] >= c0["open"] else "high"
        ib_close = float(win.iloc[-1]["close"])

        # first 120 session minutes (t_min + close arrays) — lets the optional
        # "first N-min candle green/red" filter use ANY window at runtime
        f0 = sess[sess["t_min"] <= sess["t_min"].iloc[0] + 119]

        out[dow].append({
            "date": pd.Timestamp(day), "dow": dow,
            "H": float(H), "L": float(L), "Rg": float(Rg),
            "day_open": float(sess.iloc[0]["open"]),
            "close": float(sess.iloc[-1]["close"]),     # close AT the square-off
            "first_side": first_side,
            "close_above_mid": ib_close > (H + L) / 2,
            "ft": f0["t_min"].values.astype(float),
            "fc": f0["close"].values.astype(float),
            "ph": post["high"].values.astype(float),
            "pl": post["low"].values.astype(float),
            "pt": post["t_min"].values.astype(float),
            "pc": post["close"].values.astype(float),     # post-IB closes (VWAP exit)
            "pv": post["vwap"].values.astype(float),       # post-IB session VWAP
        })
    return out


def _vwap_cross(close_px, vwap_px, is_long):
    """True when a bar CLOSES on the wrong side of VWAP for the open position:
    long → close below VWAP; short → close above VWAP."""
    if vwap_px is None or np.isnan(vwap_px):
        return False
    return (close_px < vwap_px) if is_long else (close_px > vwap_px)


def _sim_dir(ph, pl, start, entry, stop, tp1, tp2, is_long, close,
             pc=None, pv=None, vwap_exit=False):
    """
    Scale-out simulation from entry bar `start`.
    Returns (pnl_points, outcome, exit_bar_index, final_exit_price).

    Intrabar stop/target take priority (conservative); the optional VWAP-close
    exit is evaluated at the bar's CLOSE, after stop/target, and closes the whole
    remaining position (full pre-TP1, the runner post-TP1).
    """
    use_vwap = vwap_exit and pv is not None and pc is not None
    realized = 0.0
    tp1_hit = False
    cur_stop = stop
    n = len(ph)
    i = start
    while i < n:
        hi_b, lo_b = ph[i], pl[i]
        if not tp1_hit:
            stop_hit  = (lo_b <= cur_stop) if is_long else (hi_b >= cur_stop)
            tp1_reach = (hi_b >= tp1) if is_long else (lo_b <= tp1)
            tp2_reach = (hi_b >= tp2) if is_long else (lo_b <= tp2)
            if stop_hit:                                   # conservative: stop before tp
                realized += (cur_stop - entry) if is_long else (entry - cur_stop)
                return realized, "stop", i, cur_stop
            if tp1_reach:
                realized += 0.5 * ((tp1 - entry) if is_long else (entry - tp1))
                tp1_hit = True
                cur_stop = entry                           # move to breakeven
                if tp2_reach:                              # both targets same bar
                    realized += 0.5 * ((tp2 - entry) if is_long else (entry - tp2))
                    return realized, "tp1+tp2", i, tp2
        else:
            be_hit    = (lo_b <= cur_stop) if is_long else (hi_b >= cur_stop)
            tp2_reach = (hi_b >= tp2) if is_long else (lo_b <= tp2)
            if be_hit:
                return realized, "tp1+be", i, cur_stop     # runner stopped at breakeven
            if tp2_reach:
                realized += 0.5 * ((tp2 - entry) if is_long else (entry - tp2))
                return realized, "tp1+tp2", i, tp2
        # optional VWAP-close exit (whole remaining position, at this bar's close)
        if use_vwap and _vwap_cross(pc[i], pv[i], is_long):
            rem = 0.5 if tp1_hit else 1.0
            realized += rem * ((pc[i] - entry) if is_long else (entry - pc[i]))
            return realized, ("tp1+vwap" if tp1_hit else "vwap"), i, pc[i]
        i += 1
    # End-of-day: exit whatever remains at the close
    rem = 0.5 if tp1_hit else 1.0
    realized += rem * ((close - entry) if is_long else (entry - close))
    return realized, ("tp1+eod" if tp1_hit else "eod"), n - 1, close


def trade_pnl(rec, entry_pct, stop_pct, tp1_pct, tp2_pct, is_long, cutoff,
              entry_cond="any", vwap_exit=False):
    """
    One directional trade on one day. Returns (pnl, outcome, entry, risk) or None.
    entry_cond:
      "any"          — fill whenever the entry level trades (default)
      "no_breach"    — only BEFORE the reference IB boundary breaks (pure retracement;
                       the resting order is cancelled once IB High breaks for longs /
                       IB Low for shorts)
      "after_breach" — only AFTER the boundary breaks (breakout-retest: longs need the
                       IB High broken first, shorts the IB Low)
    """
    if stop_pct <= entry_pct:                              # stop must be deeper than entry
        return None
    H, L, Rg = rec["H"], rec["L"], rec["Rg"]
    if is_long:
        entry = H - entry_pct / 100 * Rg
        stop  = H - stop_pct  / 100 * Rg
        tp1   = H + tp1_pct   / 100 * Rg
        tp2   = H + tp2_pct   / 100 * Rg
    else:
        entry = L + entry_pct / 100 * Rg
        stop  = L + stop_pct  / 100 * Rg
        tp1   = L - tp1_pct   / 100 * Rg
        tp2   = L - tp2_pct   / 100 * Rg

    ph, pl, pt = rec["ph"], rec["pl"], rec["pt"]
    mask = (pl <= entry) & (entry <= ph)                   # candle trades through entry level
    if cutoff is not None:
        mask &= (pt <= cutoff)
    if entry_cond != "any":
        breach = (ph > H) if is_long else (pl < L)         # reference boundary break
        b_idx = int(np.argmax(breach)) if breach.any() else None
        order = np.arange(len(ph))
        if entry_cond == "no_breach":
            if b_idx is not None:
                mask &= (order < b_idx)                    # cancel order once breached
        else:                                              # after_breach (breakout-retest)
            if b_idx is None:
                return None
            mask &= (order >= b_idx)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    pnl, outcome, exit_i, exit_px = _sim_dir(ph, pl, idx[0], entry, stop, tp1, tp2,
                                             is_long, rec["close"],
                                             rec.get("pc"), rec.get("pv"), vwap_exit)
    extras = dict(entry_t=float(pt[idx[0]]), exit_t=float(pt[exit_i]),
                  exit_px=float(exit_px), stop_px=float(stop),
                  tp1_px=float(tp1), tp2_px=float(tp2))
    return pnl, outcome, entry, abs(entry - stop), extras


def dir_params(cfg, is_long):
    """
    Entry/stop/tp1/tp2 for one direction. If cfg['separate'] is set, long and short
    use their own long_*/short_* values; otherwise both use the shared entry/stop/tp1/tp2.
    """
    if cfg.get("separate"):
        p = "long" if is_long else "short"
        return (cfg.get(f"{p}_entry", cfg["entry"]),
                cfg.get(f"{p}_stop",  cfg["stop"]),
                cfg.get(f"{p}_tp1",   cfg["tp1"]),
                cfg.get(f"{p}_tp2",   cfg["tp2"]))
    return cfg["entry"], cfg["stop"], cfg["tp1"], cfg["tp2"]


def first_candle_dir(rec, n_min):
    """
    Direction of the session's first n-minute candle:
    close at minute n vs the session open.  +1 green · -1 red · 0 doji/unknown.
    """
    ft, fc = rec.get("ft"), rec.get("fc")
    if ft is None or fc is None or len(ft) == 0:
        return 0
    mask = ft <= ft[0] + n_min - 1
    if not mask.any():
        return 0
    diff = fc[mask][-1] - rec["day_open"]
    return 1 if diff > 0 else (-1 if diff < 0 else 0)


def dir_allowed(rec, is_long, dir_filters):
    """
    Optional IB directional filters (all default off):
      long_first_low    — LONG only if the IB LOW formed first (high later)
      long_close_above  — LONG only if the IB closed above its midpoint
      long_first_green  — LONG only if the first `candle_min`-min candle is GREEN
      short_first_high  — SHORT only if the IB HIGH formed first (low later)
      short_close_below — SHORT only if the IB closed below its midpoint
      short_first_red   — SHORT only if the first `candle_min`-min candle is RED
    """
    if not dir_filters:
        return True
    n_min = int(dir_filters.get("candle_min", 15))
    if is_long:
        if dir_filters.get("long_first_low") and rec.get("first_side") != "low":
            return False
        if dir_filters.get("long_close_above") and not rec.get("close_above_mid", True):
            return False
        if dir_filters.get("long_first_green") and first_candle_dir(rec, n_min) != 1:
            return False
    else:
        if dir_filters.get("short_first_high") and rec.get("first_side") != "high":
            return False
        if dir_filters.get("short_close_below") and rec.get("close_above_mid", False):
            return False
        if dir_filters.get("short_first_red") and first_candle_dir(rec, n_min) != -1:
            return False
    return True


def sim_day(rec, cfg, entry_cond="any", dir_filters=None, vwap_exit=False):
    """All trades for one day under its weekday config (0, 1 or 2 directional trades)."""
    if not cfg.get("trade", True):
        return []
    size_pct = rec["Rg"] / rec["day_open"] * 100
    if not (cfg["min_size"] <= size_pct <= cfg["max_size"]):
        return []
    cutoff = cfg.get("cutoff")
    trades = []
    for is_long in (True, False):
        if is_long and not cfg["allow_long"]:
            continue
        if (not is_long) and not cfg["allow_short"]:
            continue
        if not dir_allowed(rec, is_long, dir_filters):
            continue
        e, s, t1, t2 = dir_params(cfg, is_long)
        res = trade_pnl(rec, e, s, t1, t2, is_long, cutoff, entry_cond, vwap_exit)
        if res is None:
            continue
        pnl, outcome, entry, risk, ext = res
        trades.append({
            "date": rec["date"], "dow": rec["dow"],
            "direction": "long" if is_long else "short",
            "entry": round(entry, 2), "size_pct": round(size_pct, 3),
            "risk": round(risk, 2),
            "r_mult": round(pnl / risk, 3) if risk > 0 else np.nan,
            "pnl": round(pnl, 2), "outcome": outcome, "win": pnl > 0,
            # for the visual trade browser
            "entry_t": ext["entry_t"], "exit_t": ext["exit_t"],
            "exit_px": round(ext["exit_px"], 2),
            "stop_px": round(ext["stop_px"], 2),
            "tp1_px": round(ext["tp1_px"], 2), "tp2_px": round(ext["tp2_px"], 2),
            "ib_high": rec["H"], "ib_low": rec["L"],
        })
    return trades


def run(prepped: dict, configs: dict, entry_cond="any", dir_filters=None,
        vwap_exit=False) -> pd.DataFrame:
    """Run the full day-wise strategy across all weekdays. Returns trades DataFrame."""
    rows = []
    for dow, day_list in prepped.items():
        cfg = configs.get(dow)
        if not cfg:
            continue
        for rec in day_list:
            rows.extend(sim_day(rec, cfg, entry_cond, dir_filters, vwap_exit))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ─── per-weekday optimizer ────────────────────────────────────────────────────

OPT_GRID = dict(
    entry=[0, 10, 25, 50],
    stop=[50, 75, 100],
    tp1=[25, 50, 75],
    tp2=[75, 100, 125, 150],
)


def _metric(arr, name):
    if len(arr) == 0:
        return None
    wins = arr > 0
    if name == "win_rate":
        return wins.mean()
    if name == "expectancy":
        return arr.mean()
    if name == "net":
        return arr.sum()
    if name == "pf":
        gl = -arr[arr < 0].sum()
        return (arr[arr > 0].sum() / gl) if gl > 0 else np.inf
    return arr.mean()


def optimize_weekday(day_list, directions, grid, metric, min_trades, cutoff=None,
                     entry_cond="any", dir_filters=None, vwap_exit=False):
    """
    Sweep entry/stop/tp1/tp2 × direction for one weekday's days.
    `directions` is a subset of {"long","short"}. Returns a ranked DataFrame.
    """
    rows = []
    for entry in grid["entry"]:
        for stop in grid["stop"]:
            if stop <= entry:
                continue
            for tp1 in grid["tp1"]:
                for tp2 in grid["tp2"]:
                    if tp2 <= tp1:
                        continue
                    for d in directions:
                        is_long = d == "long"
                        pnls = []
                        for rec in day_list:
                            if not dir_allowed(rec, is_long, dir_filters):
                                continue
                            res = trade_pnl(rec, entry, stop, tp1, tp2, is_long,
                                            cutoff, entry_cond, vwap_exit)
                            if res is not None:
                                pnls.append(res[0])
                        if len(pnls) < min_trades:
                            continue
                        arr = np.array(pnls)
                        rows.append({
                            "direction": d, "entry": entry, "stop": stop,
                            "tp1": tp1, "tp2": tp2,
                            "trades": len(arr),
                            "win %": round((arr > 0).mean() * 100, 1),
                            "expectancy": round(arr.mean(), 2),
                            "net": round(arr.sum(), 0),
                            "pf": round(_metric(arr, "pf"), 2) if np.isfinite(_metric(arr, "pf")) else 99,
                        })
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    col = {"win_rate": "win %", "expectancy": "expectancy", "net": "net", "pf": "pf"}[metric]
    return res.sort_values(col, ascending=False).reset_index(drop=True)
