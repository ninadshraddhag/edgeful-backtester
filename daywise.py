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


def prep_days(min_df: pd.DataFrame, open_t=DEFAULT_OPEN_T, close_t=DEFAULT_CLOSE_T) -> dict:
    """
    Group minute data into per-day records keyed by weekday.
    Each record: H, L, Rg, day_open, close, ph/pl/pt (post-IB arrays), date, dow.
    open_t  : session open (IB starts here).  close_t : square-off (no carry past this).
    """
    ib_end = open_t + 60 - 1
    out = {d: [] for d in WEEKDAYS}
    for day, g0 in min_df.groupby("date_only", sort=True):
        dow = pd.Timestamp(day).day_name()
        if dow not in out:
            continue
        # session window only → enforces the square-off
        sess = g0[(g0["t_min"] >= open_t) & (g0["t_min"] <= close_t)].sort_values("t_min")
        win  = sess[sess["t_min"] <= ib_end]
        post = sess[sess["t_min"] > ib_end]
        if len(win) < 20 or post.empty:
            continue
        H, L = win["high"].max(), win["low"].min()
        Rg = H - L
        if Rg <= 0:
            continue
        out[dow].append({
            "date": pd.Timestamp(day), "dow": dow,
            "H": float(H), "L": float(L), "Rg": float(Rg),
            "day_open": float(sess.iloc[0]["open"]),
            "close": float(sess.iloc[-1]["close"]),     # close AT the square-off
            "ph": post["high"].values.astype(float),
            "pl": post["low"].values.astype(float),
            "pt": post["t_min"].values.astype(float),
        })
    return out


def _sim_dir(ph, pl, start, entry, stop, tp1, tp2, is_long, close):
    """Scale-out simulation from entry bar `start`. Returns (pnl_points, outcome)."""
    realized = 0.0
    tp1_hit = False
    cur_stop = stop
    n = len(ph)
    i = start
    while i < n:
        hi_b, lo_b = ph[i], pl[i]
        if not tp1_hit:
            if is_long:
                stop_hit  = lo_b <= cur_stop
                tp1_reach = hi_b >= tp1
                tp2_reach = hi_b >= tp2
            else:
                stop_hit  = hi_b >= cur_stop
                tp1_reach = lo_b <= tp1
                tp2_reach = lo_b <= tp2
            if stop_hit:                                   # conservative: stop before tp
                realized += (cur_stop - entry) if is_long else (entry - cur_stop)
                return realized, "stop"
            if tp1_reach:
                realized += 0.5 * ((tp1 - entry) if is_long else (entry - tp1))
                tp1_hit = True
                cur_stop = entry                           # move to breakeven
                if tp2_reach:                              # both targets same bar
                    realized += 0.5 * ((tp2 - entry) if is_long else (entry - tp2))
                    return realized, "tp1+tp2"
        else:
            if is_long:
                be_hit    = lo_b <= cur_stop
                tp2_reach = hi_b >= tp2
            else:
                be_hit    = hi_b >= cur_stop
                tp2_reach = lo_b <= tp2
            if be_hit:
                return realized, "tp1+be"                  # runner stopped at breakeven
            if tp2_reach:
                realized += 0.5 * ((tp2 - entry) if is_long else (entry - tp2))
                return realized, "tp1+tp2"
        i += 1
    # End-of-day: exit whatever remains at the close
    rem = 0.5 if tp1_hit else 1.0
    realized += rem * ((close - entry) if is_long else (entry - close))
    return realized, ("tp1+eod" if tp1_hit else "eod")


def trade_pnl(rec, entry_pct, stop_pct, tp1_pct, tp2_pct, is_long, cutoff):
    """One directional trade on one day. Returns (pnl, outcome, entry_price) or None."""
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
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    pnl, outcome = _sim_dir(ph, pl, idx[0], entry, stop, tp1, tp2, is_long, rec["close"])
    return pnl, outcome, entry


def sim_day(rec, cfg):
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
        res = trade_pnl(rec, cfg["entry"], cfg["stop"], cfg["tp1"], cfg["tp2"], is_long, cutoff)
        if res is None:
            continue
        pnl, outcome, entry = res
        trades.append({
            "date": rec["date"], "dow": rec["dow"],
            "direction": "long" if is_long else "short",
            "entry": round(entry, 2), "size_pct": round(size_pct, 3),
            "pnl": round(pnl, 2), "outcome": outcome, "win": pnl > 0,
        })
    return trades


def run(prepped: dict, configs: dict) -> pd.DataFrame:
    """Run the full day-wise strategy across all weekdays. Returns trades DataFrame."""
    rows = []
    for dow, day_list in prepped.items():
        cfg = configs.get(dow)
        if not cfg:
            continue
        for rec in day_list:
            rows.extend(sim_day(rec, cfg))
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


def optimize_weekday(day_list, directions, grid, metric, min_trades, cutoff=None):
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
                            res = trade_pnl(rec, entry, stop, tp1, tp2, is_long, cutoff)
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
