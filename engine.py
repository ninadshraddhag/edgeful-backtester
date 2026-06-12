"""
engine.py — Composable rules-based backtesting engine (Advanced mode core)
==========================================================================
Three stages, all pure (no Streamlit):

1. build_features()      — execution-timeframe frame decorated with every
   indicator the user's conditions reference (EMAs/RSIs at arbitrary periods,
   CPR, PDH/PDL, daily ATR, day context) plus HTF columns aligned with
   merge_asof so a bar only ever sees an HTF candle that closed strictly
   before it — zero forward-looking bias.

2. Condition system      — the user composes an UNLIMITED list of rows like
       Price  sweeps         PDL
       Price  above          HTF EMA 20
       Price  touches        EMA 20
       RSI    above 60
       CPR width is NARROW
       Price  moved ≥ 1×ATR above day-open
       Fair Value Gap forms
   Entry = logical AND of every row (written from the LONG perspective;
   shorts evaluate the automatic mirror: above↔below, PDH↔PDL, etc.).
   Exits reuse the same vocabulary, OR-combined ("exit when ANY is true").

3. backtest() / optimize() — event-driven sim: fixed-fractional sizing,
   structural stops (FVG-invalidation / swing / PDH-PDL / CPR edge / fixed %)
   triggered on wick-touch OR candle-close (the "SL = candle close below FVG"
   style), targets as R-multiple / opposite-liquidity / none, trailing stop,
   indicator-exit signals, intraday square-off. optimize() sweeps target ×
   stop × leak-free day filters and ranks configurations like the Edge
   Backtester's permutation optimizer.

Day-context filters expose ONLY what is knowable in real time: today's gap
(known at the open), the PREVIOUS day's kind, weekday. Today's inside/outside
classification needs the finished day's range, so it is deliberately absent.
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
import pandas as pd

import indicators as ind
import ict
import build_facts

TRADING_DAYS = 252
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# ─── condition vocabulary ─────────────────────────────────────────────────────

LEVEL_COLS = {
    "PDH": "pdh", "PDL": "pdl",
    "CPR Top": "cpr_top", "CPR Bottom": "cpr_bottom", "CPR Pivot": "cpr_p",
    "Prev Close": "prev_close",
}
LEVEL_MIRROR = {
    "PDH": "PDL", "PDL": "PDH",
    "CPR Top": "CPR Bottom", "CPR Bottom": "CPR Top",
    "CPR Pivot": "CPR Pivot", "Prev Close": "Prev Close",
}
OP_MIRROR = {
    "above": "below", "below": "above", "touches": "touches",
    "crosses above": "crosses below", "crosses below": "crosses above",
    "breaks above": "breaks below", "breaks below": "breaks above",
    "sweeps": "sweeps",
}

OPS_EMA   = ["above", "below", "touches", "crosses above", "crosses below"]
OPS_LEVEL = ["above", "below", "touches", "breaks above", "breaks below", "sweeps"]
OPS_RSI   = ["above", "below", "crosses above", "crosses below"]
OPS_ATR   = ["above day-open by", "below day-open by", "day range at least"]


# ─── feature assembly ─────────────────────────────────────────────────────────

def _align_htf(exec_frame: pd.DataFrame, htf_frame: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Backward as-of join of `cols` from htf_frame onto exec_frame with NO leakage:
    each execution bar is matched to the most recent HTF candle whose close_time
    is strictly earlier than the execution bar's own close_time.
    """
    left = exec_frame[["close_time"]].copy()
    left["_pos"] = np.arange(len(left))
    left = left.sort_values("close_time")

    right = htf_frame[["close_time"] + list(cols)].sort_values("close_time")
    merged = pd.merge_asof(left, right, on="close_time",
                           direction="backward", allow_exact_matches=False)
    merged = merged.sort_values("_pos")
    out = pd.DataFrame(index=exec_frame.index)
    for c in cols:
        out[c] = merged[c].to_numpy()
    return out


def build_features(df1m: pd.DataFrame, open_t: int, params: dict) -> pd.DataFrame:
    """
    Build the execution-timeframe feature frame from a cleaned 1-min frame.
    Indicator periods come from the user's condition rows, so only what's
    referenced is computed:
        exec_tf, htf_tf, ema_periods, htf_ema_periods, rsi_periods,
        htf_rsi_periods, cpr_period, cpr_narrow, cpr_wide, atr_period,
        fvg_min_gap_pct, ob_lookback
    """
    exec_tf = params.get("exec_tf", "5-min")
    htf_tf  = params.get("htf_tf", "15-min")
    ema_periods     = sorted({int(p) for p in params.get("ema_periods", ())})
    htf_ema_periods = sorted({int(p) for p in params.get("htf_ema_periods", ())})
    rsi_periods     = sorted({int(p) for p in params.get("rsi_periods", ())})
    htf_rsi_periods = sorted({int(p) for p in params.get("htf_rsi_periods", ())})
    atr_period = int(params.get("atr_period", 14))

    X = ind.resample_ohlc(df1m, exec_tf, open_t).copy()
    X["t_min"] = X["open_time"].dt.hour * 60 + X["open_time"].dt.minute
    X["dow"] = X["open_time"].dt.day_name()

    for p in ema_periods:
        X[f"ema_{p}"] = ind.ema(X["close"], p)
    for p in rsi_periods:
        X[f"rsi_{p}"] = ind.rsi(X["close"], p)

    # key levels & CPR (both strictly prior-period → no leak)
    levels = ind.key_levels(df1m)
    X["pdh"] = X["date_only"].map(levels["pdh"])
    X["pdl"] = X["date_only"].map(levels["pdl"])
    X["prev_close"] = X["date_only"].map(levels["prev_close"])

    # session VWAP — computed on 1-min bars, sampled at each exec bar's close
    # time (the bar's last constituent minute), so it carries no lookahead
    vwap_1m = ind.session_vwap(df1m)
    vwap_ser = pd.Series(vwap_1m.to_numpy(), index=df1m["date"].to_numpy())
    X["vwap"] = X["close_time"].map(vwap_ser)

    cpr = ind.central_pivot_range(df1m, params.get("cpr_period", "Daily"),
                                  params.get("cpr_narrow", 0.30),
                                  params.get("cpr_wide", 0.75))
    for c in ["cpr_p", "cpr_top", "cpr_bottom", "cpr_width_pct", "cpr_type"]:
        X[c] = X["date_only"].map(cpr[c])

    # daily context: ATR (Wilder, prior days only), gap type, PREV day kind
    g = df1m.groupby("date_only", sort=True)
    daily = pd.DataFrame({"o": g["open"].first(), "h": g["high"].max(),
                          "l": g["low"].min(), "c": g["close"].last()})
    pc, ph, pl = daily["c"].shift(1), daily["h"].shift(1), daily["l"].shift(1)
    tr = np.maximum(daily["h"] - daily["l"],
                    np.maximum((daily["h"] - pc).abs(), (daily["l"] - pc).abs()))
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False,
                 min_periods=atr_period).mean().shift(1)
    gap_pct = (daily["o"] - pc) / pc * 100
    thr = build_facts.GAP_THRESHOLD
    gap_type = pd.Series(np.where(gap_pct > thr, "Gap Up",
                         np.where(gap_pct < -thr, "Gap Down", "Flat")),
                         index=daily.index, dtype=object)
    gap_type[gap_pct.isna()] = "n/a"
    inside  = (daily["h"] <= ph) & (daily["l"] >= pl)
    outside = (daily["h"] > ph) & (daily["l"] < pl)
    kind = pd.Series(np.where(inside, "Inside Day",
                     np.where(outside, "Outside Day", "Normal")),
                     index=daily.index, dtype=object)
    prev_kind = kind.shift(1).fillna("n/a")

    X["atr"] = X["date_only"].map(atr)
    X["day_open"] = X["date_only"].map(daily["o"])
    X["gap_type"] = X["date_only"].map(gap_type)
    X["prev_day_kind"] = X["date_only"].map(prev_kind)

    # ICT structural signals on the execution timeframe
    fvg = ict.fair_value_gaps(X, params.get("fvg_min_gap_pct", 0.0))
    ob  = ict.order_blocks(X, params.get("ob_lookback", 5))
    X = pd.concat([X, fvg, ob], axis=1)

    # higher-timeframe columns (leak-free as-of aligned)
    H = ind.resample_ohlc(df1m, htf_tf, open_t).copy()
    htf_cols = []
    for p in htf_ema_periods:
        H[f"htf_ema_{p}"] = ind.ema(H["close"], p)
        htf_cols.append(f"htf_ema_{p}")
    for p in htf_rsi_periods:
        H[f"htf_rsi_{p}"] = ind.rsi(H["close"], p)
        htf_cols.append(f"htf_rsi_{p}")
    if htf_cols:
        X = pd.concat([X, _align_htf(X, H, htf_cols)], axis=1)

    # halve memory (matters on Streamlit Cloud for XAUUSD's multi-million rows);
    # float32 keeps ~7 significant digits — ample for index/futures/gold prices
    for c in X.columns:
        if X[c].dtype == np.float64:
            X[c] = X[c].astype(np.float32)

    X.attrs["exec_tf"], X.attrs["htf_tf"] = exec_tf, htf_tf
    X.attrs["ema_periods"] = ema_periods
    return X


# ─── condition evaluation ─────────────────────────────────────────────────────

def _price_vs(X: pd.DataFrame, lvl: pd.Series, op: str, side: str) -> np.ndarray:
    """Compare price action to a (possibly moving) level under operator `op`."""
    cl = X["close"].to_numpy(float)
    hi = X["high"].to_numpy(float)
    lo = X["low"].to_numpy(float)
    l = lvl.to_numpy(float)
    n = len(X)
    pc  = np.r_[np.nan, cl[:-1]]
    plv = np.r_[np.nan, l[:-1]]

    if op == "above":
        m = cl > l
    elif op == "below":
        m = cl < l
    elif op == "touches":
        m = (lo <= l) & (hi >= l)
    elif op == "crosses above":
        m = (cl > l) & (pc <= plv)
    elif op == "crosses below":
        m = (cl < l) & (pc >= plv)
    elif op == "breaks above":
        # first time TODAY the high exceeds the level
        prior_max = (X.groupby("date_only")["high"].cummax()
                     .groupby(X["date_only"]).shift(1).to_numpy(float))
        m = (hi > l) & ~(prior_max > l)
    elif op == "breaks below":
        prior_min = (X.groupby("date_only")["low"].cummin()
                     .groupby(X["date_only"]).shift(1).to_numpy(float))
        m = (lo < l) & ~(prior_min < l)
    elif op == "sweeps":
        # stop-hunt: wick beyond the level, close back on the trade's side
        m = (lo < l) & (cl > l) if side == "long" else (hi > l) & (cl < l)
    else:
        m = np.ones(n, dtype=bool)
    return np.where(np.isnan(l), False, m)


def evaluate_condition(X: pd.DataFrame, cond: dict, side: str = "long") -> np.ndarray:
    """
    Per-bar boolean mask for one condition row. Conditions are written from
    the LONG perspective; side="short" evaluates the automatic mirror.
    """
    t = cond["type"]
    long = side == "long"
    n = len(X)

    if t == "price_ema" or t == "price_htf_ema":
        p = int(cond.get("period", 20))
        col = (f"ema_{p}" if t == "price_ema" else f"htf_ema_{p}")
        if col not in X:
            return np.zeros(n, dtype=bool)
        op = cond.get("op", "above")
        if not long:
            op = OP_MIRROR[op]
        return _price_vs(X, X[col], op, side)

    if t == "price_level":
        name = cond.get("level", "PDL")
        op = cond.get("op", "sweeps")
        if not long:
            name, op = LEVEL_MIRROR[name], OP_MIRROR[op]
        return _price_vs(X, X[LEVEL_COLS[name]], op, side)

    if t == "price_vwap":
        if "vwap" not in X:
            return np.zeros(n, dtype=bool)
        op = cond.get("op", "above")
        if not long:
            op = OP_MIRROR[op]
        return _price_vs(X, X["vwap"], op, side)

    if t == "rsi":
        p = int(cond.get("period", 14))
        col = f"rsi_{p}" if cond.get("tf", "exec") == "exec" else f"htf_rsi_{p}"
        if col not in X:
            return np.zeros(n, dtype=bool)
        r = X[col].to_numpy(float)
        pr = np.r_[np.nan, r[:-1]]
        thr = float(cond.get("threshold", 50))
        op = cond.get("op", "above")
        if not long:                       # symmetric mirror: above 60 → below 40
            op = OP_MIRROR[op]
            thr = 100.0 - thr
        if op == "above":
            m = r > thr
        elif op == "below":
            m = r < thr
        elif op == "crosses above":
            m = (r > thr) & (pr <= thr)
        else:
            m = (r < thr) & (pr >= thr)
        return np.where(np.isnan(r), False, m)

    if t == "cpr_width":
        allowed = cond.get("allowed", ["NARROW"])
        return X["cpr_type"].isin(allowed).to_numpy()

    if t == "fvg":
        return X["bull_fvg"].to_numpy() if long else X["bear_fvg"].to_numpy()

    if t == "ob":
        return X["bull_ob"].to_numpy() if long else X["bear_ob"].to_numpy()

    if t == "atr_move":
        a = X["atr"].to_numpy(float)
        do = X["day_open"].to_numpy(float)
        cl = X["close"].to_numpy(float)
        k = float(cond.get("k", 1.0))
        op = cond.get("op", "above day-open by")
        if not long and op != "day range at least":
            op = ("below day-open by" if op == "above day-open by"
                  else "above day-open by")
        if op == "above day-open by":
            m = cl >= do + k * a
        elif op == "below day-open by":
            m = cl <= do - k * a
        else:                              # day range at least k×ATR (so far)
            run_hi = X.groupby("date_only")["high"].cummax().to_numpy(float)
            run_lo = X.groupby("date_only")["low"].cummin().to_numpy(float)
            m = (run_hi - run_lo) >= k * a
        return np.where(np.isnan(a), False, m)

    if t == "day_type":                    # direction-neutral, leak-free filters
        m = np.ones(n, dtype=bool)
        if cond.get("gaps"):
            m &= X["gap_type"].isin(cond["gaps"]).to_numpy()
        if cond.get("prev_kinds"):
            m &= X["prev_day_kind"].isin(cond["prev_kinds"]).to_numpy()
        if cond.get("dows"):
            m &= X["dow"].isin(cond["dows"]).to_numpy()
        return m

    if t == "time_window":
        tm = X["t_min"].to_numpy()
        return (tm >= int(cond.get("t1", 555))) & (tm <= int(cond.get("t2", 870)))

    if t == "time_after":                  # exit-only: square at/after a time
        return (X["t_min"].to_numpy() >= int(cond.get("t", 870)))

    return np.ones(n, dtype=bool)


def entry_signals(X, conds, long_on=True, short_on=True):
    """AND of every condition row per side; both-true bars are cancelled."""
    n = len(X)
    if not conds:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    def side_mask(side, on):
        if not on:
            return np.zeros(n, dtype=bool)
        m = np.ones(n, dtype=bool)
        for c in conds:
            m &= evaluate_condition(X, c, side)
        return m

    L, S = side_mask("long", long_on), side_mask("short", short_on)
    both = L & S
    return L & ~both, S & ~both


def exit_signals(X, conds, side):
    """OR of exit-condition rows (written for longs; shorts auto-mirrored)."""
    n = len(X)
    if not conds:
        return np.zeros(n, dtype=bool)
    m = np.zeros(n, dtype=bool)
    for c in conds:
        m |= evaluate_condition(X, c, side)
    return m


# ─── stop placement ───────────────────────────────────────────────────────────

def _initial_stop(X, i, entry, is_long, ex):
    """
    Initial stop referencing signal bar i. Structural stops are padded by
    `sl_buffer_pct` of the entry→level distance (scales with the setup).
    Falls back to swing if the chosen structure isn't usable.
    """
    kind = ex.get("sl_type", "fvg")
    buf = float(ex.get("sl_buffer_pct", 5.0)) / 100.0

    def pad(level):
        dist = abs(entry - level)
        return (level - dist * buf) if is_long else (level + dist * buf)

    def usable(level):
        return pd.notna(level) and ((level < entry) if is_long else (level > entry))

    if kind == "fvg":
        lvl = X["fvg_gap_bottom"].iloc[i] if is_long else X["fvg_gap_top"].iloc[i]
        if usable(lvl):
            return pad(lvl)
        kind = "swing"
    if kind == "pdh_pdl":
        lvl = X["pdl"].iloc[i] if is_long else X["pdh"].iloc[i]
        if usable(lvl):
            return pad(lvl)
        kind = "swing"
    if kind == "cpr_edge":
        lvl = X["cpr_bottom"].iloc[i] if is_long else X["cpr_top"].iloc[i]
        if usable(lvl):
            return pad(lvl)
        kind = "swing"
    if kind == "swing":
        lb = int(ex.get("swing_lookback", 10))
        lo = X["low"].iloc[max(0, i - lb):i + 1].min()
        hi = X["high"].iloc[max(0, i - lb):i + 1].max()
        return pad(lo if is_long else hi)

    pct = float(ex.get("sl_pct", 0.4)) / 100.0
    return entry * (1 - pct) if is_long else entry * (1 + pct)


# ─── event-driven simulation ──────────────────────────────────────────────────

def backtest(X: pd.DataFrame, strategy: dict, close_t: int = 15 * 60 + 15):
    """
    One position at a time. Per-bar priority: new-day square-off → stop →
    target → indicator-exit (close) → session square-off → trail update.
    Conservative: a bar spanning both stop and target books the stop.

    strategy = {
      long_enabled, short_enabled, entry_conds:[...],
      entry: {type: immediate|next_open},
      exit:  {sl_type, sl_trigger: touch|close, sl_buffer_pct, sl_pct,
              swing_lookback, target_mode: rr|liquidity|none, tp_R,
              exit_conds:[...], trail_on, trail_trigger_R, trail_dist_R},
      risk:  {capital, risk_pct},
      signal_filter: optional per-bar bool mask (optimizer day filters),
    }
    """
    ex = strategy.get("exit", {})
    risk = strategy.get("risk", {})
    capital0 = float(risk.get("capital", 1_000_000))
    risk_pct = float(risk.get("risk_pct", 1.0))
    tp_R = float(ex.get("tp_R", 2.0))
    target_mode = ex.get("target_mode", "rr")
    sl_trigger = ex.get("sl_trigger", "touch")
    trail_on = bool(ex.get("trail_on", False))
    trail_trigger = float(ex.get("trail_trigger_R", 1.0))
    trail_dist = float(ex.get("trail_dist_R", 1.0))
    entry_type = strategy.get("entry", {}).get("type", "immediate")

    longs, shorts = entry_signals(X, strategy.get("entry_conds", []),
                                  strategy.get("long_enabled", True),
                                  strategy.get("short_enabled", True))
    filt = strategy.get("signal_filter")
    if filt is not None:
        longs &= filt
        shorts &= filt
    ex_conds = ex.get("exit_conds") or []
    exL = exit_signals(X, ex_conds, "long")
    exS = exit_signals(X, ex_conds, "short")

    op = X["open"].to_numpy(float)
    hi = X["high"].to_numpy(float)
    lo = X["low"].to_numpy(float)
    cl = X["close"].to_numpy(float)
    tmin = X["t_min"].to_numpy()
    dates = X["date_only"].to_numpy()
    times = X["open_time"].to_numpy()
    pdh = X["pdh"].to_numpy(float)
    pdl = X["pdl"].to_numpy(float)
    n = len(X)

    sig = np.where(longs | shorts)[0]
    equity = capital0
    trades = []
    ptr = 0
    while ptr < len(sig):
        i = int(sig[ptr])
        is_long = bool(longs[i])
        if tmin[i] >= close_t or i >= n - 1:
            ptr += 1
            continue

        if entry_type == "next_open":
            e_bar = i + 1
            if e_bar >= n or dates[e_bar] != dates[i] or tmin[e_bar] >= close_t:
                ptr += 1
                continue
            entry = op[e_bar]
        else:
            e_bar = i
            entry = cl[i]

        stop = _initial_stop(X, i, entry, is_long, ex)
        rpu = abs(entry - stop)
        if rpu <= 0 or not np.isfinite(rpu):
            ptr += 1
            continue

        if target_mode == "liquidity":     # opposite liquidity: PDH long / PDL short
            lvl = pdh[e_bar] if is_long else pdl[e_bar]
            ok = np.isfinite(lvl) and ((lvl > entry) if is_long else (lvl < entry))
            target = lvl if ok else (entry + tp_R * rpu if is_long else entry - tp_R * rpu)
        elif target_mode == "rr":
            target = entry + tp_R * rpu if is_long else entry - tp_R * rpu
        else:
            target = None

        qty = (equity * risk_pct / 100.0) / rpu
        cur_stop = stop
        best = entry
        exit_price = outcome = None
        exm = exL if is_long else exS

        j = e_bar + 1
        while j < n:
            if dates[j] != dates[e_bar]:               # never carry overnight
                j -= 1
                exit_price, outcome = cl[j], "squareoff"
                break

            if is_long:
                stop_hit = (lo[j] <= cur_stop) if sl_trigger == "touch" else (cl[j] <= cur_stop)
            else:
                stop_hit = (hi[j] >= cur_stop) if sl_trigger == "touch" else (cl[j] >= cur_stop)
            if stop_hit:
                exit_price = cur_stop if sl_trigger == "touch" else cl[j]
                if abs(cur_stop - stop) < 1e-9:
                    outcome = "stop"
                elif abs(cur_stop - entry) < 1e-9:
                    outcome = "breakeven"
                else:
                    outcome = "trail"
                break

            if target is not None:
                if (is_long and hi[j] >= target) or (not is_long and lo[j] <= target):
                    exit_price, outcome = target, "target"
                    break

            if exm[j]:                                  # indicator exit at the close
                exit_price, outcome = cl[j], "exit-signal"
                break

            if tmin[j] >= close_t:
                exit_price, outcome = cl[j], "squareoff"
                break

            if trail_on:
                if is_long:
                    best = max(best, hi[j])
                    if (best - entry) >= trail_trigger * rpu:
                        cur_stop = max(cur_stop, best - trail_dist * rpu)
                else:
                    best = min(best, lo[j])
                    if (entry - best) >= trail_trigger * rpu:
                        cur_stop = min(cur_stop, best + trail_dist * rpu)
            j += 1

        if exit_price is None:                          # ran out of data
            j = n - 1
            exit_price, outcome = cl[j], "eod"

        pnl_pts = (exit_price - entry) if is_long else (entry - exit_price)
        pnl_cash = pnl_pts * qty
        equity += pnl_cash
        trades.append({
            "entry_time": pd.Timestamp(times[e_bar]),
            "exit_time": pd.Timestamp(times[min(j, n - 1)]),
            "date": pd.Timestamp(dates[e_bar]),
            "dow": pd.Timestamp(dates[e_bar]).day_name(),
            "direction": "long" if is_long else "short",
            "entry": round(float(entry), 2),
            "stop": round(float(stop), 2),
            "target": round(float(target), 2) if target is not None else np.nan,
            "exit": round(float(exit_price), 2),
            "qty": round(float(qty), 4),
            "risk_pts": round(float(rpu), 2),
            "pnl_pts": round(float(pnl_pts), 2),
            "r_mult": round(float(pnl_pts / rpu), 3),
            "pnl": round(float(pnl_cash), 2),
            "outcome": outcome,
            "win": pnl_cash > 0,
            "equity": round(float(equity), 2),
        })
        ptr = int(np.searchsorted(sig, j + 1, side="left"))

    trades_df = pd.DataFrame(trades)
    eq_df = _equity_curve(trades_df, X, capital0)
    return trades_df, eq_df


def _equity_curve(trades_df, X, capital0):
    """Daily cash equity curve over the full window (for ratio metrics)."""
    all_days = pd.to_datetime(pd.Series(sorted(pd.unique(X["date_only"]))))
    if trades_df.empty:
        return pd.DataFrame({"date": all_days, "equity": capital0})
    by_day = trades_df.groupby(trades_df["date"].dt.normalize())["pnl"].sum()
    eq = pd.Series(0.0, index=all_days.dt.normalize().values)
    eq.loc[by_day.index] = by_day.values
    equity = capital0 + eq.cumsum()
    return pd.DataFrame({"date": equity.index, "equity": equity.values})


# ─── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(trades_df: pd.DataFrame, eq_df: pd.DataFrame,
                    capital0: float = 1_000_000) -> dict | None:
    """Full institutional metric set from the trade log + daily equity curve."""
    if trades_df is None or trades_df.empty:
        return None
    pnl = trades_df["pnl"].to_numpy(float)
    wins = pnl > 0
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()

    eq = eq_df["equity"].to_numpy(float)
    dates = pd.to_datetime(eq_df["date"])
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    dd_pct = dd / peak
    max_dd = dd.min()
    max_dd_pct = dd_pct.min()

    trough_i = int(np.argmin(dd))
    peak_i = int(np.argmax(eq[:trough_i + 1])) if trough_i > 0 else 0
    rec_i = trough_i
    while rec_i < len(eq) and eq[rec_i] < peak[trough_i]:
        rec_i += 1
    dd_duration = (dates.iloc[trough_i] - dates.iloc[peak_i]).days
    recovered = rec_i < len(eq)
    recovery_days = (dates.iloc[rec_i] - dates.iloc[trough_i]).days if recovered else None

    rets = pd.Series(eq, index=dates).pct_change().dropna().to_numpy()
    sharpe = (rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)
              if len(rets) > 1 and rets.std() > 0 else float("nan"))
    downside = rets[rets < 0]
    sortino = (rets.mean() / downside.std() * np.sqrt(TRADING_DAYS)
               if len(downside) > 0 and downside.std() > 0 else float("nan"))

    span_days = max((dates.iloc[-1] - dates.iloc[0]).days, 1)
    years = span_days / 365.25
    final_eq = eq[-1]
    cagr = (final_eq / capital0) ** (1 / years) - 1 if final_eq > 0 and years > 0 else float("nan")

    return {
        "trades": int(len(pnl)),
        "net_profit": float(pnl.sum()),
        "net_pct": float(pnl.sum() / capital0 * 100),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "max_dd_pct": float(max_dd_pct * 100),
        "dd_duration_days": int(dd_duration),
        "recovery_days": recovery_days,
        "win_rate": float(wins.mean()),
        "profit_factor": float(gp / gl) if gl > 0 else float("inf"),
        "expectancy": float(pnl.mean()),
        "avg_win": float(pnl[wins].mean()) if wins.any() else 0.0,
        "avg_loss": float(pnl[~wins].mean()) if (~wins).any() else 0.0,
        "avg_r": float(trades_df["r_mult"].mean()),
        "final_equity": float(final_eq),
    }


def quick_metrics(trades_df: pd.DataFrame) -> dict:
    """Cheap trade-based metrics for optimizer ranking (no equity curve)."""
    p = trades_df["pnl"].to_numpy(float)
    wins = p > 0
    gl = -p[p < 0].sum()
    cum = np.cumsum(p)
    dd = (cum - np.maximum.accumulate(cum)).min() if len(p) else 0.0
    pf = (p[p > 0].sum() / gl) if gl > 0 else np.inf
    return {
        "trades": int(len(p)),
        "win %": round(float(wins.mean() * 100), 1),
        "expectancy": round(float(p.mean()), 0),
        "net": round(float(p.sum()), 0),
        "PF": round(float(pf), 2) if np.isfinite(pf) else 99.0,
        "max DD": round(float(dd), 0),
    }


# ─── optimizer ────────────────────────────────────────────────────────────────

def optimize(X: pd.DataFrame, strategy: dict, close_t: int,
             tp_grid, sl_grid, sweep_dow: bool, sweep_gap: bool, sweep_kind: bool,
             min_trades: int, rank_by: str, progress=None) -> pd.DataFrame:
    """
    Sweep target-R × stop-type × (leak-free) day filters over the user's fixed
    entry-condition stack; rank by the chosen metric with a min-trades guard.
    Entry signals are computed once — only exits/filters vary per combo.
    """
    dows  = ([None] + WEEKDAYS) if sweep_dow else [None]
    gaps  = [None, "Gap Up", "Gap Down", "Flat"] if sweep_gap else [None]
    kinds = ([None, "Inside Day", "Outside Day", "Normal"]) if sweep_kind else [None]
    if strategy.get("exit", {}).get("target_mode") != "rr":
        # R is only a fallback for liquidity/none targets — sweeping it would
        # just duplicate rows, so collapse the grid to the configured value
        tp_grid = [strategy.get("exit", {}).get("tp_R", 2.0)]
    combos = list(itertools.product(tp_grid, sl_grid, dows, gaps, kinds))

    dow_a = X["dow"].to_numpy()
    gap_a = X["gap_type"].to_numpy()
    kind_a = X["prev_day_kind"].to_numpy()
    n = len(X)

    rows = []
    for c_i, (tp, slt, d, gp_, kd) in enumerate(combos):
        s = copy.deepcopy(strategy)
        s["exit"]["tp_R"] = float(tp)
        s["exit"]["sl_type"] = slt
        filt = np.ones(n, dtype=bool)
        if d:
            filt &= dow_a == d
        if gp_:
            filt &= gap_a == gp_
        if kd:
            filt &= kind_a == kd
        s["signal_filter"] = filt

        trades, _ = backtest(X, s, close_t)
        if len(trades) < min_trades:
            if progress:
                progress(c_i + 1, len(combos))
            continue
        qm = quick_metrics(trades)
        rows.append({"target R": tp, "stop": slt, "weekday": d or "all",
                     "gap": gp_ or "any", "prev day": kd or "any", **qm})
        if progress:
            progress(c_i + 1, len(combos))

    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    col = {"expectancy": "expectancy", "net": "net", "win_rate": "win %",
           "pf": "PF"}[rank_by]
    return res.sort_values(col, ascending=False).reset_index(drop=True)
