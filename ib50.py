"""
ib50.py — IB Retracement strategy engine (Python port of the Pine v6 indicator)
===============================================================================
Faithful port of the "IB Retracement Indicator" virtual-trade engine. Per day:

  • Initial Balance = first `ib_min` minutes from the session open. Record IB
    high/low/range, the IB close, and which extreme formed FIRST
    (low-first = +1 bull bias, high-first = -1 bear bias; tie → bull).
  • Session VWAP (hlc3, volume-weighted with a TWAP fallback) anchored at the
    session open — exactly like the Pine `cumPV/cumVol` reset at ibStart.
  • Direction bias = AND/OR vote of the enabled filters:
        formation order · IB-close location (% of range) · price-vs-VWAP.
    The VWAP vote is evaluated per bar (it can change intraday), so bias is
    re-checked every bar — matching the Pine engine.
  • Optional breach gate (Require Breached / Not-Breached, Directional or
    EitherSide) using the after-IB break state.
  • Entry / Stop / Target are % of the IB range, measured from the IB extreme.
  • Risk-based sizing: a full stop loses `risk_usd` (-1R); P&L is reported in $
    and R. Bar-by-bar exit order: SL → TP → VWAP-close → square-off (EOD).

Pure (no Streamlit). One trade at a time per day (re-entry optional).
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
import pandas as pd

import daywise                      # green_months (month-on-month consistency)

import indicators as ind

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TRADING_DAYS = 252
VWAP_EXIT_TF = 5    # the VWAP-close exit is evaluated on this timeframe (vote/entries 1-min)


# ─── per-day preparation ──────────────────────────────────────────────────────

def prep_days(min_df: pd.DataFrame, open_t: int, close_t: int, ib_min: int,
              tf: int = 1) -> list:
    """
    Build per-day records from a cleaned minute frame. Session VWAP is anchored
    at the session open (open_t) per day — the Pine `cumPV` reset at ibStart.

    Each record carries the IB statics plus post-IB arrays (bars after the IB,
    within the trading/square-off window): t_min, high, low, close, vwap.
    """
    ib_end = open_t + ib_min - 1
    min_bars = min(20, max(3, int(ib_min / max(tf, 1) * 0.66)))
    has_vol = "volume" in min_df.columns
    out = []
    for day, g0 in min_df.groupby("date_only", sort=True):
        dow = pd.Timestamp(day).day_name()
        sess = g0[(g0["t_min"] >= open_t) & (g0["t_min"] <= close_t)].sort_values("t_min")
        win = sess[sess["t_min"] <= ib_end]
        post = sess[sess["t_min"] > ib_end]
        if len(win) < min_bars or post.empty:
            continue
        H = float(win["high"].max())
        L = float(win["low"].min())
        rng = H - L
        if rng <= 0:
            continue
        # which extreme formed first inside the IB (by minute)
        t_hi = win.loc[win["high"] >= H, "t_min"].min()
        t_lo = win.loc[win["low"] <= L, "t_min"].min()
        first_side = 1 if t_lo < t_hi else (-1 if t_hi < t_lo else 1)
        ib_close = float(win.iloc[-1]["close"])

        # first-session-candle direction (15/30/60-min): close of the first N-min
        # candle vs the session open. Up = bullish vote, down = bearish (a static
        # per-day directional vote like formation).
        day_open = float(sess.iloc[0]["open"])
        fc = {}
        for n in (15, 30, 60):
            cser = sess.loc[sess["t_min"] == open_t + n - 1, "close"]
            fc[n] = (0 if cser.empty else
                     (1 if cser.iloc[0] > day_open else (-1 if cser.iloc[0] < day_open else 0)))

        # session VWAP anchored at the session open (whole session, then sliced)
        src = (sess["high"] + sess["low"] + sess["close"]) / 3.0
        if has_vol:
            vol = pd.to_numeric(sess["volume"], errors="coerce").fillna(0.0)
            if vol.sum() > 0:
                vwap = (src * vol).cumsum() / vol.cumsum().replace(0.0, np.nan)
            else:
                vwap = src.expanding().mean()
        else:
            vwap = src.expanding().mean()
        post_mask = sess["t_min"] > ib_end
        ph_arr = post["high"].to_numpy(float)
        pl_arr = post["low"].to_numpy(float)
        # cumulative breach state (config-independent → precompute once): True at
        # bar i once the IB high/low has been pierced by bar i (inclusive)
        hb = (np.maximum.accumulate(ph_arr > H).astype(bool) if len(ph_arr)
              else np.zeros(0, dtype=bool))
        lb = (np.maximum.accumulate(pl_arr < L).astype(bool) if len(pl_arr)
              else np.zeros(0, dtype=bool))

        # 5-min VWAP step + 5-min-close marker for the VWAP-close EXIT (the
        # direction VOTE keeps the 1-min VWAP `pv`; entries/SL/TP stay 1-min)
        pv5_full, is5_full = ind.vwap5_steps(sess, open_t, VWAP_EXIT_TF)
        pm = post_mask.to_numpy()

        out.append({
            "date": pd.Timestamp(day), "dow": dow,
            "ib_high": H, "ib_low": L, "ib_range": rng, "ib_close": ib_close,
            "first_side": int(first_side), "day_open": day_open,
            "fc15": int(fc[15]), "fc30": int(fc[30]), "fc60": int(fc[60]),
            "pt": post["t_min"].to_numpy(float),
            "ph": ph_arr, "pl": pl_arr,
            "pc": post["close"].to_numpy(float),
            "pv": vwap[post_mask].to_numpy(float),     # 1-min VWAP (direction vote)
            "pv5": pv5_full[pm], "is5": is5_full[pm],  # 5-min VWAP (exit only)
            "hb": hb, "lb": lb,
        })
    return out


# ─── direction bias + breach gate (Pine semantics) ────────────────────────────

def _static_votes(rec, cfg):
    """Per-day constant votes: formation side, IB-close-location and first-candle
    direction, plus the number of enabled filters."""
    f_side = rec["first_side"] if cfg["use_formation"] else 0
    cP = cfg["close_loc_pct"] / 100.0
    c_vote = 0
    if cfg["use_closeloc"] and rec["ib_range"] > 0:
        if rec["ib_close"] > rec["ib_low"] + cP * rec["ib_range"]:
            c_vote = 1
        elif rec["ib_close"] < rec["ib_low"] + (1.0 - cP) * rec["ib_range"]:
            c_vote = -1
    fc_vote = 0
    if cfg.get("use_firstcandle", False):
        fc_vote = rec.get("fc%d" % int(cfg.get("firstcandle_min", 30)), 0)
    n_enabled = (1 if cfg["use_formation"] else 0) + \
                (1 if cfg["use_closeloc"] else 0) + (1 if cfg["use_vwap"] else 0) + \
                (1 if cfg.get("use_firstcandle", False) else 0)
    return f_side, c_vote, fc_vote, n_enabled


def _bias_at(f_side, c_vote, fc_vote, n_enabled, use_vwap, logic, cl, vw):
    """Bias for one bar; VWAP vote uses this bar's close vs VWAP."""
    if n_enabled == 0:
        return 0
    v_vote = 0
    if use_vwap and not np.isnan(vw):
        v_vote = 1 if cl > vw else (-1 if cl < vw else 0)
    n_bull = (f_side == 1) + (c_vote == 1) + (v_vote == 1) + (fc_vote == 1)
    n_bear = (f_side == -1) + (c_vote == -1) + (v_vote == -1) + (fc_vote == -1)
    if logic == "AND":
        return 1 if n_bull == n_enabled else (-1 if n_bear == n_enabled else 0)
    return 1 if (n_bull > 0 and n_bear == 0) else (-1 if (n_bear > 0 and n_bull == 0) else 0)


def _breach_ok(is_long, cfg, hb, lb):
    if not cfg["use_breach"]:
        return True
    req_breached = cfg["breach_mode"] == "Require Breached"
    if cfg["breach_scope"] == "EitherSide":
        either_br, either_in = (hb or lb), (not hb and not lb)
        return either_br if req_breached else either_in
    side = hb if is_long else lb
    return side if req_breached else (not side)


# ─── bar-by-bar virtual-trade engine ──────────────────────────────────────────

def sim_day(rec, cfg) -> list:
    """All trades for one day (0, 1, or several if re-entry). Mirrors the Pine
    per-bar order: breach update → entry (flat only) → exit (in position)."""
    rng = rec["ib_range"]
    if rng <= 0 or np.isnan(rng):
        return []
    ib_high, ib_low, ib_close = rec["ib_high"], rec["ib_low"], rec["ib_close"]
    pt, ph, pl, pc, pv = rec["pt"], rec["ph"], rec["pl"], rec["pc"], rec["pv"]
    pv5, is5 = rec["pv5"], rec["is5"]              # 5-min VWAP for the exit only
    n = len(pt)
    if n == 0:
        return []

    f_side, c_vote, fc_vote, n_enabled = _static_votes(rec, cfg)
    entry, sl, target = cfg["entry_pct"] / 100.0, cfg["sl_pct"] / 100.0, cfg["target_pct"] / 100.0
    eff_min = cfg["eff_min"]
    ew, e0, e1 = cfg["use_entry_window"], cfg["entry_start_t"], cfg["entry_end_t"]
    risk, pvval = cfg["risk_usd"], cfg.get("point_value", 1.0)
    use_vwap, logic = cfg["use_vwap"], cfg["logic"]
    reentry = cfg["allow_reentry"]
    hb, lb = rec["hb"], rec["lb"]                  # precomputed cumulative breach state
    # realism guard: require the breach to be confirmed on a PRIOR (closed) bar, so
    # the breakout candle and the retrace-entry candle can't be the same 1-min bar.
    # DEFAULT TRUE — realistic fills everywhere unless a caller explicitly opts out
    # (the Strategy-tab A/B toggle passes False to show the optimistic comparison).
    bprior = cfg.get("breach_confirm_prior", True)
    # exit controls (all default to the original behaviour):
    be_at = cfg.get("be_at", 0.0)              # breakeven: move stop to entry once the
    #                                            trade is +be_at·R in favour (0 = off)
    vwap_exit_on = cfg.get("vwap_exit", True)  # allow disabling the 5-min VWAP-close exit
    vwap_buf = cfg.get("vwap_buffer", 0.0)     # require the close this many R beyond VWAP

    traded = False
    pos = 0
    e_px = s_px = t_px = qty = np.nan
    e_i = 0
    be_armed = False
    trades = []

    for i in range(n):
        hi, lo, cl, vw, tm = ph[i], pl[i], pc[i], pv[i], int(pt[i])

        # breach state for THIS bar's entry: current-bar-inclusive (default) or, with
        # the realism guard, as-of the previous closed bar (so the entry can't fill on
        # the same candle that first pierced the IB).
        if bprior:
            hbi = hb[i - 1] if i > 0 else False
            lbi = lb[i - 1] if i > 0 else False
        else:
            hbi, lbi = hb[i], lb[i]

        # ── entry (only when flat) ───────────────────────────────────────────
        if pos == 0 and (reentry or not traded) and tm < eff_min and \
                (not ew or (e0 <= tm <= e1)):
            # when breach_confirm_prior is on, also use prior bar's close/VWAP for
            # the direction vote — prevents same-candle VWAP-cross entries
            vcl = pc[i - 1] if (bprior and i > 0) else cl
            vvw = pv[i - 1] if (bprior and i > 0) else vw
            bias = _bias_at(f_side, c_vote, fc_vote, n_enabled, use_vwap, logic, vcl, vvw)
            if bias == 1:
                eP, sP, tP = ib_high - entry * rng, ib_high - sl * rng, ib_high + target * rng
                if sP < eP and tP > eP and _breach_ok(True, cfg, hbi, lbi) \
                        and lo <= eP <= hi:
                    pos, e_px, s_px, t_px, e_i = 1, eP, sP, tP, i
                    qty = risk / (max(abs(eP - sP), 1e-9) * pvval)
                    traded, be_armed = True, False
            elif bias == -1:
                eP, sP, tP = ib_low + entry * rng, ib_low + sl * rng, ib_low - target * rng
                if sP > eP and tP < eP and _breach_ok(False, cfg, hbi, lbi) \
                        and lo <= eP <= hi:
                    pos, e_px, s_px, t_px, e_i = -1, eP, sP, tP, i
                    qty = risk / (max(abs(eP - sP), 1e-9) * pvval)
                    traded, be_armed = True, False

        # ── exit (only when in position; may be the same bar as entry) ───────
        if pos != 0:
            longp = pos == 1
            sd = abs(e_px - s_px)                       # original stop distance (R unit)
            eff_stop = e_px if be_armed else s_px       # active stop (breakeven once armed)
            hit_sl = (lo <= eff_stop) if longp else (hi >= eff_stop)
            hit_tp = (hi >= t_px) if longp else (lo <= t_px)
            # VWAP-close exit: only at a 5-min close, vs the 5-min VWAP (+ optional buffer)
            vwap_exit = (use_vwap and vwap_exit_on and is5[i] and (not np.isnan(pv5[i]))
                         and (cl < pv5[i] - vwap_buf * sd if longp
                              else cl > pv5[i] + vwap_buf * sd))
            eod = tm >= eff_min
            if hit_sl:
                exit_px, reason = eff_stop, ("BE" if be_armed else "SL")
            elif hit_tp:
                exit_px, reason = t_px, "TP"
            elif vwap_exit:
                exit_px, reason = cl, "VWAP"
            elif eod:
                exit_px, reason = cl, "EOD"
            else:
                exit_px, reason = np.nan, None
            if reason is not None:
                trades.append(_mk_trade(rec, pos, e_px, s_px, t_px, exit_px, reason,
                                        qty, risk, pt[e_i], pt[i]))
                pos = 0
            elif be_at > 0 and not be_armed:            # arm breakeven from the NEXT bar
                fav = (hi - e_px) if longp else (e_px - lo)
                if fav >= be_at * sd:
                    be_armed = True

    # leftover open position → square off at the last available close
    if pos != 0:
        trades.append(_mk_trade(rec, pos, e_px, s_px, t_px, pc[-1], "EOD",
                                qty, risk, pt[e_i], pt[-1]))
    return trades


def _mk_trade(rec, pos, e_px, s_px, t_px, exit_px, reason, qty, risk, e_t, x_t):
    stop_d = max(abs(e_px - s_px), 1e-9)
    r = (exit_px - e_px) * pos / stop_d
    pnl = r * risk
    return {
        "date": rec["date"], "dow": rec["dow"],
        "direction": "long" if pos == 1 else "short",
        "entry_t": float(e_t), "exit_t": float(x_t),
        "entry": round(float(e_px), 2), "stop": round(float(s_px), 2),
        "target": round(float(t_px), 2), "exit": round(float(exit_px), 2),
        "qty": round(float(qty), 4), "risk_usd": round(float(risk), 2),
        "r": round(float(r), 3), "pnl": round(float(pnl), 2),
        "outcome": reason, "win": pnl > 0,
        "ib_high": rec["ib_high"], "ib_low": rec["ib_low"],
    }


def run(prepped: list, cfg: dict) -> pd.DataFrame:
    rows = []
    for rec in prepped:
        rows.extend(sim_day(rec, cfg))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ─── metrics ──────────────────────────────────────────────────────────────────

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
    max_dd = -dd.min() if len(dd) else 0.0

    # daily returns for Sharpe (sum P&L per calendar day)
    by_day = trades.groupby(trades["date"].dt.normalize())["pnl"].sum().to_numpy()
    sharpe = (by_day.mean() / by_day.std() * np.sqrt(TRADING_DAYS)
              if len(by_day) > 1 and by_day.std() > 0 else float("nan"))

    return {
        "n_days": int(n_days),
        "trades": int(len(pnl)),
        "wins": int(wins.sum()), "losses": int((~wins).sum()),
        "win_rate": float(wins.mean()),
        "net_pnl": float(pnl.sum()), "net_r": float(r.sum()),
        "expectancy": float(pnl.mean()), "avg_r": float(r.mean()),
        "profit_factor": float(gp / gl) if gl > 0 else float("inf"),
        "max_dd": float(max_dd),
        "max_dd_r": float(max_dd / trades["risk_usd"].iloc[0]) if len(trades) else 0.0,
        "avg_win": float(pnl[wins].mean()) if wins.any() else 0.0,
        "avg_loss": float(pnl[~wins].mean()) if (~wins).any() else 0.0,
        "sharpe": float(sharpe),
    }


def equity_curve(trades: pd.DataFrame, prepped: list) -> pd.DataFrame:
    """Daily cumulative $ P&L over the prepped date range (0 on no-trade days)."""
    all_days = pd.to_datetime(pd.Series([r["date"] for r in prepped])).dt.normalize()
    all_days = pd.Series(sorted(all_days.unique()))
    if trades is None or trades.empty:
        return pd.DataFrame({"date": all_days, "equity": 0.0})
    by_day = trades.groupby(trades["date"].dt.normalize())["pnl"].sum()
    eq = pd.Series(0.0, index=all_days.values)
    eq.loc[by_day.index] = by_day.values
    return pd.DataFrame({"date": eq.index, "equity": eq.cumsum().values})


# ─── optimizer ────────────────────────────────────────────────────────────────

OPT_GRID = dict(entry=[0, 10, 25, 50], stop=[50, 75, 100], target=[50, 75, 100, 150])


def filter_configs(cfg: dict, sweep_form: bool, sweep_close: bool,
                   sweep_vwap: bool, sweep_logic: bool) -> list:
    """
    Deduplicated list of (use_formation, use_closeloc, use_vwap, logic) tuples to
    sweep. Combos with zero filters (no trades) are dropped; AND/OR is collapsed
    when fewer than two filters are enabled (logic is then irrelevant), so no two
    runs are effectively identical.
    """
    form_opts = [True, False] if sweep_form else [cfg["use_formation"]]
    close_opts = [True, False] if sweep_close else [cfg["use_closeloc"]]
    vwap_opts = [True, False] if sweep_vwap else [cfg["use_vwap"]]
    logic_opts = ["AND", "OR"] if sweep_logic else [cfg["logic"]]
    seen, out = set(), []
    for f in form_opts:
        for cl in close_opts:
            for vw in vwap_opts:
                if not (f or cl or vw):
                    continue
                for lg in logic_opts:
                    eff_logic = lg if (f + cl + vw) >= 2 else "AND"
                    key = (f, cl, vw, eff_logic)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(key)
    return out


def optimize(prepped: list, cfg: dict, grid: dict, min_trades: int, rank_by: str,
             sweep_form=False, sweep_close=False, sweep_vwap=False, sweep_logic=False,
             progress=None) -> pd.DataFrame:
    """
    Sweep entry% × stop% × target%  AND (optionally) the direction filters
    (formation / close-location / VWAP / AND-OR). Breach gate, close-level % and
    time windows stay as configured. Ranked by the chosen metric, min-trades
    guard. Each result row is a fully-specified, tradeable configuration.
    """
    fconfigs = filter_configs(cfg, sweep_form, sweep_close, sweep_vwap, sweep_logic)
    geom = [(e, s, t) for e in grid["entry"] for s in grid["stop"]
            for t in grid["target"] if s > e]
    combos = [(fc, g) for fc in fconfigs for g in geom]
    n_days = len(prepped)
    rows = []
    for k, ((f, cl, vw, lg), (e, s, t)) in enumerate(combos):
        c = copy.deepcopy(cfg)
        c["use_formation"], c["use_closeloc"], c["use_vwap"], c["logic"] = f, cl, vw, lg
        c["entry_pct"], c["sl_pct"], c["target_pct"] = e, s, t
        trades = run(prepped, c)
        if len(trades) >= min_trades:
            m = metrics(trades, n_days)
            gm, g, n = daywise.green_months(trades["date"], trades["pnl"])
            rows.append({
                "formation": "on" if f else "off",
                "close-loc": "on" if cl else "off",
                "vwap": "on" if vw else "off",
                "logic": lg if (f + cl + vw) >= 2 else "—",
                "entry %": e, "stop %": s, "target %": t,
                "trades": m["trades"],
                "win %": round(m["win_rate"] * 100, 1),
                "green mo %": round(gm, 1),
                "months": f"{g}/{n}",
                "net R": round(m["net_r"], 1),
                "net $": round(m["net_pnl"], 0),
                "PF": round(m["profit_factor"], 2) if np.isfinite(m["profit_factor"]) else 99,
                "expectancy $": round(m["expectancy"], 1),
                "max DD $": round(m["max_dd"], 0),
                "avg R": round(m["avg_r"], 3),
            })
        if progress:
            progress(k + 1, len(combos))
    if not rows:
        return pd.DataFrame()
    if rank_by == "green_months":
        # month-on-month consistency first, net R as the tie-break
        return (pd.DataFrame(rows)
                .sort_values(["green mo %", "net R"], ascending=False)
                .reset_index(drop=True))
    col = {"net_r": "net R", "net": "net $", "win_rate": "win %",
           "pf": "PF", "expectancy": "expectancy $"}[rank_by]
    return pd.DataFrame(rows).sort_values(col, ascending=False).reset_index(drop=True)
