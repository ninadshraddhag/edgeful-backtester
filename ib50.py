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

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TRADING_DAYS = 252


# ─── per-day preparation ──────────────────────────────────────────────────────

def prep_days(min_df: pd.DataFrame, open_t: int, close_t: int, ib_min: int) -> list:
    """
    Build per-day records from a cleaned minute frame. Session VWAP is anchored
    at the session open (open_t) per day — the Pine `cumPV` reset at ibStart.

    Each record carries the IB statics plus post-IB arrays (bars after the IB,
    within the trading/square-off window): t_min, high, low, close, vwap.
    """
    ib_end = open_t + ib_min - 1
    min_bars = min(20, max(5, int(ib_min * 0.66)))
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

        out.append({
            "date": pd.Timestamp(day), "dow": dow,
            "ib_high": H, "ib_low": L, "ib_range": rng, "ib_close": ib_close,
            "first_side": int(first_side), "day_open": float(sess.iloc[0]["open"]),
            "pt": post["t_min"].to_numpy(float),
            "ph": post["high"].to_numpy(float),
            "pl": post["low"].to_numpy(float),
            "pc": post["close"].to_numpy(float),
            "pv": vwap[post_mask].to_numpy(float),
        })
    return out


# ─── direction bias + breach gate (Pine semantics) ────────────────────────────

def _static_votes(rec, cfg):
    """Per-day constant votes: formation side and IB-close-location, plus the
    number of enabled filters."""
    f_side = rec["first_side"] if cfg["use_formation"] else 0
    cP = cfg["close_loc_pct"] / 100.0
    c_vote = 0
    if cfg["use_closeloc"] and rec["ib_range"] > 0:
        if rec["ib_close"] > rec["ib_low"] + cP * rec["ib_range"]:
            c_vote = 1
        elif rec["ib_close"] < rec["ib_low"] + (1.0 - cP) * rec["ib_range"]:
            c_vote = -1
    n_enabled = (1 if cfg["use_formation"] else 0) + \
                (1 if cfg["use_closeloc"] else 0) + (1 if cfg["use_vwap"] else 0)
    return f_side, c_vote, n_enabled


def _bias_at(f_side, c_vote, n_enabled, use_vwap, logic, cl, vw):
    """Bias for one bar; VWAP vote uses this bar's close vs VWAP."""
    if n_enabled == 0:
        return 0
    v_vote = 0
    if use_vwap and not np.isnan(vw):
        v_vote = 1 if cl > vw else (-1 if cl < vw else 0)
    n_bull = (f_side == 1) + (c_vote == 1) + (v_vote == 1)
    n_bear = (f_side == -1) + (c_vote == -1) + (v_vote == -1)
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
    n = len(pt)
    if n == 0:
        return []

    f_side, c_vote, n_enabled = _static_votes(rec, cfg)
    entry, sl, target = cfg["entry_pct"] / 100.0, cfg["sl_pct"] / 100.0, cfg["target_pct"] / 100.0
    eff_min = cfg["eff_min"]
    ew, e0, e1 = cfg["use_entry_window"], cfg["entry_start_t"], cfg["entry_end_t"]
    risk, pvval = cfg["risk_usd"], cfg.get("point_value", 1.0)
    use_vwap, logic = cfg["use_vwap"], cfg["logic"]
    reentry = cfg["allow_reentry"]

    high_br = low_br = False
    traded = False
    pos = 0
    e_px = s_px = t_px = qty = np.nan
    e_i = 0
    trades = []

    for i in range(n):
        hi, lo, cl, vw, tm = ph[i], pl[i], pc[i], pv[i], int(pt[i])
        if hi > ib_high:
            high_br = True
        if lo < ib_low:
            low_br = True

        # ── entry (only when flat) ───────────────────────────────────────────
        if pos == 0 and (reentry or not traded) and tm < eff_min and \
                (not ew or (e0 <= tm <= e1)):
            bias = _bias_at(f_side, c_vote, n_enabled, use_vwap, logic, cl, vw)
            if bias == 1:
                eP, sP, tP = ib_high - entry * rng, ib_high - sl * rng, ib_high + target * rng
                if sP < eP and tP > eP and _breach_ok(True, cfg, high_br, low_br) \
                        and lo <= eP <= hi:
                    pos, e_px, s_px, t_px, e_i = 1, eP, sP, tP, i
                    qty = risk / (max(abs(eP - sP), 1e-9) * pvval)
                    traded = True
            elif bias == -1:
                eP, sP, tP = ib_low + entry * rng, ib_low + sl * rng, ib_low - target * rng
                if sP > eP and tP < eP and _breach_ok(False, cfg, high_br, low_br) \
                        and lo <= eP <= hi:
                    pos, e_px, s_px, t_px, e_i = -1, eP, sP, tP, i
                    qty = risk / (max(abs(eP - sP), 1e-9) * pvval)
                    traded = True

        # ── exit (only when in position; may be the same bar as entry) ───────
        if pos != 0:
            longp = pos == 1
            hit_sl = (lo <= s_px) if longp else (hi >= s_px)
            hit_tp = (hi >= t_px) if longp else (lo <= t_px)
            vwap_exit = use_vwap and (not np.isnan(vw)) and (cl < vw if longp else cl > vw)
            eod = tm >= eff_min
            if hit_sl:
                exit_px, reason = s_px, "SL"
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


def optimize(prepped: list, cfg: dict, grid: dict, min_trades: int, rank_by: str,
             progress=None) -> pd.DataFrame:
    """
    Sweep entry% × stop% × target% over the fixed filter/breach configuration.
    Ranks by the chosen metric with a min-trades guard. Entry stack (filters,
    breach, times) stays as configured — only the level %s vary.
    """
    combos = [(e, s, t) for e in grid["entry"] for s in grid["stop"]
              for t in grid["target"] if s > e]
    n_days = len(prepped)
    rows = []
    for k, (e, s, t) in enumerate(combos):
        c = copy.deepcopy(cfg)
        c["entry_pct"], c["sl_pct"], c["target_pct"] = e, s, t
        trades = run(prepped, c)
        if len(trades) >= min_trades:
            m = metrics(trades, n_days)
            rows.append({
                "entry %": e, "stop %": s, "target %": t,
                "trades": m["trades"],
                "win %": round(m["win_rate"] * 100, 1),
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
    col = {"net_r": "net R", "net": "net $", "win_rate": "win %",
           "pf": "PF", "expectancy": "expectancy $"}[rank_by]
    return pd.DataFrame(rows).sort_values(col, ascending=False).reset_index(drop=True)
