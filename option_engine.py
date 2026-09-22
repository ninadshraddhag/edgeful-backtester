"""
option_engine.py — leg-based intraday option-strategy backtester on real premiums.

A strategy is 1–4 legs entered together at `entry_t` and squared off at `exit_t`
(or earlier on a position-level stop / target expressed as a % of the entry
premium). Strikes are chosen relative to the ATM at entry, resolved to the
nearest AVAILABLE strike for the traded expiry. P&L is in premium points and in
rupees (points × lot_size × lots).

cfg = {
  "underlying": "NIFTY",
  "d0","d1": date bounds,
  "expiry_which": 0,          # 0 = nearest weekly, 1 = next, ...
  "min_dte": 0,               # skip expiries closer than this many days
  "entry_t": 560,             # 09:20
  "exit_t": 915,              # 15:15 square-off
  "sl_pct": 0.0,              # position stop, % of entry premium base (0 = off)
  "tp_pct": 0.0,              # position target, % of entry premium base (0 = off)
  "lot_size": 75,
  "legs": [ {"right":"CE","side":"sell","otm":0,"lots":1},
            {"right":"PE","side":"sell","otm":0,"lots":1} ],   # short ATM straddle
}
`otm` is steps away from ATM in the OTM direction (CE→higher, PE→lower); negative
= ITM. run(cfg) → {"trades": DataFrame, "metrics": dict, "daily": Series}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import option_data as od


def _leg_grid(series: pd.DataFrame, entry_t: int, exit_t: int) -> pd.Series | None:
    """Contract close reindexed onto the entry..exit minute grid (ffill, then
    bfill the lead-in). None if the contract has no bar in the window."""
    s = series[(series["t_min"] >= entry_t) & (series["t_min"] <= exit_t)]
    if s.empty:
        return None
    grid = pd.RangeIndex(entry_t, exit_t + 1)
    return s.set_index("t_min")["close"].reindex(grid).ffill().bfill()


def simulate_day(u: str, date, cfg: dict) -> dict | None:
    ch = od.day_chain(u, date)
    if len(ch) == 0:
        return None
    exp = od.nearest_expiry(ch, date, cfg.get("min_dte", 0), cfg.get("expiry_which", 0))
    if exp is None:
        return None
    entry_t, exit_t = cfg["entry_t"], cfg["exit_t"]
    px = od.underlying_price(u, date, entry_t)
    if px is None:
        return None
    step = od.infer_step(ch, exp)
    atm = od.atm_strike(ch, exp, px)
    if atm is None:
        return None

    legs = []
    for lg in cfg["legs"]:
        sign = 1 if lg["right"] == "CE" else -1          # CE OTM = higher strike
        target = atm + sign * int(lg.get("otm", 0)) * step
        k = od.pick_strike(ch, exp, lg["right"], target)
        if k is None:
            return None
        z = _leg_grid(od.leg_series(ch, exp, k, lg["right"]), entry_t, exit_t)
        if z is None:
            return None
        legs.append({**lg, "strike": k, "px": z, "entry": float(z.iloc[0])})

    # combined P&L (premium points × lots) across the day, and the credit/debit base
    lots = np.array([lg.get("lots", 1) for lg in legs], float)
    sells = np.array([1.0 if lg["side"] == "sell" else -1.0 for lg in legs])
    entry = np.array([lg["entry"] for lg in legs])
    base = float(np.abs(np.sum(sells * entry * lots)))    # net credit(+)/debit magnitude
    combined = None
    for lg, sg, lt in zip(legs, sells, lots):
        # sell: profit when premium falls (entry-px); buy: profit when it rises
        leg_pnl = (lg["entry"] - lg["px"]) * (sg) * lt
        combined = leg_pnl if combined is None else combined.add(leg_pnl, fill_value=0.0)

    # exit: first stop/target breach after entry, else square-off at exit_t
    reason, ex_t = "square-off", exit_t
    sl = -cfg.get("sl_pct", 0.0) / 100.0 * base if cfg.get("sl_pct") else None
    tp = cfg.get("tp_pct", 0.0) / 100.0 * base if cfg.get("tp_pct") else None
    if sl is not None or tp is not None:
        for t in range(entry_t + 1, exit_t + 1):
            v = combined.get(t)
            if v is None or np.isnan(v):
                continue
            if sl is not None and v <= sl:
                reason, ex_t = "stop", t; break
            if tp is not None and v >= tp:
                reason, ex_t = "target", t; break
    pnl_pts = float(combined.loc[ex_t])
    return {
        "date": pd.Timestamp(date), "expiry": pd.Timestamp(exp),
        "dte": (pd.Timestamp(exp) - pd.Timestamp(date)).days,
        "atm": atm, "spot": round(px, 2),
        "strikes": "/".join(f"{lg['side'][0].upper()}{lg['right']}{lg['strike']}" for lg in legs),
        "credit_pts": round(base, 2), "pnl_pts": round(pnl_pts, 2),
        "pnl": round(pnl_pts * cfg["lot_size"], 2), "exit": reason,
        "exit_t": ex_t,
    }


def _metrics(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"trades": 0}
    pnl = tr["pnl"].to_numpy(float)
    eq = np.cumsum(pnl)
    dd = eq - np.maximum.accumulate(eq)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gp, gl = wins.sum(), -losses.sum()
    return {
        "trades": len(pnl),
        "win_rate": round(100 * len(wins) / len(pnl), 1),
        "total_pnl": round(pnl.sum(), 0),
        "avg_pnl": round(pnl.mean(), 1),
        "avg_win": round(wins.mean(), 1) if len(wins) else 0.0,
        "avg_loss": round(losses.mean(), 1) if len(losses) else 0.0,
        "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
        "max_dd": round(dd.min(), 0),
        "best": round(pnl.max(), 0), "worst": round(pnl.min(), 0),
    }


def run(cfg: dict) -> dict:
    u = cfg["underlying"]
    days = [d for d in od.trading_days(u) if cfg["d0"] <= d <= cfg["d1"]]
    rows = []
    for d in days:
        try:
            r = simulate_day(u, d, cfg)
        except Exception:      # noqa: BLE001 — one bad day shouldn't kill the run
            r = None
        if r is not None:
            rows.append(r)
    tr = pd.DataFrame(rows)
    daily = tr.set_index("date")["pnl"] if len(tr) else pd.Series(dtype=float)
    return {"trades": tr, "metrics": _metrics(tr), "daily": daily,
            "equity": daily.cumsum() if len(daily) else daily}
