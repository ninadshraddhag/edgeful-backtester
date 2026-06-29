"""Performance metrics computed from a trades DataFrame.

All return-based stats are in R multiples (P/L normalised by per-trade risk),
which keeps them comparable across instruments. Cash P/L is also reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {
            "trades": 0, "win_rate": 0.0, "avg_win_r": 0.0, "avg_loss_r": 0.0,
            "avg_rr": 0.0, "expectancy_r": 0.0, "profit_factor": 0.0,
            "total_r": 0.0, "total_cash": 0.0, "max_win_streak": 0,
            "max_loss_streak": 0, "max_drawdown_r": 0.0, "mom": pd.Series(dtype=float),
        }

    r = trades["r_multiple"].to_numpy(float)
    wins = r[r > 0]
    losses = r[r < 0]

    win_rate = len(wins) / len(r) * 100.0
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0          # negative
    gross_win = wins.sum()
    gross_loss = -losses.sum()                                # positive magnitude
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    expectancy = r.mean()
    # Realised reward:risk = avg win size / avg loss size.
    avg_rr = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("inf")

    win_streak, loss_streak = _streaks(r)
    equity = np.cumsum(r)
    drawdown = equity - np.maximum.accumulate(equity)
    max_dd = drawdown.min() if len(drawdown) else 0.0

    return {
        "trades": int(len(r)),
        "win_rate": round(win_rate, 2),
        "avg_win_r": round(float(avg_win), 3),
        "avg_loss_r": round(float(avg_loss), 3),
        "avg_rr": round(float(avg_rr), 3) if np.isfinite(avg_rr) else float("inf"),
        "expectancy_r": round(float(expectancy), 4),
        "profit_factor": round(float(profit_factor), 3) if np.isfinite(profit_factor) else float("inf"),
        "total_r": round(float(r.sum()), 3),
        "total_cash": round(float(trades["pnl_cash"].sum()), 2),
        "max_win_streak": int(win_streak),
        "max_loss_streak": int(loss_streak),
        "max_drawdown_r": round(float(max_dd), 3),
        "mom": month_on_month(trades),
    }


def _streaks(r: np.ndarray) -> tuple[int, int]:
    max_w = max_l = cur_w = cur_l = 0
    for x in r:
        if x > 0:
            cur_w += 1
            cur_l = 0
        elif x < 0:
            cur_l += 1
            cur_w = 0
        else:
            cur_w = cur_l = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def month_on_month(trades: pd.DataFrame) -> pd.Series:
    """Sum of R per calendar month, indexed by 'YYYY-MM'."""
    if trades.empty:
        return pd.Series(dtype=float)
    t = trades.copy()
    t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M").astype(str)
    return t.groupby("month")["r_multiple"].sum().round(3)


def format_report(metrics: dict, spec_desc: str, instrument: str) -> str:
    """Human-readable CLI block."""
    pf = metrics["profit_factor"]
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    rr = metrics["avg_rr"]
    rr_s = "inf" if rr == float("inf") else f"{rr:.2f}"
    lines = [
        "=" * 60,
        f" BACKTEST RESULT  |  {instrument}",
        f" {spec_desc}",
        "=" * 60,
        f"  Trades            : {metrics['trades']}",
        f"  Win rate          : {metrics['win_rate']:.1f}%",
        f"  Avg win / loss (R): +{metrics['avg_win_r']:.2f} / {metrics['avg_loss_r']:.2f}",
        f"  Realised RR       : {rr_s}",
        f"  Expectancy        : {metrics['expectancy_r']:+.3f} R / trade",
        f"  Profit factor     : {pf_s}",
        f"  Total P/L         : {metrics['total_r']:+.2f} R  (${metrics['total_cash']:,.0f})",
        f"  Max win streak    : {metrics['max_win_streak']}",
        f"  Max loss streak   : {metrics['max_loss_streak']}",
        f"  Max drawdown      : {metrics['max_drawdown_r']:.2f} R",
        "=" * 60,
    ]
    mom = metrics["mom"]
    if len(mom):
        lines.append("  Month-on-month P/L (R):")
        for m, v in mom.items():
            bar = "#" * int(min(abs(v), 20))
            sign = "+" if v >= 0 else "-"
            lines.append(f"    {m}  {v:+6.2f}  {sign}{bar}")
        lines.append("=" * 60)
    return "\n".join(lines)
