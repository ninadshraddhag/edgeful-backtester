"""Event-driven backtester.

Signals are computed vectorised (fast), then a single pass over the bars
simulates one position at a time:

  * Entry on the *next* bar's open after a trigger (no look-ahead).
  * Stop = entry -/+ stop_atr * ATR(at signal bar); target = rr * stop distance.
  * Intrabar: if both stop and target are touched in the same bar, the stop is
    assumed hit first (conservative).
  * Open positions are flattened at the last bar of the session day.

Returns a ``BacktestResult`` with a trades DataFrame and the equity curve in R.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators
from .strategy import StrategySpec


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series          # cumulative R, indexed by exit time
    spec: StrategySpec
    instrument: str
    point_value: float
    bars: int


def _entry_mask(df: pd.DataFrame, signals: list[str]) -> np.ndarray:
    """AND together the requested signal columns -> confluence trigger."""
    if not signals:
        return np.zeros(len(df), dtype=bool)
    mask = np.ones(len(df), dtype=bool)
    for s in signals:
        if s not in df.columns:
            raise KeyError(f"unknown signal '{s}'")
        mask &= df[s].to_numpy(dtype=bool)
    return mask


def run(df: pd.DataFrame, spec: StrategySpec, instrument: str = "NQ",
        point_value: float = 1.0) -> BacktestResult:
    data = indicators.compute_all(
        df, ema_period=spec.ema_period, atr_period=spec.atr_period,
        sweep_lookback=spec.sweep_lookback,
    )

    long_trig = _entry_mask(data, spec.long_entry)
    short_trig = _entry_mask(data, spec.short_entry)

    o = data["open"].to_numpy(float)
    h = data["high"].to_numpy(float)
    l = data["low"].to_numpy(float)
    c = data["close"].to_numpy(float)
    atr_arr = data["atr"].to_numpy(float)
    ts = data["date"].to_numpy()
    day = data["session_date"].to_numpy()
    minutes = pd.to_datetime(data["date"]).dt.strftime("%H:%M").to_numpy()

    after = spec.entry_after
    before = spec.entry_before

    n = len(data)
    trades = []
    i = 0
    trades_today = 0
    cur_day = day[0] if n else None

    while i < n - 1:
        if day[i] != cur_day:
            cur_day = day[i]
            trades_today = 0

        # last bar of the session day -> can't open a new trade that we could manage
        last_of_day = (i == n - 1) or (day[i + 1] != day[i])

        in_window = True
        if after is not None:
            in_window &= minutes[i] >= after
        if before is not None:
            in_window &= minutes[i] < before

        can_trade = (trades_today < spec.max_trades_per_day) and in_window and not last_of_day
        side = None
        if can_trade and long_trig[i] and not np.isnan(atr_arr[i]) and atr_arr[i] > 0:
            side = "long"
        elif can_trade and short_trig[i] and not np.isnan(atr_arr[i]) and atr_arr[i] > 0:
            side = "short"

        if side is None:
            i += 1
            continue

        # Enter at next bar open.
        entry_idx = i + 1
        entry_price = o[entry_idx]
        risk = spec.stop_atr * atr_arr[i]
        if side == "long":
            stop = entry_price - risk
            target = entry_price + spec.rr * risk
        else:
            stop = entry_price + risk
            target = entry_price - spec.rr * risk

        exit_idx, exit_price, reason = _simulate_exit(
            side, entry_idx, stop, target, h, l, c, day, n, spec.exit_on_session_close
        )

        if side == "long":
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price
        r_mult = pnl_pts / risk if risk else 0.0

        trades.append({
            "entry_time": pd.Timestamp(ts[entry_idx]),
            "exit_time": pd.Timestamp(ts[exit_idx]),
            "side": side,
            "entry": round(entry_price, 4),
            "stop": round(stop, 4),
            "target": round(target, 4),
            "exit": round(exit_price, 4),
            "pnl_points": round(pnl_pts, 4),
            "pnl_cash": round(pnl_pts * point_value, 2),
            "r_multiple": round(r_mult, 3),
            "reason": reason,
            "bars_held": exit_idx - entry_idx,
        })
        trades_today += 1
        i = exit_idx + 1  # one position at a time: resume after the exit

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        equity = tdf.set_index("exit_time")["r_multiple"].cumsum()
    else:
        equity = pd.Series(dtype=float)

    return BacktestResult(
        trades=tdf, equity=equity, spec=spec, instrument=instrument,
        point_value=point_value, bars=n,
    )


def _simulate_exit(side, entry_idx, stop, target, h, l, c, day, n, exit_on_close):
    """Walk forward from entry to first stop/target touch, else session-close exit."""
    j = entry_idx
    while j < n:
        end_of_day = (j == n - 1) or (day[j + 1] != day[j])
        hi, lo = h[j], l[j]

        if side == "long":
            hit_stop = lo <= stop
            hit_tgt = hi >= target
        else:
            hit_stop = hi >= stop
            hit_tgt = lo <= target

        if hit_stop and hit_tgt:
            return j, stop, "stop"          # conservative: stop first
        if hit_stop:
            return j, stop, "stop"
        if hit_tgt:
            return j, target, "target"
        if end_of_day and exit_on_close:
            return j, c[j], "session_close"
        j += 1
    return n - 1, c[n - 1], "eod"
