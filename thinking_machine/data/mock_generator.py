"""Synthetic OHLCV generator so the platform is testable with zero external data.

Produces a minute (or any timeframe) DataFrame with the exact same schema as the
real parquet files: ``date, open, high, low, close, volume``.  The series is an
intraday geometric random walk with deliberately injected structure (gaps,
displacement candles, sweep-and-reverse moves) so the ICT indicators actually
fire on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate(
    days: int = 120,
    timeframe: str = "5min",
    start: str = "2024-01-01",
    session: tuple[str, str] = ("09:30", "16:00"),
    start_price: float = 18000.0,
    seed: int | None = 7,
) -> pd.DataFrame:
    """Generate a synthetic intraday OHLCV frame.

    Parameters mirror what :func:`thinking_machine.data.loader.load_data` returns,
    so a strategy that runs on mock data runs unchanged on the real parquet.
    """
    rng = np.random.default_rng(seed)
    freq_min = _tf_minutes(timeframe)
    s_open, s_close = session

    # Build the per-day intraday grid once.
    day_index = pd.bdate_range(start=start, periods=days)
    rows = []
    price = float(start_price)

    for day in day_index:
        bars = pd.date_range(
            f"{day.date()} {s_open}", f"{day.date()} {s_close}", freq=f"{freq_min}min"
        )[:-1]  # drop the closing edge so the last bar opens before close
        n = len(bars)
        if n == 0:
            continue

        # Daily drift + vol regime so months differ from one another.
        drift = rng.normal(0, 0.0004)
        vol = abs(rng.normal(0.0012, 0.0004)) + 0.0004
        rets = rng.normal(drift, vol, n)

        # Inject a morning sweep-and-reverse roughly every third day: a spike that
        # takes out the prior bars then snaps back, which is what the liquidity
        # sweep / reversion logic is meant to catch.
        if rng.random() < 0.35 and n > 12:
            k = rng.integers(3, 10)
            rets[k] += rng.choice([-1, 1]) * vol * 6
            rets[k + 1] -= rets[k] * 0.7

        # Occasional displacement candle to create fair value gaps.
        if rng.random() < 0.5 and n > 6:
            j = rng.integers(2, n - 2)
            rets[j] += rng.choice([-1, 1]) * vol * 5

        closes = price * np.cumprod(1 + rets)
        opens = np.empty(n)
        opens[0] = price
        opens[1:] = closes[:-1]

        # Wmy wicks: high/low extend beyond the open/close body.
        wick = np.abs(rng.normal(0, vol, n)) * closes
        highs = np.maximum(opens, closes) + wick
        lows = np.minimum(opens, closes) - np.abs(rng.normal(0, vol, n)) * closes
        vols = rng.integers(50, 1500, n)

        for t, o, h, l, c, v in zip(bars, opens, highs, lows, closes, vols):
            rows.append((t.to_pydatetime(), float(o), float(h), float(l), float(c), int(v)))
        price = float(closes[-1])

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    return df


def _tf_minutes(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith("min"):
        return int(tf[:-3])
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    return int(tf)


if __name__ == "__main__":
    d = generate(days=10)
    print(d.head())
    print("rows:", len(d))
