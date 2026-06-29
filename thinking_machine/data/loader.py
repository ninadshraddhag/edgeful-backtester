"""Data access layer.

Loads the real minute parquet files that live in ``orb_ib_backtester/data`` and
resamples / session-filters them into a clean OHLCV frame for the backtester.
Falls back to the synthetic generator when ``mock=True`` or when a file is
missing, so the platform always runs.

Timestamps in the parquet files are timezone-naive exchange-local time:
  * NQ / XAUUSD  -> US Eastern (RTH 09:30-16:00)
  * NIFTY 50 / BANK NIFTY -> India IST (09:15-15:30)
"""
from __future__ import annotations

import os

import pandas as pd

from . import mock_generator

# Directory that holds the shared parquet files (the parent project's data dir).
_THIS = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(_THIS, "..", "..", "data"))

# instrument key -> (parquet filename stem, RTH session)
INSTRUMENTS: dict[str, dict] = {
    "NQ": {"file": "NQ_minute", "session": ("09:30", "16:00"), "point_value": 20.0},
    "XAUUSD": {"file": "XAUUSD_minute", "session": ("09:30", "16:00"), "point_value": 100.0},
    "NIFTY": {"file": "NIFTY 50_minute", "session": ("09:15", "15:30"), "point_value": 1.0},
    "BANKNIFTY": {"file": "BANK NIFTY_minute", "session": ("09:15", "15:30"), "point_value": 1.0},
}

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def list_instruments() -> list[str]:
    return list(INSTRUMENTS.keys())


def session_for(instrument: str) -> tuple[str, str]:
    return INSTRUMENTS.get(instrument.upper(), {}).get("session", ("09:30", "16:00"))


def point_value(instrument: str) -> float:
    return INSTRUMENTS.get(instrument.upper(), {}).get("point_value", 1.0)


def load_data(
    instrument: str = "NQ",
    timeframe: str = "5min",
    start: str | None = None,
    end: str | None = None,
    max_days: int | None = None,
    rth_only: bool = True,
    mock: bool = False,
) -> pd.DataFrame:
    """Return a tidy OHLCV frame indexed by a ``date`` column (datetime).

    Columns: ``date, open, high, low, close, volume`` plus an int ``session_date``
    helper used by the backtester for per-day logic.
    """
    instrument = instrument.upper()
    meta = INSTRUMENTS.get(instrument)
    session = meta["session"] if meta else ("09:30", "16:00")

    if mock or meta is None:
        df = mock_generator.generate(days=max_days or 120, timeframe=timeframe, session=session)
    else:
        path = os.path.join(DATA_DIR, meta["file"] + ".parquet")
        if not os.path.exists(path):
            df = mock_generator.generate(days=max_days or 120, timeframe=timeframe, session=session)
        else:
            # Indian feeds (NIFTY / BANK NIFTY) ship without a volume column, so
            # only request columns that actually exist and backfill volume=0.
            import pyarrow.parquet as pq
            available = set(pq.read_schema(path).names)
            want = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in available]
            df = pd.read_parquet(path, columns=want)
            if "volume" not in df.columns:
                df["volume"] = 0
            df = _prepare(df, timeframe, session, start, end, max_days, rth_only)

    if mock or meta is None:
        df = _finalize(df)
    return df


def _prepare(df, timeframe, session, start, end, max_days, rth_only):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]

    # Resample to the requested timeframe (1min source -> e.g. 5min bars).
    rule = _tf_rule(timeframe)
    if rule != "1min":
        df = df.resample(rule, label="left", closed="left").agg(_AGG).dropna()

    if rth_only:
        df = df.between_time(session[0], session[1], inclusive="left")

    df = df.reset_index()

    if max_days is not None:
        keep = df["date"].dt.normalize().drop_duplicates().tail(max_days)
        df = df[df["date"].dt.normalize().isin(keep)]

    return _finalize(df)


def _finalize(df):
    df = df.reset_index(drop=True)
    df["session_date"] = df["date"].dt.normalize()
    return df


def _tf_rule(tf: str) -> str:
    tf = tf.lower().strip()
    if tf in ("1min", "1m"):
        return "1min"
    if tf.endswith("min"):
        return tf
    if tf.endswith("m"):
        return tf[:-1] + "min"
    if tf.endswith("h"):
        return tf
    return tf
