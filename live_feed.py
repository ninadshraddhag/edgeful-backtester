"""
live_feed.py — market-data sources for the Live Market Statistics mode.

Two interchangeable sources, both exposing:
    today_minutes(instrument) -> DataFrame[date, open, high, low, close]
    prev_day(instrument)      -> (prev_high, prev_low, prev_close) or None

  • DhanSource — live data via the DhanHQ Data API (dhanhq SDK, lazily imported).
  • DemoSource — replays a historical day from the project CSV up to an "as-of"
    minute, so the whole pipeline is testable when the market is closed / no token.
"""
import pandas as pd
from datetime import date, timedelta

import build_facts

SQUARE_OFF_T = 15 * 60 + 15

# Dhan index identifiers (override in live_config.json if they ever change)
INSTRUMENTS = {
    "NIFTY 50":   {"security_id": "13", "segment": "IDX_I", "type": "INDEX"},
    "BANK NIFTY": {"security_id": "25", "segment": "IDX_I", "type": "INDEX"},
}


def _dhan_to_df(resp):
    """Normalise a DhanHQ candle response into date/open/high/low/close."""
    if not isinstance(resp, dict) or resp.get("status") not in ("success", None):
        raise RuntimeError(f"Dhan API error: {resp}")
    d = resp.get("data", resp)
    ts = d.get("timestamp") or d.get("start_Time") or d.get("time")
    if ts is None:
        raise RuntimeError(f"Unexpected Dhan response shape: {list(d)[:8]}")
    dt = pd.to_datetime(pd.Series(ts, dtype="float64"), unit="s", utc=True) \
           .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return pd.DataFrame({
        "date": dt, "open": d["open"], "high": d["high"],
        "low": d["low"], "close": d["close"],
    })


class DhanSource:
    def __init__(self, client_id, access_token, instruments=None):
        self.instruments = instruments or INSTRUMENTS
        try:
            from dhanhq import dhanhq
        except ImportError as e:
            raise RuntimeError("Dhan SDK not installed — run:  pip install dhanhq") from e
        self.dhan = dhanhq(client_id, access_token)

    def _meta(self, instrument):
        m = self.instruments.get(instrument)
        if not m:
            raise ValueError(f"No Dhan id configured for '{instrument}'")
        return m

    def today_minutes(self, instrument):
        m = self._meta(instrument)
        today = date.today().isoformat()
        resp = self.dhan.intraday_minute_data(
            security_id=m["security_id"], exchange_segment=m["segment"],
            instrument_type=m["type"], from_date=today, to_date=today)
        return _dhan_to_df(resp)

    def prev_day(self, instrument):
        m = self._meta(instrument)
        to_d = date.today()
        from_d = to_d - timedelta(days=12)
        resp = self.dhan.historical_daily_data(
            security_id=m["security_id"], exchange_segment=m["segment"],
            instrument_type=m["type"], from_date=from_d.isoformat(), to_date=to_d.isoformat())
        df = _dhan_to_df(resp)
        df = df[df["date"].dt.date < to_d]
        if df.empty:
            return None
        last = df.iloc[-1]
        return float(last["high"]), float(last["low"]), float(last["close"])


class DemoSource:
    """Replay a historical session from a preloaded (cleaned) minute DataFrame."""
    def __init__(self, full_df, asof_date, asof_t):
        self.df = full_df
        self.asof_date = asof_date
        self.asof_t = asof_t

    def today_minutes(self, instrument):
        d = self.df[(self.df["date_only"] == self.asof_date) & (self.df["t_min"] <= self.asof_t)]
        return d[["date", "open", "high", "low", "close"]].copy()

    def prev_day(self, instrument):
        days = sorted(self.df["date_only"].unique())
        if self.asof_date not in days:
            return None
        i = days.index(self.asof_date)
        if i == 0:
            return None
        g = self.df[(self.df["date_only"] == days[i - 1]) & (self.df["t_min"] <= SQUARE_OFF_T)]
        if g.empty:
            return None
        return float(g["high"].max()), float(g["low"].min()), float(g.iloc[-1]["close"])
