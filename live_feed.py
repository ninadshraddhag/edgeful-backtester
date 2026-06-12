"""
live_feed.py — market-data sources for the Live Market Statistics mode.

Two interchangeable sources, both exposing:
    today_minutes(instrument) -> DataFrame[date, open, high, low, close]
    prev_day(instrument)      -> (prev_high, prev_low, prev_close) or None

  • KotakNeoSource — live data via the Kotak Neo API (neo-api-client SDK,
                     lazily imported).  Requires a pre-authenticated NeoAPI
                     client object — the 3-step login (consumer_key/secret →
                     mobilenumber+password → OTP) is handled in app.py and the
                     authenticated client is stored in st.session_state.
  • DemoSource     — replays a historical day from the project CSV up to an
                     "as-of" minute, so the whole pipeline is testable when the
                     market is closed / no token.
"""
import pandas as pd
from datetime import date, timedelta

import build_facts

SQUARE_OFF_T = 15 * 60 + 15

# Kotak Neo instrument tokens (NSE standard tokens; update if master data changes)
KOTAK_INSTRUMENTS = {
    "NIFTY 50":   {"instrument_token": "26000", "exchange": "nse_cm"},
    "BANK NIFTY": {"instrument_token": "26009", "exchange": "nse_cm"},
}


def _kotak_to_df(resp):
    """
    Parse a neo-api-client historical_candle_data response into a clean
    date/open/high/low/close DataFrame.

    The SDK returns either:
      {"Success": [[datetime_str, O, H, L, C, V], ...]}
    or occasionally a bare list.  We handle both shapes plus dict-of-rows.
    """
    candles = None
    if isinstance(resp, dict):
        # try common top-level keys
        candles = (resp.get("Success") or resp.get("data")
                   or resp.get("candles") or resp.get("result"))
    elif isinstance(resp, list):
        candles = resp

    if not candles:
        raise RuntimeError(f"No candle data from Kotak Neo: {str(resp)[:300]}")

    rows = []
    for c in candles:
        if isinstance(c, (list, tuple)) and len(c) >= 5:
            dt = pd.to_datetime(c[0])
            o, h, l, cl = float(c[1]), float(c[2]), float(c[3]), float(c[4])
        elif isinstance(c, dict):
            dt = pd.to_datetime(
                c.get("time") or c.get("date") or c.get("datetime"))
            o = float(c["open"]); h = float(c["high"])
            l = float(c["low"]);  cl = float(c["close"])
        else:
            continue
        # strip timezone info (IST timestamps come as "+05:30")
        if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
            try:
                dt = dt.tz_convert("Asia/Kolkata").tz_localize(None)
            except Exception:
                dt = dt.replace(tzinfo=None)
        rows.append({"date": dt, "open": o, "high": h, "low": l, "close": cl})

    if not rows:
        raise RuntimeError("No parseable candles in Kotak Neo response.")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


class KotakNeoSource:
    """
    Wraps an already-authenticated neo_api_client.NeoAPI instance.
    The NeoAPI client must have completed session_2fa() before being passed here.
    """
    def __init__(self, client, instruments=None):
        self.client = client
        self.instruments = instruments or KOTAK_INSTRUMENTS

    def _meta(self, instrument):
        m = self.instruments.get(instrument)
        if not m:
            raise ValueError(f"No Kotak Neo config for '{instrument}'")
        return m

    def today_minutes(self, instrument):
        m = self._meta(instrument)
        today = date.today().strftime("%d/%m/%Y")
        resp = self.client.historical_candle_data(
            instrument_token=m["instrument_token"],
            from_date=today, to_date=today,
            resample="1", exchange=m["exchange"])
        return _kotak_to_df(resp)

    def prev_day(self, instrument):
        """
        Returns (high, low, close) for the last completed trading day.
        Fetches a rolling 10-day window of 1-min candles and aggregates by day
        to avoid needing a separate daily-candle endpoint.
        """
        m = self._meta(instrument)
        to_d = date.today() - timedelta(days=1)
        from_d = to_d - timedelta(days=10)
        resp = self.client.historical_candle_data(
            instrument_token=m["instrument_token"],
            from_date=from_d.strftime("%d/%m/%Y"),
            to_date=to_d.strftime("%d/%m/%Y"),
            resample="1", exchange=m["exchange"])
        df = _kotak_to_df(resp)
        if df.empty:
            return None
        df["_day"] = df["date"].dt.date
        last_day = df["_day"].max()
        g = df[df["_day"] == last_day]
        if g.empty:
            return None
        return float(g["high"].max()), float(g["low"].min()), float(g.iloc[-1]["close"])


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
