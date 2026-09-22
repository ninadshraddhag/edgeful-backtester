"""
build_options_store.py — one-time preprocessing of the raw NSE options archive
into a compact, query-friendly parquet store used by the option backtester.

Source (local only, ~6.6 GB of daily CSVs):
    <ARCHIVE>/nifty_data/nifty_options/<YYYY>/<M>/nifty_options_DD_MM_YYYY.csv
    <ARCHIVE>/nifty_data/nifty_fut/...      (symbol NIFTY-I, continuous front)
    <ARCHIVE>/nifty_data/nifty_spot/...     (symbol NIFTY)
    ...and the banknifty_data mirror.

Each options CSV is ONE trading day holding the WHOLE chain, minute bars:
    date,time,symbol,open,high,low,close,oi,volume
    symbol = NIFTY04JAN2418300PE = <underlying><DDMMMYY expiry><strike><CE|PE>

Output (data_options/, gitignored — stays offline):
    data_options/<UND>/<YYYY-MM>.parquet   chain: dt,expiry,strike,right,o,h,l,c,oi,volume
    data_options/<UND>_spot_minute.parquet continuous spot minute OHLC
    data_options/<UND>_fut_minute.parquet  continuous front-future minute OHLCV

Idempotent: existing monthly parquets are skipped, so a re-run resumes.
Run:  python build_options_store.py                # everything
      python build_options_store.py NIFTY 2024     # one underlying / year (testing)
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.environ.get("OPTIONS_ARCHIVE", r"C:\Users\ninad\Downloads\archive (6)")
OUT = os.path.join(HERE, "data_options")

UNDERLYINGS = {
    "NIFTY":     {"dir": "nifty_data",     "prefix": "nifty"},
    "BANKNIFTY": {"dir": "banknifty_data", "prefix": "banknifty"},
}
SYM = re.compile(r"^(NIFTY|BANKNIFTY)(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")


def _parse_options(df: pd.DataFrame) -> pd.DataFrame:
    """Raw one-day chain CSV → structured columns (drops unparseable symbols)."""
    df = df.dropna(subset=["symbol"])
    m = df["symbol"].astype(str).str.extract(SYM)
    ok = m[0].notna()
    df, m = df[ok], m[ok]
    return pd.DataFrame({
        "dt":     pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str)),
        "expiry": pd.to_datetime(m[1], format="%d%b%y"),
        "strike": m[2].astype("int32"),
        "right":  m[3].astype("category"),
        "open":   pd.to_numeric(df["open"], errors="coerce").astype("float32"),
        "high":   pd.to_numeric(df["high"], errors="coerce").astype("float32"),
        "low":    pd.to_numeric(df["low"], errors="coerce").astype("float32"),
        "close":  pd.to_numeric(df["close"], errors="coerce").astype("float32"),
        "oi":     pd.to_numeric(df["oi"], errors="coerce").fillna(0).astype("int32"),
        "volume": pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int32"),
    }).dropna(subset=["open", "high", "low", "close"])


def build_options(u: str, only_year: str | None = None):
    cfg = UNDERLYINGS[u]
    base = os.path.join(ARCHIVE, cfg["dir"], f"{cfg['prefix']}_options")
    outdir = os.path.join(OUT, u)
    os.makedirs(outdir, exist_ok=True)
    years = sorted(d for d in os.listdir(base) if d.isdigit())
    for y in years:
        if only_year and y != only_year:
            continue
        ydir = os.path.join(base, y)
        for mo in sorted((d for d in os.listdir(ydir) if d.isdigit()), key=int):
            mdir = os.path.join(ydir, mo)
            outp = os.path.join(outdir, f"{y}-{int(mo):02d}.parquet")
            if os.path.exists(outp):
                print("  skip", os.path.basename(outp)); continue
            files = glob.glob(os.path.join(mdir, "*.csv"))
            if not files:
                continue
            t = time.time()
            parts = []
            for f in files:
                try:
                    parts.append(_parse_options(pd.read_csv(f)))
                except Exception as e:      # noqa: BLE001 — log and skip a bad day
                    print("   ERR", os.path.basename(f), e)
            if not parts:
                continue
            allm = pd.concat(parts, ignore_index=True).sort_values("dt").reset_index(drop=True)
            allm.to_parquet(outp, compression="zstd", index=False)
            print("  wrote %-22s %8d rows  %5.1fs" % (os.path.basename(outp), len(allm), time.time() - t))


def build_series(u: str, kind: str):
    """Continuous spot/fut minute OHLC → one parquet. Fut keeps the front '-I' symbol."""
    cfg = UNDERLYINGS[u]
    base = os.path.join(ARCHIVE, cfg["dir"], f"{cfg['prefix']}_{kind}")
    if not os.path.isdir(base):
        return
    outp = os.path.join(OUT, f"{u}_{kind}_minute.parquet")
    if os.path.exists(outp):
        print("  skip", os.path.basename(outp)); return
    files = glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)
    parts = []
    t = time.time()
    for f in files:
        try:
            d = pd.read_csv(f)
            if kind == "fut" and "symbol" in d.columns:      # front-month continuous only
                d = d[d["symbol"].astype(str).str.endswith("-I")]
            if d.empty:
                continue
            row = {
                "dt":   pd.to_datetime(d["date"].astype(str) + " " + d["time"].astype(str)),
                "open": pd.to_numeric(d["open"], errors="coerce").astype("float32"),
                "high": pd.to_numeric(d["high"], errors="coerce").astype("float32"),
                "low":  pd.to_numeric(d["low"], errors="coerce").astype("float32"),
                "close": pd.to_numeric(d["close"], errors="coerce").astype("float32"),
            }
            if "volume" in d.columns:
                row["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0).astype("int32")
            parts.append(pd.DataFrame(row))
        except Exception as e:      # noqa: BLE001
            print("   ERR", os.path.basename(f), e)
    if not parts:
        return
    allm = (pd.concat(parts, ignore_index=True)
            .dropna(subset=["open", "high", "low", "close"])
            .drop_duplicates("dt").sort_values("dt").reset_index(drop=True))
    allm.to_parquet(outp, compression="zstd", index=False)
    print("  wrote %-26s %8d rows  %5.1fs" % (os.path.basename(outp), len(allm), time.time() - t))


def main():
    only_u = sys.argv[1].upper() if len(sys.argv) > 1 else None
    only_y = sys.argv[2] if len(sys.argv) > 2 else None
    os.makedirs(OUT, exist_ok=True)
    for u in UNDERLYINGS:
        if only_u and u != only_u:
            continue
        print(f"[{u}]")
        build_series(u, "spot")
        build_series(u, "fut")
        build_options(u, only_y)
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
