"""
data_store.py — central instrument registry (local + cloud friendly)
====================================================================
One place that answers "which instruments exist and where is their data?",
so every mode (Edge / Day-wise / Advanced / Probabilities / Live) sees the
same list — including instruments the user uploads (NQ, XAUUSD, …).

Sources, merged in this order (later wins on name collisions):
  1. LEGACY  — the original local Windows paths (C:\\NIFTY ...csv); only
               listed when the file actually exists, so cloud deploys
               simply skip them.
  2. data/   — any *_minute.csv / *_minute.parquet inside the repo's data
               folder. This is what gets committed to GitHub and what
               Streamlit Cloud reads. Parquet preferred (≈5× smaller).
               Uploads from the UI are saved here too, so they persist
               and appear in every mode.

Per-instrument metadata (data/instruments.json):
  open_t — session-open override in minutes-since-midnight. Needed for
           24-hour instruments where auto-detection of the first candle
           would not give the cash open (e.g. NQ in EST → 09:30 = 570).
           Timestamps in uploaded files are treated as EXCHANGE-LOCAL
           time (naive): upload NQ/XAUUSD data already converted to EST
           and set the open to 09:30.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
META_FILE = os.path.join(DATA_DIR, "instruments.json")

LEGACY = {
    "NIFTY 50":   r"C:\NIFTY 50_minute.csv",
    "BANK NIFTY": r"C:\NIFTY BANK_minute.csv",
}


def _name_from_file(fname: str) -> str:
    stem = os.path.splitext(os.path.basename(fname))[0]
    return re.sub(r"_minute$", "", stem, flags=re.IGNORECASE).strip()


def discover() -> dict:
    """{instrument name: data file path} across all sources. Cheap to call."""
    out = {}
    for n, p in LEGACY.items():
        if os.path.exists(p):
            out[n] = p
    if os.path.isdir(DATA_DIR):
        files = sorted(os.listdir(DATA_DIR))
        # csv first so a parquet of the same name overrides it
        for ext in (".csv", ".parquet"):
            for f in files:
                if f.lower().endswith(ext):
                    out[_name_from_file(f)] = os.path.join(DATA_DIR, f)
    return out


def read_minute(path: str) -> pd.DataFrame:
    """Raw minute OHLC frame from csv or parquet (clean with build_facts.clean_min)."""
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_upload(name: str, data: bytes) -> str:
    """Persist an uploaded minute CSV into data/ so every mode can use it."""
    os.makedirs(DATA_DIR, exist_ok=True)
    safe = re.sub(r"[^\w \-]", "", name).strip() or "INSTRUMENT"
    path = os.path.join(DATA_DIR, f"{safe}_minute.csv")
    with open(path, "wb") as fh:
        fh.write(data)
    return path


# ─── per-instrument metadata (session-open overrides) ─────────────────────────

def _load_meta() -> dict:
    if os.path.exists(META_FILE):
        try:
            return json.load(open(META_FILE, encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_meta(meta: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    json.dump(meta, open(META_FILE, "w", encoding="utf-8"), indent=2)


def session_open(name: str, auto_open_t: int) -> int:
    """The instrument's session-open minute: stored override, else auto-detected."""
    ov = _load_meta().get(name, {}).get("open_t")
    return int(ov) if ov else int(auto_open_t)


def set_open_override(name: str, open_t: int | None, auto_open_t: int | None = None):
    """
    Persist a session-open override (e.g. NQ → 570 = 09:30 EST). Passing the
    auto-detected value (or None) clears the override instead of storing it.
    """
    meta = _load_meta()
    entry = meta.get(name, {})
    if open_t is None or (auto_open_t is not None and int(open_t) == int(auto_open_t)):
        entry.pop("open_t", None)
    else:
        entry["open_t"] = int(open_t)
    if entry:
        meta[name] = entry
    else:
        meta.pop(name, None)
    _save_meta(meta)
