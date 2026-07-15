"""
build_facts.py
================
Crunches 10 years of NIFTY 50 & BANK NIFTY minute data into a per-day
"facts table" describing ORB (15-min) and IB (60-min) behaviour.

Per trading day it records:
  • ORB / IB high, low, range
  • which extreme formed FIRST inside the window  (high-first / low-first)
  • whether each side was BREACHED after the window, which broke first,
    both / one / neither
  • EXTENSION reached on each side, in units of range
        up_ext = (max high after window − range_high) / range
        dn_ext = (range_low − min low after window) / range
  • RETRACEMENT after a break, in units of range
        up_retr = (range_high − min low AFTER the high first broke) / range
        dn_retr = (max high AFTER the low first broke − range_low) / range
  • Previous-Day High / Low (PDH / PDL) and whether today broke them
  • whether YESTERDAY was an inside day  (prev_inside)  → inside-day breakout setup
  • day-of-week, gap type (Gap Up / Down / Flat), Inside / Outside day

Output:
  analysis/facts.csv      – one row per (instrument, day)
  analysis/facts.parquet
"""

import os
import numpy as np
import pandas as pd

import data_store

# Backward-compat alias — the live registry is data_store.discover(), which
# merges these legacy local paths with everything in data/ (incl. uploaded
# instruments such as NQ / XAUUSD).
FILES = data_store.LEGACY

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

DEFAULT_OPEN_T  = 9 * 60 + 15    # 09:15 (NSE). Auto-detected per instrument when None.
DEFAULT_CLOSE_T = 15 * 60 + 15   # 15:15 square-off — nothing carries past this.
ORB_MIN  = 15
IB_MIN   = 60
GAP_THRESHOLD = 0.15             # % vs prev close to qualify as Gap Up/Down

# Clean gap-% buckets (mirrored for negative gaps), most-positive first.
GAP_BUCKET_ORDER = ["> +1%", "+0.5% to +1%", "+0.2% to +0.5%", "0 to +0.2%",
                    "0 to -0.2%", "-0.2% to -0.5%", "-0.5% to -1%", "< -1%"]


def gap_bucket_label(gp):
    """Map a gap % to its bucket label (0.2 / 0.5 / 1.0 boundaries, signed)."""
    if pd.isna(gp):
        return "n/a"
    neg, a = gp < 0, abs(gp)
    if a >= 1.0:
        return "< -1%" if neg else "> +1%"
    if a >= 0.5:
        return "-0.5% to -1%" if neg else "+0.5% to +1%"
    if a >= 0.2:
        return "-0.2% to -0.5%" if neg else "+0.2% to +0.5%"
    return "0 to -0.2%" if neg else "0 to +0.2%"


def _first_extreme_side(win, hi, lo):
    t_hi = win.loc[win["high"] >= hi, "t_min"].min()
    t_lo = win.loc[win["low"]  <= lo, "t_min"].min()
    if t_hi < t_lo:
        return "high"
    if t_lo < t_hi:
        return "low"
    c = win.loc[win["t_min"] == t_hi].iloc[0]
    return "low" if c["close"] >= c["open"] else "high"


def _breach(post, hi, lo):
    """WICK breach: a level counts as broken the instant price trades through it."""
    if post.empty:
        return False, False, "none"
    hi_mask = post["high"] > hi
    lo_mask = post["low"]  < lo
    hb, lb = bool(hi_mask.any()), bool(lo_mask.any())
    t_hi = post.loc[hi_mask, "t_min"].min() if hb else np.nan
    t_lo = post.loc[lo_mask, "t_min"].min() if lb else np.nan
    if hb and lb:
        first = "high" if t_hi < t_lo else ("low" if t_lo < t_hi else "high")
    elif hb:
        first = "high"
    elif lb:
        first = "low"
    else:
        first = "none"
    return hb, lb, first


def _breach_close(post_tf, hi, lo):
    """CLOSE breach: a level counts as broken only when a resampled (e.g. 3-min)
    candle CLOSES beyond it — stricter than the wick definition."""
    if post_tf is None or post_tf.empty:
        return False, False, "none"
    hi_mask = post_tf["close"] > hi
    lo_mask = post_tf["close"] < lo
    hb, lb = bool(hi_mask.any()), bool(lo_mask.any())
    t_hi = post_tf.loc[hi_mask, "t_min"].min() if hb else np.nan
    t_lo = post_tf.loc[lo_mask, "t_min"].min() if lb else np.nan
    if hb and lb:
        first = "high" if t_hi < t_lo else ("low" if t_lo < t_hi else "high")
    elif hb:
        first = "high"
    elif lb:
        first = "low"
    else:
        first = "none"
    return hb, lb, first


def _excursion(post, hi, lo, rng):
    """Extension reached each side + retracement after the first break."""
    if post.empty or rng <= 0:
        return np.nan, np.nan, np.nan, np.nan
    up_ext = (post["high"].max() - hi) / rng
    dn_ext = (lo - post["low"].min()) / rng

    up_break = post[post["high"] > hi]
    if not up_break.empty:
        t0 = up_break["t_min"].iloc[0]
        after = post[post["t_min"] >= t0]
        up_retr = (hi - after["low"].min()) / rng
    else:
        up_retr = np.nan

    dn_break = post[post["low"] < lo]
    if not dn_break.empty:
        t0 = dn_break["t_min"].iloc[0]
        after = post[post["t_min"] >= t0]
        dn_retr = (after["high"].max() - lo) / rng
    else:
        dn_retr = np.nan

    return up_ext, dn_ext, up_retr, dn_retr


def _pd_breaks(g, pdh, pdl):
    """
    Did TODAY (same session only) trade AT/THROUGH the Previous-Day High / Low,
    and which level was reached first. TOUCH semantics via cumulative cross:
    the level counts once the day's running range brackets it (price has been
    on both sides → it traded through the level). A day that OPENS beyond a
    level (e.g. gaps above PDH) only counts after pulling back to it. Robust
    to fast markets where no single candle straddles the level.
    """
    if pd.isna(pdh):
        return False, False, "none"
    cmh, cml = g["high"].cummax(), g["low"].cummin()
    hi_mask = (cmh >= pdh) & (cml <= pdh)
    lo_mask = (cmh >= pdl) & (cml <= pdl)
    bh, bl = bool(hi_mask.any()), bool(lo_mask.any())
    t_h = g.loc[hi_mask, "t_min"].min() if bh else np.nan
    t_l = g.loc[lo_mask, "t_min"].min() if bl else np.nan
    if bh and bl:
        first = "pdh" if t_h < t_l else ("pdl" if t_l < t_h else "pdh")
    elif bh:
        first = "pdh"
    elif bl:
        first = "pdl"
    else:
        first = "none"
    return bh, bl, first


def clean_min(df):
    """Normalise an OHLC minute DataFrame: lowercase cols, parse date, add t_min.

    Memory diet (Streamlit Cloud has ~1 GB): floats stored as float32 (halves
    the numeric footprint; plenty of precision for prices) and date_only holds
    SHARED per-day date objects instead of one Python object per row (saves
    >100 MB on the 3.8M-row XAUUSD frame).
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:                 # kept for VWAP (optional column)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["high"] >= df["low"]) & (df["open"] > 0)]
    df = df.sort_values("date").reset_index(drop=True)
    for c in df.columns:
        if c != "date" and pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
    dd = df["date"].dt.date
    canon = {v: v for v in pd.unique(dd)}      # one shared object per day
    df["date_only"] = dd.map(canon)
    df["t_min"] = (df["date"].dt.hour * 60 + df["date"].dt.minute).astype("int16")
    return df


def detect_open_t(df):
    """Session open = the most common first-candle minute across days (auto-detect)."""
    firsts = df.groupby("date_only")["t_min"].min()
    return int(firsts.mode().iloc[0])


def to_timeframe(df, tf=5, open_t=None):
    """
    Resample a cleaned 1-min frame to `tf`-minute bars, anchored at the session
    open so the first bucket starts exactly at open_t and buckets never cross a
    day boundary. Returns the same column shape as clean_min (date, open, high,
    low, close, [volume], date_only, t_min) where `date`/`t_min` are the bar's
    OPEN time. `tf<=1` returns the frame unchanged.
    """
    if tf is None or tf <= 1:
        return df
    if open_t is None:
        open_t = detect_open_t(df)
    bucket = (df["t_min"].to_numpy() - open_t) // tf
    g = df.groupby([df["date_only"].to_numpy(), bucket], sort=True)
    agg = {"date": "first", "open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = g.agg(agg).reset_index(drop=True)
    out["date_only"] = out["date"].dt.date
    out["t_min"] = out["date"].dt.hour * 60 + out["date"].dt.minute
    return out.sort_values("date").reset_index(drop=True)


def build_from_minute(df, name, open_t=None, close_t=DEFAULT_CLOSE_T,
                      orb_min=ORB_MIN, ib_min=IB_MIN):
    """
    Build the per-day facts table from minute data for ANY instrument.
    open_t  : session-open minute (auto-detected if None) → IB/ORB windows start here.
    close_t : square-off minute (default 15:15) → nothing after this counts; no carry.
    ib_min  : Initial-Balance window length in minutes (default 60); orb_min likewise.
    """
    if "t_min" not in df.columns:
        df = clean_min(df)
    if open_t is None:
        open_t = detect_open_t(df)
    orb_end = open_t + orb_min - 1
    ib_end  = open_t + ib_min  - 1
    min_ib_bars = min(20, max(5, int(ib_min * 0.66)))

    recs = []
    prev_close = prev_high = prev_low = None
    prev_inside = False

    for day, g0 in df.groupby("date_only", sort=True):
        # session window only: [open_t, close_t]  → enforces the 15:15 square-off
        sess = g0[(g0["t_min"] >= open_t) & (g0["t_min"] <= close_t)].sort_values("t_min")
        if sess.empty:
            continue
        g = sess
        orb = g[g["t_min"] <= orb_end]
        ib  = g[g["t_min"] <= ib_end]
        if len(orb) < 5 or len(ib) < min_ib_bars:
            prev_close, prev_high, prev_low = g.iloc[-1]["close"], g["high"].max(), g["low"].min()
            prev_inside = False
            continue

        day_open  = g.iloc[0]["open"]
        day_close = g.iloc[-1]["close"]      # close AT the square-off
        day_high  = g["high"].max()
        day_low   = g["low"].min()

        pdh, pdl = prev_high, prev_low      # Previous-Day High / Low

        if prev_close:
            gap_pct = (day_open - prev_close) / prev_close * 100
            if gap_pct > GAP_THRESHOLD:    gap_type = "Gap Up"
            elif gap_pct < -GAP_THRESHOLD: gap_type = "Gap Down"
            else:                          gap_type = "Flat"
            inside  = (day_high <= prev_high) and (day_low >= prev_low)
            outside = (day_high >  prev_high) and (day_low <  prev_low)
        else:
            gap_pct, gap_type, inside, outside = np.nan, "n/a", False, False

        # Previous-Day level breaks (today)
        broke_pdh, broke_pdl, pd_first = _pd_breaks(g, pdh, pdl)

        # first-candle direction: close of the first N minutes vs session open
        fc = {}
        for nmin in (15, 30, 60):
            w = g[g["t_min"] <= open_t + nmin - 1]
            fc[nmin] = bool(w.iloc[-1]["close"] > day_open) if len(w) else False

        rec = {
            "instrument": name,
            "date": pd.Timestamp(day),
            "dow":  pd.Timestamp(day).day_name(),
            "day_open": day_open, "day_close": day_close,
            "day_high": day_high, "day_low": day_low,
            "prev_close": prev_close,
            "pdh": pdh, "pdl": pdl,
            "gap_pct": round(gap_pct, 3) if prev_close else np.nan,
            "gap_type": gap_type,
            "gap_bucket": gap_bucket_label(gap_pct if prev_close else np.nan),
            "inside_day": inside, "outside_day": outside,
            "prev_inside": prev_inside,        # yesterday was an inside day → breakout setup
            "broke_pdh": broke_pdh, "broke_pdl": broke_pdl,
            "broke_pd_both": broke_pdh and broke_pdl,
            "broke_pd_none": (not broke_pdh) and (not broke_pdl),
            "broke_pd_first": pd_first,
            "first15_bull": fc[15], "first30_bull": fc[30], "first60_bull": fc[60],
        }

        for tag, win, end_t in (("orb", orb, orb_end), ("ib", ib, ib_end)):
            hi, lo = win["high"].max(), win["low"].min()
            rng = hi - lo
            first_side = _first_extreme_side(win, hi, lo)
            win_close = win.iloc[-1]["close"]               # close of the window's last candle
            post = g[g["t_min"] > end_t]
            hb, lb, bfirst = _breach(post, hi, lo)
            up_ext, dn_ext, up_retr, dn_retr = _excursion(post, hi, lo, rng)
            bc = 2 if (hb and lb) else (1 if (hb or lb) else 0)
            # 3-min-CLOSE breach variant (stricter): resample post to 3-min bars
            post3 = to_timeframe(post, 3, open_t) if not post.empty else post
            hb_c, lb_c, bfirst_c = _breach_close(post3, hi, lo)
            bc_c = 2 if (hb_c and lb_c) else (1 if (hb_c or lb_c) else 0)

            rec.update({
                f"{tag}_high": hi, f"{tag}_low": lo, f"{tag}_range": round(rng, 2),
                f"{tag}_close": win_close,
                f"{tag}_close_above_mid": bool(win_close > (hi + lo) / 2),
                # close location within the window range: 0 = at low, 1 = at high
                f"{tag}_close_loc": round((win_close - lo) / rng, 4) if rng > 0 else 0.5,
                f"{tag}_first_side": first_side,
                f"{tag}_high_break": hb, f"{tag}_low_break": lb,
                f"{tag}_both_break": hb and lb,
                f"{tag}_one_side": bc == 1, f"{tag}_no_break": bc == 0,
                f"{tag}_break_first": bfirst, f"{tag}_break_count": bc,
                # 3-min-close breach variant (suffix _c)
                f"{tag}_high_break_c": hb_c, f"{tag}_low_break_c": lb_c,
                f"{tag}_both_break_c": hb_c and lb_c,
                f"{tag}_one_side_c": bc_c == 1, f"{tag}_no_break_c": bc_c == 0,
                f"{tag}_break_first_c": bfirst_c, f"{tag}_break_count_c": bc_c,
                f"{tag}_up_ext":  round(up_ext, 4)  if pd.notna(up_ext)  else np.nan,
                f"{tag}_dn_ext":  round(dn_ext, 4)  if pd.notna(dn_ext)  else np.nan,
                f"{tag}_up_retr": round(up_retr, 4) if pd.notna(up_retr) else np.nan,
                f"{tag}_dn_retr": round(dn_retr, 4) if pd.notna(dn_retr) else np.nan,
            })

        recs.append(rec)
        prev_close, prev_high, prev_low = day_close, day_high, day_low
        prev_inside = inside

    return pd.DataFrame(recs)


def sessionize(df, open_t, close_t):
    """
    Return (df2, open_t2, close_t2) with the session window guaranteed NOT to
    cross midnight, so build_from_minute's per-calendar-day grouping works.

    Same-day sessions (close_t >= open_t) pass through unchanged. Cross-midnight
    sessions (e.g. Globex 18:00 → 16:00 next day, Asia 21:30 → 03:00) are
    time-shifted forward so the open lands at 00:00 of the CLOSE date — i.e.
    the session is labeled by the day it ends (futures trading-day convention:
    Sunday 18:00 Globex open belongs to Monday).
    """
    if close_t >= open_t:
        return df, open_t, close_t
    shift = 1440 - open_t
    df2 = df.copy()
    df2["date"] = df2["date"] + pd.Timedelta(minutes=shift)
    df2["date_only"] = df2["date"].dt.date
    df2["t_min"] = df2["date"].dt.hour * 60 + df2["date"].dt.minute
    return df2, 0, close_t + shift


def build_session_facts(df, name, session, orb_min=ORB_MIN, ib_min=IB_MIN):
    """
    Facts table for a NAMED session of a 24-hour instrument (NY / Globex /
    London / Asia — see data_store.SESSIONS_24H). Gap %, PDH/PDL and
    inside/outside flags all chain session-vs-SAME-session (London gap =
    London open vs previous London close, etc.) because each session is
    built independently.
    """
    s = data_store.SESSIONS_24H[session]
    if "t_min" not in df.columns:
        df = clean_min(df)
    df2, o2, c2 = sessionize(df, s["open_t"], s["close_t"])
    return build_from_minute(df2, name, o2, c2, orb_min=orb_min, ib_min=ib_min)


def build_instrument(name, path, open_t=None, close_t=None):
    print(f"\n[{name}] reading {path} ...")
    df = clean_min(data_store.read_minute(path))
    # session times: explicit arg > CODE defaults > auto-detect.
    # The instruments.json override layer is deliberately IGNORED here — stale
    # UI overrides have twice poisoned the canonical facts table (NQ built on
    # 03:00-07:00, XAUUSD truncated at noon). facts.csv must always reflect the
    # canonical sessions in data_store.SESSION_DEFAULTS.
    d = data_store.SESSION_DEFAULTS.get(name, {})
    ot = open_t if open_t is not None else d.get("open_t", detect_open_t(df))
    ct = close_t if close_t is not None else d.get("close_t", DEFAULT_CLOSE_T)
    print(f"   {len(df):,} candles over {df['date_only'].nunique():,} days; "
          f"open={ot} close={ct}")
    out = build_from_minute(df, name, ot, ct)
    print(f"   built {len(out):,} usable day-records")
    return out


def main():
    files = data_store.discover()
    frames = [build_instrument(n, p) for n, p in files.items() if os.path.exists(p)]
    facts = pd.concat(frames, ignore_index=True)

    csv_path = os.path.join(OUT_DIR, "facts.csv")
    facts.to_csv(csv_path, index=False)
    try:
        facts.to_parquet(os.path.join(OUT_DIR, "facts.parquet"), index=False)
    except Exception as e:
        print(f"   (parquet skipped: {e})")

    print(f"\nSaved facts table:")
    print(f"   {csv_path}")
    print(f"   {len(facts):,} rows  x  {facts.shape[1]} columns")
    print(f"   date range : {facts['date'].min().date()} -> {facts['date'].max().date()}")


if __name__ == "__main__":
    main()
