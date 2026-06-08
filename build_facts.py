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

FILES = {
    "NIFTY 50":   r"C:\NIFTY 50_minute.csv",
    "BANK NIFTY": r"C:\NIFTY BANK_minute.csv",
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

OPEN_T   = 9 * 60 + 15          # 09:15
ORB_MIN  = 15
IB_MIN   = 60
GAP_THRESHOLD = 0.15            # % vs prev close to qualify as Gap Up/Down


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
    """Did the full day break Previous-Day High / Low, and which first."""
    if pd.isna(pdh):
        return False, False, "none"
    hi_mask = g["high"] > pdh
    lo_mask = g["low"]  < pdl
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


def build_instrument(name, path):
    print(f"\n[{name}] reading {path} ...")
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["high"] >= df["low"]) & (df["open"] > 0)]
    df = df.sort_values("date").reset_index(drop=True)
    df["date_only"] = df["date"].dt.date
    df["t_min"] = df["date"].dt.hour * 60 + df["date"].dt.minute
    print(f"   {len(df):,} candles over {df['date_only'].nunique():,} days")

    orb_end = OPEN_T + ORB_MIN - 1
    ib_end  = OPEN_T + IB_MIN  - 1

    recs = []
    prev_close = prev_high = prev_low = None
    prev_inside = False

    for day, g in df.groupby("date_only", sort=True):
        g = g.sort_values("t_min")
        orb = g[(g["t_min"] >= OPEN_T) & (g["t_min"] <= orb_end)]
        ib  = g[(g["t_min"] >= OPEN_T) & (g["t_min"] <= ib_end)]
        if len(orb) < 5 or len(ib) < 20:
            prev_close, prev_high, prev_low = g.iloc[-1]["close"], g["high"].max(), g["low"].min()
            prev_inside = False
            continue

        day_open  = g.iloc[0]["open"]
        day_close = g.iloc[-1]["close"]
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
            "inside_day": inside, "outside_day": outside,
            "prev_inside": prev_inside,        # yesterday was an inside day → breakout setup
            "broke_pdh": broke_pdh, "broke_pdl": broke_pdl,
            "broke_pd_both": broke_pdh and broke_pdl,
            "broke_pd_none": (not broke_pdh) and (not broke_pdl),
            "broke_pd_first": pd_first,
        }

        for tag, win, end_t in (("orb", orb, orb_end), ("ib", ib, ib_end)):
            hi, lo = win["high"].max(), win["low"].min()
            rng = hi - lo
            first_side = _first_extreme_side(win, hi, lo)
            post = g[g["t_min"] > end_t]
            hb, lb, bfirst = _breach(post, hi, lo)
            up_ext, dn_ext, up_retr, dn_retr = _excursion(post, hi, lo, rng)
            bc = 2 if (hb and lb) else (1 if (hb or lb) else 0)

            rec.update({
                f"{tag}_high": hi, f"{tag}_low": lo, f"{tag}_range": round(rng, 2),
                f"{tag}_first_side": first_side,
                f"{tag}_high_break": hb, f"{tag}_low_break": lb,
                f"{tag}_both_break": hb and lb,
                f"{tag}_one_side": bc == 1, f"{tag}_no_break": bc == 0,
                f"{tag}_break_first": bfirst, f"{tag}_break_count": bc,
                f"{tag}_up_ext":  round(up_ext, 4)  if pd.notna(up_ext)  else np.nan,
                f"{tag}_dn_ext":  round(dn_ext, 4)  if pd.notna(dn_ext)  else np.nan,
                f"{tag}_up_retr": round(up_retr, 4) if pd.notna(up_retr) else np.nan,
                f"{tag}_dn_retr": round(dn_retr, 4) if pd.notna(dn_retr) else np.nan,
            })

        recs.append(rec)
        prev_close, prev_high, prev_low = day_close, day_high, day_low
        prev_inside = inside

    out = pd.DataFrame(recs)
    print(f"   built {len(out):,} usable day-records")
    return out


def main():
    frames = [build_instrument(n, p) for n, p in FILES.items() if os.path.exists(p)]
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
