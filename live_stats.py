"""
live_stats.py — turn today's (partial) session into a day-type classification and
look up the matching historical probabilities from facts.csv.

Pure functions (no Streamlit), so they're unit-testable with DemoSource data.
"""
import numpy as np
import pandas as pd

import build_facts
import prob_app

ORB_MIN, IB_MIN = 15, 60
EXT_LEVELS = [0.25, 0.5, 1.0, 1.5, 2.0]


def classify_live(today_df, prev_hlc, open_t, close_t, now_t=None,
                  ib_min=IB_MIN, orb_min=ORB_MIN):
    """Features known so far from today's session (capped at now_t)."""
    prev_high, prev_low, prev_close = prev_hlc if prev_hlc else (None, None, None)
    df = today_df if "t_min" in today_df.columns else build_facts.clean_min(today_df)
    df = df[(df["t_min"] >= open_t) & (df["t_min"] <= close_t)].sort_values("t_min")

    feat = {"open_t": open_t, "close_t": close_t, "ib_min": ib_min,
            "prev_high": prev_high, "prev_low": prev_low, "prev_close": prev_close}
    if now_t is None:
        now_t = int(df["t_min"].max()) if len(df) else open_t
    feat["now_t"] = now_t
    df = df[df["t_min"] <= now_t]
    if df.empty:
        feat["phase"] = "pre-open"
        return feat

    orb_end, ib_end = open_t + orb_min - 1, open_t + ib_min - 1
    feat["day_open"] = float(df.iloc[0]["open"])
    feat["price"]    = float(df.iloc[-1]["close"])
    feat["dow"]      = pd.Timestamp(df.iloc[0]["date"]).day_name()

    if prev_close:
        gp = (feat["day_open"] - prev_close) / prev_close * 100
        feat["gap_pct"]  = round(gp, 3)
        feat["gap_type"] = ("Gap Up" if gp > build_facts.GAP_THRESHOLD
                            else "Gap Down" if gp < -build_facts.GAP_THRESHOLD else "Flat")
        feat["broke_pdh"] = bool(df["high"].max() > prev_high)
        feat["broke_pdl"] = bool(df["low"].min() < prev_low)
    else:
        feat["gap_pct"] = feat["gap_type"] = None

    # ORB
    orb = df[df["t_min"] <= orb_end]
    if now_t >= orb_end and len(orb) >= 5:
        oh, ol = float(orb["high"].max()), float(orb["low"].min())
        feat.update(orb_high=oh, orb_low=ol, orb_range=oh - ol,
                    orb_first_side=build_facts._first_extreme_side(orb, oh, ol))

    # IB
    ib = df[df["t_min"] <= ib_end]
    if now_t >= ib_end and len(ib) >= min(20, max(5, int(ib_min * 0.66))):
        ih, il = float(ib["high"].max()), float(ib["low"].min())
        rg = ih - il
        ib_close = float(ib.iloc[-1]["close"])
        feat.update(ib_high=ih, ib_low=il, ib_range=rg,
                    ib_size_pct=round(rg / feat["day_open"] * 100, 3) if feat["day_open"] else None,
                    ib_first_side=build_facts._first_extreme_side(ib, ih, il),
                    ib_close=ib_close, ib_mid=(ih + il) / 2,
                    ib_close_above_mid=bool(ib_close > (ih + il) / 2))
        post = df[df["t_min"] > ib_end]
        feat["broke_ib_high"] = bool(post["high"].max() > ih) if len(post) else False
        feat["broke_ib_low"]  = bool(post["low"].min()  < il) if len(post) else False
        feat["phase"] = "post-IB · live" if len(post) else "IB just complete"
    elif now_t >= orb_end:
        feat["phase"] = "IB forming"
    else:
        feat["phase"] = "ORB forming"
    return feat


def live_probabilities(facts, feat):
    """Conditional historical probabilities for today's emerging day type."""
    out = {}
    base = facts[facts["dow"] == feat.get("dow")]
    gt = feat.get("gap_type")
    gapset = base[base["gap_type"] == gt] if gt else base
    out["n_gap"] = len(gapset)
    out["gap_slice"] = prob_app.stats(gapset, "ib")

    fs = feat.get("ib_first_side")
    matched = gapset
    if fs:
        matched = gapset[gapset["ib_first_side"] == fs]
        out["n_matched"] = len(matched)
        out["matched"] = prob_app.stats(matched, "ib")
        opp = "low" if fs == "high" else "high"
        out["fade_opp"] = opp
        if len(matched):
            out["fade_opp_break"] = float(matched[f"ib_{opp}_break"].mean())
            bf = matched["ib_break_first"].value_counts(normalize=True)
            out["fade_opp_first"] = float(bf.get(opp, 0.0))

        # close-vs-midpoint confirmation of the fade
        cam = feat.get("ib_close_above_mid")
        if cam is not None and "ib_close_above_mid" in matched.columns and len(matched):
            slc = matched[matched["ib_close_above_mid"] == cam]
            # the fade is "confirmed" when the close sits beyond the midpoint, away
            # from the first-formed extreme (low-first -> close above mid; high-first
            # -> close below mid)
            confirmed = (fs == "low" and cam) or (fs == "high" and not cam)
            out["mid"] = {
                "close_above_mid": bool(cam), "confirmed": bool(confirmed),
                "opp": opp, "n": int(len(slc)),
                "prob": float(slc[f"ib_{opp}_break"].mean()) if len(slc) else None,
                "base": float(matched[f"ib_{opp}_break"].mean()),
            }

    out["pd_table"] = prob_app.gap_pd_table(base)

    if feat.get("ib_range"):
        src = matched if (fs and len(matched) >= 30) else gapset
        ue, de = src["ib_up_ext"].dropna(), src["ib_dn_ext"].dropna()
        out["ext_src_n"] = len(src)
        out["ext"] = [{
            "level": L,
            "up_price": feat["ib_high"] + L * feat["ib_range"],
            "up_p": float((ue >= L).mean()) if len(ue) else None,
            "dn_price": feat["ib_low"] - L * feat["ib_range"],
            "dn_p": float((de >= L).mean()) if len(de) else None,
        } for L in EXT_LEVELS]
    return out
