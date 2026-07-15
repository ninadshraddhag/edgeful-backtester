"""
ORB / IB Probability Explorer
==============================
Ask probabilistic questions of 10 years of NIFTY 50 & BANK NIFTY data.

Filters (= your question):
   instrument · setup (ORB/IB) · date range · day of week · gap type ·
   inside/outside day · which extreme formed first · ORB/IB size

Answers:
   • break probabilities (HIGH / LOW / BOTH / ONE / NEITHER) + edge vs baseline
   • "first-move-fades" conditional
   • EXTENSION reach probabilities (0.1–2.0 × range, both sides)
   • RETRACEMENT probabilities after a break (0.1–0.9 × range)
   • Previous-Day High/Low: inside-day breakout, gap-up→PDH, gap-down→PDL

Data layer:  analysis/facts.csv  (build_facts.py)
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import build_facts
import data_store

# NOTE: set_page_config is NOT called at import time so this module can be embedded
# as a mode inside app.py. It is only set when run standalone (see _standalone()).

HERE   = os.path.dirname(os.path.abspath(__file__))
FACTS  = os.path.join(HERE, "analysis", "facts.csv")
DOW_ORDER  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
GAP_ORDER  = ["Gap Up", "Flat", "Gap Down"]
KIND_ORDER = ["Inside Day", "Normal", "Outside Day"]
GAP_BUCKETS = build_facts.GAP_BUCKET_ORDER          # clean ±0.2/0.5/1% bands


# ─── data ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_facts(mtime):           # mtime arg busts the cache when facts.csv changes
    df = pd.read_csv(FACTS, parse_dates=["date"])
    df["day_kind"] = np.where(df["inside_day"], "Inside Day",
                     np.where(df["outside_day"], "Outside Day", "Normal"))
    return df


@st.cache_data(show_spinner=True, max_entries=2)
def load_facts_custom(ib_min, _mtimes):
    """Recompute the facts table live for a non-default IB duration (cached)."""
    import build_facts
    import data_store
    frames = []
    for name, path in data_store.discover().items():
        if os.path.exists(path):
            mdf = build_facts.clean_min(data_store.read_minute(path))
            open_t = data_store.session_open(name, build_facts.detect_open_t(mdf))
            frames.append(build_facts.build_from_minute(
                mdf, name, open_t, build_facts.DEFAULT_CLOSE_T, ib_min=ib_min))
    df = pd.concat(frames, ignore_index=True)
    df["day_kind"] = np.where(df["inside_day"], "Inside Day",
                     np.where(df["outside_day"], "Outside Day", "Normal"))
    return df


@st.cache_data(show_spinner=True, max_entries=6)
def load_facts_session(inst, session, ib_min, mtime):
    """Facts table for one NAMED session of a 24-h instrument (cached).
    Gap/PDH/PDL chain session-vs-same-session (London vs London, Asia vs Asia)."""
    path = data_store.discover()[inst]
    mdf = build_facts.clean_min(data_store.read_minute(path))
    out = build_facts.build_session_facts(mdf, inst, session, ib_min=ib_min)
    out["day_kind"] = np.where(out["inside_day"], "Inside Day",
                      np.where(out["outside_day"], "Outside Day", "Normal"))
    return out


@st.cache_data(show_spinner=False, max_entries=1)
def _minute_and_open(inst, mtime):
    """Cleaned minute frame + auto-detected open for the day-browser charts."""
    path = data_store.discover()[inst]
    mdf = build_facts.clean_min(data_store.read_minute(path))
    return mdf, build_facts.detect_open_t(mdf)


def build_facts_inline():
    build_facts.main()          # builds for every discovered instrument


# ─── natural-language → filters ───────────────────────────────────────────────

def parse_query(q: str) -> dict:
    q = q.lower()
    upd = {}
    days = [d.capitalize() for d in DOW_ORDER if d.lower() in q]
    if days:
        upd["f_dow"] = days
    gaps = []
    if "gap up"   in q: gaps.append("Gap Up")
    if "gap down" in q: gaps.append("Gap Down")
    if "flat"     in q: gaps.append("Flat")
    if gaps:
        upd["f_gap"] = gaps
    kinds = []
    if "inside day"  in q: kinds.append("Inside Day")
    if "outside day" in q: kinds.append("Outside Day")
    if kinds:
        upd["f_kind"] = kinds
    if "high formed first" in q or "high first" in q:
        upd["f_first"] = "High formed first"
    elif "low formed first" in q or "low first" in q:
        upd["f_first"] = "Low formed first"
    if "bank" in q:
        upd["f_instrument"] = "BANK NIFTY"
    elif "nq" in q or "nasdaq" in q:
        upd["f_instrument"] = "NQ"
    elif "xau" in q or "gold" in q:
        upd["f_instrument"] = "XAUUSD"
    elif "nifty" in q:
        upd["f_instrument"] = "NIFTY 50"
    if "orb" in q or "opening range" in q:
        upd["f_setup"] = "ORB (15 min)"
    elif "ib" in q or "initial balance" in q or "hour" in q:
        upd["f_setup"] = "IB"
    if "globex" in q or "overnight" in q:
        upd["f_session"] = "Globex"
    elif "london" in q:
        upd["f_session"] = "London"
    elif "asia" in q or "tokyo" in q:
        upd["f_session"] = "Asia"
    elif "new york" in q or " ny " in f" {q} ":
        upd["f_session"] = "NY"
    return upd


# ─── stats ────────────────────────────────────────────────────────────────────

def stats(sub, tag):
    n = len(sub)
    if n == 0:
        return None
    bf = sub[f"{tag}_break_first"].value_counts(normalize=True)
    return {
        "n": n,
        "high": sub[f"{tag}_high_break"].mean(),
        "low":  sub[f"{tag}_low_break"].mean(),
        "both": sub[f"{tag}_both_break"].mean(),
        "one":  sub[f"{tag}_one_side"].mean(),
        "none": sub[f"{tag}_no_break"].mean(),
        "first_high": bf.get("high", 0.0),
        "first_low":  bf.get("low", 0.0),
    }


def pct(x):
    return f"{x*100:.1f}%"


def gap_pd_table(s):
    rows = []
    for gt in GAP_ORDER:
        x = s[s["gap_type"] == gt]
        if len(x) == 0:
            continue
        rows.append({
            "Gap": gt, "days": len(x),
            "reach PDH %": round(x["broke_pdh"].mean()*100, 1),
            "reach PDL %": round(x["broke_pdl"].mean()*100, 1),
            "either %":    round((x["broke_pdh"] | x["broke_pdl"]).mean()*100, 1),
            "both %":      round(x["broke_pd_both"].mean()*100, 1),
            "neither %":   round(x["broke_pd_none"].mean()*100, 1),
        })
    return pd.DataFrame(rows)


# ─── app ──────────────────────────────────────────────────────────────────────

def render():
    st.title("🎲 ORB / IB Probability Explorer")
    st.caption("All loaded instruments (NIFTY 50 · BANK NIFTY · NQ · XAUUSD) · "
               "NQ/XAUUSD support session views: NY · Globex · London · Asia (ET) · "
               "set filters to ask a question, or type one below")

    if not os.path.exists(FACTS):
        st.error("Facts table not found.")
        if st.button("Build it now (reads the two C:\\ CSVs, ~1 min)"):
            with st.spinner("Crunching 10 years of minute data…"):
                build_facts_inline()
            st.cache_data.clear()
            st.rerun()
        st.stop()

    # IB duration (widget lives in the sidebar below; read its state here so the
    # right facts table is loaded before anything renders)
    ib_min = int(st.session_state.get("f_ibmin", 60))
    if ib_min == 60:
        df = load_facts(os.path.getmtime(FACTS))
    else:
        import data_store as _ds
        mtimes = tuple(os.path.getmtime(p) for p in _ds.discover().values()
                       if os.path.exists(p))
        with st.spinner(f"Computing facts for a {ib_min}-min IB (first time only)…"):
            df = load_facts_custom(ib_min, mtimes)

    # ── natural-language box ──────────────────────────────────────────────────
    with st.form("nlq"):
        c1, c2 = st.columns([6, 1])
        q = c1.text_input("Ask a question",
                          placeholder="e.g.  'Bank Nifty Mondays gap up, high formed first'   ·   'Nifty outside day IB'",
                          label_visibility="collapsed")
        submitted = c2.form_submit_button("Ask ▶", use_container_width=True)
    if submitted and q.strip():
        for k, v in parse_query(q).items():
            st.session_state[k] = v
        st.rerun()

    # ── defaults ──────────────────────────────────────────────────────────────
    st.session_state.setdefault("f_instrument", "NIFTY 50")
    st.session_state.setdefault("f_setup", "IB")
    st.session_state.setdefault("f_ibmin", 60)
    st.session_state.setdefault("f_dow", DOW_ORDER.copy())
    st.session_state.setdefault("f_gap", GAP_ORDER.copy())
    st.session_state.setdefault("f_kind", KIND_ORDER.copy())
    st.session_state.setdefault("f_first", "Either")

    # ── sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters  =  your question")
        instruments = sorted(df["instrument"].unique())   # every instrument in facts
        if st.session_state.get("f_instrument") not in instruments:
            st.session_state["f_instrument"] = instruments[0]
        st.radio("Instrument", instruments, key="f_instrument")
        inst = st.session_state["f_instrument"]

        # session picker — 24-hour instruments only (NQ / XAUUSD)
        session = "NY"
        if data_store.has_sessions(inst):
            if st.session_state.get("f_session") not in data_store.SESSIONS_24H:
                st.session_state["f_session"] = "NY"
            session = st.radio(
                "Session", list(data_store.SESSIONS_24H.keys()),
                format_func=lambda k: data_store.SESSIONS_24H[k]["label"],
                key="f_session",
                help="IB = first 60 min of the session, ORB = first 15 min. "
                     "Cross-midnight sessions (Globex, Asia) are labeled by their "
                     "CLOSE date — the Globex session opening Sunday 18:00 counts "
                     "as Monday. Gap % is measured vs the SAME session's previous "
                     "close (London vs London, Asia vs Asia).")

        st.radio("Setup", ["IB", "ORB (15 min)"], key="f_setup")
        st.number_input("IB duration (min)", 10, 240, step=10, key="f_ibmin",
                        help="Length of the Initial Balance window from the open, in "
                             "10-min steps (e.g. 40, 50). Non-60 values recompute the "
                             "stats live (cached).")
        tag  = "ib" if st.session_state["f_setup"].startswith("IB") else "orb"

        # data slice for the chosen instrument + session
        if session != "NY" and data_store.has_sessions(inst):
            _path = data_store.discover().get(inst)
            with st.spinner(f"Building {inst} {session}-session stats "
                            "(first time only, then cached)…"):
                df_inst = load_facts_session(inst, session, ib_min,
                                             os.path.getmtime(_path))
        else:
            df_inst = df[df["instrument"] == inst].copy()
        if "gap_bucket" not in df_inst.columns:      # back-compat with old facts.csv
            df_inst["gap_bucket"] = df_inst["gap_pct"].map(build_facts.gap_bucket_label)
        if df_inst.empty:
            st.warning("No sessions found for this instrument/session combination.")
            st.stop()

        st.divider()
        dmin = df_inst["date"].min().date()
        dmax = df_inst["date"].max().date()
        date_sel = st.date_input("Date range", value=(dmin, dmax),
                                 min_value=dmin, max_value=dmax,
                                 key=f"f_dates_{inst}_{session}")

        st.multiselect("Day of week", DOW_ORDER, key="f_dow")
        st.multiselect("Gap type", GAP_ORDER, key="f_gap")
        sel_buckets = st.multiselect(
            "Gap % bucket", GAP_BUCKETS, default=GAP_BUCKETS,
            key=f"f_gapbucket_{inst}_{session}",
            help="Opening gap vs the previous session's close, in clean bands "
                 "(±0–0.2% · 0.2–0.5% · 0.5–1% · 1%+). Deselect bands to focus — "
                 "e.g. keep only '+0.2% to +0.5%' for small gap-ups.")
        st.multiselect("Day kind", KIND_ORDER, key="f_kind")
        st.radio("Which extreme formed first",
                 ["Either", "High formed first", "Low formed first"], key="f_first")

        fcc = st.columns([1, 1.4])
        fc_dur = fcc[0].selectbox("First candle", [15, 30, 60], index=1,
                                  format_func=lambda x: f"{x} min", key="f_fc_dur")
        fc_dir = fcc[1].radio("Direction", ["Any", "Bullish", "Bearish"],
                              key="f_fc_dir", horizontal=True,
                              help="Filter by the direction of the session's first "
                                   "candle (close of the first N minutes vs the open). "
                                   "E.g. first 30-min candle bullish → long-bias days only.")

        st.divider()
        size_col = f"{tag}_range"
        # IB/ORB size as a % of price (volatility-normalized, comparable across
        # instruments) instead of raw points
        size_pct_all = (df_inst[size_col] / df_inst["day_open"] * 100).dropna()
        s_lo = float(np.floor(size_pct_all.min() * 100) / 100) if len(size_pct_all) else 0.0
        s_hi = float(np.ceil(size_pct_all.quantile(0.995) * 100) / 100) if len(size_pct_all) else 2.0
        size_rng = st.slider(f"{tag.upper()} size (% of price)", s_lo, s_hi, (s_lo, s_hi),
                             0.01, key=f"f_sizepct_{tag}_{inst}_{session}_{ib_min}",
                             help="Opening-range width as a % of price — a "
                                  "volatility-normalized size filter (e.g. 0.20%–0.40%).")
        st.caption("Filter days by IB/ORB size as a % of price.")

        st.divider()
        st.markdown("**Extension / Retracement (× range)**")
        bull_ext  = st.slider("Bull extension level",  0.1, 2.0, 0.5, 0.05, key="f_bull_ext")
        bear_ext  = st.slider("Bear extension level",  0.1, 2.0, 0.5, 0.05, key="f_bear_ext")
        bull_retr = st.slider("Bull retracement level", 0.1, 0.9, 0.5, 0.05, key="f_bull_retr")
        bear_retr = st.slider("Bear retracement level", 0.1, 0.9, 0.5, 0.05, key="f_bear_retr")

        if st.button("↺ Reset category filters", use_container_width=True):
            for k in ["f_dow", "f_gap", "f_kind", "f_first"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── apply filters ─────────────────────────────────────────────────────────
    scope = df_inst.copy()                       # single instrument + session
    if isinstance(date_sel, (tuple, list)) and len(date_sel) == 2:
        d0, d1 = pd.Timestamp(date_sel[0]), pd.Timestamp(date_sel[1]) + pd.Timedelta(days=1)
        scope = scope[(scope["date"] >= d0) & (scope["date"] < d1)]

    baseline = scope.copy()                      # date-scoped, all categories
    pdset = scope[scope["dow"].isin(st.session_state["f_dow"])].copy()   # for PD-level stats

    sub = scope[scope["dow"].isin(st.session_state["f_dow"])]
    sub = sub[sub["gap_type"].isin(st.session_state["f_gap"])]
    sub = sub[sub["day_kind"].isin(st.session_state["f_kind"])]
    if st.session_state["f_first"] == "High formed first":
        sub = sub[sub[f"{tag}_first_side"] == "high"]
    elif st.session_state["f_first"] == "Low formed first":
        sub = sub[sub[f"{tag}_first_side"] == "low"]
    fc_col = f"first{fc_dur}_bull"
    if fc_dir != "Any":
        if fc_col in sub.columns:
            sub = sub[sub[fc_col] == (fc_dir == "Bullish")]
        else:
            st.warning("First-candle data missing — rebuild the facts table "
                       "(`python build_facts.py`) to enable this filter.")
    _sz_pct = sub[size_col] / sub["day_open"] * 100
    sub = sub[(_sz_pct >= size_rng[0]) & (_sz_pct <= size_rng[1])]
    sub_all_buckets = sub.copy()                 # for the per-bucket breakdown table
    if len(sel_buckets) < len(GAP_BUCKETS):
        sub = sub[sub["gap_bucket"].isin(sel_buckets)]

    # ── headline sentence ─────────────────────────────────────────────────────
    setup_label = f"IB ({ib_min} min)" if tag == "ib" else "ORB (15 min)"
    bits = [inst]
    if data_store.has_sessions(inst):
        bits.append(f"{session} session")
    bits.append(setup_label)
    if isinstance(date_sel, (tuple, list)) and len(date_sel) == 2:
        bits.append(f"{date_sel[0]}→{date_sel[1]}")
    if len(st.session_state["f_dow"]) < 5:  bits.append("/".join(st.session_state["f_dow"]))
    if len(st.session_state["f_gap"]) < 3:  bits.append("/".join(st.session_state["f_gap"]))
    if len(sel_buckets) < len(GAP_BUCKETS):
        bits.append("gap " + " / ".join(sel_buckets))
    if len(st.session_state["f_kind"]) < 3: bits.append("/".join(st.session_state["f_kind"]))
    if st.session_state["f_first"] != "Either": bits.append(st.session_state["f_first"])
    if fc_dir != "Any": bits.append(f"first {fc_dur}m {fc_dir.lower()}")
    st.subheader("  ·  ".join(str(b) for b in bits))

    s = stats(sub, tag)
    b = stats(baseline, tag)
    if s is None or b is None:
        st.warning("No days match these filters. Loosen them.")
        st.stop()
    if s["n"] < 30:
        st.warning(f"⚠ Only {s['n']} matching days — treat as indicative, not reliable.")

    # ── break probabilities ───────────────────────────────────────────────────
    st.markdown("#### Break probabilities (rest of day)")
    c = st.columns(6)
    c[0].metric("Matching days", f"{s['n']:,}", f"of {b['n']:,}")
    c[1].metric("HIGH breaks", pct(s["high"]), f"{(s['high']-b['high'])*100:+.1f} pp")
    c[2].metric("LOW breaks",  pct(s["low"]),  f"{(s['low']-b['low'])*100:+.1f} pp")
    c[3].metric("BOTH sides",  pct(s["both"]), f"{(s['both']-b['both'])*100:+.1f} pp")
    c[4].metric("ONE side",    pct(s["one"]),  f"{(s['one']-b['one'])*100:+.1f} pp")
    c[5].metric("NEITHER",     pct(s["none"]), f"{(s['none']-b['none'])*100:+.1f} pp",
                delta_color="inverse")
    st.caption("Δ vs this instrument's baseline over the same date range (pp = percentage points).")

    cc = st.columns(2)
    cc[0].metric("Breaks HIGH first", pct(s["first_high"]))
    cc[1].metric("Breaks LOW first",  pct(s["first_low"]))

    # ── first-move-fades ──────────────────────────────────────────────────────
    st.markdown("#### “First-move-fades” — which side breaks FIRST, given which "
                "extreme FORMED first")
    st.caption("FORMED first = which extreme printed first inside the window "
               "(formation). Breaks FIRST = which boundary was breached first "
               "after the window. Only the breaks-first sequencing is shown.")
    hf = sub[sub[f"{tag}_first_side"] == "high"]
    lf = sub[sub[f"{tag}_first_side"] == "low"]
    fm = st.columns(2)
    with fm[0]:
        if len(hf):
            fm[0].metric(f"HIGH formed first → LOW breaks FIRST  (n={len(hf):,})",
                         pct((hf[f"{tag}_break_first"] == "low").mean()))
        else:
            st.info("No 'high formed first' days in this slice.")
    with fm[1]:
        if len(lf):
            fm[1].metric(f"LOW formed first → HIGH breaks FIRST  (n={len(lf):,})",
                         pct((lf[f"{tag}_break_first"] == "high").mean()))
        else:
            st.info("No 'low formed first' days in this slice.")

    # ── close vs midpoint (momentum confirmation) ─────────────────────────────
    st.markdown(f"#### {tag.upper()} close vs midpoint — confirmation of the fade")
    mid_col = f"{tag}_close_above_mid"
    if mid_col in sub.columns:
        # NB: local names must not shadow the baseline stats dict `b` used below
        lf_conf = lf[lf[mid_col] == True]      # low formed first AND closed above mid
        hf_conf = hf[hf[mid_col] == False]     # high formed first AND closed below mid
        mp = st.columns(2)
        with mp[0]:
            if len(lf_conf):
                base_a = pct((lf[f"{tag}_break_first"] == "high").mean()) if len(lf) else "—"
                mp[0].metric("LOW formed first + close ABOVE mid → HIGH breaks FIRST  "
                             f"(n={len(lf_conf):,})",
                             pct((lf_conf[f"{tag}_break_first"] == "high").mean()),
                             f"vs {base_a} for all low-formed-first days")
            else:
                st.info("No days: low formed first + close above mid.")
        with mp[1]:
            if len(hf_conf):
                base_b = pct((hf[f"{tag}_break_first"] == "low").mean()) if len(hf) else "—"
                mp[1].metric("HIGH formed first + close BELOW mid → LOW breaks FIRST  "
                             f"(n={len(hf_conf):,})",
                             pct((hf_conf[f"{tag}_break_first"] == "low").mean()),
                             f"vs {base_b} for all high-formed-first days")
            else:
                st.info("No days: high formed first + close below mid.")
        st.caption(f"Close = last 1-min close of the {tag.upper()} window; midpoint = "
                   "(window high + low) / 2. Closing back beyond the midpoint confirms "
                   "the first move has already been faded.")
    else:
        st.info("Rebuild the facts table (`python build_facts.py`) to enable this stat.")

    # ── close location (75% / 25%) — stronger confirmation ────────────────────
    loc_col = f"{tag}_close_loc"
    if loc_col in sub.columns:
        thr = st.slider(f"Close-location threshold — how deep into the {tag.upper()} "
                        "range the close must be", 55, 95, 75, 5, key=f"f_closeloc_{tag}",
                        help="75 = close in the upper 75% of the range for the bullish "
                             "check (and lower 25% for the bearish check). Stronger than "
                             "the midpoint (50).")
        up_thr, dn_thr = thr / 100.0, 1.0 - thr / 100.0
        st.markdown(f"#### {tag.upper()} close beyond {thr}% of range — stronger fade confirmation")
        lf_q = lf[lf[loc_col] >= up_thr]       # low first AND close in upper thr%
        hf_q = hf[hf[loc_col] <= dn_thr]       # high first AND close in lower (100-thr)%
        qp = st.columns(2)
        with qp[0]:
            if len(lf_q):
                base_a = pct((lf[f"{tag}_break_first"] == "high").mean()) if len(lf) else "—"
                qp[0].metric(f"LOW first + close ≥{thr}% → HIGH breaks FIRST  (n={len(lf_q):,})",
                             pct((lf_q[f"{tag}_break_first"] == "high").mean()),
                             f"vs {base_a} for all low-first days")
            else:
                st.info(f"No days: low first + close ≥{thr}% of range.")
        with qp[1]:
            if len(hf_q):
                base_b = pct((hf[f"{tag}_break_first"] == "low").mean()) if len(hf) else "—"
                qp[1].metric(f"HIGH first + close ≤{100-thr}% → LOW breaks FIRST  (n={len(hf_q):,})",
                             pct((hf_q[f"{tag}_break_first"] == "low").mean()),
                             f"vs {base_b} for all high-first days")
            else:
                st.info(f"No days: high first + close ≤{100-thr}% of range.")
        st.caption(f"Close location = (close − low) ÷ range. A close in the upper {thr}% "
                   "after the LOW formed first is a stronger 'already-faded' signal than "
                   "just clearing the midpoint — the deeper the reversal close, the more "
                   "the first move is confirmed dead.")
    else:
        st.info("Rebuild the facts table to enable the close-location (75%) stat.")

    # ── EXTENSIONS & RETRACEMENTS ─────────────────────────────────────────────
    # Measured ONLY on SINGLE-BREAK days — days where exactly ONE side of the
    # window broke. Bull = only the HIGH broke (clean upside day), Bear = only
    # the LOW broke. Both-sides chop days are excluded so the numbers describe
    # genuine directional days.
    st.divider()
    st.markdown(f"#### Extensions & Retracements  ·  measured in × {tag.upper()} range")
    hi_only = sub[sub[f"{tag}_high_break"] & ~sub[f"{tag}_low_break"]]
    lo_only = sub[sub[f"{tag}_low_break"] & ~sub[f"{tag}_high_break"]]
    n_hi, n_lo = len(hi_only), len(lo_only)
    up_ext = hi_only[f"{tag}_up_ext"].dropna()
    dn_ext = lo_only[f"{tag}_dn_ext"].dropna()
    up_rt  = hi_only[f"{tag}_up_retr"].dropna()
    dn_rt  = lo_only[f"{tag}_dn_retr"].dropna()
    st.caption(f"SINGLE-BREAK days only (exactly one side broke, no both-sides chop) → "
               f"**{n_hi:,}** only-HIGH-broke days (bull) · **{n_lo:,}** only-LOW-broke "
               f"days (bear), of {len(sub):,} days in this slice.")

    e = st.columns(4)
    e[0].metric(f"Bull ext ≥ {bull_ext:.2f}×",
                pct((up_ext >= bull_ext).mean()) if len(up_ext) else "—",
                f"of {n_hi:,} only-HIGH-broke days",
                help="Given ONLY the high broke (single-break upside day), "
                     "P(price reaches range_high + level×range).")
    e[1].metric(f"Bear ext ≥ {bear_ext:.2f}×",
                pct((dn_ext >= bear_ext).mean()) if len(dn_ext) else "—",
                f"of {n_lo:,} only-LOW-broke days",
                help="Given ONLY the low broke (single-break downside day), "
                     "P(price reaches range_low − level×range).")
    e[2].metric(f"Bull retr ≥ {bull_retr:.2f}×",
                pct((up_rt >= bull_retr).mean()) if len(up_rt) else "—",
                f"of {len(up_rt):,} only-HIGH-broke days",
                help="Given ONLY the high broke, P(pullback ≥ level×range back "
                     "below the range high).")
    e[3].metric(f"Bear retr ≥ {bear_retr:.2f}×",
                pct((dn_rt >= bear_retr).mean()) if len(dn_rt) else "—",
                f"of {len(dn_rt):,} only-LOW-broke days",
                help="Given ONLY the low broke, P(bounce ≥ level×range back "
                     "above the range low).")

    # extension probability curve
    levels = np.round(np.arange(0.1, 2.01, 0.1), 2)
    bull_curve = [(up_ext >= L).mean()*100 if len(up_ext) else 0 for L in levels]
    bear_curve = [(dn_ext >= L).mean()*100 if len(dn_ext) else 0 for L in levels]
    fig_e = go.Figure()
    fig_e.add_scatter(x=levels, y=bull_curve, mode="lines+markers",
                      name=f"Bull ext · only HIGH broke (n={n_hi:,})",
                      line=dict(color="#4CAF50", width=2))
    fig_e.add_scatter(x=levels, y=bear_curve, mode="lines+markers",
                      name=f"Bear ext · only LOW broke (n={n_lo:,})",
                      line=dict(color="#F44336", width=2))
    fig_e.add_vline(x=bull_ext, line_dash="dot", line_color="#4CAF50")
    fig_e.add_vline(x=bear_ext, line_dash="dot", line_color="#F44336")
    fig_e.update_layout(title="P(reach extension ≥ x) given that side broke first",
                        xaxis_title="extension (× range)", yaxis_title="% of days",
                        height=340, template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_e, use_container_width=True)

    # ── PREVIOUS-DAY LEVELS (PDH / PDL) ───────────────────────────────────────
    st.divider()
    st.markdown("#### Previous-Day High / Low  (PDH / PDL)")
    st.caption("Same-session INTRADAY touches only — PDH/PDL = the PREVIOUS session's "
               "high/low, measured until this session's close. TOUCH semantics: price "
               "must actually TRADE AT the level that day — a day that opens beyond "
               "PDH/PDL only counts after pulling back to touch it. Nothing carries "
               "beyond the session.")

    # headline: PD-touch odds for the CURRENT slice (ALL sidebar filters applied)
    _pd_e = (sub["broke_pdh"] | sub["broke_pdl"])
    hp = st.columns(4)
    hp[0].metric("Touch PDH", pct(sub["broke_pdh"].mean()))
    hp[1].metric("Touch PDL", pct(sub["broke_pdl"].mean()))
    hp[2].metric("Touch EITHER", pct(_pd_e.mean()),
                 help="PDH or PDL (or both) touched within the same session.")
    hp[3].metric("Touch NEITHER", pct(1 - _pd_e.mean()))
    st.caption(f"Current slice ({len(sub):,} days — every sidebar filter applied). "
               "The three setup cards below respond to date & day-of-week only.")

    pc = st.columns(3)
    # inside-day breakout: yesterday was an inside day
    ins = pdset[pdset["prev_inside"] == True]
    with pc[0]:
        st.markdown("**Inside-day breakout** (prev day was inside)")
        if len(ins):
            st.metric("Touch PDH", pct(ins["broke_pdh"].mean()),
                      f"touch PDL: {pct(ins['broke_pdl'].mean())}")
            st.caption(f"EITHER: {pct((ins['broke_pdh'] | ins['broke_pdl']).mean())} · "
                       f"both: {pct(ins['broke_pd_both'].mean())} · "
                       f"neither: {pct(ins['broke_pd_none'].mean())} · n={len(ins):,}")
        else:
            st.info("No inside-day setups in this slice.")
    # gap up → touch PDH
    gu = pdset[pdset["gap_type"] == "Gap Up"]
    with pc[1]:
        st.markdown("**Gap Up → touch PDH**")
        if len(gu):
            st.metric("Touch PDH", pct(gu["broke_pdh"].mean()), f"n={len(gu):,}",
                      help="Price trades AT the level the same session (touch). "
                           "A day opening above PDH only counts after pulling back.")
            above = gu[gu["day_open"] > gu["pdh"]]
            below = gu[gu["day_open"] <= gu["pdh"]]
            st.caption(
                f"opens ABOVE PDH: {pct(len(above)/len(gu))}"
                + (f" → pulls back to touch: {pct(above['broke_pdh'].mean())}" if len(above) else "")
                + (f" · opens below → rallies to touch: {pct(below['broke_pdh'].mean())}" if len(below) else ""))
        else:
            st.info("No Gap Up days in this slice.")
    # gap down → touch PDL
    gd = pdset[pdset["gap_type"] == "Gap Down"]
    with pc[2]:
        st.markdown("**Gap Down → touch PDL**")
        if len(gd):
            st.metric("Touch PDL", pct(gd["broke_pdl"].mean()), f"n={len(gd):,}",
                      help="Price trades AT the level the same session (touch). "
                           "A day opening below PDL only counts after bouncing back.")
            belowL = gd[gd["day_open"] < gd["pdl"]]
            aboveL = gd[gd["day_open"] >= gd["pdl"]]
            st.caption(
                f"opens BELOW PDL: {pct(len(belowL)/len(gd))}"
                + (f" → bounces to touch: {pct(belowL['broke_pdl'].mean())}" if len(belowL) else "")
                + (f" · opens above → drops to touch: {pct(aboveL['broke_pdl'].mean())}" if len(aboveL) else ""))
        else:
            st.info("No Gap Down days in this slice.")

    with st.expander("PDH / PDL reach rates by gap type"):
        st.dataframe(gap_pd_table(pdset), use_container_width=True, hide_index=True)

    # ── break stats by gap % bucket ───────────────────────────────────────────
    st.divider()
    st.markdown("#### Break stats by gap % bucket")
    st.caption("Every bucket in the current slice (before the gap-bucket filter) — "
               "spot which gap sizes carry the edge at a glance.")
    bkt_rows = []
    for bkt in GAP_BUCKETS:
        x = sub_all_buckets[sub_all_buckets["gap_bucket"] == bkt]
        if len(x) == 0:
            continue
        bkt_rows.append({
            "Gap bucket": bkt, "days": len(x),
            "HIGH breaks %":  round(x[f"{tag}_high_break"].mean() * 100, 1),
            "LOW breaks %":   round(x[f"{tag}_low_break"].mean() * 100, 1),
            "BOTH %":         round(x[f"{tag}_both_break"].mean() * 100, 1),
            "NEITHER %":      round(x[f"{tag}_no_break"].mean() * 100, 1),
            "reach PDH %":    round(x["broke_pdh"].mean() * 100, 1),
            "reach PDL %":    round(x["broke_pdl"].mean() * 100, 1),
            "PD either %":    round((x["broke_pdh"] | x["broke_pdl"]).mean() * 100, 1),
        })
    if bkt_rows:
        st.dataframe(pd.DataFrame(bkt_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No days with a measurable gap in this slice.")

    # ── charts: break probs + by day of week ──────────────────────────────────
    st.divider()
    g1, g2 = st.columns(2)
    with g1:
        labels = ["HIGH", "LOW", "BOTH", "ONE", "NEITHER"]
        fig = go.Figure()
        fig.add_bar(name="Filtered", x=labels,
                    y=[s["high"]*100, s["low"]*100, s["both"]*100, s["one"]*100, s["none"]*100],
                    marker_color="#2196F3",
                    text=[f"{v*100:.0f}%" for v in (s["high"], s["low"], s["both"], s["one"], s["none"])],
                    textposition="outside")
        fig.add_bar(name="Baseline", x=labels,
                    y=[b["high"]*100, b["low"]*100, b["both"]*100, b["one"]*100, b["none"]*100],
                    marker_color="#B0BEC5")
        fig.update_layout(title="Break probabilities vs baseline", barmode="group",
                          height=360, template="plotly_dark", yaxis_title="%",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        present = [d for d in DOW_ORDER if d in sub["dow"].unique()]
        dow_tbl = (sub.groupby("dow")[[f"{tag}_high_break", f"{tag}_low_break"]]
                   .mean().reindex(present))
        fig2 = go.Figure()
        fig2.add_bar(name="HIGH breaks", x=dow_tbl.index, y=dow_tbl[f"{tag}_high_break"]*100,
                     marker_color="#4CAF50")
        fig2.add_bar(name="LOW breaks", x=dow_tbl.index, y=dow_tbl[f"{tag}_low_break"]*100,
                     marker_color="#F44336")
        fig2.update_layout(title="Break rate by day of week (current slice)", barmode="group",
                           height=360, template="plotly_dark", yaxis_title="%",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig2, use_container_width=True)

    # ── slice table + download ────────────────────────────────────────────────
    show_cols = ["date", "dow", "gap_type", "gap_bucket", "day_kind", f"{tag}_range",
                 f"{tag}_first_side", f"{tag}_high_break", f"{tag}_low_break",
                 f"{tag}_up_ext", f"{tag}_dn_ext", "broke_pdh", "broke_pdl"]
    with st.expander(f"Matching days ({s['n']:,})"):
        st.dataframe(sub[show_cols].sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)
    st.download_button("⬇ Download matching days (CSV)",
                       sub[show_cols].to_csv(index=False).encode(),
                       file_name="prob_slice.csv", mime="text/csv")

    # ── day browser: chart every matching day with its levels ────────────────
    _render_day_browser(sub, inst, session, tag, ib_min)


# ─── day browser ──────────────────────────────────────────────────────────────

def _fmt_x(v, unit="×"):
    return "—" if pd.isna(v) else f"{v:.2f}{unit}"


def _render_day_browser(sub, inst, session, tag, ib_min):
    """Step through every matching day with a clean candlestick chart + levels."""
    st.divider()
    st.markdown("#### 📅 Day browser — every matching day, charted with levels")
    if sub.empty:
        return

    srt = sub.sort_values("date")
    days = srt["date"].dt.date.tolist()
    by_day = srt.set_index(srt["date"].dt.date)
    day_dow = dict(zip(days, srt["dow"]))

    path = data_store.discover().get(inst)
    if not path or not os.path.exists(path):
        st.info("Minute data for this instrument is not available — charts disabled.")
        return
    mdf, auto_open = _minute_and_open(inst, os.path.getmtime(path))

    if data_store.has_sessions(inst):
        sdef = data_store.SESSIONS_24H[session]
        o, c = sdef["open_t"], sdef["close_t"]
    else:
        # canonical times only (same resolution as build_facts.build_instrument)
        d = data_store.SESSION_DEFAULTS.get(inst, {})
        o = d.get("open_t", auto_open)
        c = d.get("close_t", build_facts.DEFAULT_CLOSE_T)
    cross = c < o                                   # session crosses midnight

    key = f"brw_{inst}_{session}_{tag}"
    if st.session_state.get(f"{key}_day") not in days:
        st.session_state[f"{key}_day"] = days[-1]

    nav = st.columns([1, 1, 4, 2])
    cur = days.index(st.session_state[f"{key}_day"])
    # buttons render BEFORE the selectbox, so writing its state here is legal
    if nav[0].button("◀ Prev day", key=f"{key}_prev", disabled=(cur == 0),
                     use_container_width=True):
        st.session_state[f"{key}_day"] = days[max(cur - 1, 0)]
    cur = days.index(st.session_state[f"{key}_day"])
    if nav[1].button("Next day ▶", key=f"{key}_next", disabled=(cur == len(days) - 1),
                     use_container_width=True):
        st.session_state[f"{key}_day"] = days[min(cur + 1, len(days) - 1)]
    nav[2].selectbox("Day", days, key=f"{key}_day",
                     format_func=lambda d: f"{d}  ·  {day_dow.get(d, '')}",
                     label_visibility="collapsed")
    tf = nav[3].selectbox("Chart timeframe", [1, 5, 15], index=1,
                          format_func=lambda x: f"{x}-min candles",
                          key=f"{key}_tf", label_visibility="collapsed")

    day = st.session_state[f"{key}_day"]
    r = by_day.loc[day]
    if isinstance(r, pd.DataFrame):
        r = r.iloc[0]
    cur = days.index(day)

    d0 = pd.Timestamp(day)
    start = (d0 - pd.Timedelta(days=1) if cross else d0) + pd.Timedelta(minutes=o)
    end = d0 + pd.Timedelta(minutes=c)
    g = mdf[(mdf["date"] >= start) & (mdf["date"] <= end)]
    if g.empty:
        st.warning("No minute data for this session.")
        return
    if tf > 1:
        g = build_facts.to_timeframe(g, tf, open_t=o)

    hi, lo = r[f"{tag}_high"], r[f"{tag}_low"]
    rng = r[f"{tag}_range"]
    fig = go.Figure(go.Candlestick(
        x=g["date"], open=g["open"], high=g["high"], low=g["low"], close=g["close"],
        increasing_line_color="#25F08A", decreasing_line_color="#FF5C5C",
        increasing_fillcolor="rgba(37,240,138,0.9)",
        decreasing_fillcolor="rgba(255,92,92,0.9)",
        line=dict(width=1), name=""))

    # levels — clipped to a sane band around the day's range so far-away levels
    # (e.g. PDH on a big gap day) don't stretch the chart
    lim_lo = float(g["low"].min()) - 1.0 * rng
    lim_hi = float(g["high"].max()) + 1.0 * rng
    win_lab = f"{tag.upper()}"
    lines = [
        (hi,            "#25F08A", "solid", f"{win_lab} High"),
        (lo,            "#FF5C5C", "solid", f"{win_lab} Low"),
        ((hi + lo) / 2, "#8b93a7", "dot",   "Mid"),
        (hi + 0.5 * rng, "#1db06b", "dot",  "+0.5×"),
        (hi + 1.0 * rng, "#1db06b", "dash", "+1.0×"),
        (lo - 0.5 * rng, "#c74848", "dot",  "−0.5×"),
        (lo - 1.0 * rng, "#c74848", "dash", "−1.0×"),
    ]
    if pd.notna(r.get("pdh")):
        lines += [(r["pdh"], "#FFB74D", "dashdot", "PDH"),
                  (r["pdl"], "#BA68C8", "dashdot", "PDL")]
    for y, colr, dash, txt in lines:
        if pd.isna(y) or not (lim_lo <= y <= lim_hi):
            continue
        fig.add_hline(y=y, line_color=colr, line_width=1, line_dash=dash,
                      annotation_text=txt, annotation_position="right",
                      annotation_font=dict(color=colr, size=11))

    win_min = ib_min if tag == "ib" else 15
    fig.add_vrect(x0=start, x1=start + pd.Timedelta(minutes=win_min),
                  fillcolor="rgba(37,240,138,0.05)", line_width=0)

    sess_bit = f" · {session} session" if data_store.has_sessions(inst) else ""
    fig.update_layout(
        title=f"{inst}{sess_bit} · {day} ({r['dow']})",
        height=560, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False, showlegend=False,
        margin=dict(t=48, b=10),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
    st.plotly_chart(fig, use_container_width=True)

    gp = r.get("gap_pct")
    st.caption(
        f"Day **{cur + 1} of {len(days)}** matching days"
        + (f" · gap {gp:+.2f}% ({r.get('gap_bucket', '—')})" if pd.notna(gp) else "")
        + f" · formed first: **{str(r[f'{tag}_first_side']).upper()}**"
        + f" · broke first: **{str(r[f'{tag}_break_first']).upper()}**"
        + f" · up-ext {_fmt_x(r[f'{tag}_up_ext'])} · dn-ext {_fmt_x(r[f'{tag}_dn_ext'])}"
        + f" · PDH {'touched' if r.get('broke_pdh') else 'not touched'}"
        + f" · PDL {'touched' if r.get('broke_pdl') else 'not touched'}")


def _standalone():
    st.set_page_config(page_title="ORB/IB Probability Explorer", page_icon="🎲", layout="wide")
    import ui_theme
    ui_theme.apply()
    render()


if __name__ == "__main__":
    _standalone()
