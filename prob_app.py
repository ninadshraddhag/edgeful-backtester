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

# NOTE: set_page_config is NOT called at import time so this module can be embedded
# as a mode inside app.py. It is only set when run standalone (see _standalone()).

HERE   = os.path.dirname(os.path.abspath(__file__))
FACTS  = os.path.join(HERE, "analysis", "facts.csv")
DOW_ORDER  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
GAP_ORDER  = ["Gap Up", "Flat", "Gap Down"]
KIND_ORDER = ["Inside Day", "Normal", "Outside Day"]


# ─── data ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_facts(mtime):           # mtime arg busts the cache when facts.csv changes
    df = pd.read_csv(FACTS, parse_dates=["date"])
    df["day_kind"] = np.where(df["inside_day"], "Inside Day",
                     np.where(df["outside_day"], "Outside Day", "Normal"))
    return df


@st.cache_data(show_spinner=True)
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


def build_facts_inline():
    import build_facts
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
            "both %":      round(x["broke_pd_both"].mean()*100, 1),
            "neither %":   round(x["broke_pd_none"].mean()*100, 1),
        })
    return pd.DataFrame(rows)


# ─── app ──────────────────────────────────────────────────────────────────────

def render():
    st.title("🎲 ORB / IB Probability Explorer")
    st.caption("All loaded instruments (NIFTY 50 · BANK NIFTY · NQ · XAUUSD) · "
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
        st.radio("Setup", ["IB", "ORB (15 min)"], key="f_setup")
        st.number_input("IB duration (min)", 10, 240, step=10, key="f_ibmin",
                        help="Length of the Initial Balance window from the open, in "
                             "10-min steps (e.g. 40, 50). Non-60 values recompute the "
                             "stats live (cached).")
        tag  = "ib" if st.session_state["f_setup"].startswith("IB") else "orb"
        inst = st.session_state["f_instrument"]

        st.divider()
        dmin = df["date"].min().date()
        dmax = df["date"].max().date()
        date_sel = st.date_input("Date range", value=(dmin, dmax),
                                 min_value=dmin, max_value=dmax, key="f_dates")

        st.multiselect("Day of week", DOW_ORDER, key="f_dow")
        st.multiselect("Gap type", GAP_ORDER, key="f_gap")
        # precise gap-% band — drag to a specific gap, e.g. +0.2% to +0.8%
        _gp = df.loc[df["instrument"] == inst, "gap_pct"].dropna()
        g_min = float(np.clip(np.floor(_gp.min() * 10) / 10, -10.0, 0.0)) if len(_gp) else -5.0
        g_max = float(np.clip(np.ceil(_gp.max() * 10) / 10, 0.0, 10.0)) if len(_gp) else 5.0
        gap_rng = st.slider("Gap % range  (up +, down −)", g_min, g_max, (g_min, g_max),
                            0.05, key=f"f_gap_pct_{inst}",
                            help="Filter days by the exact opening-gap %. Drag the "
                                 "handles to a band — e.g. +0.2% to +0.8% for a specific "
                                 "gap-up day, or −0.5% to −0.1% for a small gap down.")
        st.multiselect("Day kind", KIND_ORDER, key="f_kind")
        st.radio("Which extreme formed first",
                 ["Either", "High formed first", "Low formed first"], key="f_first")

        st.divider()
        base_inst = df[df["instrument"] == inst]
        size_col = f"{tag}_range"
        # IB/ORB size as a % of price (volatility-normalized, comparable across
        # instruments) instead of raw points
        size_pct_all = (base_inst[size_col] / base_inst["day_open"] * 100).dropna()
        s_lo = float(np.floor(size_pct_all.min() * 100) / 100) if len(size_pct_all) else 0.0
        s_hi = float(np.ceil(size_pct_all.quantile(0.995) * 100) / 100) if len(size_pct_all) else 2.0
        size_rng = st.slider(f"{tag.upper()} size (% of price)", s_lo, s_hi, (s_lo, s_hi),
                             0.01, key=f"f_sizepct_{tag}_{inst}_{ib_min}",
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
    scope = df[df["instrument"] == inst].copy()
    if isinstance(date_sel, (tuple, list)) and len(date_sel) == 2:
        d0, d1 = pd.Timestamp(date_sel[0]), pd.Timestamp(date_sel[1]) + pd.Timedelta(days=1)
        scope = scope[(scope["date"] >= d0) & (scope["date"] < d1)]

    baseline = scope.copy()                      # date-scoped, all categories
    pdset = scope[scope["dow"].isin(st.session_state["f_dow"])].copy()   # for PD-level stats

    sub = scope[scope["dow"].isin(st.session_state["f_dow"])]
    sub = sub[sub["gap_type"].isin(st.session_state["f_gap"])]
    sub = sub[(sub["gap_pct"] >= gap_rng[0]) & (sub["gap_pct"] <= gap_rng[1])]
    sub = sub[sub["day_kind"].isin(st.session_state["f_kind"])]
    if st.session_state["f_first"] == "High formed first":
        sub = sub[sub[f"{tag}_first_side"] == "high"]
    elif st.session_state["f_first"] == "Low formed first":
        sub = sub[sub[f"{tag}_first_side"] == "low"]
    _sz_pct = sub[size_col] / sub["day_open"] * 100
    sub = sub[(_sz_pct >= size_rng[0]) & (_sz_pct <= size_rng[1])]

    # ── headline sentence ─────────────────────────────────────────────────────
    setup_label = f"IB ({ib_min} min)" if tag == "ib" else "ORB (15 min)"
    bits = [inst, setup_label]
    if isinstance(date_sel, (tuple, list)) and len(date_sel) == 2:
        bits.append(f"{date_sel[0]}→{date_sel[1]}")
    if len(st.session_state["f_dow"]) < 5:  bits.append("/".join(st.session_state["f_dow"]))
    if len(st.session_state["f_gap"]) < 3:  bits.append("/".join(st.session_state["f_gap"]))
    if gap_rng[0] > g_min or gap_rng[1] < g_max:
        bits.append(f"gap {gap_rng[0]:+.2f}%…{gap_rng[1]:+.2f}%")
    if len(st.session_state["f_kind"]) < 3: bits.append("/".join(st.session_state["f_kind"]))
    if st.session_state["f_first"] != "Either": bits.append(st.session_state["f_first"])
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
    st.markdown("#### “First-move-fades” — what breaks given which extreme formed first")
    hf = sub[sub[f"{tag}_first_side"] == "high"]
    lf = sub[sub[f"{tag}_first_side"] == "low"]
    fm = st.columns(2)
    with fm[0]:
        if len(hf):
            fm[0].metric(f"HIGH first → LOW breaks  (n={len(hf):,})",
                         pct(hf[f"{tag}_low_break"].mean()),
                         f"low breaks first: {pct((hf[f'{tag}_break_first']=='low').mean())}")
        else:
            st.info("No 'high first' days in this slice.")
    with fm[1]:
        if len(lf):
            fm[1].metric(f"LOW first → HIGH breaks  (n={len(lf):,})",
                         pct(lf[f"{tag}_high_break"].mean()),
                         f"high breaks first: {pct((lf[f'{tag}_break_first']=='high').mean())}")
        else:
            st.info("No 'low first' days in this slice.")

    # ── close vs midpoint (momentum confirmation) ─────────────────────────────
    st.markdown(f"#### {tag.upper()} close vs midpoint — confirmation of the fade")
    mid_col = f"{tag}_close_above_mid"
    if mid_col in sub.columns:
        # NB: local names must not shadow the baseline stats dict `b` used below
        lf_conf = lf[lf[mid_col] == True]      # low first AND window closed above mid
        hf_conf = hf[hf[mid_col] == False]     # high first AND window closed below mid
        mp = st.columns(2)
        with mp[0]:
            if len(lf_conf):
                base_a = pct(lf[f"{tag}_high_break"].mean()) if len(lf) else "—"
                mp[0].metric(f"LOW first + close ABOVE mid → HIGH breaks  (n={len(lf_conf):,})",
                             pct(lf_conf[f"{tag}_high_break"].mean()),
                             f"vs {base_a} for all low-first days")
            else:
                st.info("No days: low first + close above mid.")
        with mp[1]:
            if len(hf_conf):
                base_b = pct(hf[f"{tag}_low_break"].mean()) if len(hf) else "—"
                mp[1].metric(f"HIGH first + close BELOW mid → LOW breaks  (n={len(hf_conf):,})",
                             pct(hf_conf[f"{tag}_low_break"].mean()),
                             f"vs {base_b} for all high-first days")
            else:
                st.info("No days: high first + close below mid.")
        st.caption(f"Close = last 1-min close of the {tag.upper()} window; midpoint = "
                   "(window high + low) / 2. Closing back beyond the midpoint confirms "
                   "the first move has already been faded.")
    else:
        st.info("Rebuild the facts table (`python build_facts.py`) to enable this stat.")

    # ── EXTENSIONS & RETRACEMENTS ─────────────────────────────────────────────
    # Directionally conditioned so the numbers are interpretable: a BULL extension
    # is only meaningful on days the HIGH broke FIRST (a genuine upside breakout),
    # and a BEAR extension only on days the LOW broke first. Mixing both directions
    # into one pool would dilute the reading. Retracements follow the same rule —
    # a bull retracement (pullback after the high broke) is measured on high-first
    # days, a bear retracement on low-first days.
    st.divider()
    st.markdown(f"#### Extensions & Retracements  ·  measured in × {tag.upper()} range")
    hi_first = sub[sub[f"{tag}_break_first"] == "high"]
    lo_first = sub[sub[f"{tag}_break_first"] == "low"]
    n_hi, n_lo = len(hi_first), len(lo_first)
    up_ext = hi_first[f"{tag}_up_ext"].dropna()
    dn_ext = lo_first[f"{tag}_dn_ext"].dropna()
    up_rt  = hi_first[f"{tag}_up_retr"].dropna()
    dn_rt  = lo_first[f"{tag}_dn_retr"].dropna()
    st.caption(f"Conditioned on which side broke **first** → "
               f"**{n_hi:,}** high-first days (bull) · **{n_lo:,}** low-first days (bear), "
               f"of {len(sub):,} in this slice.")

    e = st.columns(4)
    e[0].metric(f"Bull ext ≥ {bull_ext:.2f}×",
                pct((up_ext >= bull_ext).mean()) if len(up_ext) else "—",
                f"of {n_hi:,} high-first days",
                help="Given the HIGH broke first, P(price reaches "
                     "range_high + level×range).")
    e[1].metric(f"Bear ext ≥ {bear_ext:.2f}×",
                pct((dn_ext >= bear_ext).mean()) if len(dn_ext) else "—",
                f"of {n_lo:,} low-first days",
                help="Given the LOW broke first, P(price reaches "
                     "range_low − level×range).")
    e[2].metric(f"Bull retr ≥ {bull_retr:.2f}×",
                pct((up_rt >= bull_retr).mean()) if len(up_rt) else "—",
                f"of {len(up_rt):,} high-first days",
                help="Given the HIGH broke first, P(pullback ≥ level×range below "
                     "the high).")
    e[3].metric(f"Bear retr ≥ {bear_retr:.2f}×",
                pct((dn_rt >= bear_retr).mean()) if len(dn_rt) else "—",
                f"of {len(dn_rt):,} low-first days",
                help="Given the LOW broke first, P(bounce ≥ level×range above "
                     "the low).")

    # extension probability curve
    levels = np.round(np.arange(0.1, 2.01, 0.1), 2)
    bull_curve = [(up_ext >= L).mean()*100 if len(up_ext) else 0 for L in levels]
    bear_curve = [(dn_ext >= L).mean()*100 if len(dn_ext) else 0 for L in levels]
    fig_e = go.Figure()
    fig_e.add_scatter(x=levels, y=bull_curve, mode="lines+markers",
                      name=f"Bull ext · high-first (n={n_hi:,})",
                      line=dict(color="#4CAF50", width=2))
    fig_e.add_scatter(x=levels, y=bear_curve, mode="lines+markers",
                      name=f"Bear ext · low-first (n={n_lo:,})",
                      line=dict(color="#F44336", width=2))
    fig_e.add_vline(x=bull_ext, line_dash="dot", line_color="#4CAF50")
    fig_e.add_vline(x=bear_ext, line_dash="dot", line_color="#F44336")
    fig_e.update_layout(title="P(reach extension ≥ x) given that side broke first",
                        xaxis_title="extension (× range)", yaxis_title="% of days",
                        height=340, template="plotly_white",
                        legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_e, use_container_width=True)

    # ── PREVIOUS-DAY LEVELS (PDH / PDL) ───────────────────────────────────────
    st.divider()
    st.markdown("#### Previous-Day High / Low  (PDH / PDL)")
    st.caption("Same-day INTRADAY touches only — measured within today's session up to the "
               "15:15 square-off (PDH/PDL = previous session's high/low). Nothing carries to "
               "the next day. Responds to date & day-of-week filters.")

    pc = st.columns(3)
    # inside-day breakout: yesterday was an inside day
    ins = pdset[pdset["prev_inside"] == True]
    with pc[0]:
        st.markdown("**Inside-day breakout** (prev day was inside)")
        if len(ins):
            st.metric("Break PDH", pct(ins["broke_pdh"].mean()),
                      f"break PDL: {pct(ins['broke_pdl'].mean())}")
            st.caption(f"both: {pct(ins['broke_pd_both'].mean())} · "
                       f"neither: {pct(ins['broke_pd_none'].mean())} · n={len(ins):,}")
        else:
            st.info("No inside-day setups in this slice.")
    # gap up → reach PDH
    gu = pdset[pdset["gap_type"] == "Gap Up"]
    with pc[1]:
        st.markdown("**Gap Up → reach PDH**")
        if len(gu):
            st.metric("Reach PDH", pct(gu["broke_pdh"].mean()), f"n={len(gu):,}")
            st.caption(f"also reach PDL same day: {pct(gu['broke_pdl'].mean())}")
        else:
            st.info("No Gap Up days in this slice.")
    # gap down → reach PDL
    gd = pdset[pdset["gap_type"] == "Gap Down"]
    with pc[2]:
        st.markdown("**Gap Down → reach PDL**")
        if len(gd):
            st.metric("Reach PDL", pct(gd["broke_pdl"].mean()), f"n={len(gd):,}")
            st.caption(f"also reach PDH same day: {pct(gd['broke_pdh'].mean())}")
        else:
            st.info("No Gap Down days in this slice.")

    with st.expander("PDH / PDL reach rates by gap type"):
        st.dataframe(gap_pd_table(pdset), use_container_width=True, hide_index=True)

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
                          height=360, template="plotly_white", yaxis_title="%",
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
                           height=360, template="plotly_white", yaxis_title="%",
                           legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig2, use_container_width=True)

    # ── slice table + download ────────────────────────────────────────────────
    show_cols = ["date", "dow", "gap_type", "day_kind", f"{tag}_range",
                 f"{tag}_first_side", f"{tag}_high_break", f"{tag}_low_break",
                 f"{tag}_up_ext", f"{tag}_dn_ext", "broke_pdh", "broke_pdl"]
    with st.expander(f"Matching days ({s['n']:,})"):
        st.dataframe(sub[show_cols].sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)
    st.download_button("⬇ Download matching days (CSV)",
                       sub[show_cols].to_csv(index=False).encode(),
                       file_name="prob_slice.csv", mime="text/csv")


def _standalone():
    st.set_page_config(page_title="ORB/IB Probability Explorer", page_icon="🎲", layout="wide")
    render()


if __name__ == "__main__":
    _standalone()
