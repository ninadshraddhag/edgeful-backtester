"""
option_mode.py — "🎯 NIFTY Option Strategy" mode UI.

Three legs (T1 Trend-Ride, GF Gap-Fade, ORB), a global BUY-ATM / SELL-0.8Δ
variant toggle, per-leg optimizer, trade browser (spot chart + premium curve),
and a portfolio view (equity, correlation, monthly heatmap, green months).
Reuses option_strategy (engine) + options_pricing. Local-first mode.
"""
import copy
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import build_facts
import data_store
import option_strategy as ostr
import options_pricing as op

HERE_PATHS = {"NIFTY 50": None, "BANK NIFTY": None}
LEG_NAMES = {"T1": "T1 · Trend-Ride", "GF": "GF · Gap-Fade", "ORB": "ORB · IB breakout"}


@st.cache_data(show_spinner=True, max_entries=2)
def _prep(instrument, mpath, mtime):
    mdf = build_facts.clean_min(data_store.read_minute(mpath))
    return ostr.prep_days(mdf)


def _cfg_from_state() -> dict:
    s = st.session_state
    cfg = copy.deepcopy(ostr.DEFAULTS)
    cfg["iv"] = s["opt_iv"] / 100.0
    cfg["lots"] = int(s["opt_lots"])
    cfg["variant"] = "sell" if s["opt_variant"].startswith("Sell") else "buy"
    cfg["sell_delta"] = s["opt_selldelta"]
    cfg["slippage_pts"] = s["opt_slip"]
    cfg["cost_mult"] = 1.5 if s["opt_stress_cost"] else 1.0
    cfg["slip_mult"] = 2.0 if s["opt_stress_slip"] else 1.0
    for leg in ostr.LEGS:
        cfg[leg]["on"] = s[f"opt_{leg}_on"]
    cfg["T1"].update(ema_gap_pct=s["opt_T1_ema"], stop_pct=s["opt_T1_stop"],
                     trail_pct=s["opt_T1_trail"], cutoff=s["opt_T1_cut"] * 60,
                     use_vwap=s["opt_T1_vwap"])
    cfg["GF"].update(gap_min=s["opt_GF_gmin"], gap_max=s["opt_GF_gmax"],
                     target_buf_pct=s["opt_GF_buf"], stop_pct=s["opt_GF_stop"])
    cfg["ORB"].update(stop_cap_pct=s["opt_ORB_cap"], trail_pct=s["opt_ORB_trail"],
                      cutoff=s["opt_ORB_cut"] * 60, use_vwap=s["opt_ORB_vwap"])
    return cfg


def _metric_cards(m, label):
    st.markdown(f"**{label}**")
    c = st.columns(8)
    c[0].metric("Trades", f"{m['n']:,}")
    c[1].metric("Expectancy", f"₹{m['exp']:,.0f}")
    c[2].metric("Win %", f"{m['win']:.1f}%")
    c[3].metric("Profit factor", f"{m['pf']:.2f}" if np.isfinite(m["pf"]) else "∞")
    c[4].metric("Green months", f"{m['green_pct']:.0f}%", help=f"{m['green']} months green")
    c[5].metric("Max red streak", f"{m['max_red_streak']} mo",
                help="Longest run of consecutive losing months — the worst "
                     "drought you'd have had to sit through.")
    c[6].metric("Daily Sharpe", f"{m['sharpe']:.2f}")
    c[7].metric("Max DD", f"₹{m['max_dd']:,.0f}")


def _monthly_heatmap(trades, title):
    t = trades.copy()
    t["y"] = pd.to_datetime(t["date"]).dt.year
    t["m"] = pd.to_datetime(t["date"]).dt.month
    piv = t.pivot_table(index="y", columns="m", values="pnl", aggfunc="sum").reindex(
        columns=range(1, 13))
    z = piv.values
    mx = np.nanmax(np.abs(z)) or 1
    fig = go.Figure(go.Heatmap(
        z=z, x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                "Oct", "Nov", "Dec"], y=[str(i) for i in piv.index],
        colorscale=[[0, "#c0392b"], [0.5, "#1c2330"], [1, "#25F08A"]],
        zmid=0, zmin=-mx, zmax=mx,
        text=np.where(np.isnan(z), "", np.round(z / 1000, 1)),
        texttemplate="%{text}", textfont={"size": 10},
        hovertemplate="%{y} %{x}: ₹%{z:,.0f}<extra></extra>"))
    fig.update_layout(title=f"{title} — monthly P&L (₹'000)", height=max(220, 34 * len(piv) + 90),
                      template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=44, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _equity(port, legs):
    fig = go.Figure()
    colors = {"T1": "#42A5F5", "GF": "#FFB74D", "ORB": "#AB47BC", "ALL": "#25F08A"}
    for leg in legs + (["ALL"] if len(legs) > 1 else []):
        t = port.get(leg)
        if t is None or t.empty:
            continue
        d = t.groupby(pd.to_datetime(t["date"]).dt.normalize())["pnl"].sum().cumsum()
        fig.add_scatter(x=d.index, y=d.values, mode="lines", name=leg,
                        line=dict(color=colors.get(leg), width=2.4 if leg == "ALL" else 1.4))
    fig.update_layout(title="Equity curve (₹, 1 lot/leg)", height=360, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)


def _trade_browser(days, port, cfg):
    st.markdown("#### 🔍 Trade browser")
    legs_present = [l for l in ostr.LEGS if l in port and not port[l].empty]
    if not legs_present:
        return
    cc = st.columns([1, 3])
    leg = cc[0].selectbox("Leg", legs_present, key="opt_tb_leg")
    tdf = port[leg].sort_values("date").reset_index(drop=True)
    labels = [f"{r.date.date()} · {r.dir} · {r.opt} · ₹{r.pnl:,.0f} ({r.reason})"
              for r in tdf.itertuples()]
    i = cc[1].selectbox("Trade", range(len(tdf)),
                        format_func=lambda k: labels[k], key="opt_tb_i")
    tr = tdf.iloc[i]
    rec = ostr.find_day(days, tr["date"])
    if rec is None:
        st.info("Minute data unavailable for this day.")
        return

    # spot candles for the session + markers
    ps = ostr.premium_series(rec, tr, cfg)
    tt = rec["t"]
    fig = go.Figure(go.Scatter(x=tt, y=rec["cl"], mode="lines",
                               line=dict(color="#8b93a7", width=1.2), name="spot"))
    for y, colr, lab in [(rec["PDH"], "#FFB74D", "PDH"), (rec["PDL"], "#BA68C8", "PDL"),
                         (rec["PDC"], "#607D8B", "PDC"), (rec["ib_hi"], "#26A69A", "IB-H"),
                         (rec["ib_lo"], "#EF5350", "IB-L")]:
        fig.add_hline(y=y, line_color=colr, line_width=1, line_dash="dot",
                      annotation_text=lab, annotation_font=dict(color=colr, size=10))
    fig.add_scatter(x=[tr["entry_t"]], y=[tr["entry_spot"]], mode="markers",
                    marker=dict(color="#25F08A", size=13, symbol="triangle-up"), name="entry")
    fig.add_scatter(x=[tr["exit_t"]], y=[tr["exit_spot"]], mode="markers",
                    marker=dict(color="#F23645", size=13, symbol="x"), name="exit")
    fig.update_layout(title=f"{tr['leg']} {tr['dir']} · {tr['date'].date()} · spot",
                      height=340, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", xaxis_title="minute of day",
                      margin=dict(t=40, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    if not ps.empty:
        f2 = go.Figure(go.Scatter(x=ps["t_min"], y=ps["premium"], mode="lines",
                                  line=dict(color="#25F08A", width=2), name="premium"))
        f2.update_layout(title=f"Option premium — {tr['opt']} {tr['strike']:.0f} "
                               f"(entry ₹{tr['entry_prem']}, exit ₹{tr['exit_prem']}, "
                               f"P&L ₹{tr['pnl']:,.0f})",
                         height=230, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10))
        st.plotly_chart(f2, use_container_width=True)


def render():
    st.title("🎯 NIFTY Option Strategy")
    st.caption("Three diversified intraday legs on synthetic ATM options · "
               "long-ATM (buy) or short-0.8Δ (sell) · 1 lot · flat 15:15 · "
               "stops in % of PDC. Synthetic Black-Scholes premiums (flat IV) — "
               "not live option data; numbers are a faithful model, not a promise.")

    # ── sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Option strategy")
        insts = [n for n in ("NIFTY 50", "BANK NIFTY") if n in data_store.discover()]
        instrument = st.selectbox("Instrument", insts, key="opt_inst")
        mpath = data_store.discover()[instrument]
        mtime = os.path.getmtime(mpath)

        st.session_state.setdefault("opt_variant", "Buy ATM (0.5Δ)")
        st.radio("Position style", ["Buy ATM (0.5Δ)", "Sell 0.8Δ (opposite)"],
                 key="opt_variant",
                 help="Buy = long the ATM option in the trade direction (long gamma). "
                      "Sell = short a 0.8-delta OPPOSITE option (bullish→short PUT), "
                      "collecting premium (short gamma, +theta).")
        st.slider("Flat IV %", 6.0, 30.0, 13.0, 0.5, key="opt_iv")
        st.number_input("Sell delta", 0.2, 0.9, 0.8, 0.05, key="opt_selldelta",
                        help="Target |delta| of the option sold in the Sell variant.")
        st.number_input("Lots / leg", 1, 50, 1, key="opt_lots")

        with st.expander("💸 Costs & stress"):
            st.number_input("Slippage (premium pts / side)", 0.0, 5.0, 0.5, 0.25,
                            key="opt_slip")
            st.toggle("Stress: costs ×1.5", key="opt_stress_cost")
            st.toggle("Stress: slippage ×2", key="opt_stress_slip")

        split = st.selectbox("Period", list(ostr.SPLITS.keys()), index=1, key="opt_split")
        if "Test" in split:
            st.warning("🔒 TEST 2025–26 is your reserved hold-out. Every look "
                       "burns it. Use only for a final one-shot check.")

    # ── leg config ─────────────────────────────────────────────────────────────
    st.subheader("Legs")
    defs = ostr.DEFAULTS
    with st.expander(f"⚙️ {LEG_NAMES['T1']}", expanded=False):
        st.toggle("Trade this leg", value=defs["T1"]["on"], key="opt_T1_on")
        r = st.columns(5)
        r[0].number_input("EMA gap %", 0.0, 0.5, defs["T1"]["ema_gap_pct"], 0.01, key="opt_T1_ema")
        r[1].number_input("Stop % PDC", 0.1, 2.0, defs["T1"]["stop_pct"], 0.1, key="opt_T1_stop")
        r[2].number_input("Trail % PDC", 0.3, 3.0, defs["T1"]["trail_pct"], 0.1, key="opt_T1_trail")
        r[3].number_input("Cutoff hr", 10, 15, defs["T1"]["cutoff"] // 60, key="opt_T1_cut")
        r[4].toggle("VWAP filter", value=True, key="opt_T1_vwap")
        st.caption("First 3-min close beyond PDH/PDL before cutoff · EMA9−EMA21 gate · "
                   "initial stop then trail · no target.")
    with st.expander(f"⚙️ {LEG_NAMES['GF']}", expanded=False):
        st.toggle("Trade this leg", value=defs["GF"]["on"], key="opt_GF_on")
        r = st.columns(4)
        r[0].number_input("Gap min %", 0.1, 1.0, defs["GF"]["gap_min"], 0.05, key="opt_GF_gmin")
        r[1].number_input("Gap max %", 0.3, 2.0, defs["GF"]["gap_max"], 0.05, key="opt_GF_gmax")
        r[2].number_input("Target buffer % PDC", 0.0, 0.3, defs["GF"]["target_buf_pct"], 0.01,
                          key="opt_GF_buf")
        r[3].number_input("Stop % PDC", 0.3, 2.0, defs["GF"]["stop_pct"], 0.1, key="opt_GF_stop")
        st.caption("Gap in band · enter 09:21 · fade toward PDC±buffer · stop % of PDC.")
    with st.expander(f"⚙️ {LEG_NAMES['ORB']}", expanded=False):
        st.toggle("Trade this leg", value=defs["ORB"]["on"], key="opt_ORB_on")
        r = st.columns(4)
        r[0].number_input("Stop cap % PDC", 0.3, 2.0, defs["ORB"]["stop_cap_pct"], 0.1, key="opt_ORB_cap")
        r[1].number_input("Trail % PDC", 0.3, 3.0, defs["ORB"]["trail_pct"], 0.1, key="opt_ORB_trail")
        r[2].number_input("Cutoff hr", 11, 15, defs["ORB"]["cutoff"] // 60, key="opt_ORB_cut")
        r[3].toggle("VWAP filter", value=True, key="opt_ORB_vwap")
        st.caption("First 3-min close beyond the 60-min IB before cutoff · stop = "
                   "opposite IB side capped at stop-cap · trail.")

    with st.spinner(f"Indexing {instrument} (first run cached)…"):
        days_all = _prep(instrument, mpath, mtime)
    d0, d1 = ostr.SPLITS[st.session_state["opt_split"]]
    days = ostr.filter_days(days_all, d0, d1)
    if not days:
        st.warning("No trading days in this period.")
        return

    run = st.button("▶ Run option strategy", type="primary")
    if run:
        import credits
        if credits.try_charge("daywise_run"):
            cfg = _cfg_from_state()
            st.session_state["opt_res"] = (ostr.run_portfolio(days, cfg), cfg,
                                           st.session_state["opt_split"])

    res = st.session_state.get("opt_res")
    if res:
        port, cfg, split_name = res
        legs = [l for l in ostr.LEGS if l in port and not port[l].empty]
        if not legs:
            st.warning("No trades — check leg toggles and gap/EMA gates.")
            return
        variant_lbl = "SELL 0.8Δ" if cfg["variant"] == "sell" else "BUY ATM"
        st.divider()
        st.subheader(f"{instrument} · {split_name} · {variant_lbl} · {len(legs)} leg(s)")

        _metric_cards(ostr.metrics(port["ALL"]), "🎯 Portfolio (all enabled legs)")
        cols = st.columns(len(legs))
        for c, leg in zip(cols, legs):
            with c:
                m = ostr.metrics(port[leg])
                st.metric(LEG_NAMES[leg], f"₹{m['exp']:,.0f}/tr",
                          f"PF {m['pf']:.2f} · {m['green_pct']:.0f}% green · n={m['n']}")

        _equity(port, legs)
        _monthly_heatmap(port["ALL"], "Portfolio")

        dc, mc = ostr.correlations(port)
        if dc is not None:
            st.markdown("#### Leg correlation (P&L streams — the diversification check)")
            g = st.columns(2)
            g[0].caption("Daily"); g[0].dataframe(dc, use_container_width=True)
            g[1].caption("Monthly"); g[1].dataframe(mc, use_container_width=True)
            st.caption("Near-zero cross-correlation = genuine diversification. T1 and "
                       "ORB are breakout cousins (correlated); GF is independent.")

        _trade_browser(days, port, cfg)

        with st.expander("📄 All trades"):
            st.dataframe(port["ALL"].sort_values("date", ascending=False),
                         use_container_width=True, hide_index=True)
            st.download_button("⬇ Download trades (CSV)",
                               port["ALL"].to_csv(index=False).encode(),
                               file_name=f"option_strategy_{instrument.replace(' ','_')}.csv",
                               mime="text/csv")

    # ── per-leg optimizer ──────────────────────────────────────────────────────
    st.divider()
    with st.expander("🔬 Per-leg optimizer — sweep one leg's parameters"):
        is_sell = st.session_state.get("opt_variant", "").startswith("Sell")
        oc = st.columns(6 if is_sell else 5)
        oleg = oc[0].selectbox("Leg", ostr.LEGS, format_func=lambda l: LEG_NAMES[l],
                               key="opt_opt_leg")
        orank = oc[1].selectbox("Rank by", ["exp", "pf", "green_months", "sharpe", "net"],
                                key="opt_opt_rank")
        omin = oc[2].number_input("Min trades", 20, 3000, 100, 10, key="opt_opt_min")
        ogreen = oc[3].number_input("Min green months %", 0, 100, 0, 5, key="opt_opt_green")
        sweep_delta = False
        if is_sell:
            sweep_delta = oc[4].toggle("Sweep sell Δ", key="opt_opt_swd",
                                       help="Also sweep the sold option's delta over "
                                            "0.2–0.8 to find whether a lower (OTM) delta "
                                            "makes premium-selling profitable.")
        run_col = oc[5] if is_sell else oc[4]
        st.caption(f"Sweeps {LEG_NAMES[oleg]} over {oleg}'s grid on the selected period "
                   f"({split_name if res else st.session_state['opt_split']}), using the "
                   "current Buy/Sell + cost settings."
                   + (" · **Sell Δ swept 0.2→0.8**" if sweep_delta else ""))
        if run_col.button("🔎 Optimize", key="opt_opt_run"):
            import credits
            if credits.try_charge("ib50_optimizer"):
                cfg = _cfg_from_state()
                prog = st.progress(0.0, text="Sweeping…")
                rdf = ostr.optimize_leg(days, cfg, oleg, rank=orank,
                                        min_trades=int(omin), min_green=int(ogreen),
                                        sweep_delta=sweep_delta,
                                        progress=lambda k, n: prog.progress(k / n,
                                                 text=f"{k}/{n} configs…"))
                prog.empty()
                st.session_state["opt_opt_res"] = (oleg, rdf)
        r = st.session_state.get("opt_opt_res")
        if r and not r[1].empty:
            st.markdown(f"**Top configs — {LEG_NAMES[r[0]]}** (all shown, so you can "
                        "check for parameter cliffs)")
            st.dataframe(r[1], use_container_width=True, hide_index=True)
        elif r:
            st.warning("No config met the min-trades / min-green thresholds.")

    st.caption("⚠️ Educational research only. Synthetic flat-IV premiums exclude "
               "event-day IV crush and spread blow-ups that hit exactly these days. "
               "Paper-trade 3–6 months before any capital.")
