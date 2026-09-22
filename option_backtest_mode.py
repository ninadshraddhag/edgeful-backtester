"""
option_mode.py — Streamlit UI for the real-data option-strategy backtester.

Local-only: when data_options/ is absent (e.g. Streamlit Cloud) it shows a
one-line setup hint instead of the controls.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import option_data as od
import option_engine as oe

ACCENT = "#25F08A"

# preset → list of legs; `w` = strangle/condor width in strike-steps OTM
def _presets(w: int):
    return {
        "Short Straddle (sell ATM CE+PE)":  [("CE", "sell", 0), ("PE", "sell", 0)],
        "Short Strangle (sell OTM CE+PE)":  [("CE", "sell", w), ("PE", "sell", w)],
        "Long Straddle (buy ATM CE+PE)":    [("CE", "buy", 0), ("PE", "buy", 0)],
        "Long Strangle (buy OTM CE+PE)":    [("CE", "buy", w), ("PE", "buy", w)],
        "Iron Condor (sell OTM, buy wings)": [("CE", "sell", w), ("PE", "sell", w),
                                              ("CE", "buy", 2 * w), ("PE", "buy", 2 * w)],
        "Buy CE (directional)":             [("CE", "buy", 0)],
        "Buy PE (directional)":             [("PE", "buy", 0)],
        "Sell CE (directional)":            [("CE", "sell", 0)],
        "Sell PE (directional)":            [("PE", "sell", 0)],
    }


def _hm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def render():
    st.subheader("🧩 Option Strategy Backtester  ·  real NSE premiums")
    unders = od.available()
    if not unders:
        st.info(
            "**No local options data found.** This mode runs on the offline NSE "
            "options archive. Build the store once (local machine):\n\n"
            "```\npython build_options_store.py\n```\n"
            "It converts the daily CSVs in `data_options/` and this tab lights up "
            "automatically. (The ~1 GB store stays local — it is not deployed.)")
        return

    c = st.columns([1, 1, 1, 1])
    u = c[0].selectbox("Underlying", unders, key="opt_u")
    lo, hi = od.date_range(u)
    d0 = c[1].date_input("From", value=max(lo, pd.Timestamp(hi).replace(month=1, day=1).date()),
                         min_value=lo, max_value=hi, key="opt_d0")
    d1 = c[2].date_input("To", value=hi, min_value=lo, max_value=hi, key="opt_d1")
    lot = c[3].number_input("Lot size", 1, 10000, od.LOT_SIZE.get(u, 75), key="opt_lot",
                            help="Contracts per lot. NSE has changed this over the years — "
                                 "set it to the value for your period.")

    st.markdown("**Strategy**")
    s = st.columns([2, 1, 1])
    width = s[1].number_input("Strangle/condor width (strike steps OTM)", 1, 30, 3, key="opt_w")
    lots = s[2].number_input("Lots", 1, 100, 1, key="opt_lots")
    presets = _presets(int(width))
    preset = s[0].selectbox("Preset", list(presets), key="opt_preset")
    legs = [{"right": r, "side": sd, "otm": o, "lots": int(lots)} for (r, sd, o) in presets[preset]]

    st.markdown("**Timing & risk**")
    t = st.columns([1, 1, 1, 1, 1])
    entry_t = t[0].selectbox("Entry", [555, 560, 570, 585, 600, 615, 645], index=1,
                             format_func=_hm, key="opt_entry")
    exit_t = t[1].selectbox("Square-off", [900, 915, 920, 925, 930], index=1,
                            format_func=_hm, key="opt_exit")
    which = t[2].selectbox("Expiry", [0, 1, 2], format_func=lambda i: ["nearest", "next", "3rd"][i],
                           key="opt_which")
    sl = t[3].number_input("Stop % of credit", 0, 500, 0, key="opt_sl",
                           help="Exit the whole position if its loss reaches this % of the "
                                "entry premium. 0 = no stop.")
    tp = t[4].number_input("Target % of credit", 0, 500, 0, key="opt_tp")

    if not st.button("▶ Run option backtest", type="primary", key="opt_run"):
        st.caption(f"{u}: data {lo} → {hi}. Pick a strategy and run.")
        return

    cfg = dict(underlying=u, d0=d0, d1=d1, expiry_which=int(which), min_dte=0,
               entry_t=int(entry_t), exit_t=int(exit_t), sl_pct=float(sl), tp_pct=float(tp),
               lot_size=int(lot), legs=legs)
    with st.spinner("Simulating on real premiums…"):
        res = oe.run(cfg)
    tr, m = res["trades"], res["metrics"]
    if m.get("trades", 0) == 0:
        st.warning("No trades in that window — try a wider date range or a different expiry rule.")
        return

    k = st.columns(5)
    k[0].metric("Total P&L", f"₹{m['total_pnl']:,.0f}")
    k[1].metric("Trades", m["trades"])
    k[2].metric("Win rate", f"{m['win_rate']}%")
    k[3].metric("Profit factor", m["profit_factor"])
    k[4].metric("Max drawdown", f"₹{m['max_dd']:,.0f}")
    k2 = st.columns(5)
    k2[0].metric("Avg / trade", f"₹{m['avg_pnl']:,.0f}")
    k2[1].metric("Avg win", f"₹{m['avg_win']:,.0f}")
    k2[2].metric("Avg loss", f"₹{m['avg_loss']:,.0f}")
    k2[3].metric("Best day", f"₹{m['best']:,.0f}")
    k2[4].metric("Worst day", f"₹{m['worst']:,.0f}")

    eq = res["equity"]
    fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, mode="lines",
                               line=dict(color=ACCENT, width=2), fill="tozeroy",
                               fillcolor="rgba(37,240,138,0.08)"))
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=30, b=10),
                      title="Equity curve (cumulative ₹)", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Trade log**")
    show = tr[["date", "dte", "spot", "atm", "strikes", "credit_pts", "pnl_pts", "pnl", "exit"]].copy()
    show["date"] = show["date"].dt.date
    st.dataframe(show, use_container_width=True, height=360, hide_index=True)
    st.download_button("⬇ Download trades (CSV)", show.to_csv(index=False).encode(),
                       file_name=f"{u}_option_backtest.csv", mime="text/csv", key="opt_dl")
