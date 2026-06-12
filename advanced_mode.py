"""
advanced_mode.py — "Advanced Backtesting" Streamlit mode (clean redesign)
=========================================================================
Two clearly separated sections, exactly as a strategy is written on paper:

  1️⃣ ENTRY CONDITIONS — an unlimited, composable list of rows
     ("Price sweeps PDL", "Price above HTF EMA 20", "RSI above 60",
      "CPR is NARROW", "Price breaks above PDH", "1-min FVG forms", …).
     Rows are written from the LONG perspective; shorts auto-mirror.
     All rows must be true together (logical AND).

  2️⃣ EXIT CONDITIONS — stop loss (structural or fixed, triggered on wick
     touch OR candle close — the "SL = candle close below FVG" style),
     target (strict R:R, opposite liquidity PDH/PDL, or none), optional
     trailing stop, optional indicator exits ("close crosses below EMA 9").

Plus: a day-type playbook fed by the same facts.csv as the Probabilities tab,
an Edge-Backtester-style permutation optimizer, and a per-trade visual
browser that draws each trade (entry/SL/target/FVG zones) on the chart.

No IB inputs here — the mode has its own slim data sidebar.
"""
from __future__ import annotations

import os
from datetime import time as dtime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import build_facts
import data_store
import engine

HERE = os.path.dirname(os.path.abspath(__file__))
FACTS = os.path.join(HERE, "analysis", "facts.csv")

EXEC_TFS = ["1-min", "3-min", "5-min", "15-min", "30-min", "1-hour"]
HTF_TFS  = ["5-min", "15-min", "30-min", "1-hour", "Daily", "Weekly"]

SUBJECTS = {
    "price_ema":     "Price vs EMA",
    "price_htf_ema": "Price vs HTF EMA",
    "price_level":   "Price vs key level (PDH/PDL/CPR)",
    "rsi":           "RSI",
    "cpr_width":     "CPR width type",
    "fvg":           "Fair Value Gap forms",
    "ob":            "Order Block forms",
    "atr_move":      "ATR move",
    "day_type":      "Day type filter",
    "time_window":   "Time window",
}
EXIT_SUBJECTS = {
    "price_ema":     "Price vs EMA",
    "price_htf_ema": "Price vs HTF EMA",
    "price_level":   "Price vs key level (PDH/PDL/CPR)",
    "rsi":           "RSI",
    "time_after":    "Time reached (exit at/after)",
}
LEVELS = list(engine.LEVEL_COLS.keys())
GAPS = ["Gap Up", "Gap Down", "Flat"]
KINDS = ["Inside Day", "Outside Day", "Normal"]

# seeded starter rule = the user's sample trade: sweep PDL + FVG confirmation
ENTRY_SEED = {0: {"type": "price_level", "op": "sweeps", "level": "PDL"},
              1: {"type": "fvg"}}


def _pct(x):
    return f"{x*100:.1f}%"


# ─── cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load(mpath, mtime):
    return build_facts.clean_min(data_store.read_minute(mpath))


@st.cache_data(show_spinner=False)
def _bounds(mpath, mtime):
    d = _load(mpath, mtime)["date_only"]
    return d.min(), d.max(), build_facts.detect_open_t(_load(mpath, mtime))


@st.cache_data(show_spinner=True)
def _features(mpath, mtime, open_t, sig):
    df = _load(mpath, mtime)
    params = dict(sig)
    return engine.build_features(df, open_t, params)


@st.cache_data(show_spinner=False)
def _facts(mtime):
    return pd.read_csv(FACTS, parse_dates=["date"])


# ─── slim data sidebar (NO IB) ────────────────────────────────────────────────

def _data_sidebar():
    st.header("Data")
    paths = data_store.discover()
    names = list(paths.keys())
    if not names:
        st.error("No instrument data found — add a *_minute.csv/.parquet to data/.")
        st.stop()
    instrument = st.selectbox("Instrument", names + ["➕ Upload new instrument…"],
                              key="adv_inst")

    if instrument == "➕ Upload new instrument…":
        nm = st.text_input("Instrument name (e.g. XAUUSD, NQ)", key="adv_nm")
        f = st.file_uploader("Minute CSV — date, open, high, low, close "
                             "(timestamps in exchange-local time, e.g. EST for NQ)",
                             type="csv", key="adv_nf")
        if nm and f is not None:
            data_store.save_upload(nm.strip(), f.getbuffer())
            st.success(f"Added {nm.strip()} — available in every mode.")
            st.rerun()
        st.stop()

    mpath = paths[instrument]
    mtime = os.path.getmtime(mpath)
    dmin, dmax, auto_t = _bounds(mpath, mtime)

    # session open/close with persistent per-instrument overrides (NQ → 09:30
    # / 16:00 ET). Widget keys scoped per instrument to avoid cross-market leaks.
    cur_t = data_store.session_open(instrument, auto_t)
    so = st.time_input("Session open", value=dtime(cur_t // 60, cur_t % 60),
                       key=f"adv_so_{instrument}",
                       help="Auto-detected; override for 24h instruments "
                            "(e.g. NQ → 09:30 EST). Remembered per instrument.")
    open_t = so.hour * 60 + so.minute
    if open_t != cur_t:
        data_store.set_open_override(instrument, open_t, auto_t)

    default_close = 15 * 60 + 15
    cur_c = data_store.session_close(instrument, default_close)
    sq = st.time_input("Square-off (force exit)",
                       value=dtime(cur_c // 60, cur_c % 60),
                       key=f"adv_sq_{instrument}",
                       help="Remembered per instrument (NSE 15:15 · NQ/XAUUSD 16:00 ET).")
    close_t = sq.hour * 60 + sq.minute
    if close_t != cur_c:
        data_store.set_close_override(instrument, close_t, default_close)
    dr = st.date_input("Date range", value=(dmin, dmax),
                       min_value=dmin, max_value=dmax, key="adv_dates")
    d0, d1 = (dr if isinstance(dr, (tuple, list)) and len(dr) == 2 else (dmin, dmax))
    st.caption(f"Session open **{open_t//60:02d}:{open_t%60:02d}**"
               f"{' (auto)' if open_t == auto_t else ' (override)'} · "
               "intraday only, no overnight carry")
    return instrument, mpath, mtime, open_t, close_t, d0, d1


def _settings_sidebar():
    st.header("Timeframes")
    exec_tf = st.selectbox("Execution timeframe", EXEC_TFS, index=2, key="adv_exec_tf",
                           help="Bars the strategy trades on. Pick 1-min for "
                                "1-min FVG entries.")
    htf_tf = st.selectbox("Higher timeframe (HTF)", HTF_TFS, index=3, key="adv_htf_tf",
                          help="Used by 'Price vs HTF EMA' / HTF RSI rows. Aligned "
                               "with zero lookahead.")
    st.header("CPR")
    c = st.columns(2)
    cpr_narrow = c[0].number_input("NARROW < %", 0.01, 5.0, 0.30, 0.05, key="adv_cprn",
                                   help="CPR width below this % of price → NARROW.")
    cpr_wide = c[1].number_input("WIDE > %", 0.05, 10.0, 0.75, 0.05, key="adv_cprw")
    cpr_period = st.selectbox("CPR basis", ["Daily", "Weekly"], key="adv_cprb")

    st.header("Detection")
    c2 = st.columns(2)
    atr_period = c2[0].number_input("ATR period (days)", 2, 100, 14, key="adv_atrp")
    fvg_min = c2[1].number_input("FVG min gap %", 0.0, 5.0, 0.0, 0.01, key="adv_fvgm",
                                 help="Ignore imbalances smaller than this % of price.")
    return dict(exec_tf=exec_tf, htf_tf=htf_tf, cpr_period=cpr_period,
                cpr_narrow=float(cpr_narrow), cpr_wide=float(cpr_wide),
                atr_period=int(atr_period), fvg_min_gap_pct=float(fvg_min),
                ob_lookback=5)


# ─── dynamic condition rows ───────────────────────────────────────────────────

def _row(prefix, rid, subjects, dflt):
    """One condition row. Returns (cond_dict, delete_clicked)."""
    keys = list(subjects.keys())
    with st.container(border=True):
        c = st.columns([1.5, 3.3, 0.3])
        subj = c[0].selectbox("Condition", keys, key=f"{prefix}{rid}_type",
                              index=keys.index(dflt.get("type", keys[0])),
                              format_func=lambda k: subjects[k],
                              label_visibility="collapsed")
        cond = {"type": subj}
        box = c[1]

        if subj in ("price_ema", "price_htf_ema"):
            cc = box.columns([1.4, 1])
            cond["op"] = cc[0].selectbox("op", engine.OPS_EMA, key=f"{prefix}{rid}_op",
                                         index=engine.OPS_EMA.index(dflt.get("op", "above")),
                                         label_visibility="collapsed")
            cond["period"] = int(cc[1].number_input("period", 2, 500,
                                                    int(dflt.get("period", 20)),
                                                    key=f"{prefix}{rid}_p",
                                                    label_visibility="collapsed"))
        elif subj == "price_level":
            cc = box.columns([1.4, 1])
            cond["op"] = cc[0].selectbox("op", engine.OPS_LEVEL, key=f"{prefix}{rid}_op2",
                                         index=engine.OPS_LEVEL.index(dflt.get("op", "sweeps")),
                                         label_visibility="collapsed")
            cond["level"] = cc[1].selectbox("level", LEVELS, key=f"{prefix}{rid}_lvl",
                                            index=LEVELS.index(dflt.get("level", "PDL")),
                                            label_visibility="collapsed")
        elif subj == "rsi":
            cc = box.columns([0.8, 1.2, 0.7, 0.7])
            cond["tf"] = cc[0].selectbox("tf", ["exec", "HTF"], key=f"{prefix}{rid}_tf",
                                         label_visibility="collapsed")
            cond["op"] = cc[1].selectbox("op", engine.OPS_RSI, key=f"{prefix}{rid}_rop",
                                         label_visibility="collapsed")
            cond["threshold"] = float(cc[2].number_input("thr", 1.0, 99.0,
                                                         float(dflt.get("threshold", 60)),
                                                         key=f"{prefix}{rid}_thr",
                                                         label_visibility="collapsed"))
            cond["period"] = int(cc[3].number_input("rp", 2, 100,
                                                    int(dflt.get("period", 14)),
                                                    key=f"{prefix}{rid}_rp",
                                                    label_visibility="collapsed"))
        elif subj == "cpr_width":
            cond["allowed"] = box.multiselect("CPR types", ["NARROW", "MEDIUM", "WIDE"],
                                              default=dflt.get("allowed", ["NARROW"]),
                                              key=f"{prefix}{rid}_cw",
                                              label_visibility="collapsed")
        elif subj == "fvg":
            box.caption("Bullish FVG → long · bearish FVG → short (auto-mirrored). "
                        "Signal fires on the confirming candle.")
        elif subj == "ob":
            box.caption("Bullish Order Block → long · bearish → short (auto-mirrored).")
        elif subj == "atr_move":
            cc = box.columns([1.6, 0.8])
            cond["op"] = cc[0].selectbox("op", engine.OPS_ATR, key=f"{prefix}{rid}_aop",
                                         label_visibility="collapsed")
            cond["k"] = float(cc[1].number_input("k", 0.1, 10.0,
                                                 float(dflt.get("k", 1.0)), 0.1,
                                                 key=f"{prefix}{rid}_ak",
                                                 label_visibility="collapsed"))
        elif subj == "day_type":
            cc = box.columns(3)
            cond["gaps"] = cc[0].multiselect("Gap", GAPS, key=f"{prefix}{rid}_g")
            cond["prev_kinds"] = cc[1].multiselect("Prev day", KINDS, key=f"{prefix}{rid}_k")
            cond["dows"] = cc[2].multiselect("Weekday", engine.WEEKDAYS,
                                             key=f"{prefix}{rid}_d")
        elif subj == "time_window":
            cc = box.columns(2)
            t1 = cc[0].time_input("from", value=dtime(9, 30), key=f"{prefix}{rid}_t1",
                                  label_visibility="collapsed")
            t2 = cc[1].time_input("to", value=dtime(14, 30), key=f"{prefix}{rid}_t2",
                                  label_visibility="collapsed")
            cond["t1"], cond["t2"] = t1.hour * 60 + t1.minute, t2.hour * 60 + t2.minute
        elif subj == "time_after":
            t = box.time_input("at", value=dtime(14, 30), key=f"{prefix}{rid}_ta",
                               label_visibility="collapsed")
            cond["t"] = t.hour * 60 + t.minute

        deleted = c[2].button("🗑", key=f"{prefix}{rid}_del", help="Remove this condition")
    return cond, deleted


def _condition_rows(state_key, subjects, seed):
    """Render a dynamic add/remove list of condition rows; return cond dicts."""
    ids = st.session_state.setdefault(state_key, list(seed.keys()))
    st.session_state.setdefault(f"{state_key}_next", (max(ids) + 1) if ids else 0)

    conds, to_del = [], None
    for rid in ids:
        cond, deleted = _row(f"{state_key}_", rid, subjects, seed.get(rid, {}))
        if deleted:
            to_del = rid
        else:
            conds.append(cond)
    if to_del is not None:
        ids.remove(to_del)
        st.rerun()
    if st.button("➕ Add condition", key=f"{state_key}_add"):
        ids.append(st.session_state[f"{state_key}_next"])
        st.session_state[f"{state_key}_next"] += 1
        st.rerun()
    return conds


def _describe(cond):
    """Plain-English rendering of one condition row (long perspective)."""
    t = cond["type"]
    if t == "price_ema":
        return f"price {cond['op']} EMA {cond['period']}"
    if t == "price_htf_ema":
        return f"price {cond['op']} HTF EMA {cond['period']}"
    if t == "price_level":
        return f"price {cond['op']} {cond['level']}"
    if t == "rsi":
        return f"{cond['tf']} RSI({cond['period']}) {cond['op']} {cond['threshold']:.0f}"
    if t == "cpr_width":
        return f"CPR is {'/'.join(cond.get('allowed') or ['any'])}"
    if t == "fvg":
        return "FVG forms"
    if t == "ob":
        return "Order Block forms"
    if t == "atr_move":
        return f"price {cond['op']} {cond['k']:g}×ATR"
    if t == "day_type":
        bits = [", ".join(cond[k]) for k in ("gaps", "prev_kinds", "dows") if cond.get(k)]
        return "day is " + (" & ".join(bits) if bits else "any")
    if t == "time_window":
        return f"time {cond['t1']//60:02d}:{cond['t1']%60:02d}–{cond['t2']//60:02d}:{cond['t2']%60:02d}"
    if t == "time_after":
        return f"time ≥ {cond['t']//60:02d}:{cond['t']%60:02d}"
    return t


# ─── feature requirements from conditions ─────────────────────────────────────

def _collect_params(settings, entry_conds, exit_conds):
    ema_p, htf_ema_p, rsi_p, htf_rsi_p = set(), set(), set(), set()
    for c in entry_conds + exit_conds:
        if c["type"] == "price_ema":
            ema_p.add(int(c["period"]))
        elif c["type"] == "price_htf_ema":
            htf_ema_p.add(int(c["period"]))
        elif c["type"] == "rsi":
            (rsi_p if c.get("tf", "exec") == "exec" else htf_rsi_p).add(int(c["period"]))
    return {**settings,
            "ema_periods": tuple(sorted(ema_p)),
            "htf_ema_periods": tuple(sorted(htf_ema_p)),
            "rsi_periods": tuple(sorted(rsi_p)),
            "htf_rsi_periods": tuple(sorted(htf_rsi_p))}


# ─── day-type playbook (probabilities integration) ────────────────────────────

def _playbook(instrument, d0, d1, entry_conds):
    if not os.path.exists(FACTS):
        st.caption("`analysis/facts.csv` missing — run `python build_facts.py` to "
                   "enable the playbook.")
        return
    f = _facts(os.path.getmtime(FACTS))
    f = f[(f["instrument"] == instrument)
          & (f["date"] >= pd.Timestamp(d0)) & (f["date"] <= pd.Timestamp(d1))].copy()
    f["day_kind"] = np.where(f["inside_day"], "Inside Day",
                    np.where(f["outside_day"], "Outside Day", "Normal"))

    # apply any day-type filters the user added as entry conditions
    note = []
    for c in entry_conds:
        if c["type"] == "day_type":
            if c.get("gaps"):
                f = f[f["gap_type"].isin(c["gaps"])]
                note.append("gap=" + "/".join(c["gaps"]))
            if c.get("dows"):
                f = f[f["dow"].isin(c["dows"])]
                note.append("dow=" + "/".join(c["dows"]))
    if f.empty:
        st.warning("No historical days match the current day-type filters.")
        return

    k = st.columns(5)
    k[0].metric("Matching days", f"{len(f):,}")
    k[1].metric("Reach PDH", _pct(f["broke_pdh"].mean()))
    k[2].metric("Reach PDL", _pct(f["broke_pdl"].mean()))
    k[3].metric("IB high breaks", _pct(f["ib_high_break"].mean()))
    k[4].metric("IB low breaks", _pct(f["ib_low_break"].mean()))

    g1, g2 = st.columns(2)
    by_gap = (f.groupby("gap_type")
              .agg(days=("date", "size"), **{"reach PDH %": ("broke_pdh", "mean"),
                                             "reach PDL %": ("broke_pdl", "mean")}))
    by_gap[["reach PDH %", "reach PDL %"]] = (by_gap[["reach PDH %", "reach PDL %"]] * 100).round(1)
    with g1:
        st.markdown("**By today's gap** — known at the open")
        st.dataframe(by_gap, use_container_width=True)
    by_kind = (f.groupby("day_kind")
               .agg(days=("date", "size"), **{"reach PDH %": ("broke_pdh", "mean"),
                                              "reach PDL %": ("broke_pdl", "mean")}))
    by_kind[["reach PDH %", "reach PDL %"]] = (by_kind[["reach PDH %", "reach PDL %"]] * 100).round(1)
    with g2:
        st.markdown("**By day kind** — context only (known after the close)")
        st.dataframe(by_kind, use_container_width=True)
    st.caption(("Filters: " + ", ".join(note) + " · " if note else "")
               + "Same data as the **Probabilities** mode — use it for the deep dive, "
                 "then encode the edge here as day-type / level conditions.")


# ─── results: metrics, charts, log ────────────────────────────────────────────

def _show_metrics(m):
    k = st.columns(5)
    k[0].metric("Net Profit", f"₹{m['net_profit']:,.0f}", f"{m['net_pct']:+.1f}%")
    k[1].metric("CAGR", _pct(m["cagr"]) if np.isfinite(m["cagr"]) else "—")
    k[2].metric("Sharpe", f"{m['sharpe']:.2f}" if np.isfinite(m["sharpe"]) else "—")
    k[3].metric("Sortino", f"{m['sortino']:.2f}" if np.isfinite(m["sortino"]) else "—")
    k[4].metric("Profit Factor", f"{m['profit_factor']:.2f}"
                if np.isfinite(m["profit_factor"]) else "∞")
    k2 = st.columns(5)
    k2[0].metric("Win Rate", _pct(m["win_rate"]))
    k2[1].metric("Max Drawdown", f"{m['max_dd_pct']:.1f}%", f"₹{m['max_dd']:,.0f}",
                 delta_color="inverse")
    rec = "not recovered" if m["recovery_days"] is None else f"{m['recovery_days']}d to recover"
    k2[2].metric("DD Duration", f"{m['dd_duration_days']}d", rec, delta_color="off")
    k2[3].metric("Trades", f"{m['trades']:,}")
    k2[4].metric("Avg R / trade", f"{m['avg_r']:+.2f}R")


def _equity_tab(eq_df):
    fig = go.Figure()
    fig.add_scatter(x=eq_df["date"], y=eq_df["equity"], mode="lines",
                    line=dict(color="#2196F3", width=2),
                    fill="tozeroy", fillcolor="rgba(33,150,243,0.08)", name="Equity")
    peak = np.maximum.accumulate(eq_df["equity"].to_numpy())
    fig.add_scatter(x=eq_df["date"], y=peak, mode="lines",
                    line=dict(color="#90A4AE", width=1, dash="dot"), name="Peak")
    fig.update_layout(height=340, template="plotly_white", yaxis_title="₹",
                      margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)


def _trade_chart(V, tr, ema_cols):
    """Single-day chart of one trade: candles, EMAs, levels, FVGs, the trade."""
    x = V["open_time"]
    fig = go.Figure()
    fig.add_candlestick(x=x, open=V["open"], high=V["high"], low=V["low"],
                        close=V["close"], name="Price",
                        increasing_line_color="#26A69A", decreasing_line_color="#EF5350")
    palette = ["#FB8C00", "#8E24AA", "#1E88E5", "#00897B"]
    for i, col in enumerate(ema_cols):
        if col in V and V[col].notna().any():
            fig.add_scatter(x=x, y=V[col], mode="lines", name=col.replace("ema_", "EMA "),
                            line=dict(width=1.3, color=palette[i % len(palette)]))
    for col, nm, color in [("pdh", "PDH", "#455A64"), ("pdl", "PDL", "#455A64"),
                           ("cpr_top", "CPR top", "#7E57C2"),
                           ("cpr_bottom", "CPR bot", "#7E57C2")]:
        v = V[col].dropna()
        if len(v):
            fig.add_hline(y=float(v.iloc[0]), line=dict(width=1, dash="dash", color=color),
                          annotation_text=nm, annotation_font_size=10)
    # FVG zones that confirmed during this day
    for _, r in V[(V["bull_fvg"]) | (V["bear_fvg"])].iterrows():
        bull = bool(r["bull_fvg"])
        fig.add_shape(type="rect", x0=r["open_time"], x1=x.iloc[-1],
                      y0=r["fvg_gap_bottom"], y1=r["fvg_gap_top"],
                      fillcolor=("rgba(38,166,154,0.13)" if bull else "rgba(239,83,80,0.13)"),
                      line_width=0, layer="below")
    # the trade itself
    long = tr["direction"] == "long"
    win_c = "#2E7D32" if tr["win"] else "#C62828"
    fig.add_scatter(x=[tr["entry_time"]], y=[tr["entry"]], mode="markers+text",
                    marker=dict(symbol="triangle-up" if long else "triangle-down",
                                size=14, color="#2E7D32" if long else "#C62828",
                                line=dict(width=1, color="white")),
                    text=["ENTRY"], textposition="bottom center" if long else "top center",
                    textfont=dict(size=10), showlegend=False)
    fig.add_scatter(x=[tr["exit_time"]], y=[tr["exit"]], mode="markers+text",
                    marker=dict(symbol="x", size=11, color=win_c),
                    text=[f"EXIT ({tr['outcome']})"], textposition="top center",
                    textfont=dict(size=10), showlegend=False)
    fig.add_scatter(x=[tr["entry_time"], tr["exit_time"]], y=[tr["entry"], tr["exit"]],
                    mode="lines", line=dict(width=1.2, color=win_c, dash="dot"),
                    showlegend=False, hoverinfo="skip")
    fig.add_shape(type="line", x0=tr["entry_time"], x1=tr["exit_time"],
                  y0=tr["stop"], y1=tr["stop"],
                  line=dict(width=1.5, color="#C62828", dash="dash"))
    if pd.notna(tr.get("target")):
        fig.add_shape(type="line", x0=tr["entry_time"], x1=tr["exit_time"],
                      y0=tr["target"], y1=tr["target"],
                      line=dict(width=1.5, color="#2E7D32", dash="dash"))
    fig.update_layout(height=520, template="plotly_white",
                      xaxis_rangeslider_visible=False, margin=dict(t=25, b=10),
                      legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)


def _trade_browser(X, trades, ema_cols):
    t = trades.reset_index(drop=True)
    labels = [f"#{i+1} · {r.entry_time:%d %b %Y %H:%M} · {r.direction.upper()} · "
              f"{r.r_mult:+.2f}R · {r.outcome}" for i, r in t.iterrows()]
    sel = st.selectbox("Trade", range(len(t)), format_func=lambda i: labels[i],
                       key="adv_tb_sel")
    tr = t.iloc[sel]
    V = X[X["date_only"] == tr["date"].date()]
    if V.empty:
        st.info("No chart data for this trade's day.")
        return
    c = st.columns(6)
    c[0].metric("Direction", tr["direction"].upper())
    c[1].metric("Entry", f"{tr['entry']:,.1f}")
    c[2].metric("Stop", f"{tr['stop']:,.1f}")
    c[3].metric("Exit", f"{tr['exit']:,.1f}")
    c[4].metric("R multiple", f"{tr['r_mult']:+.2f}R")
    c[5].metric("P&L", f"₹{tr['pnl']:,.0f}")
    _trade_chart(V, tr, ema_cols)
    st.caption("▲/▼ entry · ✕ exit · red dashed = initial stop · green dashed = target · "
               "shaded = FVG zones · dashed h-lines = PDH/PDL & CPR.")


def _window_chart(X, trades, ema_cols, d0, d1):
    cc = st.columns([1, 3])
    default_day = (trades["date"].dt.date.value_counts().idxmax()
                   if not trades.empty else d1)
    win = cc[0].date_input("Window", value=(default_day, default_day),
                           min_value=d0, max_value=d1, key="adv_win2")
    cd0, cd1 = (win if isinstance(win, (tuple, list)) and len(win) == 2 else (win, win))
    cc[1].caption("Multi-day overview — pick a tight window for readable candles.")
    V = X[(X["date_only"] >= cd0) & (X["date_only"] <= cd1)]
    if V.empty:
        st.info("No bars in this window.")
        return
    if len(V) > 4000:
        V = V.iloc[-4000:]
    x = V["open_time"]
    fig = go.Figure()
    fig.add_candlestick(x=x, open=V["open"], high=V["high"], low=V["low"],
                        close=V["close"], name="Price",
                        increasing_line_color="#26A69A", decreasing_line_color="#EF5350")
    palette = ["#FB8C00", "#8E24AA", "#1E88E5"]
    for i, col in enumerate(ema_cols):
        if col in V:
            fig.add_scatter(x=x, y=V[col], mode="lines", name=col.replace("ema_", "EMA "),
                            line=dict(width=1.2, color=palette[i % len(palette)]))
    for col, nm in [("pdh", "PDH"), ("pdl", "PDL"),
                    ("cpr_top", "CPR top"), ("cpr_bottom", "CPR bot")]:
        if V[col].notna().any():
            fig.add_scatter(x=x, y=V[col], mode="lines", name=nm, opacity=0.6,
                            line=dict(width=1, dash="dash"), connectgaps=False)
    tin = trades[(trades["date"].dt.date >= cd0) & (trades["date"].dt.date <= cd1)]
    for _, tr in tin.iterrows():
        long = tr["direction"] == "long"
        win_c = "#2E7D32" if tr["win"] else "#C62828"
        fig.add_scatter(x=[tr["entry_time"]], y=[tr["entry"]], mode="markers",
                        marker=dict(symbol="triangle-up" if long else "triangle-down",
                                    size=11, color="#2E7D32" if long else "#C62828"),
                        showlegend=False, hovertext=f"{tr['direction']} {tr['entry']}",
                        hoverinfo="text")
        fig.add_scatter(x=[tr["exit_time"]], y=[tr["exit"]], mode="markers",
                        marker=dict(symbol="x", size=9, color=win_c), showlegend=False,
                        hovertext=f"exit {tr['exit']} ({tr['outcome']})", hoverinfo="text")
    fig.update_layout(height=520, template="plotly_white",
                      xaxis_rangeslider_visible=False, margin=dict(t=25, b=10),
                      legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(tin)} trade(s) in window.")


def _trade_log(trades):
    f = st.columns(4)
    dirs = f[0].multiselect("Direction", ["long", "short"], default=["long", "short"],
                            key="adv_log_dir")
    dows = f[1].multiselect("Weekday", engine.WEEKDAYS, default=engine.WEEKDAYS,
                            key="adv_log_dow")
    outs = f[2].multiselect("Outcome", sorted(trades["outcome"].unique()),
                            default=sorted(trades["outcome"].unique()),
                            key="adv_log_out")
    res = f[3].selectbox("Result", ["all", "wins", "losses"], key="adv_log_res")
    t = trades[trades["direction"].isin(dirs) & trades["dow"].isin(dows)
               & trades["outcome"].isin(outs)]
    if res == "wins":
        t = t[t["win"]]
    elif res == "losses":
        t = t[~t["win"]]
    st.dataframe(t.sort_values("entry_time", ascending=False),
                 use_container_width=True, hide_index=True, height=380)
    st.download_button("⬇ Download trades (CSV)", trades.to_csv(index=False).encode(),
                       file_name="advanced_trades.csv", mime="text/csv")


# ─── optimizer ────────────────────────────────────────────────────────────────

def _optimizer(X, strategy, close_t):
    st.caption("Sweeps target-R × stop-type × leak-free day filters over your fixed "
               "entry stack and ranks every configuration (like the Edge Backtester).")
    o = st.columns(4)
    rank = o[0].selectbox("Rank by", ["expectancy", "net", "win_rate", "pf"],
                          key="adv_opt_rank")
    min_tr = int(o[1].number_input("Min trades", 10, 2000, 50, 10, key="adv_opt_min"))
    tp_grid = o[2].multiselect("Target R grid", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                               default=[1.0, 1.5, 2.0, 3.0], key="adv_opt_tp")
    sl_grid = o[3].multiselect("Stop types", ["fvg", "swing", "pdh_pdl", "cpr_edge",
                                              "fixed_pct"],
                               default=["fvg", "swing", "fixed_pct"], key="adv_opt_sl")
    o2 = st.columns(3)
    sw_gap = o2[0].checkbox("Sweep gap type", value=True, key="adv_opt_g")
    sw_kind = o2[1].checkbox("Sweep prev-day kind", value=True, key="adv_opt_k")
    sw_dow = o2[2].checkbox("Sweep weekday", value=False, key="adv_opt_d")

    if st.button("🔎 Run optimization", key="adv_opt_run"):
        if not tp_grid or not sl_grid:
            st.warning("Pick at least one target and one stop type.")
            return
        prog = st.progress(0.0, text="Sweeping configurations…")

        def cb(done, total):
            if done % 5 == 0 or done == total:
                prog.progress(done / total, text=f"{done}/{total} configurations…")

        res = engine.optimize(X, strategy, close_t, tp_grid, sl_grid,
                              sw_dow, sw_gap, sw_kind, min_tr, rank, progress=cb)
        prog.empty()
        st.session_state["adv_opt_res"] = res

    res = st.session_state.get("adv_opt_res")
    if res is not None:
        if res.empty:
            st.warning("No configuration met the minimum-trades threshold.")
        else:
            st.success(f"{len(res):,} valid configurations · top 40 shown")
            st.dataframe(res.head(40), use_container_width=True, hide_index=True)
            st.download_button("⬇ Download all results", res.to_csv(index=False).encode(),
                               file_name="advanced_optimizer.csv", mime="text/csv")
            st.caption("In-sample optimum — validate the best rows on a date sub-range "
                       "before trusting them. Day filters use only leak-free context "
                       "(today's gap, previous day's kind, weekday).")


# ─── main entry point ─────────────────────────────────────────────────────────

def render(*_):
    st.title("🏛️ Advanced Backtesting")
    st.caption("Composable entry/exit rules · shorts auto-mirror · zero lookahead · "
               "fixed-fractional risk · intraday square-off")

    with st.sidebar:
        instrument, mpath, mtime, open_t, close_t, d0, d1 = _data_sidebar()
        st.divider()
        settings = _settings_sidebar()

    # ── 1️⃣ entry conditions ──────────────────────────────────────────────────
    st.subheader("1️⃣ Entry Conditions")
    ec = st.columns([1, 1, 2])
    long_on = ec[0].toggle("LONG trades", value=True, key="adv_long_on")
    short_on = ec[1].toggle("SHORT trades", value=True, key="adv_short_on")
    entry_type = ec[2].selectbox("Entry fill", ["immediate", "next_open"],
                                 format_func=lambda x: {"immediate": "Signal candle close (immediate)",
                                                        "next_open": "Next candle open"}[x],
                                 key="adv_entry_type")
    st.caption("Write rules for the **LONG** side — shorts mirror automatically "
               "(above↔below · PDH↔PDL · bullish↔bearish). ALL rows must be true "
               "together (AND).")
    entry_conds = _condition_rows("adv_ec", SUBJECTS, ENTRY_SEED)

    if entry_conds:
        rule = " AND ".join(_describe(c) for c in entry_conds)
        st.info(f"**LONG when:** {rule}" + ("  ·  *(shorts mirrored)*" if short_on else ""))

    with st.expander("📊 Day-type playbook — historical odds from the Probabilities data"):
        _playbook(instrument, d0, d1, entry_conds)

    # ── 2️⃣ exit conditions ───────────────────────────────────────────────────
    st.subheader("2️⃣ Exit Conditions")
    x1, x2, x3 = st.columns(3)
    with x1.container(border=True):
        st.markdown("**Stop loss**")
        sl_type = st.selectbox("Placement", ["fvg", "swing", "pdh_pdl", "cpr_edge",
                                             "fixed_pct"],
                               format_func=lambda x: {"fvg": "FVG invalidation",
                                                      "swing": "Recent swing",
                                                      "pdh_pdl": "Beyond PDL/PDH",
                                                      "cpr_edge": "Beyond CPR edge",
                                                      "fixed_pct": "Fixed %"}[x],
                               key="adv_sl_type")
        sl_trigger = st.radio("Trigger", ["close", "touch"], horizontal=True,
                              format_func=lambda x: {"close": "Candle close beyond",
                                                     "touch": "Wick touch"}[x],
                              key="adv_sl_trig",
                              help="'Candle close beyond' = the sample-trade style: "
                                   "SL only if a candle CLOSES past the level.")
        sl_buf = st.number_input("Buffer (% of structure dist.)", 0.0, 50.0, 5.0, 1.0,
                                 key="adv_sl_buf")
        sl_pct = st.number_input("Fixed stop (% price)", 0.05, 5.0, 0.4, 0.05,
                                 key="adv_sl_pct")
        swing_lb = st.number_input("Swing lookback (bars)", 2, 100, 10, key="adv_swlb")
    with x2.container(border=True):
        st.markdown("**Target**")
        target_mode = st.radio("Mode", ["rr", "liquidity", "none"],
                               format_func=lambda x: {"rr": "Strict Risk:Reward",
                                                      "liquidity": "Opposite liquidity (PDH/PDL)",
                                                      "none": "No fixed target"}[x],
                               key="adv_tmode",
                               help="Opposite liquidity = the sample trade's "
                                    "'Target: immediate liquidity' (PDH for longs).")
        tp_R = st.number_input("R multiple (or fallback)", 0.5, 10.0, 2.0, 0.5,
                               key="adv_tpR")
    with x3.container(border=True):
        st.markdown("**Trailing stop**")
        trail_on = st.toggle("Enable trailing", value=False, key="adv_trail")
        trail_trig = st.number_input("Trigger (R in profit)", 0.1, 10.0, 1.0, 0.1,
                                     key="adv_ttrig", disabled=not trail_on)
        trail_dist = st.number_input("Distance (R behind best)", 0.1, 10.0, 1.0, 0.1,
                                     key="adv_tdist", disabled=not trail_on)

    with st.expander("➕ Indicator-based exits (optional) — exit when ANY is true"):
        st.caption("Written for LONG positions (e.g. *price crosses below EMA 9*); "
                   "shorts mirror automatically.")
        exit_conds = _condition_rows("adv_xc", EXIT_SUBJECTS, {})

    # ── capital & run ─────────────────────────────────────────────────────────
    cr = st.columns([1, 1, 2])
    capital = cr[0].number_input("Capital (₹)", 50_000, 100_000_000, 1_000_000, 50_000,
                                 key="adv_capital")
    risk_pct = cr[1].number_input("Risk per trade (% equity)", 0.1, 10.0, 1.0, 0.1,
                                  key="adv_riskpct")
    cr[2].caption("Fixed-fractional sizing: each trade risks this % of current equity "
                  "at the initial stop.")

    strategy = {
        "long_enabled": long_on, "short_enabled": short_on,
        "entry_conds": entry_conds,
        "entry": {"type": entry_type},
        "exit": dict(sl_type=sl_type, sl_trigger=sl_trigger,
                     sl_buffer_pct=float(sl_buf), sl_pct=float(sl_pct),
                     swing_lookback=int(swing_lb), target_mode=target_mode,
                     tp_R=float(tp_R), exit_conds=exit_conds,
                     trail_on=bool(trail_on), trail_trigger_R=float(trail_trig),
                     trail_dist_R=float(trail_dist)),
        "risk": dict(capital=float(capital), risk_pct=float(risk_pct)),
    }

    run = st.button("▶ Run backtest", type="primary", use_container_width=True)
    if run:
        if not entry_conds:
            st.warning("Add at least one entry condition.")
            st.stop()
        params = _collect_params(settings, entry_conds, exit_conds)
        sig = tuple(sorted(params.items()))
        with st.spinner("Building features (cached after first run)…"):
            X = _features(mpath, mtime, open_t, sig)
        Xw = X[(X["date_only"] >= d0) & (X["date_only"] <= d1)].reset_index(drop=True)
        if Xw.empty:
            st.warning("No data in the selected date range.")
            st.stop()
        with st.spinner("Simulating…"):
            trades, eq = engine.backtest(Xw, strategy, close_t=close_t)
        ema_cols = [f"ema_{p}" for p in params["ema_periods"]]
        keep = (["open_time", "date_only", "open", "high", "low", "close",
                 "bull_fvg", "bear_fvg", "fvg_gap_bottom", "fvg_gap_top",
                 "cpr_top", "cpr_bottom", "pdh", "pdl", "t_min", "dow",
                 "gap_type", "prev_day_kind"] + ema_cols)
        st.session_state["adv_res"] = dict(
            trades=trades, eq=eq, capital=float(capital), ema_cols=ema_cols,
            X=Xw[keep].copy(), strategy=strategy, close_t=close_t, d0=d0, d1=d1,
            sig=sig, mpath=mpath, mtime=mtime, open_t=open_t)

    res = st.session_state.get("adv_res")
    if res is None:
        st.info("Build your entry/exit rules above, then **Run backtest**. The seeded "
                "rule is your sample trade: sweep of PDL + FVG confirmation, SL on "
                "candle close beyond the FVG.")
        return

    trades, eq = res["trades"], res["eq"]
    st.divider()
    if trades.empty:
        st.warning("No trades generated — loosen the conditions or widen the dates.")
        return

    m = engine.compute_metrics(trades, eq, res["capital"])
    _show_metrics(m)

    t1, t2, t3, t4, t5 = st.tabs(["📈 Equity", "🔍 Trade browser", "🗺 Chart window",
                                  "📋 Trade log", "🔬 Optimizer"])
    with t1:
        _equity_tab(eq)
    with t2:
        _trade_browser(res["X"], trades, res["ema_cols"])
    with t3:
        _window_chart(res["X"], trades, res["ema_cols"], res["d0"], res["d1"])
    with t4:
        _trade_log(trades)
    with t5:
        # the optimizer re-simulates, so it needs the FULL feature frame
        # (cached — this is instant after the first build)
        Xf = _features(res["mpath"], res["mtime"], res["open_t"], res["sig"])
        Xf = Xf[(Xf["date_only"] >= res["d0"])
                & (Xf["date_only"] <= res["d1"])].reset_index(drop=True)
        _optimizer(Xf, res["strategy"], res["close_t"])
