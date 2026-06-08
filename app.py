"""
ORB & IB Strategy Backtester
Opening Range Breakout (configurable duration) + Initial Balance (60-min)
Instruments: NIFTY 50, BANK NIFTY
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import time

st.set_page_config(
    page_title="ORB & IB Backtester",
    page_icon="📈",
    layout="wide",
)

# ─── constants ────────────────────────────────────────────────────────────────
MARKET_OPEN  = time(9, 15)
MARKET_CLOSE = time(15, 15)

STRATEGIES = {
    "ORB Breakout — Long":          "orb_long",
    "ORB Breakout — Short":         "orb_short",
    "ORB Breakout — Both Sides":    "orb_both",
    "IB Breakout — Long":           "ib_long",
    "IB Breakout — Short":          "ib_short",
    "IB Breakout — Both Sides":     "ib_both",
    "ORB confirmed by IB — Long":   "orb_ib_long",
    "ORB confirmed by IB — Short":  "orb_ib_short",
    "Failed ORB Reversal — Long":   "failed_orb_long",
    "Failed ORB Reversal — Short":  "failed_orb_short",
    "IB Retest — Long":             "ib_retest_long",
    "IB Retest — Short":            "ib_retest_short",
}

SL_OPTIONS = {
    "Full ORB/IB Range":  "range",
    "Half ORB/IB Range":  "half_range",
    "Fixed 50 pts":       "fixed_50",
    "Fixed 100 pts":      "fixed_100",
}

# Default CSV locations — if present, the dashboard auto-loads them so no
# manual upload is required. Uploaded files (if any) always take priority.
DEFAULT_PATHS = {
    "NIFTY 50":   r"C:\NIFTY 50_minute.csv",
    "BANK NIFTY": r"C:\NIFTY BANK_minute.csv",
}

# ─── data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_csv_path(path):
    """Load a CSV directly from a filesystem path (used for auto-load)."""
    df = pd.read_csv(path, parse_dates=["date"])
    return _clean(df)


@st.cache_data(show_spinner=False)
def load_csv(file_bytes, filename):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes), parse_dates=["date"])
    return _clean(df)


def _clean(df):
    df.columns = df.columns.str.strip().str.lower()
    df = df.sort_values("date").reset_index(drop=True)
    # drop rows with any missing OHLC
    df = df.dropna(subset=["open", "high", "low", "close"])
    # sanity checks
    df = df[(df["high"] >= df["low"]) & (df["open"] > 0) & (df["close"] > 0)]
    df["date_only"] = df["date"].dt.date
    df["t_min"]     = df["date"].dt.hour * 60 + df["date"].dt.minute   # minutes since midnight
    return df.reset_index(drop=True)


# ─── level computation ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_levels(df, orb_min=15, ib_min=60):
    """Return a per-day DataFrame with ORB and IB levels."""
    open_t = 9 * 60 + 15          # 9:15 in minutes
    orb_end_t = open_t + orb_min - 1
    ib_end_t  = open_t + ib_min  - 1

    records = []
    for day, grp in df.groupby("date_only"):
        orb_data = grp[(grp["t_min"] >= open_t) & (grp["t_min"] <= orb_end_t)]
        ib_data  = grp[(grp["t_min"] >= open_t) & (grp["t_min"] <= ib_end_t)]
        if orb_data.empty or ib_data.empty:
            continue
        records.append({
            "date":      day,
            "orb_high":  orb_data["high"].max(),
            "orb_low":   orb_data["low"].min(),
            "orb_range": orb_data["high"].max() - orb_data["low"].min(),
            "orb_open":  orb_data.iloc[0]["open"],
            "ib_high":   ib_data["high"].max(),
            "ib_low":    ib_data["low"].min(),
            "ib_range":  ib_data["high"].max() - ib_data["low"].min(),
        })
    return pd.DataFrame(records)


# ─── stop-loss & target helpers ───────────────────────────────────────────────

def get_sl_target(entry: float, direction: str, ref_range: float,
                  sl_type: str, rr: float):
    if sl_type == "range":
        sl_pts = ref_range
    elif sl_type == "half_range":
        sl_pts = ref_range * 0.5
    elif sl_type == "fixed_50":
        sl_pts = 50.0
    else:                          # fixed_100
        sl_pts = 100.0

    if direction == "long":
        return entry - sl_pts, entry + sl_pts * rr, sl_pts
    else:
        return entry + sl_pts, entry - sl_pts * rr, sl_pts


# ─── single-trade simulator ───────────────────────────────────────────────────

def simulate_trade(entry_bar, entry_price, direction, sl, target, day_df):
    """
    Scan candles after entry_bar on the same day.
    Returns (exit_time, exit_price, outcome) or None if no exit found.
    """
    for _, c in day_df[day_df["date"] > entry_bar].iterrows():
        if c["date"].time() >= MARKET_CLOSE:
            return c["date"], c["open"], "time_exit"

        if direction == "long":
            if c["low"] <= sl:
                return c["date"], sl, "sl_hit"
            if c["high"] >= target:
                return c["date"], target, "target_hit"
        else:
            if c["high"] >= sl:
                return c["date"], sl, "sl_hit"
            if c["low"] <= target:
                return c["date"], target, "target_hit"
    return None


# ─── core backtesting engine ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_backtest(df, levels, strategy, rr, orb_min, ib_min, sl_type):

    open_t   = 9 * 60 + 15
    orb_end_t = open_t + orb_min
    ib_end_t  = open_t + ib_min

    trades = []

    for _, lv in levels.iterrows():
        day = lv["date"]
        day_df = df[df["date_only"] == day].copy()
        if day_df.empty:
            continue

        oh, ol, o_rng = lv["orb_high"], lv["orb_low"], lv["orb_range"]
        ih, il, i_rng = lv["ib_high"],  lv["ib_low"],  lv["ib_range"]

        # candles available after formation period
        if strategy in ("orb_long", "orb_short", "orb_both",
                        "failed_orb_long", "failed_orb_short"):
            trade_start_t = orb_end_t
        else:
            # ib_*, orb_ib_* and ib_retest_* all need the full IB to form first
            trade_start_t = ib_end_t

        avail = day_df[day_df["t_min"] >= trade_start_t]
        if avail.empty:
            continue

        def add_trade(bar, entry, direction, ref_range, ref_level):
            sl, target, sl_pts = get_sl_target(entry, direction, ref_range, sl_type, rr)
            result = simulate_trade(bar, entry, direction, sl, target, day_df)
            if result is None:
                return
            exit_time, exit_price, outcome = result
            pnl = (exit_price - entry) if direction == "long" else (entry - exit_price)
            trades.append({
                "date":        day,
                "strategy":    strategy,
                "direction":   direction,
                "entry_time":  bar,
                "entry":       entry,
                "sl":          sl,
                "target":      target,
                "sl_pts":      sl_pts,
                "ref_level":   ref_level,
                "exit_time":   exit_time,
                "exit_price":  exit_price,
                "outcome":     outcome,
                "pnl":         round(pnl, 2),
                "win":         pnl > 0,
            })

        # ── state flags ───────────────────────────────────────────────────────
        orb_long_done   = False
        orb_short_done  = False
        ib_long_done    = False
        ib_short_done   = False
        orb_h_broken    = False   # for failed-ORB
        orb_l_broken    = False
        ib_h_broken     = False   # for IB retest
        ib_l_broken     = False
        retest_done_l   = False
        retest_done_s   = False

        for _, row in avail.iterrows():
            h, l, bar = row["high"], row["low"], row["date"]

            # early exit once all expected trades for both-side strategies are done
            if strategy == "orb_both" and orb_long_done and orb_short_done:
                break
            if strategy == "ib_both" and ib_long_done and ib_short_done:
                break

            # ── ORB Long ──────────────────────────────────────────────────────
            if strategy in ("orb_long", "orb_both") and not orb_long_done:
                if h > oh:
                    orb_long_done = True
                    add_trade(bar, oh, "long", o_rng, oh)
                    if strategy == "orb_long":
                        break

            # ── ORB Short ─────────────────────────────────────────────────────
            if strategy in ("orb_short", "orb_both") and not orb_short_done:
                if l < ol:
                    orb_short_done = True
                    add_trade(bar, ol, "short", o_rng, ol)
                    if strategy == "orb_short":
                        break
                    if strategy == "orb_both" and orb_long_done:
                        break

            # ── IB Long ───────────────────────────────────────────────────────
            if strategy in ("ib_long", "ib_both") and not ib_long_done:
                if h > ih:
                    ib_long_done = True
                    add_trade(bar, ih, "long", i_rng, ih)
                    if strategy == "ib_long":
                        break

            # ── IB Short ──────────────────────────────────────────────────────
            if strategy in ("ib_short", "ib_both") and not ib_short_done:
                if l < il:
                    ib_short_done = True
                    add_trade(bar, il, "short", i_rng, il)
                    if strategy == "ib_short":
                        break
                    if strategy == "ib_both" and ib_long_done:
                        break

            # ── ORB confirmed by IB — Long (ORB high AND IB high both broken) ─
            if strategy == "orb_ib_long" and not orb_long_done:
                if h > oh and h > ih:
                    orb_long_done = True
                    entry = max(oh, ih)
                    add_trade(bar, entry, "long", o_rng, entry)
                    break

            # ── ORB confirmed by IB — Short ───────────────────────────────────
            if strategy == "orb_ib_short" and not orb_short_done:
                if l < ol and l < il:
                    orb_short_done = True
                    entry = min(ol, il)
                    add_trade(bar, entry, "short", o_rng, entry)
                    break

            # ── Failed ORB Reversal — Short ───────────────────────────────────
            # Price breaks ORB High (fake out) then falls below ORB Low → short
            if strategy == "failed_orb_short":
                if not orb_h_broken and h > oh:
                    orb_h_broken = True
                if orb_h_broken and not orb_short_done and l < ol:
                    orb_short_done = True
                    add_trade(bar, ol, "short", o_rng, ol)
                    break

            # ── Failed ORB Reversal — Long ────────────────────────────────────
            # Price breaks ORB Low (fake out) then rises above ORB High → long
            if strategy == "failed_orb_long":
                if not orb_l_broken and l < ol:
                    orb_l_broken = True
                if orb_l_broken and not orb_long_done and h > oh:
                    orb_long_done = True
                    add_trade(bar, oh, "long", o_rng, oh)
                    break

            # ── IB Retest Long ────────────────────────────────────────────────
            # IB High broken, price pulls back close to IB High, then bounces
            if strategy == "ib_retest_long":
                if not ib_h_broken and h > ih:
                    ib_h_broken = True
                if ib_h_broken and not retest_done_l:
                    zone = max(i_rng * 0.12, 20)
                    if abs(l - ih) <= zone:      # price touched IB High area
                        retest_done_l = True
                        add_trade(bar, ih, "long", i_rng * 0.5, ih)
                        break

            # ── IB Retest Short ───────────────────────────────────────────────
            if strategy == "ib_retest_short":
                if not ib_l_broken and l < il:
                    ib_l_broken = True
                if ib_l_broken and not retest_done_s:
                    zone = max(i_rng * 0.12, 20)
                    if abs(h - il) <= zone:
                        retest_done_s = True
                        add_trade(bar, il, "short", i_rng * 0.5, il)
                        break

    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ─── metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    pnl  = trades["pnl"]
    wins = trades["win"]

    cum   = pnl.cumsum()
    dd    = cum - cum.cummax()
    gp    = pnl[pnl > 0].sum()
    gl    = pnl[pnl < 0].abs().sum()
    pf    = gp / gl if gl > 0 else float("inf")

    return {
        "total_trades":  len(trades),
        "win_rate":       wins.mean(),
        "net_pnl":        pnl.sum(),
        "max_drawdown":   dd.min(),
        "profit_factor":  pf,
        "avg_win":        pnl[pnl > 0].mean() if wins.any() else 0,
        "avg_loss":       pnl[pnl < 0].mean() if (~wins).any() else 0,
        "best_trade":     pnl.max(),
        "worst_trade":    pnl.min(),
        "avg_trade":      pnl.mean(),
        "winners":        int(wins.sum()),
        "losers":         int((~wins).sum()),
        "cumulative":     cum,
        "drawdown":       dd,
    }


# ─── chart helpers ────────────────────────────────────────────────────────────

def equity_chart(trades: pd.DataFrame, title: str):
    t = trades.sort_values("exit_time")
    cum = t["pnl"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t["exit_time"], y=cum,
        mode="lines", name="Equity",
        line=dict(color="#2196F3", width=2),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.10)",
    ))
    fig.update_layout(title=title, xaxis_title="Date",
                      yaxis_title="Cumulative PnL (pts)",
                      height=320, template="plotly_white",
                      margin=dict(t=40, b=30))
    return fig


def drawdown_chart(trades: pd.DataFrame, title: str):
    t   = trades.sort_values("exit_time")
    cum = t["pnl"].cumsum()
    dd  = cum - cum.cummax()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t["exit_time"], y=dd,
        mode="lines", name="Drawdown",
        line=dict(color="#F44336", width=1.5),
        fill="tozeroy", fillcolor="rgba(244,67,54,0.12)",
    ))
    fig.update_layout(title=title, xaxis_title="Date",
                      yaxis_title="Drawdown (pts)",
                      height=220, template="plotly_white",
                      margin=dict(t=40, b=30))
    return fig


def monthly_chart(trades: pd.DataFrame, title: str):
    t = trades.copy()
    t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M").astype(str)
    monthly = t.groupby("month")["pnl"].sum().reset_index()
    colors  = ["#4CAF50" if v >= 0 else "#F44336" for v in monthly["pnl"]]
    fig = go.Figure(go.Bar(x=monthly["month"], y=monthly["pnl"],
                           marker_color=colors))
    fig.update_layout(title=title, height=300, template="plotly_white",
                      xaxis_tickangle=-45, margin=dict(t=40, b=60))
    return fig


# ─── strategy description table ───────────────────────────────────────────────

STRATEGY_DESCRIPTIONS = {
    "orb_long":         "Buy when price breaks above the ORB high. SL at ORB low.",
    "orb_short":        "Sell when price breaks below the ORB low. SL at ORB high.",
    "orb_both":         "Take whichever ORB side breaks first each day.",
    "ib_long":          "Buy when price breaks above the IB (60-min) high.",
    "ib_short":         "Sell when price breaks below the IB low.",
    "ib_both":          "Take whichever IB side breaks first each day.",
    "orb_ib_long":      "Buy only when BOTH ORB High and IB High are broken simultaneously — strong bull filter.",
    "orb_ib_short":     "Sell only when BOTH ORB Low and IB Low are broken simultaneously — strong bear filter.",
    "failed_orb_long":  "ORB Low breaks first (fake-out bear), then price reclaims ORB High → long entry.",
    "failed_orb_short": "ORB High breaks first (fake-out bull), then price drops below ORB Low → short entry.",
    "ib_retest_long":   "IB High is broken, price retests that level from above → long (breakout retest).",
    "ib_retest_short":  "IB Low is broken, price retests that level from below → short (breakout retest).",
}


# ─── main app ─────────────────────────────────────────────────────────────────

def main():
    st.title("ORB & IB Strategy Backtester")
    st.caption("Opening Range Breakout · Initial Balance · 12 strategy permutations · NIFTY 50 & BANK NIFTY")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    import os

    with st.sidebar:
        st.header("Data")
        auto = {k: v for k, v in DEFAULT_PATHS.items() if os.path.exists(v)}
        if auto:
            st.success("Auto-loaded: " + ", ".join(auto.keys()))
            st.caption("Upload below to override with your own files.")
        nifty_file  = st.file_uploader("NIFTY 50 (minute CSV)",    type="csv", key="n50")
        bnifty_file = st.file_uploader("BANK NIFTY (minute CSV)",  type="csv", key="bnf")

        st.header("Strategy")
        strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()))
        strategy_key  = STRATEGIES[strategy_name]
        instrument    = st.radio("Instrument", ["NIFTY 50", "BANK NIFTY", "Both"], horizontal=True)

        st.header("Parameters")
        rr         = st.slider("Risk : Reward", 1.0, 5.0, 2.0, 0.5)
        orb_min    = st.slider("ORB duration (min)", 5, 30, 15, 5)
        ib_min     = st.slider("IB duration (min)",  30, 120, 60, 15)
        sl_label   = st.selectbox("Stop Loss type", list(SL_OPTIONS.keys()))
        sl_type    = SL_OPTIONS[sl_label]

        st.header("Date Filter")
        use_filter = st.checkbox("Enable date filter")

        run = st.button("▶  Run Backtest", type="primary", use_container_width=True)

    # ── Load files (uploads take priority, else auto-load from C:\) ───────────
    available = {}
    # 1) auto-load any default CSVs that exist on disk
    for name, path in DEFAULT_PATHS.items():
        if os.path.exists(path):
            try:
                available[name] = load_csv_path(path)
            except Exception as e:
                st.sidebar.error(f"Could not auto-load {name}: {e}")
    # 2) uploads override auto-loaded data
    if nifty_file:
        available["NIFTY 50"]   = load_csv(nifty_file.read(),  nifty_file.name)
    if bnifty_file:
        available["BANK NIFTY"] = load_csv(bnifty_file.read(), bnifty_file.name)

    if not available:
        _show_landing()
        return

    # ── Date filter UI (needs data to know range) ─────────────────────────────
    date_range = None
    if use_filter:
        all_dates = pd.concat(
            [df["date_only"].apply(pd.Timestamp) for df in available.values()]
        )
        d_min = all_dates.min().date()
        d_max = all_dates.max().date()
        with st.sidebar:
            date_range = st.date_input("Date range", value=(d_min, d_max),
                                       min_value=d_min, max_value=d_max)

    # ── Show data summary before running ─────────────────────────────────────
    if not run:
        _show_data_summary(available)
        return

    # ── Select instruments ────────────────────────────────────────────────────
    if instrument == "Both":
        targets = list(available.keys())
    else:
        targets = [instrument] if instrument in available else list(available.keys())

    # ── Run backtests ─────────────────────────────────────────────────────────
    for inst in targets:
        df = available[inst].copy()

        if date_range and len(date_range) == 2:
            s, e = pd.Timestamp(date_range[0]).date(), pd.Timestamp(date_range[1]).date()
            df = df[(df["date_only"] >= s) & (df["date_only"] <= e)]

        if df.empty:
            st.warning(f"No data for {inst} in the selected date range.")
            continue

        with st.spinner(f"Computing ORB/IB levels for {inst}…"):
            levels = compute_levels(df, orb_min, ib_min)

        with st.spinner(f"Running backtest for {inst} — {strategy_name}…"):
            trades = run_backtest(df, levels, strategy_key, rr, orb_min, ib_min, sl_type)

        _display_results(inst, strategy_name, strategy_key, trades, levels)


# ─── landing page ─────────────────────────────────────────────────────────────

def _show_landing():
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Expected CSV format")
        st.code(
            "date,open,high,low,close,volume\n"
            "2015-01-09 09:15:00,8285.45,8295.9,8285.45,8292.1,0\n"
            "2015-01-09 09:16:00,8292.6,8293.6,8287.2,8288.15,0\n"
            "…",
            language="text"
        )
    with c2:
        st.subheader("Strategy guide")
        rows = [{"Strategy": k, "Logic": STRATEGY_DESCRIPTIONS[v]}
                for k, v in STRATEGIES.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _show_data_summary(available):
    st.subheader("Data loaded — configure & click **▶ Run Backtest**")
    for name, df in available.items():
        with st.expander(f"{name}  ({len(df):,} candles)", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Trading Days", f"{df['date_only'].nunique():,}")
            c2.metric("From", str(df['date_only'].min()))
            c3.metric("To",   str(df['date_only'].max()))
            st.dataframe(df.head(8)[["date","open","high","low","close","volume"]],
                         use_container_width=True, hide_index=True)


# ─── result display ───────────────────────────────────────────────────────────

def _display_results(inst, strategy_name, strategy_key, trades, levels):
    st.divider()
    st.subheader(f"{inst}  ·  {strategy_name}")
    st.caption(STRATEGY_DESCRIPTIONS.get(strategy_key, ""))

    if trades.empty:
        st.warning("No trades were generated. Try a different strategy, date range, or SL type.")
        return

    m = compute_metrics(trades)

    # ── Metric cards row 1 ────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    wr = m["win_rate"]
    c1.metric("Total Trades",  f"{m['total_trades']:,}")
    c2.metric("Win Rate",      f"{wr*100:.1f}%",
              delta=f"{(wr-0.5)*100:+.1f}% vs 50%",
              delta_color="normal")
    c3.metric("Net PnL (pts)", f"{m['net_pnl']:,.0f}",
              delta_color="normal")
    c4.metric("Max Drawdown",  f"{m['max_drawdown']:,.0f}")
    c5.metric("Profit Factor", f"{m['profit_factor']:.2f}")
    c6.metric("Avg Trade",     f"{m['avg_trade']:,.1f}")

    # ── Metric cards row 2 ────────────────────────────────────────────────────
    c7, c8, c9, c10, c11, c12 = st.columns(6)
    c7.metric("Winners",    f"{m['winners']}")
    c8.metric("Losers",     f"{m['losers']}")
    c9.metric("Avg Win",    f"{m['avg_win']:,.1f}")
    c10.metric("Avg Loss",  f"{m['avg_loss']:,.1f}")
    c11.metric("Best Trade",  f"{m['best_trade']:,.1f}")
    c12.metric("Worst Trade", f"{m['worst_trade']:,.1f}")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.plotly_chart(equity_chart(trades,   f"{inst} — Equity Curve"),
                    use_container_width=True)
    st.plotly_chart(drawdown_chart(trades, f"{inst} — Drawdown"),
                    use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        fig_hist = px.histogram(
            trades, x="pnl", nbins=40,
            color_discrete_sequence=["#4CAF50"],
            title="PnL Distribution",
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        fig_hist.update_layout(height=280, template="plotly_white",
                               margin=dict(t=40, b=30))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        oc = trades["outcome"].value_counts()
        fig_pie = px.pie(
            values=oc.values, names=oc.index,
            title="Exit Breakdown",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.plotly_chart(monthly_chart(trades, f"{inst} — Monthly PnL (pts)"),
                    use_container_width=True)

    # ── Win rate by direction ─────────────────────────────────────────────────
    if trades["direction"].nunique() > 1:
        wr_dir = (
            trades.groupby("direction")
            .agg(trades=("win", "count"), win_rate=("win", "mean"), net_pnl=("pnl", "sum"))
            .reset_index()
        )
        wr_dir["win_rate"] = (wr_dir["win_rate"] * 100).round(1).astype(str) + "%"
        wr_dir["net_pnl"]  = wr_dir["net_pnl"].round(0)
        st.dataframe(wr_dir, use_container_width=True, hide_index=True)

    # ── ORB/IB range stats ────────────────────────────────────────────────────
    with st.expander("ORB / IB Level Statistics"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**ORB range distribution**")
            st.dataframe(
                levels["orb_range"].describe().rename("pts").round(1).to_frame(),
                use_container_width=True,
            )
        with col_b:
            st.markdown("**IB range distribution**")
            st.dataframe(
                levels["ib_range"].describe().rename("pts").round(1).to_frame(),
                use_container_width=True,
            )

    # ── Trade log ─────────────────────────────────────────────────────────────
    with st.expander(f"Trade Log  ({len(trades)} trades)"):
        cols = ["date", "direction", "entry_time", "entry", "sl", "target",
                "exit_time", "exit_price", "outcome", "pnl", "win"]
        st.dataframe(
            trades[cols].sort_values("entry_time", ascending=False),
            use_container_width=True, hide_index=True,
        )

    # ── Download ──────────────────────────────────────────────────────────────
    csv = trades.to_csv(index=False).encode()
    st.download_button(
        f"⬇ Download {inst} trade log",
        data=csv,
        file_name=f"{inst.replace(' ','_')}_{strategy_key}_trades.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
