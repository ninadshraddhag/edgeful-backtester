"""
ORB / IB Edge Backtester  +  Permutation Optimizer
===================================================
Trades the probability EDGES discovered in the Probability Explorer:

  • First-move-fade   : whichever extreme forms first, the OPPOSITE side
                        breaks ~75% (IB) / ~85% (ORB)  → fade it
  • Gap continuation  : Gap Up → reach PDH 77% ,  Gap Down → reach PDL 83%
  • Plain breakout    : long on range-high break / short on range-low break
  • + filters: day-of-week, gap type, inside/outside day, first-side, range size

Targets & stops are expressed in × range  (extension / retracement units),
so 1.0 = one full opening-range width.

The OPTIMIZER sweeps permutations of {side-logic × filters × target × stop}
and ranks them by expectancy / win-rate / profit-factor, with a min-trade
guard, to surface the highest-probability configurations.

Data:  C:\\NIFTY 50_minute.csv , C:\\NIFTY BANK_minute.csv  (auto-loaded)
       analysis/facts.csv  (build_facts.py) — for the day classifications
"""
import os, itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import daywise
import prob_app
import build_facts
from datetime import time as dtime

st.set_page_config(page_title="ORB/IB Strategy Suite", page_icon="📈", layout="wide")

HERE  = os.path.dirname(os.path.abspath(__file__))
FACTS = os.path.join(HERE, "analysis", "facts.csv")
PATHS = {
    "NIFTY 50":   r"C:\NIFTY 50_minute.csv",
    "BANK NIFTY": r"C:\NIFTY BANK_minute.csv",
}

DEFAULT_OPEN_T  = 9 * 60 + 15    # 09:15 (NSE); auto-detected per instrument
DEFAULT_CLOSE_T = 15 * 60 + 15   # 15:15 square-off — no positions/levels carry overnight
T_GRID   = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]      # target (× range)
S_GRID   = [0.25, 0.5, 0.75, 1.0]                # stop   (× range)
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

SIDE_LOGICS = {
    "First-move fade":   "fade",     # high-first → short, low-first → long
    "Long @ high break": "long",
    "Short @ low break": "short",
    "Gap continuation":  "gap",      # gap up → long, gap down → short, flat skip
}


# ─── data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_min(path, mtime):
    return build_facts.clean_min(pd.read_csv(path))


@st.cache_data(show_spinner=False)
def load_facts(mtime):
    df = pd.read_csv(FACTS, parse_dates=["date"])
    df["day_kind"] = np.where(df["inside_day"], "Inside Day",
                     np.where(df["outside_day"], "Outside Day", "Normal"))
    return df


@st.cache_data(show_spinner=False)
def get_open_t(mpath, mtime):
    """Auto-detect the session-open minute for any instrument."""
    return daywise.detect_open_t(load_min(mpath, mtime))


@st.cache_data(show_spinner=False)
def get_date_bounds(mpath, mtime):
    d = load_min(mpath, mtime)["date_only"]
    return d.min(), d.max()


@st.cache_data(show_spinner=True)
def get_facts(instrument, mpath, mtime, open_t, close_t):
    """
    Per-day facts for ANY instrument & session. Uses the prebuilt facts.csv for the
    default indices at the default 09:15/15:15 session; otherwise computes live.
    """
    if (instrument in PATHS and open_t == DEFAULT_OPEN_T
            and close_t == DEFAULT_CLOSE_T and os.path.exists(FACTS)):
        f = load_facts(os.path.getmtime(FACTS))
        sub = f[f["instrument"] == instrument].copy()
        if len(sub):
            return sub
    facts = build_facts.build_from_minute(load_min(mpath, mtime), instrument, open_t, close_t)
    facts["day_kind"] = np.where(facts["inside_day"], "Inside Day",
                        np.where(facts["outside_day"], "Outside Day", "Normal"))
    return facts


# ─── first-passage cache (the engine) ─────────────────────────────────────────

@st.cache_data(show_spinner=True)
def build_passage(instrument, tag, mpath, mtime, open_t, close_t):
    """
    Per-day first-passage table: for a long-on-high-break and a short-on-low-break,
    the minute each target/stop level (× range) is first reached. This resolves
    target-vs-stop ordering exactly for ANY (target, stop) pair without re-simulating.
    Capped at the square-off (close_t) — nothing carries past it.
    """
    mdf = load_min(mpath, mtime)
    meta = get_facts(instrument, mpath, mtime, open_t, close_t).set_index("date")
    end_t = (open_t + 15 - 1) if tag == "orb" else (open_t + 60 - 1)

    rows = []
    for day, g0 in mdf.groupby("date_only", sort=True):
        sess = g0[(g0["t_min"] >= open_t) & (g0["t_min"] <= close_t)].sort_values("t_min")
        win  = sess[sess["t_min"] <= end_t]
        post = sess[sess["t_min"] > end_t]
        if len(win) < 5 or post.empty:
            continue
        hi, lo = win["high"].max(), win["low"].min()
        rng = hi - lo
        if rng <= 0:
            continue
        close = sess.iloc[-1]["close"]          # close AT square-off
        ph, pl, pt = post["high"].values, post["low"].values, post["t_min"].values

        rec = {"date": pd.Timestamp(day), "range": rng, "day_close": close}

        # LONG: enter at hi when high first exceeds hi
        le = np.where(ph > hi)[0]
        if len(le):
            et = pt[le[0]]
            rec["L_entry"] = hi
            for t in T_GRID:
                idx = np.where((ph >= hi + t * rng) & (pt >= et))[0]
                rec[f"L_T_{t}"] = pt[idx[0]] if len(idx) else np.nan
            for s in S_GRID:
                idx = np.where((pl <= hi - s * rng) & (pt >= et))[0]
                rec[f"L_S_{s}"] = pt[idx[0]] if len(idx) else np.nan
        else:
            rec["L_entry"] = np.nan

        # SHORT: enter at lo when low first breaks lo
        se = np.where(pl < lo)[0]
        if len(se):
            et = pt[se[0]]
            rec["S_entry"] = lo
            for t in T_GRID:
                idx = np.where((pl <= lo - t * rng) & (pt >= et))[0]
                rec[f"S_T_{t}"] = pt[idx[0]] if len(idx) else np.nan
            for s in S_GRID:
                idx = np.where((ph >= lo + s * rng) & (pt >= et))[0]
                rec[f"S_S_{s}"] = pt[idx[0]] if len(idx) else np.nan
        else:
            rec["S_entry"] = np.nan

        rows.append(rec)

    P = pd.DataFrame(rows).set_index("date")
    keep = ["dow", "gap_type", "day_kind", "prev_inside", f"{tag}_first_side",
            "broke_pdh", "broke_pdl"]
    P = P.join(meta[keep]).rename(columns={f"{tag}_first_side": "first_side"})
    return P.reset_index()


# ─── pnl construction ─────────────────────────────────────────────────────────

def side_pnl(P, side, t, s):
    """Per-day pnl (points) for taking `side` every day with target t, stop s."""
    pre = "L" if side == "long" else "S"
    entry = P[f"{pre}_entry"].values
    rng   = P["range"].values
    close = P["day_close"].values
    fpt   = P[f"{pre}_T_{t}"].values
    fps   = P[f"{pre}_S_{s}"].values

    entered = ~np.isnan(entry)
    fpt_i = np.where(np.isnan(fpt), np.inf, fpt)
    fps_i = np.where(np.isnan(fps), np.inf, fps)

    tgt_first  = entered & (fpt_i <= fps_i) & np.isfinite(fpt_i)
    stop_first = entered & (fps_i <  fpt_i) & np.isfinite(fps_i)
    neither    = entered & ~np.isfinite(fpt_i) & ~np.isfinite(fps_i)

    if side == "long":
        time_pnl = close - entry
    else:
        time_pnl = entry - close

    pnl = np.full(len(P), np.nan)
    pnl[tgt_first]  =  t * rng[tgt_first]
    pnl[stop_first] = -s * rng[stop_first]
    pnl[neither]    =  time_pnl[neither]
    return pnl


def logic_pnl(P, logic, t, s):
    """Combine long/short pnl according to the side-logic; nan = no trade that day."""
    lp = side_pnl(P, "long", t, s)
    sp = side_pnl(P, "short", t, s)
    if logic == "long":
        return lp
    if logic == "short":
        return sp
    if logic == "fade":          # high formed first → short, low first → long
        return np.where(P["first_side"].values == "high", sp, lp)
    if logic == "gap":           # gap up → long, gap down → short, else skip
        out = np.full(len(P), np.nan)
        gu = P["gap_type"].values == "Gap Up"
        gd = P["gap_type"].values == "Gap Down"
        out[gu] = lp[gu]
        out[gd] = sp[gd]
        return out
    return lp


# ─── filtering & metrics ──────────────────────────────────────────────────────

def filter_mask(P, dows, gaps, kinds, first):
    m = np.ones(len(P), dtype=bool)
    if dows:  m &= P["dow"].isin(dows).values
    if gaps:  m &= P["gap_type"].isin(gaps).values
    if kinds: m &= P["day_kind"].isin(kinds).values
    if first == "high": m &= (P["first_side"].values == "high")
    if first == "low":  m &= (P["first_side"].values == "low")
    return m


def metrics(pnl_ordered):
    """pnl_ordered: 1-D array of per-trade pnl in date order (no NaNs)."""
    n = len(pnl_ordered)
    if n == 0:
        return None
    wins = pnl_ordered > 0
    gp = pnl_ordered[pnl_ordered > 0].sum()
    gl = -pnl_ordered[pnl_ordered < 0].sum()
    cum = np.cumsum(pnl_ordered)
    dd  = cum - np.maximum.accumulate(cum)
    return {
        "trades": n,
        "win_rate": wins.mean(),
        "expectancy": pnl_ordered.mean(),
        "net": pnl_ordered.sum(),
        "pf": (gp / gl) if gl > 0 else np.inf,
        "max_dd": dd.min(),
        "avg_win": pnl_ordered[wins].mean() if wins.any() else 0,
        "avg_loss": pnl_ordered[~wins].mean() if (~wins).any() else 0,
        "cum": cum,
    }


def evaluate(P, logic, t, s, dows, gaps, kinds, first):
    pnl = logic_pnl(P, logic, t, s)
    mask = filter_mask(P, dows, gaps, kinds, first) & ~np.isnan(pnl)
    return pnl[mask], P.loc[mask, "date"].values


# ─── UI helpers ───────────────────────────────────────────────────────────────

def pct(x): return f"{x*100:.1f}%"

EDGE_NOTE = """
**Edges in play (10-yr, both indices):**
- **First-move fade** — high-first → low breaks ~75% (IB) / ~87% (ORB); symmetric for low-first.
- **Gap continuation** — Gap Up → reach PDH **77%**, Gap Down → reach PDL **83%**.
- **Outside day** — strong downside follow-through (IB low breaks ~78%).
- **Inside-day** — next day breaks PDH **62%** vs PDL 48%.
"""


def resolve_path(instrument, uploaded):
    if uploaded is not None:
        tmp = os.path.join(HERE, "analysis", f"_uploaded_{instrument.replace(' ', '_')}.csv")
        with open(tmp, "wb") as f:
            f.write(uploaded.getbuffer())
        return tmp
    return PATHS.get(instrument)


def data_sidebar(key):
    """
    Shared sidebar block: instrument picker (incl. uploads like XAUUSD / NQ),
    auto-detected session open, square-off time and date range.
    Returns (instrument, mpath, mtime, open_t, close_t, (d0, d1)).
    """
    st.session_state.setdefault("custom_instruments", {})
    st.header("Data")
    names = list(PATHS.keys()) + list(st.session_state["custom_instruments"].keys())
    choice = st.selectbox("Instrument", names + ["➕ Upload new instrument…"],
                          key=f"{key}_inst")

    if choice == "➕ Upload new instrument…":
        nm = st.text_input("Instrument name (e.g. XAUUSD, NQ)", key=f"{key}_nm")
        f  = st.file_uploader("Minute CSV — columns: date, open, high, low, close",
                              type="csv", key=f"{key}_nf")
        if nm and f is not None:
            path = os.path.join(HERE, "analysis", f"_inst_{nm.replace(' ', '_')}.csv")
            with open(path, "wb") as fh:
                fh.write(f.getbuffer())
            st.session_state["custom_instruments"][nm] = path
            st.success(f"Added {nm}. Select it above.")
            st.rerun()
        st.info("Name the instrument and upload its minute CSV to add it.")
        st.stop()

    mpath = PATHS.get(choice) or st.session_state["custom_instruments"].get(choice)
    if not mpath or not os.path.exists(mpath):
        st.warning(f"No data file for {choice}.")
        st.stop()
    mtime = os.path.getmtime(mpath)

    open_t = get_open_t(mpath, mtime)
    sq = st.time_input("Square-off (force exit)", value=dtime(15, 15), key=f"{key}_sq")
    close_t = sq.hour * 60 + sq.minute
    st.caption(f"Session open auto-detected **{open_t//60:02d}:{open_t%60:02d}** · "
               f"IB = first 60 min · square-off **{sq.strftime('%H:%M')}** (no overnight carry)")

    dmin, dmax = get_date_bounds(mpath, mtime)
    dr = st.date_input("Date range", value=(dmin, dmax),
                       min_value=dmin, max_value=dmax, key=f"{key}_dates")
    d0, d1 = (dr if isinstance(dr, (tuple, list)) and len(dr) == 2 else (dmin, dmax))
    return choice, mpath, mtime, open_t, close_t, (d0, d1)


def _filter_prepped(prepped, d0, d1):
    return {dow: [r for r in lst if d0 <= r["date"].date() <= d1]
            for dow, lst in prepped.items()}


# ─── day-wise IB retracement mode ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def prep_daywise(instrument, mpath, mtime, open_t, close_t):
    mdf = load_min(mpath, mtime)
    return daywise.prep_days(mdf, open_t, close_t)


DW_FIELDS = ["trade", "entry", "stop", "tp1", "tp2", "min_size", "max_size",
             "allow_long", "allow_short", "cutoff_on", "cutoff_t"]


def _dw_init_state():
    """Seed per-weekday widget state from DEFAULT_CONFIG once."""
    for day in daywise.WEEKDAYS:
        d = daywise.DEFAULT_CONFIG[day]
        for f in ["trade", "entry", "stop", "tp1", "tp2", "min_size",
                  "max_size", "allow_long", "allow_short"]:
            st.session_state.setdefault(f"dw_{day}_{f}", d[f])
        st.session_state.setdefault(f"dw_{day}_cutoff_on", d["cutoff"] is not None)
        st.session_state.setdefault(f"dw_{day}_cutoff_t",
                                    dtime(d["cutoff"] // 60, d["cutoff"] % 60)
                                    if d["cutoff"] else dtime(13, 0))


def _dw_read_config():
    cfg = {}
    for day in daywise.WEEKDAYS:
        g = lambda f: st.session_state[f"dw_{day}_{f}"]
        cutoff = None
        if g("cutoff_on"):
            t = g("cutoff_t")
            cutoff = t.hour * 60 + t.minute
        cfg[day] = dict(trade=g("trade"), entry=g("entry"), stop=g("stop"),
                        tp1=g("tp1"), tp2=g("tp2"), min_size=g("min_size"),
                        max_size=g("max_size"), allow_long=g("allow_long"),
                        allow_short=g("allow_short"), cutoff=cutoff)
    return cfg


def _dw_weekday_panel(day):
    d = st.session_state
    top = st.columns([1.2, 1, 1])
    top[0].toggle("Trade this day", key=f"dw_{day}_trade")
    top[1].toggle("Allow long", key=f"dw_{day}_allow_long")
    top[2].toggle("Allow short", key=f"dw_{day}_allow_short")

    r1 = st.columns(4)
    r1[0].number_input("Entry retr %", 0, 100, key=f"dw_{day}_entry", step=5,
                       help="Retracement entry level. 0 = enter at the IB boundary (breakout).")
    r1[1].number_input("Stop retr %", 0, 200, key=f"dw_{day}_stop", step=5,
                       help="Deeper retracement. Must be greater than entry %. 100 = opposite IB boundary.")
    r1[2].number_input("Target 1 (ext %)", 0, 500, key=f"dw_{day}_tp1", step=5,
                       help="First profit target, extension beyond the boundary. Half position exits here.")
    r1[3].number_input("Target 2 (ext %)", 0, 500, key=f"dw_{day}_tp2", step=5,
                       help="Second target for the runner (stop moves to breakeven after TP1).")

    r2 = st.columns(4)
    r2[0].number_input("Min IB size %", 0.0, 10.0, key=f"dw_{day}_min_size", step=0.1,
                       help="IB range as % of price (range ÷ open × 100).")
    r2[1].number_input("Max IB size %", 0.0, 10.0, key=f"dw_{day}_max_size", step=0.1)
    r2[2].toggle("Entry cutoff", key=f"dw_{day}_cutoff_on")
    r2[3].time_input("Cutoff time", key=f"dw_{day}_cutoff_t",
                     disabled=not d[f"dw_{day}_cutoff_on"])

    if d[f"dw_{day}_stop"] <= d[f"dw_{day}_entry"]:
        st.warning("Stop % must be greater than entry % — this day will be skipped.")


def daywise_mode():
    st.caption("Edgeful-style day-wise IB retracement · entry/stop in % of IB range "
               "(retracement) · targets in % of IB range (extension) · half at TP1, "
               "runner to TP2 with stop at breakeven")
    _dw_init_state()

    with st.sidebar:
        instrument, mpath, mtime, open_t, close_t, (d0, d1) = data_sidebar("dw")

    with st.spinner(f"Indexing {instrument} minute data (first run is cached)…"):
        prepped = prep_daywise(instrument, mpath, mtime, open_t, close_t)
    prepped = _filter_prepped(prepped, d0, d1)
    if not any(prepped.values()):
        st.warning("No trading days in the selected date range.")
        st.stop()

    # ── per-weekday config panels ─────────────────────────────────────────────
    st.subheader("Day-wise configuration")
    cc = st.columns([1, 1, 3])
    if cc[0].button("📋 Copy Monday → all days", use_container_width=True):
        src = {f: st.session_state[f"dw_Monday_{f}"] for f in
               ["entry", "stop", "tp1", "tp2", "min_size", "max_size",
                "allow_long", "allow_short", "trade", "cutoff_on", "cutoff_t"]}
        for day in daywise.WEEKDAYS:
            for f, v in src.items():
                st.session_state[f"dw_{day}_{f}"] = v
        st.rerun()
    if cc[1].button("↺ Reset to defaults", use_container_width=True):
        for day in daywise.WEEKDAYS:
            for f in DW_FIELDS:
                st.session_state.pop(f"dw_{day}_{f}", None)
        st.rerun()

    for day in daywise.WEEKDAYS:
        cfg = _dw_read_config()[day]
        flags = []
        if not cfg["trade"]: flags.append("off")
        if cfg["allow_long"]: flags.append("long")
        if cfg["allow_short"]: flags.append("short")
        label = f"{day}  ·  entry {cfg['entry']}% / stop {cfg['stop']}% · " \
                f"TP {cfg['tp1']}%/{cfg['tp2']}% · {'/'.join(flags) or 'no direction'}"
        with st.expander(label, expanded=(day == "Monday")):
            _dw_weekday_panel(day)

    st.divider()
    run = st.button("▶ Run day-wise backtest", type="primary")

    if run:
        configs = _dw_read_config()
        trades = daywise.run(prepped, configs)
        _dw_show_results(instrument, trades)

    # ── per-weekday optimizer ─────────────────────────────────────────────────
    st.divider()
    with st.expander("🔬 Per-weekday optimizer — find the best config for one weekday"):
        oc = st.columns(4)
        opt_day = oc[0].selectbox("Weekday", daywise.WEEKDAYS, key="dw_opt_day")
        opt_metric = oc[1].selectbox("Rank by",
                                     ["expectancy", "win_rate", "net", "pf"], key="dw_opt_metric")
        opt_min = oc[2].number_input("Min trades", 20, 1000, 100, 10, key="dw_opt_min")
        opt_dirs = oc[3].multiselect("Directions", ["long", "short"],
                                     default=["long", "short"], key="dw_opt_dirs")
        if st.button("🔎 Optimize this weekday", key="dw_opt_run"):
            if not opt_dirs:
                st.warning("Pick at least one direction.")
            else:
                with st.spinner(f"Sweeping {opt_day} configurations…"):
                    res = daywise.optimize_weekday(
                        prepped[opt_day], opt_dirs, daywise.OPT_GRID,
                        opt_metric, int(opt_min))
                if res.empty:
                    st.warning("No configs met the minimum-trades threshold.")
                else:
                    st.session_state["dw_opt_result"] = (opt_day, res)

        if "dw_opt_result" in st.session_state:
            od, res = st.session_state["dw_opt_result"]
            st.markdown(f"**Top configs for {od}** (showing 20)")
            st.dataframe(res.head(20), use_container_width=True, hide_index=True)
            best = res.iloc[0]
            if st.button(f"✅ Apply best {od} config to the panel above"):
                st.session_state[f"dw_{od}_entry"] = int(best["entry"])
                st.session_state[f"dw_{od}_stop"] = int(best["stop"])
                st.session_state[f"dw_{od}_tp1"] = int(best["tp1"])
                st.session_state[f"dw_{od}_tp2"] = int(best["tp2"])
                st.session_state[f"dw_{od}_allow_long"] = best["direction"] == "long"
                st.session_state[f"dw_{od}_allow_short"] = best["direction"] == "short"
                st.session_state[f"dw_{od}_trade"] = True
                del st.session_state["dw_opt_result"]
                st.rerun()


def _dw_show_results(instrument, trades):
    st.divider()
    if trades.empty:
        st.warning("No trades generated. Check that days are enabled, a direction is "
                   "allowed, and the IB-size filter isn't excluding everything.")
        return

    pnl = trades.sort_values("date")["pnl"].values
    m = metrics(pnl)
    st.subheader(f"{instrument} · day-wise results")

    k = st.columns(6)
    k[0].metric("Win Rate", pct(m["win_rate"]))
    k[1].metric("Max Drawdown", f"{m['max_dd']:,.0f} pts")
    k[2].metric("Net P&L", f"{m['net']:,.0f} pts")
    k[3].metric("Expectancy", f"{m['expectancy']:.2f} pts/trade")
    k[4].metric("Profit Factor", f"{m['pf']:.2f}" if np.isfinite(m["pf"]) else "∞")
    k[5].metric("Trades", f"{m['trades']:,}")

    # equity curve
    t = trades.sort_values("date")
    fig = go.Figure()
    fig.add_scatter(x=pd.to_datetime(t["date"]), y=np.cumsum(t["pnl"].values),
                    mode="lines", line=dict(color="#2196F3", width=2),
                    fill="tozeroy", fillcolor="rgba(33,150,243,0.10)")
    fig.update_layout(title="Equity curve (points, 1 unit / trade)", height=340,
                      template="plotly_white", yaxis_title="cumulative pts")
    st.plotly_chart(fig, use_container_width=True)

    # per-weekday breakdown
    wd = (trades.groupby("dow")
          .agg(trades=("pnl", "size"), win_rate=("win", "mean"),
               net=("pnl", "sum"), expectancy=("pnl", "mean"))
          .reindex([d for d in daywise.WEEKDAYS if d in trades["dow"].unique()]))
    wd["win_rate"] = (wd["win_rate"] * 100).round(1)
    wd[["net", "expectancy"]] = wd[["net", "expectancy"]].round(2)
    g1, g2 = st.columns([1, 1])
    with g1:
        st.markdown("**By weekday**")
        st.dataframe(wd, use_container_width=True)
    with g2:
        st.markdown("**Exit breakdown**")
        oc = trades["outcome"].value_counts()
        figp = go.Figure(go.Bar(x=oc.values, y=oc.index, orientation="h",
                                marker_color="#4CAF50"))
        figp.update_layout(height=260, template="plotly_white", margin=dict(t=10, b=10))
        st.plotly_chart(figp, use_container_width=True)

    with st.expander(f"Trade log ({len(trades):,})"):
        st.dataframe(trades.sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)
    st.download_button("⬇ Download trades (CSV)", trades.to_csv(index=False).encode(),
                       file_name=f"daywise_{instrument.replace(' ', '_')}.csv",
                       mime="text/csv")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(FACTS):
        st.title("📈 ORB / IB Strategy Suite")
        st.error("analysis/facts.csv missing — run `python build_facts.py` first.")
        st.stop()

    mode = st.sidebar.radio(
        "Mode",
        ["Edge Backtester", "Day-wise IB Retracement", "Probabilities"],
        key="app_mode")
    st.sidebar.divider()
    if mode == "Probabilities":
        prob_app.render()
        return

    st.title("📈 ORB / IB Strategy Suite")
    if mode == "Day-wise IB Retracement":
        daywise_mode()
        return

    st.caption("Trades the probability edges · targets & stops in × opening-range · "
               "permutation optimizer to find the best configuration")

    with st.sidebar:
        instrument, mpath, mtime, open_t, close_t, (d0, d1) = data_sidebar("edge")
        setup = st.radio("Setup", ["IB (60 min)", "ORB (15 min)"])
        tag = "ib" if setup.startswith("IB") else "orb"
        st.markdown(EDGE_NOTE)

    with st.spinner(f"Indexing {instrument} {setup} (first run is cached)…"):
        P = build_passage(instrument, tag, mpath, mtime, open_t, close_t)

    P = P[(P["date"] >= pd.Timestamp(d0)) & (P["date"] <= pd.Timestamp(d1))].reset_index(drop=True)
    if P.empty:
        st.warning("No trading days in the selected date range.")
        st.stop()

    tab1, tab2 = st.tabs(["🎯 Single Strategy", "🔬 Optimizer (permutations)"])

    # ============================ SINGLE STRATEGY ============================
    with tab1:
        c = st.columns(4)
        logic_label = c[0].selectbox("Side logic", list(SIDE_LOGICS.keys()))
        logic = SIDE_LOGICS[logic_label]
        t = c[1].select_slider("Target (× range)", T_GRID, value=1.0)
        s = c[2].select_slider("Stop (× range)",   S_GRID, value=0.5)
        first = c[3].selectbox("First-side filter", ["Either", "high", "low"])

        c2 = st.columns(3)
        dows  = c2[0].multiselect("Day of week", DOW_ORDER, default=DOW_ORDER)
        gaps  = c2[1].multiselect("Gap type", ["Gap Up", "Flat", "Gap Down"],
                                  default=["Gap Up", "Flat", "Gap Down"])
        kinds = c2[2].multiselect("Day kind", ["Inside Day", "Normal", "Outside Day"],
                                  default=["Inside Day", "Normal", "Outside Day"])

        first_f = "" if first == "Either" else first
        pnl, dates = evaluate(P, logic, t, s, dows, gaps, kinds, first_f)
        m = metrics(pnl)

        st.divider()
        if m is None:
            st.warning("No trades for this configuration.")
        else:
            if m["trades"] < 30:
                st.warning(f"⚠ Only {m['trades']} trades — low confidence.")
            k = st.columns(6)
            k[0].metric("Win Rate", pct(m["win_rate"]))
            k[1].metric("Max Drawdown", f"{m['max_dd']:,.0f} pts")
            k[2].metric("Expectancy", f"{m['expectancy']:.1f} pts/trade")
            k[3].metric("Net P&L", f"{m['net']:,.0f} pts")
            k[4].metric("Profit Factor", f"{m['pf']:.2f}" if np.isfinite(m["pf"]) else "∞")
            k[5].metric("Trades", f"{m['trades']:,}")

            k2 = st.columns(3)
            k2[0].metric("Avg Win", f"{m['avg_win']:.1f} pts")
            k2[1].metric("Avg Loss", f"{m['avg_loss']:.1f} pts")
            rr = (m["avg_win"] / abs(m["avg_loss"])) if m["avg_loss"] else 0
            k2[2].metric("Reward : Risk (realised)", f"{rr:.2f}")

            order = np.argsort(dates)
            fig = go.Figure()
            fig.add_scatter(x=pd.to_datetime(dates[order]), y=np.cumsum(pnl[order]),
                            mode="lines", line=dict(color="#2196F3", width=2),
                            fill="tozeroy", fillcolor="rgba(33,150,243,0.10)")
            fig.update_layout(title=f"Equity curve — {logic_label} · target {t}× / stop {s}×",
                              height=360, template="plotly_white",
                              yaxis_title="cumulative pts")
            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"{instrument} · {setup} · {logic_label} · "
                       f"target {t}× / stop {s}× (R:R {t/s:.1f}) · {m['trades']:,} trades")

    # ============================ OPTIMIZER ============================
    with tab2:
        st.markdown("Sweep permutations and rank by your chosen metric. "
                    "Each row is a fully-specified, tradeable rule.")
        o = st.columns(4)
        sweep_logics = o[0].multiselect("Side logics", list(SIDE_LOGICS.keys()),
                                        default=list(SIDE_LOGICS.keys()))
        SORT_COLS = {"Expectancy": "expectancy", "Net P&L": "net",
                     "Win rate": "win %", "Profit factor": "PF"}
        sort_label  = o[1].selectbox("Rank by", list(SORT_COLS.keys()))
        sort_metric = SORT_COLS[sort_label]
        min_trades = o[2].number_input("Min trades", 20, 2000, 100, 10)
        sweep_dow  = o[3].checkbox("Also sweep each weekday", value=False)

        o2 = st.columns(3)
        sweep_gap   = o2[0].checkbox("Sweep gap type", value=True)
        sweep_kind  = o2[1].checkbox("Sweep day kind", value=True)
        sweep_first = o2[2].checkbox("Sweep first-side", value=True)

        if st.button("🔎 Run optimization", type="primary"):
            gap_opts   = [None, "Gap Up", "Gap Down", "Flat"] if sweep_gap else [None]
            kind_opts  = [None, "Inside Day", "Outside Day", "Normal"] if sweep_kind else [None]
            first_opts = ["", "high", "low"] if sweep_first else [""]
            dow_opts   = [None] + DOW_ORDER if sweep_dow else [None]

            combos = list(itertools.product(
                [SIDE_LOGICS[l] for l in sweep_logics],
                T_GRID, S_GRID, dow_opts, gap_opts, kind_opts, first_opts))
            inv = {v: k for k, v in SIDE_LOGICS.items()}

            results = []
            prog = st.progress(0.0, text=f"Evaluating {len(combos):,} permutations…")
            # precompute pnl per (logic, t, s) to avoid recompute across filters
            cache = {}
            for i, (lg, t, s, d, gp, kd, fs) in enumerate(combos):
                key = (lg, t, s)
                if key not in cache:
                    cache[key] = logic_pnl(P, lg, t, s)
                pnl_all = cache[key]
                dows  = [d] if d else DOW_ORDER
                gaps  = [gp] if gp else ["Gap Up", "Flat", "Gap Down"]
                kinds = [kd] if kd else ["Inside Day", "Normal", "Outside Day"]
                mask = filter_mask(P, dows, gaps, kinds, fs) & ~np.isnan(pnl_all)
                if mask.sum() < min_trades:
                    continue
                m = metrics(pnl_all[mask])
                if m is None:
                    continue
                results.append({
                    "side logic": inv[lg],
                    "target×": t, "stop×": s, "R:R": round(t / s, 2),
                    "weekday": d or "all",
                    "gap": gp or "any", "day kind": kd or "any",
                    "first": fs or "any",
                    "trades": m["trades"],
                    "win %": round(m["win_rate"] * 100, 1),
                    "expectancy": round(m["expectancy"], 2),
                    "net": round(m["net"], 0),
                    "PF": round(m["pf"], 2) if np.isfinite(m["pf"]) else 99,
                    "max DD": round(m["max_dd"], 0),
                })
                if i % 200 == 0:
                    prog.progress(i / len(combos), text=f"{i:,}/{len(combos):,} permutations…")
            prog.empty()

            if not results:
                st.warning("No configurations met the minimum-trades threshold. Lower it.")
            else:
                res = pd.DataFrame(results)
                res = res.sort_values(sort_metric, ascending=False).reset_index(drop=True)
                st.success(f"{len(res):,} valid configurations · showing top 40 by {sort_label}")
                st.dataframe(res.head(40), use_container_width=True, hide_index=True,
                             column_config={
                                 "win %": st.column_config.NumberColumn(format="%.1f"),
                                 "expectancy": st.column_config.NumberColumn(format="%.2f"),
                             })
                st.download_button("⬇ Download all results",
                                   res.to_csv(index=False).encode(),
                                   file_name=f"optimizer_{instrument}_{tag}.csv",
                                   mime="text/csv")
                st.caption("Expectancy = avg pts/trade. Prefer configs with high trades + "
                           "positive expectancy + shallow max-DD; treat extreme PF on few "
                           "trades with suspicion (overfitting).")


if __name__ == "__main__":
    main()
