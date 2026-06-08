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

st.set_page_config(page_title="ORB/IB Edge Backtester", page_icon="📈", layout="wide")

HERE  = os.path.dirname(os.path.abspath(__file__))
FACTS = os.path.join(HERE, "analysis", "facts.csv")
PATHS = {
    "NIFTY 50":   r"C:\NIFTY 50_minute.csv",
    "BANK NIFTY": r"C:\NIFTY BANK_minute.csv",
}

OPEN_T   = 9 * 60 + 15
ORB_END  = OPEN_T + 15 - 1
IB_END   = OPEN_T + 60 - 1
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
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["high"] >= df["low"]) & (df["open"] > 0)]
    df = df.sort_values("date").reset_index(drop=True)
    df["date_only"] = df["date"].dt.date
    df["t_min"] = df["date"].dt.hour * 60 + df["date"].dt.minute
    return df


@st.cache_data(show_spinner=False)
def load_facts(mtime):
    df = pd.read_csv(FACTS, parse_dates=["date"])
    df["day_kind"] = np.where(df["inside_day"], "Inside Day",
                     np.where(df["outside_day"], "Outside Day", "Normal"))
    return df


# ─── first-passage cache (the engine) ─────────────────────────────────────────

@st.cache_data(show_spinner=True)
def build_passage(instrument, tag, mpath, mtime, facts_mtime):
    """
    Per-day first-passage table: for a long-on-high-break and a short-on-low-break,
    the minute each target/stop level (× range) is first reached. This resolves
    target-vs-stop ordering exactly for ANY (target, stop) pair without re-simulating.
    """
    mdf = load_min(mpath, mtime)
    meta = load_facts(facts_mtime)
    meta = meta[meta["instrument"] == instrument].set_index("date")
    end_t = ORB_END if tag == "orb" else IB_END

    rows = []
    for day, g in mdf.groupby("date_only", sort=True):
        g = g.sort_values("t_min")
        win  = g[(g["t_min"] >= OPEN_T) & (g["t_min"] <= end_t)]
        post = g[g["t_min"] > end_t]
        if len(win) < 5 or post.empty:
            continue
        hi, lo = win["high"].max(), win["low"].min()
        rng = hi - lo
        if rng <= 0:
            continue
        close = g.iloc[-1]["close"]
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


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("📈 ORB / IB Edge Backtester")
    st.caption("Trades the probability edges · targets & stops in × opening-range · "
               "permutation optimizer to find the best configuration")

    if not os.path.exists(FACTS):
        st.error("analysis/facts.csv missing — run `python build_facts.py` first.")
        st.stop()

    with st.sidebar:
        st.header("Data")
        instrument = st.radio("Instrument", list(PATHS.keys()))
        setup = st.radio("Setup", ["IB (60 min)", "ORB (15 min)"])
        tag = "ib" if setup.startswith("IB") else "orb"
        up = st.file_uploader(f"Override {instrument} CSV", type="csv")
        st.markdown(EDGE_NOTE)

    mpath = resolve_path(instrument, up)
    if not mpath or not os.path.exists(mpath):
        st.warning(f"No minute CSV found for {instrument}. Upload one in the sidebar.")
        st.stop()

    with st.spinner(f"Indexing {instrument} {setup} (first run is cached)…"):
        P = build_passage(instrument, tag, mpath, os.path.getmtime(mpath),
                          os.path.getmtime(FACTS))

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
        sort_metric = o[1].selectbox("Rank by",
                                     ["expectancy", "net", "win_rate", "pf"])
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
                asc = sort_metric in ()           # all our metrics: higher = better
                res = res.sort_values(sort_metric, ascending=False).reset_index(drop=True)
                st.success(f"{len(res):,} valid configurations · showing top 40 by {sort_metric}")
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
