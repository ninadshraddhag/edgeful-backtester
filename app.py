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
import os, itertools, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import daywise
import prob_app
import build_facts
import data_store
import live_feed
import live_stats
import report
from datetime import time as dtime

st.set_page_config(page_title="Backtester Pro", page_icon="📈", layout="wide")

HERE  = os.path.dirname(os.path.abspath(__file__))
FACTS = os.path.join(HERE, "analysis", "facts.csv")
# Dynamic instrument registry: legacy C:\ files + repo data/ folder + uploads
# (NQ, XAUUSD, …). Re-evaluated every script run, so new uploads appear at once.
PATHS = data_store.discover()

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
    return build_facts.clean_min(data_store.read_minute(path))


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
def get_facts(instrument, mpath, mtime, open_t, close_t, ib_min=60):
    """
    Per-day facts for ANY instrument, session & IB duration. Uses the prebuilt
    facts.csv for the default indices at 09:15/15:15 with a 60-min IB; otherwise
    computes live (cached).
    """
    std_open = data_store.session_open(instrument, get_open_t(mpath, mtime))
    std_close = data_store.session_close(instrument, DEFAULT_CLOSE_T)
    if (instrument in PATHS and open_t == std_open and ib_min == 60
            and close_t == std_close and os.path.exists(FACTS)):
        f = load_facts(os.path.getmtime(FACTS))
        sub = f[f["instrument"] == instrument].copy()
        if len(sub):
            return sub
    facts = build_facts.build_from_minute(load_min(mpath, mtime), instrument,
                                          open_t, close_t, ib_min=ib_min)
    facts["day_kind"] = np.where(facts["inside_day"], "Inside Day",
                        np.where(facts["outside_day"], "Outside Day", "Normal"))
    return facts


# ─── first-passage cache (the engine) ─────────────────────────────────────────

@st.cache_data(show_spinner=True)
def build_passage(instrument, tag, mpath, mtime, open_t, close_t, ib_min=60):
    """
    Per-day first-passage table: for a long-on-high-break and a short-on-low-break,
    the minute each target/stop level (× range) is first reached. This resolves
    target-vs-stop ordering exactly for ANY (target, stop) pair without re-simulating.
    Capped at the square-off (close_t) — nothing carries past it.
    """
    mdf = load_min(mpath, mtime)
    meta = get_facts(instrument, mpath, mtime, open_t, close_t, ib_min).set_index("date")
    end_t = (open_t + 15 - 1) if tag == "orb" else (open_t + ib_min - 1)

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

        rec = {"date": pd.Timestamp(day), "range": rng, "day_close": close,
               "win_hi": hi, "win_lo": lo}

        # LONG: enter at hi when high first exceeds hi
        le = np.where(ph > hi)[0]
        if len(le):
            et = pt[le[0]]
            rec["L_entry"] = hi
            rec["L_entry_t"] = et
            for t in T_GRID:
                idx = np.where((ph >= hi + t * rng) & (pt >= et))[0]
                rec[f"L_T_{t}"] = pt[idx[0]] if len(idx) else np.nan
            for s in S_GRID:
                idx = np.where((pl <= hi - s * rng) & (pt >= et))[0]
                rec[f"L_S_{s}"] = pt[idx[0]] if len(idx) else np.nan
        else:
            rec["L_entry"] = np.nan
            rec["L_entry_t"] = np.nan

        # SHORT: enter at lo when low first breaks lo
        se = np.where(pl < lo)[0]
        if len(se):
            et = pt[se[0]]
            rec["S_entry"] = lo
            rec["S_entry_t"] = et
            for t in T_GRID:
                idx = np.where((pl <= lo - t * rng) & (pt >= et))[0]
                rec[f"S_T_{t}"] = pt[idx[0]] if len(idx) else np.nan
            for s in S_GRID:
                idx = np.where((ph >= lo + s * rng) & (pt >= et))[0]
                rec[f"S_S_{s}"] = pt[idx[0]] if len(idx) else np.nan
        else:
            rec["S_entry"] = np.nan
            rec["S_entry_t"] = np.nan

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


# ─── trade log construction (entry/exit times) ────────────────────────────────

def hhmm(m):
    """Minutes-since-midnight → 'HH:MM' (NaN-safe)."""
    if pd.isna(m):
        return ""
    m = int(m)
    return f"{m // 60:02d}:{m % 60:02d}"


def edge_trades(P, logic, t, s, dows, gaps, kinds, first, close_t):
    """
    Full per-trade log for one Edge configuration — entry/exit prices AND times.
    Mirrors side_pnl()/logic_pnl() exactly (conservative stop-first tie rule),
    so the summed pnl always matches the metrics shown above it.
    """
    mask = filter_mask(P, dows, gaps, kinds, first)
    rows = []
    for _, r in P[mask].iterrows():
        # which side does this logic take today?
        if logic == "long":
            side = "long"
        elif logic == "short":
            side = "short"
        elif logic == "fade":
            side = "short" if r["first_side"] == "high" else "long"
        elif logic == "gap":
            if r["gap_type"] == "Gap Up":
                side = "long"
            elif r["gap_type"] == "Gap Down":
                side = "short"
            else:
                continue                      # flat day → no trade
        else:
            side = "long"

        pre = "L" if side == "long" else "S"
        entry = r[f"{pre}_entry"]
        if pd.isna(entry):
            continue                          # that side never triggered today
        rng = r["range"]
        fpt = r[f"{pre}_T_{t}"]
        fps = r[f"{pre}_S_{s}"]
        fpt_i = np.inf if pd.isna(fpt) else fpt
        fps_i = np.inf if pd.isna(fps) else fps

        if np.isfinite(fpt_i) and fpt_i <= fps_i:          # target first
            outcome, exit_t = "target", fpt
            exit_px = entry + t * rng if side == "long" else entry - t * rng
            pnl = t * rng
        elif np.isfinite(fps_i):                           # stop first
            outcome, exit_t = "stop", fps
            exit_px = entry - s * rng if side == "long" else entry + s * rng
            pnl = -s * rng
        else:                                              # neither → square-off
            outcome, exit_t = "squareoff", close_t
            exit_px = r["day_close"]
            pnl = (exit_px - entry) if side == "long" else (entry - exit_px)

        risk = s * rng
        rows.append({
            "date": r["date"], "dow": r["dow"], "side": side,
            "entry time": hhmm(r[f"{pre}_entry_t"]), "exit time": hhmm(exit_t),
            "entry": round(float(entry), 2), "exit": round(float(exit_px), 2),
            "target px": round(float(entry + t * rng if side == "long" else entry - t * rng), 2),
            "stop px": round(float(entry - s * rng if side == "long" else entry + s * rng), 2),
            "range": round(float(rng), 2),
            "outcome": outcome, "pnl": round(float(pnl), 2),
            "r_mult": round(float(pnl / risk), 3) if risk > 0 else np.nan,
            "win": pnl > 0,
            "gap": r["gap_type"], "day kind": r["day_kind"],
            "first side": r["first_side"],
            "win_hi": float(r["win_hi"]), "win_lo": float(r["win_lo"]),
            "_entry_t": r[f"{pre}_entry_t"], "_exit_t": exit_t,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["date", "_entry_t"]).reset_index(drop=True)


EDGE_LOG_COLS = ["date", "dow", "side", "entry time", "exit time", "entry", "exit",
                 "target px", "stop px", "range", "outcome", "pnl", "r_mult",
                 "gap", "day kind", "first side"]


# ── clean, TradingView-style chart primitives (shared by all trade browsers) ───
CANDLE_UP, CANDLE_DN = "#089981", "#F23645"     # TradingView green / red


def _clean_candles(fig, x, day_df):
    fig.add_candlestick(
        x=x, open=day_df["open"], high=day_df["high"], low=day_df["low"],
        close=day_df["close"], name="Price", line=dict(width=1),
        increasing=dict(line=dict(color=CANDLE_UP, width=1), fillcolor=CANDLE_UP),
        decreasing=dict(line=dict(color=CANDLE_DN, width=1), fillcolor=CANDLE_DN))


def _clean_layout(fig, height=480, title=None):
    fig.update_layout(
        height=height, template="plotly_white", xaxis_rangeslider_visible=False,
        margin=dict(t=42 if title else 14, b=16, l=8, r=8),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
        dragmode="pan", showlegend=False,
        title=(dict(text=title, x=0, xanchor="left", font=dict(size=13)) if title else None),
        xaxis=dict(showgrid=False, showline=True, linecolor="rgba(0,0,0,0.18)",
                   ticks="outside", tickcolor="rgba(0,0,0,0.18)",
                   rangebreaks=[dict(bounds=["sat", "mon"])]),
        yaxis=dict(side="right", showgrid=True, gridcolor="rgba(0,0,0,0.05)",
                   zeroline=False, showline=False))


def _edge_day_chart(day_df, tr, open_t, end_t):
    """One trade drawn on its day's minute chart — clean candles, OR box, TP/SL."""
    day = pd.Timestamp(tr["date"])
    x = day_df["date"]
    fig = go.Figure()
    _clean_candles(fig, x, day_df)
    # opening-range box (ORB/IB window) — subtle
    fig.add_shape(type="rect",
                  x0=day + pd.Timedelta(minutes=open_t), x1=day + pd.Timedelta(minutes=end_t),
                  y0=tr["win_lo"], y1=tr["win_hi"],
                  fillcolor="rgba(96,125,139,0.10)", line=dict(width=1, color="#90A4AE"))
    x_entry = day + pd.Timedelta(minutes=int(tr["_entry_t"]))
    x_exit = day + pd.Timedelta(minutes=int(tr["_exit_t"]))
    for px, color in [(tr["target px"], "#2E7D32"), (tr["stop px"], "#C62828")]:
        fig.add_shape(type="line", x0=x_entry, x1=x_exit, y0=px, y1=px,
                      line=dict(width=1.3, color=color, dash="dash"))
    long = tr["side"] == "long"
    win_c = "#2E7D32" if tr["win"] else "#C62828"
    fig.add_scatter(x=[x_entry], y=[tr["entry"]], mode="markers",
                    marker=dict(symbol="triangle-up" if long else "triangle-down",
                                size=13, color="#2E7D32" if long else "#C62828",
                                line=dict(width=1, color="white")),
                    showlegend=False, hovertext=f"entry {tr['entry time']} · {tr['entry']:.2f}",
                    hoverinfo="text")
    fig.add_scatter(x=[x_exit], y=[tr["exit"]], mode="markers",
                    marker=dict(symbol="x", size=10, color=win_c), showlegend=False,
                    hovertext=f"exit {tr['exit time']} · {tr['exit']:.2f} ({tr['outcome']})",
                    hoverinfo="text")
    fig.add_scatter(x=[x_entry, x_exit], y=[tr["entry"], tr["exit"]], mode="lines",
                    line=dict(width=1, color=win_c, dash="dot"),
                    showlegend=False, hoverinfo="skip")
    _clean_layout(fig, height=480,
                  title=f"{day:%a %d %b %Y} · {tr['side'].upper()} · "
                        f"{tr['pnl']:+.1f} pts ({tr['r_mult']:+.2f}R)")
    st.plotly_chart(fig, use_container_width=True)


def _edge_trade_browser(trades, mpath, mtime, open_t, end_t, close_t, key):
    """Pick any trade → see it on that day's minute chart."""
    t_ = trades.reset_index(drop=True)
    labels = [f"#{i+1} · {r['date']:%d %b %Y} {r['entry time']} · {r['side'].upper()} · "
              f"{r['outcome']} · {r['pnl']:+.1f} pts" for i, r in t_.iterrows()]
    sel = st.selectbox("Trade", range(len(t_)), format_func=lambda i: labels[i],
                       key=f"{key}_tb")
    tr = t_.iloc[sel]
    mdf = load_min(mpath, mtime)
    dd = mdf[(mdf["date_only"] == pd.Timestamp(tr["date"]).date())
             & (mdf["t_min"] >= open_t) & (mdf["t_min"] <= close_t)]
    if dd.empty:
        st.info("No minute data for this trade's day.")
        return
    _edge_day_chart(dd, tr, open_t, end_t)
    st.caption("Shaded box = ORB/IB window · green dashed = target · red dashed = stop · "
               "▲/▼ entry · ✕ exit.")


# ─── confluence portfolio (combine strategies) ────────────────────────────────

def _confl_label(c):
    inv = {v: k for k, v in SIDE_LOGICS.items()}
    return (f"{inv.get(c['logic'], c['logic'])} · {c['t']}×/{c['s']}× · "
            f"{c['dow'] or 'all days'} · gap:{c['gap'] or 'any'} · "
            f"kind:{c['kind'] or 'any'} · first:{c['first'] or 'any'}")


CONFL_FILE = os.path.join(HERE, "analysis", "confluence_presets.json")


def _confl_store_load():
    if os.path.exists(CONFL_FILE):
        try:
            return json.load(open(CONFL_FILE, encoding="utf-8"))
        except Exception:
            pass
    return {"presets": {}, "last": []}


def _confl_store_save(data):
    os.makedirs(os.path.dirname(CONFL_FILE), exist_ok=True)
    json.dump(data, open(CONFL_FILE, "w", encoding="utf-8"), indent=2)


def _confl_components():
    """Session component list; on first access restore the last-used set from disk."""
    if "edge_confl" not in st.session_state:
        st.session_state["edge_confl"] = _confl_store_load().get("last") or []
    return st.session_state["edge_confl"]


def _confl_persist_last():
    store = _confl_store_load()
    store["last"] = st.session_state.get("edge_confl", [])
    _confl_store_save(store)


def _confl_add(specs):
    comps = _confl_components()
    added = 0
    for s_ in specs:
        if s_ not in comps:
            comps.append(s_)
            added += 1
    if added:
        _confl_persist_last()
    return added


def confluence_tab(P, instrument, tag, mpath, mtime, open_t, end_t, close_t):
    st.markdown("Combine several edge configurations into ONE portfolio — "
                "uncorrelated strategies smooth the equity curve. Add components "
                "from the **Single Strategy** tab or tick rows in the **Optimizer**.")
    comps = _confl_components()

    # ── save / load named confluences ─────────────────────────────────────────
    store = _confl_store_load()
    with st.expander("💾 Save / Load confluences", expanded=False):
        s_ = st.columns([2, 1, 2, 1, 1])
        pname = s_[0].text_input("Confluence name", key="confl_pname",
                                 placeholder="e.g. confluence-1")
        if s_[1].button("Save", key="confl_psave", use_container_width=True):
            if not comps:
                st.warning("Nothing to save — add strategies first.")
            elif pname.strip():
                store["presets"][pname.strip()] = comps
                store["last"] = comps
                _confl_store_save(store)
                st.success(f"Saved confluence '{pname.strip()}' ({len(comps)} strategies).")
            else:
                st.warning("Enter a name first.")
        names = list(store.get("presets", {}).keys())
        sel = s_[2].selectbox("Saved confluences", ["—"] + names, key="confl_psel")
        if s_[3].button("Load", key="confl_pload", use_container_width=True) and sel != "—":
            st.session_state["edge_confl"] = list(store["presets"][sel])
            _confl_persist_last()
            st.rerun()
        if s_[4].button("Delete", key="confl_pdel", use_container_width=True) and sel != "—":
            store["presets"].pop(sel, None)
            _confl_store_save(store)
            st.rerun()
        st.caption("Saved to `analysis/confluence_presets.json` — the last list "
                   "auto-restores on launch. Note: on the cloud app, saves last until "
                   "the app restarts; save locally for permanence.")

    if not comps:
        st.info("No strategies added yet. Use “➕ Add this strategy to Confluence” on the "
                "Single Strategy tab, tick optimizer rows and click “➕ Add ticked to "
                "Confluence”, or load a saved confluence above.")
        return

    cur = [c for c in comps if c["instrument"] == instrument and c["tag"] == tag]
    other = len(comps) - len(cur)
    list_df = pd.DataFrame({"#": range(1, len(comps) + 1),
                            "strategy": [_confl_label(c) for c in comps],
                            "instrument": [c["instrument"] for c in comps],
                            "setup": [c["tag"].upper() for c in comps]})
    st.dataframe(list_df, use_container_width=True, hide_index=True)
    cdel = st.columns([2, 1, 1])
    rm = cdel[0].multiselect("Remove #", list(range(1, len(comps) + 1)), key="confl_rm")
    if cdel[1].button("Remove selected", use_container_width=True) and rm:
        st.session_state["edge_confl"] = [c for i, c in enumerate(comps, 1) if i not in rm]
        _confl_persist_last()
        st.rerun()
    if cdel[2].button("Clear all", use_container_width=True):
        st.session_state["edge_confl"] = []
        _confl_persist_last()
        st.rerun()
    if other:
        st.caption(f"⚠ {other} component(s) belong to a different instrument/setup and are "
                   f"excluded here — switch the sidebar to run them.")
    if not cur:
        st.warning(f"No components for {instrument} · {tag.upper()}.")
        return

    frames = []
    for i, c in enumerate(cur, 1):
        tr = edge_trades(P, c["logic"], c["t"], c["s"],
                         [c["dow"]] if c["dow"] else DOW_ORDER,
                         [c["gap"]] if c["gap"] else ["Gap Up", "Flat", "Gap Down"],
                         [c["kind"]] if c["kind"] else ["Inside Day", "Normal", "Outside Day"],
                         c["first"] or "", close_t)
        if tr.empty:
            continue
        tr["strategy"] = f"#{i}"
        frames.append(tr)
    if not frames:
        st.warning("No trades from the current components in this date range.")
        return
    T = pd.concat(frames).sort_values(["date", "_entry_t"]).reset_index(drop=True)

    pnl = T["pnl"].to_numpy()
    m = metrics(pnl)
    k = st.columns(6)
    k[0].metric("Strategies", len(frames))
    k[1].metric("Trades", f"{m['trades']:,}")
    k[2].metric("Win Rate", pct(m["win_rate"]))
    k[3].metric("Net P&L", f"{m['net']:,.0f} pts")
    k[4].metric("Expectancy", f"{m['expectancy']:.2f} pts/trade")
    k[5].metric("Profit Factor", f"{m['pf']:.2f}" if np.isfinite(m["pf"]) else "∞")
    rr = (m["avg_win"] / abs(m["avg_loss"])) if m["avg_loss"] else float("inf")
    k2 = st.columns(4)
    k2[0].metric("Max Drawdown", f"{m['max_dd']:,.0f} pts")
    k2[1].metric("Avg Win", f"{m['avg_win']:.1f} pts")
    k2[2].metric("Avg Loss", f"{m['avg_loss']:.1f} pts")
    k2[3].metric("Avg RR (win/loss)", f"{rr:.2f}" if np.isfinite(rr) else "∞")

    # combined equity + drawdown
    cum = np.cumsum(pnl)
    dd = cum - np.maximum.accumulate(cum)
    dts = pd.to_datetime(T["date"])
    fig = go.Figure()
    fig.add_scatter(x=dts, y=cum, mode="lines", name="Portfolio equity",
                    line=dict(color="#2196F3", width=2),
                    fill="tozeroy", fillcolor="rgba(33,150,243,0.08)")
    fig.update_layout(title="Combined equity (points, 1 unit per trade)", height=320,
                      template="plotly_white", margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    figd = go.Figure()
    figd.add_scatter(x=dts, y=dd, mode="lines", name="Drawdown",
                     line=dict(color="#E53935", width=1.5),
                     fill="tozeroy", fillcolor="rgba(229,57,53,0.15)")
    figd.update_layout(title="Drawdown (points)", height=220, template="plotly_white",
                       margin=dict(t=40, b=10))
    st.plotly_chart(figd, use_container_width=True)

    # per-strategy contribution
    st.markdown("**Per-strategy contribution**")
    contrib = (T.groupby("strategy")
               .agg(trades=("pnl", "size"), net=("pnl", "sum"),
                    win_rate=("win", "mean"), expectancy=("pnl", "mean")))
    contrib["win_rate"] = (contrib["win_rate"] * 100).round(1)
    contrib[["net", "expectancy"]] = contrib[["net", "expectancy"]].round(1)
    st.dataframe(contrib, use_container_width=True)

    with st.expander(f"📋 All portfolio trades ({len(T):,})"):
        st.dataframe(T[["strategy"] + EDGE_LOG_COLS].sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True, height=360)
    st.download_button("⬇ Download portfolio trades (CSV)",
                       T[["strategy"] + EDGE_LOG_COLS].to_csv(index=False).encode(),
                       file_name=f"confluence_{instrument.replace(' ', '_')}_{tag}.csv",
                       mime="text/csv")
    with st.expander("🔍 Trade browser"):
        _edge_trade_browser(T, mpath, mtime, open_t, end_t, close_t, key="confl")


# ─── UI helpers ───────────────────────────────────────────────────────────────

APP_CSS = """
<style>
/* ——— typography ——— */
h1 { letter-spacing: -0.02em; font-weight: 800; }
h2, h3 { letter-spacing: -0.01em; }

/* ——— metric cards ——— */
[data-testid="stMetric"] {
  background: linear-gradient(180deg, #ffffff 0%, #f6f9fc 100%);
  border: 1px solid #e3e9f2;
  border-radius: 14px;
  padding: 14px 16px 10px 16px;
  box-shadow: 0 1px 3px rgba(16, 42, 67, 0.06);
}
[data-testid="stMetricLabel"] { color: #5b6b7f; }
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.01em; }

/* ——— tabs ——— */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
  background: #eef2f7;
  border-radius: 10px 10px 0 0;
  padding: 8px 16px;
  font-weight: 600;
}
.stTabs [aria-selected="true"] {
  background: #ffffff;
  border-bottom: 3px solid #1565C0;
}

/* ——— expanders, dataframes, containers ——— */
[data-testid="stExpander"] {
  border: 1px solid #e3e9f2;
  border-radius: 12px;
  background: #ffffff;
}
[data-testid="stDataFrame"] { border: 1px solid #e3e9f2; border-radius: 12px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }

/* ——— buttons ——— */
.stButton button, .stDownloadButton button { border-radius: 10px; font-weight: 600; }
.stButton button[kind="primary"] {
  background: linear-gradient(90deg, #1565C0 0%, #1E88E5 100%);
  border: none;
  box-shadow: 0 2px 10px rgba(21, 101, 192, 0.35);
}
.stButton button[kind="primary"]:hover { box-shadow: 0 4px 14px rgba(21, 101, 192, 0.5); }

/* ——— sidebar ——— */
[data-testid="stSidebar"] {
  border-right: 1px solid #e3e9f2;
  background: linear-gradient(180deg, #f7f9fc 0%, #eef2f7 100%);
}
</style>
"""

BRAND_HTML = """
<div style="padding:4px 0 10px 0">
  <span style="font-size:1.55rem;font-weight:800;letter-spacing:-0.02em;
        background:linear-gradient(90deg,#1565C0,#26A69A);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    ⚡ BACKTESTER PRO</span><br>
  <span style="color:#7b8aa0;font-size:0.78rem;letter-spacing:0.06em;
        text-transform:uppercase;">Quant Backtesting Suite</span>
</div>
"""


def inject_style():
    """Global look & feel — called once per run, themes every mode."""
    st.markdown(APP_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(BRAND_HTML, unsafe_allow_html=True)


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
    Shared sidebar block: instrument picker (incl. persistent uploads like
    XAUUSD / NQ), session open (auto-detected, overridable for EST markets),
    square-off time and date range.
    Returns (instrument, mpath, mtime, open_t, close_t, ib_min, (d0, d1)).
    """
    st.header("Data")
    paths = data_store.discover()
    names = list(paths.keys())
    choice = st.selectbox("Instrument", names + ["➕ Upload new instrument…"],
                          key=f"{key}_inst")

    if choice == "➕ Upload new instrument…":
        nm = st.text_input("Instrument name (e.g. XAUUSD, NQ)", key=f"{key}_nm")
        f  = st.file_uploader("Minute CSV — columns: date, open, high, low, close. "
                              "Timestamps in EXCHANGE-LOCAL time (e.g. EST for NQ).",
                              type="csv", key=f"{key}_nf")
        if nm and f is not None:
            data_store.save_upload(nm.strip(), f.getbuffer())
            st.success(f"Added {nm.strip()} — it is now available in every mode. "
                       "Select it above.")
            st.rerun()
        st.info("Name the instrument and upload its minute CSV to add it. It is "
                "saved to the data/ folder, so it persists and shows up in ALL modes.")
        st.stop()

    mpath = paths.get(choice)
    if not mpath or not os.path.exists(mpath):
        st.warning(f"No data file for {choice}.")
        st.stop()
    mtime = os.path.getmtime(mpath)

    # session open: auto-detect, with a persistent per-instrument override —
    # needed for 24h instruments (e.g. NQ: auto-detect sees the Globex open,
    # set 09:30 EST here for the cash-session IB). Widget keys are scoped per
    # instrument so switching instruments can't leak times across markets.
    wkey = f"{key}_{choice}"
    auto_t = get_open_t(mpath, mtime)
    cur_t = data_store.session_open(choice, auto_t)
    so = st.time_input("Session open (IB starts here)",
                       value=dtime(cur_t // 60, cur_t % 60), key=f"{wkey}_so",
                       help="Auto-detected from the data. Override for 24-hour "
                            "instruments: e.g. NQ → 09:30 (EST cash open). The "
                            "override is remembered per instrument.")
    open_t = so.hour * 60 + so.minute
    if open_t != cur_t:
        data_store.set_open_override(choice, open_t, auto_t)

    cur_c = data_store.session_close(choice, DEFAULT_CLOSE_T)
    sq = st.time_input("Square-off (force exit)",
                       value=dtime(cur_c // 60, cur_c % 60), key=f"{wkey}_sq",
                       help="Remembered per instrument (NSE 15:15 · NQ/XAUUSD 16:00 ET).")
    close_t = sq.hour * 60 + sq.minute
    if close_t != cur_c:
        data_store.set_close_override(choice, close_t, DEFAULT_CLOSE_T)
    ib_min = int(st.number_input("IB duration (min)", 10, 240, 60, 10, key=f"{key}_ibmin",
                                 help="Initial Balance window length from the session open."))
    st.caption(f"Session open **{open_t//60:02d}:{open_t%60:02d}**"
               f"{' (auto)' if open_t == auto_t else ' (override)'} · "
               f"IB = first {ib_min} min · square-off **{sq.strftime('%H:%M')}** "
               "(no overnight carry)")

    dmin, dmax = get_date_bounds(mpath, mtime)
    dr = st.date_input("Date range", value=(dmin, dmax),
                       min_value=dmin, max_value=dmax, key=f"{key}_dates")
    d0, d1 = (dr if isinstance(dr, (tuple, list)) and len(dr) == 2 else (dmin, dmax))
    return choice, mpath, mtime, open_t, close_t, ib_min, (d0, d1)


def _filter_prepped(prepped, d0, d1):
    return {dow: [r for r in lst if d0 <= r["date"].date() <= d1]
            for dow, lst in prepped.items()}


# ─── day-wise IB retracement mode ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def prep_daywise(instrument, mpath, mtime, open_t, close_t, ib_min=60):
    mdf = load_min(mpath, mtime)
    return daywise.prep_days(mdf, open_t, close_t, ib_min)


PRESETS_FILE = os.path.join(HERE, "analysis", "daywise_presets.json")
SHARED_NUM = ["entry", "stop", "tp1", "tp2"]
DIR_NUM = [f"{p}_{x}" for p in ("long", "short") for x in ("entry", "stop", "tp1", "tp2")]
DW_FIELDS = (["trade", "allow_long", "allow_short", "separate",
              "min_size", "max_size", "cutoff_on", "cutoff_t"] + SHARED_NUM + DIR_NUM)


def _load_presets():
    if os.path.exists(PRESETS_FILE):
        try:
            return json.load(open(PRESETS_FILE, encoding="utf-8"))
        except Exception:
            pass
    return {"presets": {}, "last": None}


def _save_presets(data):
    os.makedirs(os.path.dirname(PRESETS_FILE), exist_ok=True)
    json.dump(data, open(PRESETS_FILE, "w", encoding="utf-8"), indent=2, default=str)


def _dw_seed_defaults():
    """setdefault every per-weekday widget key from DEFAULT_CONFIG (fills any gaps)."""
    for day in daywise.WEEKDAYS:
        d = daywise.DEFAULT_CONFIG[day]
        for f in ["trade", "allow_long", "allow_short"]:
            st.session_state.setdefault(f"dw_{day}_{f}", d[f])
        st.session_state.setdefault(f"dw_{day}_separate", False)
        for f in SHARED_NUM:
            st.session_state.setdefault(f"dw_{day}_{f}", d[f])
        for p in ("long", "short"):
            for x in ("entry", "stop", "tp1", "tp2"):
                st.session_state.setdefault(f"dw_{day}_{p}_{x}", d[x])
        st.session_state.setdefault(f"dw_{day}_min_size", d["min_size"])
        st.session_state.setdefault(f"dw_{day}_max_size", d["max_size"])
        st.session_state.setdefault(f"dw_{day}_cutoff_on", d["cutoff"] is not None)
        st.session_state.setdefault(f"dw_{day}_cutoff_t",
                                    dtime(d["cutoff"] // 60, d["cutoff"] % 60)
                                    if d["cutoff"] else dtime(13, 0))


def _dw_apply_config(configs):
    """Write a saved 5-day config dict back into the widget session_state keys."""
    for day, c in configs.items():
        if day not in daywise.WEEKDAYS:
            continue
        st.session_state[f"dw_{day}_trade"]       = c.get("trade", True)
        st.session_state[f"dw_{day}_allow_long"]  = c.get("allow_long", True)
        st.session_state[f"dw_{day}_allow_short"] = c.get("allow_short", False)
        st.session_state[f"dw_{day}_separate"]    = c.get("separate", False)
        for f in SHARED_NUM:
            st.session_state[f"dw_{day}_{f}"] = c.get(f, 0)
        for p in ("long", "short"):
            for x in ("entry", "stop", "tp1", "tp2"):
                st.session_state[f"dw_{day}_{p}_{x}"] = c.get(f"{p}_{x}", c.get(x, 0))
        st.session_state[f"dw_{day}_min_size"] = c.get("min_size", 0.0)
        st.session_state[f"dw_{day}_max_size"] = c.get("max_size", 5.0)
        cutoff = c.get("cutoff")
        st.session_state[f"dw_{day}_cutoff_on"] = cutoff is not None
        st.session_state[f"dw_{day}_cutoff_t"] = (dtime(int(cutoff) // 60, int(cutoff) % 60)
                                                  if cutoff else dtime(13, 0))


def _dw_init_state():
    """On first load, restore the last-saved config from disk; then fill any gaps."""
    if not st.session_state.get("dw_init_done"):
        data = _load_presets()
        if data.get("last"):
            _dw_apply_config(data["last"])
        st.session_state["dw_init_done"] = True
    _dw_seed_defaults()


def _dw_read_config():
    cfg = {}
    for day in daywise.WEEKDAYS:
        g = lambda f: st.session_state[f"dw_{day}_{f}"]
        cutoff = None
        if g("cutoff_on"):
            t = g("cutoff_t")
            cutoff = t.hour * 60 + t.minute
        c = dict(trade=g("trade"), allow_long=g("allow_long"), allow_short=g("allow_short"),
                 separate=g("separate"),
                 entry=g("entry"), stop=g("stop"), tp1=g("tp1"), tp2=g("tp2"),
                 min_size=g("min_size"), max_size=g("max_size"), cutoff=cutoff)
        for p in ("long", "short"):
            for x in ("entry", "stop", "tp1", "tp2"):
                c[f"{p}_{x}"] = g(f"{p}_{x}")
        cfg[day] = c
    return cfg


def _dw_param_row(day, pfx):
    r = st.columns(4)
    r[0].number_input("Entry retr %", 0, 100, key=f"dw_{day}_{pfx}entry", step=5,
                      help="Retracement entry level. 0 = enter at the IB boundary (breakout).")
    r[1].number_input("Stop retr %", 0, 200, key=f"dw_{day}_{pfx}stop", step=5,
                      help="Deeper retracement; must exceed entry %. 100 = opposite IB boundary.")
    r[2].number_input("Target 1 (ext %)", 0, 500, key=f"dw_{day}_{pfx}tp1", step=5,
                      help="First target (extension beyond the boundary). Half position exits here.")
    r[3].number_input("Target 2 (ext %)", 0, 500, key=f"dw_{day}_{pfx}tp2", step=5,
                      help="Second target for the runner (stop → breakeven after TP1).")


def _dw_weekday_panel(day):
    d = st.session_state
    top = st.columns([1.1, 1, 1, 1.5])
    top[0].toggle("Trade this day", key=f"dw_{day}_trade")
    top[1].toggle("Allow long", key=f"dw_{day}_allow_long")
    top[2].toggle("Allow short", key=f"dw_{day}_allow_short")
    top[3].toggle("Separate long/short SL & targets", key=f"dw_{day}_separate")

    sep = d[f"dw_{day}_separate"]
    if not sep:
        _dw_param_row(day, "")
    else:
        st.markdown("**Long** — entry/stop off IB High, targets above")
        _dw_param_row(day, "long_")
        st.markdown("**Short** — entry/stop off IB Low, targets below")
        _dw_param_row(day, "short_")

    r2 = st.columns(4)
    r2[0].number_input("Min IB size %", 0.0, 10.0, key=f"dw_{day}_min_size", step=0.1,
                       help="IB range as % of price (range ÷ open × 100).")
    r2[1].number_input("Max IB size %", 0.0, 10.0, key=f"dw_{day}_max_size", step=0.1)
    r2[2].toggle("Entry cutoff", key=f"dw_{day}_cutoff_on")
    r2[3].time_input("Cutoff time", key=f"dw_{day}_cutoff_t",
                     disabled=not d[f"dw_{day}_cutoff_on"])

    bad = lambda pfx: d[f"dw_{day}_{pfx}stop"] <= d[f"dw_{day}_{pfx}entry"]
    if not sep:
        if bad(""):
            st.warning("Stop % must be greater than entry % — this day will be skipped.")
    else:
        msgs = [p for p, pfx in (("long", "long_"), ("short", "short_"))
                if d[f"dw_{day}_allow_{p}"] and bad(pfx)]
        if msgs:
            st.warning(f"Stop % must exceed entry % for {', '.join(msgs)} — "
                       "those trades will be skipped.")


def daywise_mode():
    # Apply any deferred config writes from the optimizer "Apply" buttons. This must
    # run BEFORE the weekday widgets are instantiated (Streamlit forbids writing a
    # widget's session_state key once the widget exists in the same run).
    pend = st.session_state.pop("dw_pending", None)
    if pend:
        for k, v in pend.items():
            st.session_state[k] = v

    st.caption("Edgeful-style day-wise IB retracement · entry/stop in % of IB range "
               "(retracement) · targets in % of IB range (extension) · half at TP1, "
               "runner to TP2 with stop at breakeven")
    _dw_init_state()

    ENTRY_CONDS = {
        "Any touch (default)": "any",
        "Only if IB High/Low NOT yet breached (pure retracement)": "no_breach",
        "Only AFTER breakout — High broken for longs / Low for shorts (retest)": "after_breach",
    }
    with st.sidebar:
        instrument, mpath, mtime, open_t, close_t, ib_min, (d0, d1) = data_sidebar("dw")
        ec_label = st.radio("Entry condition", list(ENTRY_CONDS.keys()), key="dw_entrycond",
                            help="Pure retracement: the resting order is cancelled the "
                                 "moment the reference IB boundary breaks. Retest: entries "
                                 "only count after the boundary has broken first.")
        entry_cond = ENTRY_CONDS[ec_label]

        st.markdown("**🎯 IB directional filters** *(optional)*")
        fc_min = int(st.select_slider(
            "First-candle window (min)", options=[5, 10, 15, 30, 45, 60, 90, 120],
            value=15, key="dw_flt_fcmin",
            help="Window for the green/red first-candle filters below: candle = "
                 "session open → close at this many minutes in."))
        dir_filters = {
            "candle_min": fc_min,
            "long_first_low": st.checkbox(
                "LONG only if IB LOW formed first", key="dw_flt_ll",
                help="Low forms first, high later → bullish bias; otherwise skip longs."),
            "long_close_above": st.checkbox(
                "LONG only if IB closes ABOVE midpoint", key="dw_flt_lc",
                help="IB's last candle closes above (IB High+Low)/2 → bullish confirmation."),
            "long_first_green": st.checkbox(
                f"LONG only if first {fc_min}-min candle is GREEN", key="dw_flt_lg",
                help="Close at the end of the window above the session open → bullish bias."),
            "short_first_high": st.checkbox(
                "SHORT only if IB HIGH formed first", key="dw_flt_sh",
                help="High forms first, low later → bearish bias; otherwise skip shorts."),
            "short_close_below": st.checkbox(
                "SHORT only if IB closes BELOW midpoint", key="dw_flt_sc",
                help="IB's last candle closes below the midpoint → bearish confirmation."),
            "short_first_red": st.checkbox(
                f"SHORT only if first {fc_min}-min candle is RED", key="dw_flt_sr",
                help="Close at the end of the window below the session open → bearish bias."),
        }
        if any(v for k, v in dir_filters.items() if k != "candle_min"):
            st.caption("Filters apply to the backtest AND both optimizers.")

        vwap_exit = st.checkbox(
            "🟠 Exit on VWAP close", key="dw_vwap_exit",
            help="Optional momentum exit: close the position when a post-IB candle "
                 "CLOSES on the wrong side of session VWAP — long exits on a close "
                 "below VWAP, short on a close above. Intrabar stop/target still take "
                 "priority; this exits the remainder at that bar's close.")

    with st.spinner(f"Indexing {instrument} minute data (first run is cached)…"):
        prepped = prep_daywise(instrument, mpath, mtime, open_t, close_t, ib_min)
    prepped = _filter_prepped(prepped, d0, d1)
    if not any(prepped.values()):
        st.warning("No trading days in the selected date range.")
        st.stop()

    # ── save / load settings ──────────────────────────────────────────────────
    st.subheader("Day-wise configuration")
    presets = _load_presets()
    with st.expander("💾 Save / Load settings", expanded=False):
        s = st.columns([2, 1, 2, 1, 1])
        pname = s[0].text_input("Preset name", key="dw_preset_name",
                                placeholder="e.g. my-nifty-setup")
        if s[1].button("Save", use_container_width=True):
            if pname.strip():
                cfgs = _dw_read_config()
                presets["presets"][pname.strip()] = cfgs
                presets["last"] = cfgs
                _save_presets(presets)
                st.success(f"Saved preset '{pname.strip()}'.")
            else:
                st.warning("Enter a preset name first.")
        names = list(presets.get("presets", {}).keys())
        sel = s[2].selectbox("Saved presets", ["—"] + names, key="dw_preset_sel")
        if s[3].button("Load", use_container_width=True) and sel != "—":
            _dw_apply_config(presets["presets"][sel])
            presets["last"] = presets["presets"][sel]
            _save_presets(presets)
            st.rerun()
        if s[4].button("Delete", use_container_width=True) and sel != "—":
            presets["presets"].pop(sel, None)
            _save_presets(presets)
            st.rerun()
        st.caption("Settings auto-restore on next launch. Saved to "
                   "`analysis/daywise_presets.json`.")

    cc = st.columns([1, 1, 3])
    if cc[0].button("📋 Copy Monday → all days", use_container_width=True):
        for day in daywise.WEEKDAYS:
            for f in DW_FIELDS:
                st.session_state[f"dw_{day}_{f}"] = st.session_state[f"dw_Monday_{f}"]
        st.rerun()
    if cc[1].button("↺ Reset to defaults", use_container_width=True):
        for day in daywise.WEEKDAYS:
            for f in DW_FIELDS:
                st.session_state.pop(f"dw_{day}_{f}", None)
        st.session_state["dw_init_done"] = False
        st.rerun()

    for day in daywise.WEEKDAYS:
        cfg = _dw_read_config()[day]
        flags = []
        if not cfg["trade"]: flags.append("off")
        if cfg["allow_long"]: flags.append("long")
        if cfg["allow_short"]: flags.append("short")
        if cfg.get("separate"):
            params = (f"L {cfg['long_entry']}/{cfg['long_stop']}→{cfg['long_tp1']}/{cfg['long_tp2']} · "
                      f"S {cfg['short_entry']}/{cfg['short_stop']}→{cfg['short_tp1']}/{cfg['short_tp2']}")
        else:
            params = f"entry {cfg['entry']}% / stop {cfg['stop']}% · TP {cfg['tp1']}%/{cfg['tp2']}%"
        label = f"{day}  ·  {params} · {'/'.join(flags) or 'no direction'}"
        with st.expander(label, expanded=(day == "Monday")):
            _dw_weekday_panel(day)

    st.divider()
    run = st.button("▶ Run day-wise backtest", type="primary")

    if run:
        configs = _dw_read_config()
        presets["last"] = configs          # auto-persist most recent config
        _save_presets(presets)
        trades = daywise.run(prepped, configs, entry_cond, dir_filters, vwap_exit)
        with st.spinner("Building PDF report…"):
            try:
                pdf_bytes = report.build_daywise_pdf(
                    instrument, (d0, d1), open_t, close_t, configs, trades,
                    ib_min=ib_min, entry_cond=entry_cond)
            except Exception as e:
                pdf_bytes = None
                st.warning(f"PDF generation failed: {e}")
        st.session_state["dw_results"] = dict(
            instrument=instrument, trades=trades, pdf=pdf_bytes, period=(d0, d1),
            mpath=mpath, mtime=mtime, open_t=open_t, close_t=close_t, ib_min=ib_min)

    res = st.session_state.get("dw_results")
    if res is not None:
        _dw_show_results(res)
        if res.get("pdf"):
            st.download_button(
                "📄 Download PDF report (system + performance + all trades)",
                res["pdf"],
                file_name=(f"daywise_report_{res['instrument'].replace(' ', '_')}_"
                           f"{res['period'][0]}_{res['period'][1]}.pdf"),
                mime="application/pdf", key="dw_pdf_dl")

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
                        opt_metric, int(opt_min), entry_cond=entry_cond,
                        dir_filters=dir_filters, vwap_exit=vwap_exit)
                if res.empty:
                    st.warning("No configs met the minimum-trades threshold.")
                else:
                    st.session_state["dw_opt_result"] = (opt_day, res)

        if "dw_opt_result" in st.session_state:
            od, res = st.session_state["dw_opt_result"]
            st.markdown(f"**Top configs for {od}** (showing 20)")
            st.dataframe(res.head(20), use_container_width=True, hide_index=True)
            best = res.iloc[0]
            bdir = best["direction"]
            if st.button(f"✅ Apply best {od} config ({bdir}) to the panel above"):
                pend = {}
                if st.session_state.get(f"dw_{od}_separate"):
                    # write into that direction's own params; leave the other side intact
                    for x in ("entry", "stop", "tp1", "tp2"):
                        pend[f"dw_{od}_{bdir}_{x}"] = int(best[x])
                    pend[f"dw_{od}_allow_{bdir}"] = True
                else:
                    for x in ("entry", "stop", "tp1", "tp2"):
                        pend[f"dw_{od}_{x}"] = int(best[x])
                    pend[f"dw_{od}_allow_long"] = bdir == "long"
                    pend[f"dw_{od}_allow_short"] = bdir == "short"
                pend[f"dw_{od}_trade"] = True
                st.session_state["dw_pending"] = pend       # applied at top of next run
                st.session_state.pop("dw_opt_result", None)
                st.rerun()

    # ── combined optimizer (all weekdays at once) ─────────────────────────────
    with st.expander("🧩 Combined optimizer — best config for EVERY weekday at once"):
        oc2 = st.columns(3)
        co_metric = oc2[0].selectbox("Rank by",
                                     ["expectancy", "win_rate", "net", "pf"],
                                     key="dw_co_metric")
        co_min = oc2[1].number_input("Min trades / weekday", 20, 1000, 100, 10,
                                     key="dw_co_min")
        co_dirs = oc2[2].multiselect("Directions", ["long", "short"],
                                     default=["long", "short"], key="dw_co_dirs")
        if st.button("🔎 Optimize all 5 weekdays (~1 min)", key="dw_co_run"):
            if not co_dirs:
                st.warning("Pick at least one direction.")
            else:
                prog = st.progress(0.0, text="Optimizing weekdays…")
                rows = []
                for i, day in enumerate(daywise.WEEKDAYS):
                    res = daywise.optimize_weekday(
                        prepped[day], co_dirs, daywise.OPT_GRID,
                        co_metric, int(co_min), entry_cond=entry_cond,
                        dir_filters=dir_filters, vwap_exit=vwap_exit)
                    if not res.empty:
                        b = res.iloc[0]
                        rows.append({"Day": day, "direction": b["direction"],
                                     "entry": int(b["entry"]), "stop": int(b["stop"]),
                                     "tp1": int(b["tp1"]), "tp2": int(b["tp2"]),
                                     "trades": int(b["trades"]), "win %": b["win %"],
                                     "expectancy": b["expectancy"],
                                     "net": b["net"], "pf": b["pf"]})
                    prog.progress((i + 1) / len(daywise.WEEKDAYS),
                                  text=f"{day} done ({i + 1}/{len(daywise.WEEKDAYS)})")
                prog.empty()
                st.session_state["dw_co_result"] = pd.DataFrame(rows)

        co = st.session_state.get("dw_co_result")
        if co is not None and len(co):
            st.markdown("**Best configuration per weekday**")
            st.dataframe(co, use_container_width=True, hide_index=True)
            tot = int(co["trades"].sum())
            wwin = float((co["win %"] * co["trades"]).sum() / tot) if tot else 0.0
            st.caption(f"Combined: {tot:,} trades · weighted win rate {wwin:.1f}% · "
                       f"total net {co['net'].sum():,.0f} pts. In-sample optimum — "
                       "validate on a date sub-range before trusting it.")
            if st.button("✅ Apply ALL best configs to the panels above", key="dw_co_apply"):
                pend = {}
                for _, r in co.iterrows():
                    day = r["Day"]
                    for x in ("entry", "stop", "tp1", "tp2"):
                        pend[f"dw_{day}_{x}"] = int(r[x])
                    pend[f"dw_{day}_allow_long"] = r["direction"] == "long"
                    pend[f"dw_{day}_allow_short"] = r["direction"] == "short"
                    pend[f"dw_{day}_separate"] = False
                    pend[f"dw_{day}_trade"] = True
                st.session_state["dw_pending"] = pend        # applied at top of next run
                st.session_state.pop("dw_co_result", None)
                st.rerun()


def _dw_day_chart(day_df, tr, open_t, ib_min):
    """One day-wise trade on its day's minute chart: IB box, entry/stop/TP1/TP2
    levels and entry/exit markers."""
    day = pd.Timestamp(tr["date"])
    x = day_df["date"]
    fig = go.Figure()
    _clean_candles(fig, x, day_df)
    # IB box — subtle
    fig.add_shape(type="rect",
                  x0=day + pd.Timedelta(minutes=open_t),
                  x1=day + pd.Timedelta(minutes=open_t + ib_min - 1),
                  y0=tr["ib_low"], y1=tr["ib_high"],
                  fillcolor="rgba(96,125,139,0.10)", line=dict(width=1, color="#90A4AE"))
    x_entry = day + pd.Timedelta(minutes=int(tr["entry_t"]))
    x_exit = day + pd.Timedelta(minutes=int(tr["exit_t"]))
    # levels drawn over the trade's lifetime
    for px, color, dash in [(tr["stop_px"], "#C62828", "dash"),
                            (tr["tp1_px"], "#66BB6A", "dash"),
                            (tr["tp2_px"], "#2E7D32", "dash"),
                            (tr["entry"], "#90A4AE", "dot")]:
        fig.add_shape(type="line", x0=x_entry, x1=x_exit, y0=px, y1=px,
                      line=dict(width=1.3, color=color, dash=dash))
    long = tr["direction"] == "long"
    win_c = "#2E7D32" if tr["win"] else "#C62828"
    fig.add_scatter(x=[x_entry], y=[tr["entry"]], mode="markers",
                    marker=dict(symbol="triangle-up" if long else "triangle-down",
                                size=13, color="#2E7D32" if long else "#C62828",
                                line=dict(width=1, color="white")),
                    showlegend=False,
                    hovertext=f"entry {hhmm(tr['entry_t'])} · {tr['entry']:.2f}",
                    hoverinfo="text")
    fig.add_scatter(x=[x_exit], y=[tr["exit_px"]], mode="markers",
                    marker=dict(symbol="x", size=10, color=win_c), showlegend=False,
                    hovertext=f"exit {hhmm(tr['exit_t'])} · {tr['exit_px']:.2f} "
                              f"({tr['outcome']})", hoverinfo="text")
    fig.add_scatter(x=[x_entry, x_exit], y=[tr["entry"], tr["exit_px"]], mode="lines",
                    line=dict(width=1, color=win_c, dash="dot"),
                    showlegend=False, hoverinfo="skip")
    _clean_layout(fig, height=480,
                  title=f"{day:%a %d %b %Y} · {tr['direction'].upper()} · "
                        f"{tr['pnl']:+.1f} pts ({tr['r_mult']:+.2f}R)")
    st.plotly_chart(fig, use_container_width=True)


def _dw_trade_browser(res):
    trades = res["trades"]
    if "entry_t" not in trades.columns:
        st.info("Re-run the backtest to enable the visual browser (older results "
                "lack trade times).")
        return
    t_ = trades.reset_index(drop=True)
    labels = [f"#{i+1} · {r['date']:%d %b %Y} {hhmm(r['entry_t'])} · "
              f"{r['direction'].upper()} · {r['outcome']} · {r['pnl']:+.1f} pts"
              for i, r in t_.iterrows()]
    sel = st.selectbox("Trade", range(len(t_)), format_func=lambda i: labels[i],
                       key="dw_tb")
    tr = t_.iloc[sel]
    mdf = load_min(res["mpath"], res["mtime"])
    dd = mdf[(mdf["date_only"] == pd.Timestamp(tr["date"]).date())
             & (mdf["t_min"] >= res["open_t"]) & (mdf["t_min"] <= res["close_t"])]
    if dd.empty:
        st.info("No minute data for this trade's day.")
        return
    _dw_day_chart(dd, tr, res["open_t"], res["ib_min"])
    st.caption("Shaded box = IB window · grey dotted = entry level · red dashed = stop · "
               "light green = TP1 (half exits, stop → breakeven) · dark green = TP2 · "
               "▲/▼ entry · ✕ final exit.")


def _dw_show_results(res):
    instrument, trades = res["instrument"], res["trades"]
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

    rr = (m["avg_win"] / abs(m["avg_loss"])) if m["avg_loss"] else float("inf")
    avg_r = trades["r_mult"].mean() if "r_mult" in trades.columns else float("nan")
    k2 = st.columns(4)
    k2[0].metric("Avg Win", f"{m['avg_win']:.1f} pts")
    k2[1].metric("Avg Loss", f"{m['avg_loss']:.1f} pts")
    k2[2].metric("Avg RR (win/loss)", f"{rr:.2f}" if np.isfinite(rr) else "∞",
                 help="Average winning trade ÷ average losing trade.")
    k2[3].metric("Avg R / trade", f"{avg_r:+.2f}R" if np.isfinite(avg_r) else "—",
                 help="Mean PnL per unit of initial risk (entry−stop distance).")

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
        show = trades.copy()
        if "entry_t" in show.columns:
            show.insert(3, "entry time", show["entry_t"].map(hhmm))
            show.insert(4, "exit time", show["exit_t"].map(hhmm))
            show = show.drop(columns=["entry_t", "exit_t", "ib_high", "ib_low"],
                             errors="ignore")
        st.dataframe(show.sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)
    st.download_button("⬇ Download trades (CSV)", trades.to_csv(index=False).encode(),
                       file_name=f"daywise_{instrument.replace(' ', '_')}.csv",
                       mime="text/csv")
    with st.expander("🔍 Trade browser — any trade on its day's candlestick chart"):
        _dw_trade_browser(res)


# ─── live market statistics mode ──────────────────────────────────────────────

LIVE_CFG = os.path.join(HERE, "live_config.json")


def _load_live_cfg():
    if os.path.exists(LIVE_CFG):
        try:
            raw = json.load(open(LIVE_CFG, encoding="utf-8"))
            # migrate old Dhan-only configs gracefully
            if "kotak" not in raw:
                raw["kotak"] = {"consumer_key": "", "consumer_secret": "",
                                "mobile": "", "password": ""}
            return raw
        except Exception:
            pass
    return {"kotak": {"consumer_key": "", "consumer_secret": "",
                      "mobile": "", "password": ""}, "instruments": {}}


def _save_live_cfg(cfg):
    json.dump(cfg, open(LIVE_CFG, "w", encoding="utf-8"), indent=2)


def _badge(label, value, tone="#455A64"):
    return (f"<span style='background:{tone};color:#fff;padding:3px 10px;"
            f"border-radius:12px;font-size:0.85em;margin-right:6px;white-space:nowrap'>"
            f"{label}: <b>{value}</b></span>")


def _live_sample_day_chart(day_df, row, open_t, ib_min, close_t):
    """One matched day's session as clean candlesticks: IB box, IB high/low,
    PDH/PDL, with a descriptive title."""
    day = pd.Timestamp(row["date"])
    x = day_df["date"]
    fig = go.Figure()
    _clean_candles(fig, x, day_df)
    # IB window box
    ib_x0 = day + pd.Timedelta(minutes=open_t)
    ib_x1 = day + pd.Timedelta(minutes=open_t + ib_min - 1)
    if pd.notna(row.get("ib_high")) and pd.notna(row.get("ib_low")):
        fig.add_shape(type="rect", x0=ib_x0, x1=ib_x1,
                      y0=row["ib_low"], y1=row["ib_high"],
                      fillcolor="rgba(96,125,139,0.10)", line=dict(width=1, color="#90A4AE"))
        fig.add_annotation(x=ib_x0, y=row["ib_high"], text="IB", showarrow=False,
                           yshift=8, font=dict(size=10, color="#607D8B"))
        # IB high/low extended across the rest of the session
        fig.add_shape(type="line", x0=ib_x1, x1=x.iloc[-1], y0=row["ib_high"],
                      y1=row["ib_high"], line=dict(width=1, dash="dot", color="#2E7D32"))
        fig.add_shape(type="line", x0=ib_x1, x1=x.iloc[-1], y0=row["ib_low"],
                      y1=row["ib_low"], line=dict(width=1, dash="dot", color="#C62828"))
    # PDH / PDL — subtle
    for lvl, nm in [(row.get("pdh"), "PDH"), (row.get("pdl"), "PDL")]:
        if pd.notna(lvl):
            fig.add_hline(y=float(lvl), line=dict(width=1, dash="dash", color="#6A1B9A"),
                          annotation_text=nm, annotation_font_size=9)
    first = (row.get("ib_first_side") or "?").upper()
    brk = (row.get("ib_break_first") or "none")
    ue, de = row.get("ib_up_ext"), row.get("ib_dn_ext")
    ext_txt = (f"up {ue:.2f}×R" if pd.notna(ue) else "—") + " / " + \
              (f"dn {de:.2f}×R" if pd.notna(de) else "—")
    _clean_layout(fig, height=460,
                  title=f"{day:%a %d %b %Y}  ·  {row.get('gap_type','')}  ·  "
                        f"{first} formed first  ·  broke {brk} first  ·  ext {ext_txt}")
    st.plotly_chart(fig, use_container_width=True)


def _live_sample_section(instrument, feat, probs, open_t):
    """🗂 Sample days — pick any matched day to see its clean candlestick chart."""
    sample = probs.get("sample")
    if sample is None or sample.empty:
        return
    label = probs.get("sample_label", "matching days")
    ib_min = int(feat.get("ib_min", 60))
    close_t = feat.get("close_t", 15 * 60 + 15)

    with st.expander(f"🗂 Sample days — the **{len(sample):,}** matching days behind "
                     f"these odds  ·  {label}", expanded=False):
        df = sample.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=False).reset_index(drop=True)
        df["IB Size %"] = (df["ib_range"] / df["day_open"] * 100).round(3)

        # one-click picker: choose any day → see its candlestick
        st.caption("👆 Pick any matching day to see its session as a clean "
                   "candlestick chart (IB box · IB high/low · PDH/PDL).")
        labels = [f"{r.date:%d %b %Y} · {str(r.dow)[:3]} · {r.gap_type} · "
                  f"{r.ib_first_side} first · broke {r.ib_break_first} · "
                  f"ext up {r.ib_up_ext:.2f}/dn {r.ib_dn_ext:.2f}×R"
                  for r in df.itertuples()]
        sel = st.selectbox("Matched day", range(len(df)),
                           format_func=lambda i: labels[i], key="live_sample_pick")
        row = df.iloc[sel]

        mpath = data_store.discover().get(instrument)
        if mpath:
            full = load_min(mpath, os.path.getmtime(mpath))
            dd = full[(full["date_only"] == row["date"].date())
                      & (full["t_min"] >= open_t) & (full["t_min"] <= close_t)]
            if dd.empty:
                st.info("No minute data for this day.")
            else:
                _live_sample_day_chart(dd, row, open_t, ib_min, close_t)
                k = st.columns(5)
                k[0].metric("IB High", f"{row['ib_high']:,.1f}")
                k[1].metric("IB Low", f"{row['ib_low']:,.1f}")
                k[2].metric("IB Range", f"{row['ib_range']:,.1f}")
                k[3].metric("Broke IB-H", "Yes" if row["ib_high_break"] else "No")
                k[4].metric("Broke IB-L", "Yes" if row["ib_low_break"] else "No")
        else:
            st.info("Minute history for this instrument isn't available to chart.")

        # full list + CSV
        tbl = pd.DataFrame({
            "Date": df["date"].dt.strftime("%Y-%m-%d"),
            "Weekday": df["dow"].str[:3],
            "Gap": df["gap_type"],
            "IB Range": df["ib_range"].round(1),
            "IB Size %": df["IB Size %"],
            "First side": df["ib_first_side"],
            "Broke first": df["ib_break_first"],
            "Broke IB-H": df["ib_high_break"],
            "Broke IB-L": df["ib_low_break"],
            "Up ext ×R": df["ib_up_ext"].round(2),
            "Down ext ×R": df["ib_dn_ext"].round(2),
            "Reached PDH": df["broke_pdh"],
            "Reached PDL": df["broke_pdl"],
        })
        with st.expander(f"📋 All {len(tbl):,} matched days — full detail"):
            st.dataframe(tbl, use_container_width=True, hide_index=True, height=300)
        st.download_button(
            "⬇ Download sample days (CSV)", tbl.to_csv(index=False).encode(),
            file_name=f"sample_days_{instrument.replace(' ', '_')}.csv",
            mime="text/csv", key="live_sample_dl")


def _live_render(instrument, feat, probs, open_t):
    ist = pd.Timestamp.now(tz="Asia/Kolkata")
    st.markdown(f"#### {instrument} &nbsp;·&nbsp; {feat.get('phase','—')} "
                f"&nbsp;·&nbsp; {ist.strftime('%H:%M:%S')} IST")

    if "day_open" not in feat:
        st.info("Session not started yet (pre-open). Levels will populate after the open.")
        return

    # ── day-forming badges ────────────────────────────────────────────────────
    gtone = {"Gap Up": "#2E7D32", "Gap Down": "#C62828", "Flat": "#455A64"}.get(feat.get("gap_type"), "#455A64")
    badges = [_badge("Day", feat["dow"]),
              _badge("Price", f"{feat['price']:.1f}")]
    if feat.get("gap_type"):
        badges.append(_badge("Gap", f"{feat['gap_type']} ({feat['gap_pct']:+.2f}%)", gtone))
    if feat.get("ib_first_side"):
        side = feat["ib_first_side"].upper()
        badges.append(_badge("IB first", f"{side} formed first",
                             "#2E7D32" if side == "LOW" else "#C62828"))
        badges.append(_badge("IB size", f"{feat['ib_size_pct']:.2f}% of price"))
    for lab, key in [("PDH", "broke_pdh"), ("PDL", "broke_pdl"),
                     ("IB-H", "broke_ib_high"), ("IB-L", "broke_ib_low")]:
        if key in feat:
            hit = feat[key]
            badges.append(_badge(lab, "BROKEN" if hit else "intact",
                                 "#6A1B9A" if hit else "#455A64"))
    if "ib_close_above_mid" in feat:
        badges.append(_badge("IB close",
                             "above mid" if feat["ib_close_above_mid"] else "below mid",
                             "#00695C"))
    st.markdown(" ".join(badges), unsafe_allow_html=True)

    if "ib_first_side" not in feat:
        ibm = feat.get("ib_min", 60)
        st.info(f"IB (first {ibm} min) still forming — full probabilities unlock at "
                f"{(open_t+ibm)//60:02d}:{(open_t+ibm)%60:02d}. Showing gap-only stats below.")

    # ── live probability cards ────────────────────────────────────────────────
    st.markdown("##### Live probabilities — historical odds for this exact day type")
    s = probs.get("matched") or probs.get("gap_slice")
    nlab = probs.get("n_matched", probs.get("n_gap", 0))
    if s:
        c = st.columns(5)
        c[0].metric("IB HIGH breaks", pct(s["high"]))
        c[1].metric("IB LOW breaks", pct(s["low"]))
        c[2].metric("BOTH sides", pct(s["both"]))
        c[3].metric("ONE side", pct(s["one"]))
        c[4].metric("Sample days", f"{nlab:,}")
        if "fade_opp_break" in probs:
            fs = feat["ib_first_side"].upper(); opp = probs["fade_opp"].upper()
            st.success(f"**First-move-fade:** {fs} formed first → **{pct(probs['fade_opp_break'])} "
                       f"chance the {opp} breaks** (breaks first {pct(probs.get('fade_opp_first',0))}), "
                       f"n={probs.get('n_matched',0):,}.")
        mid = probs.get("mid")
        if mid and mid.get("prob") is not None:
            where = "ABOVE" if mid["close_above_mid"] else "BELOW"
            opp = mid["opp"].upper()
            delta = (mid["prob"] - mid["base"]) * 100
            if mid["confirmed"]:
                st.success(f"**Midpoint confirmation:** IB closed **{where} its midpoint** "
                           f"→ **{pct(mid['prob'])} chance the {opp} breaks** "
                           f"({delta:+.1f} pp vs {pct(mid['base'])} for all matching days), "
                           f"n={mid['n']:,}. The close confirms the fade.")
            else:
                st.warning(f"**Midpoint:** IB closed {where} its midpoint — this does *not* "
                           f"confirm the fade. P({opp} breaks) on such days = "
                           f"{pct(mid['prob'])} (n={mid['n']:,}).")
        if nlab < 30:
            st.warning(f"⚠ Only {nlab} historical matches — low confidence.")

    # ── IB-size conditioned odds (gap + day + IB-size band) ───────────────────
    sz = probs.get("size")
    if sz and sz.get("stats"):
        ss = sz["stats"]
        st.markdown(f"##### Conditioned on IB size — today's IB is **{sz['today_pct']:.2f}%** "
                    f"of price → **{sz['bucket']}** of historical "
                    f"{feat.get('gap_type','')} {feat.get('dow','')} days")
        c = st.columns(5)
        c[0].metric("IB HIGH breaks", pct(ss["high"]))
        c[1].metric("IB LOW breaks", pct(ss["low"]))
        c[2].metric("BOTH sides", pct(ss["both"]))
        c[3].metric("ONE side", pct(ss["one"]))
        c[4].metric("Sample days", f"{sz['n']:,}")
        st.caption(f"Size terciles for this gap+day: ≤{sz['q1']:.2f}% narrow · "
                   f"≥{sz['q2']:.2f}% wide"
                   + (" · also matched on IB first-side" if sz.get("with_first_side")
                      else " · first-side not applied (kept sample size up)"))

    # ── sample days behind the odds (chart + table) ──────────────────────────
    _live_sample_section(instrument, feat, probs, open_t)

    # ── PDH/PDL reach for today's gap ─────────────────────────────────────────
    pdt = probs.get("pd_table")
    if pdt is not None and not pdt.empty and feat.get("gap_type"):
        row = pdt[pdt["Gap"] == feat["gap_type"]]
        if not row.empty:
            r = row.iloc[0]
            cc = st.columns(2)
            cc[0].metric(f"{feat['gap_type']} → reach PDH", f"{r['reach PDH %']:.1f}%")
            cc[1].metric(f"{feat['gap_type']} → reach PDL", f"{r['reach PDL %']:.1f}%")

    # ── today's levels + extension targets ────────────────────────────────────
    rows = []
    if feat.get("ib_high"):
        rows += [("IB High", feat["ib_high"]), ("IB Low", feat["ib_low"])]
    if feat.get("orb_high"):
        rows += [("ORB High", feat["orb_high"]), ("ORB Low", feat["orb_low"])]
    if feat.get("prev_high"):
        rows += [("PDH", feat["prev_high"]), ("PDL", feat["prev_low"]),
                 ("Prev close", feat["prev_close"])]
    if rows:
        lv = pd.DataFrame(rows, columns=["Level", "Price"]).round(1)
        ext = probs.get("ext")
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**Today's key levels**")
            st.dataframe(lv, use_container_width=True, hide_index=True)
        with cR:
            if ext:
                st.markdown("**Extension targets — price & historical reach %**")
                et = pd.DataFrame([{
                    "× range": e["level"],
                    "Bull @": round(e["up_price"], 1),
                    "reach %": None if e["up_p"] is None else round(e["up_p"] * 100, 1),
                    "Bear @": round(e["dn_price"], 1),
                    "reach %.": None if e["dn_p"] is None else round(e["dn_p"] * 100, 1),
                } for e in ext])
                st.dataframe(et, use_container_width=True, hide_index=True)
                n_hi = probs.get("ext_n_hi"); n_lo = probs.get("ext_n_lo")
                if n_hi is not None:
                    st.caption(f"Bull reach % conditioned on **{n_hi:,}** high-broke-first "
                               f"days · Bear on **{n_lo:,}** low-broke-first days.")

    # ── retracement probabilities (pull-back entry depth after a breakout) ─────
    retr = probs.get("retr")
    if retr:
        st.markdown("##### Retracement probabilities — pull-back depth for a "
                    "retracement ENTRY")
        st.info("ℹ️ **A retracement is only valid while the IB boundary is NOT yet "
                "breached.** These are forward-looking odds *if* the breakout then "
                "happens: after the IB **high** breaks, how far price pulls back DOWN "
                "(long entry); after the IB **low** breaks, how far it bounces UP "
                "(short entry). “reach %” = chance the pull-back is at least this deep "
                "(i.e. your retracement entry fills).")
        hb, lb = feat.get("broke_ib_high"), feat.get("broke_ib_low")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(f"**Bull retracement** · after IB-high break · "
                        f"n={probs.get('retr_n_hi', 0):,}")
            if hb:
                st.warning("⚠ IB **high already broken** today — the pure-retracement "
                           "long is no longer valid (breakout occurred). Shown for "
                           "reference / breakout-retest only.")
            else:
                st.success("✅ IB high intact — retracement long still valid.")
            bt = pd.DataFrame([{
                "× range": e["level"],
                "Pull-back to": round(e["up_price"], 1),
                "reach %": None if e["up_p"] is None else round(e["up_p"] * 100, 1),
            } for e in retr])
            st.dataframe(bt, use_container_width=True, hide_index=True)
        with rc2:
            st.markdown(f"**Bear retracement** · after IB-low break · "
                        f"n={probs.get('retr_n_lo', 0):,}")
            if lb:
                st.warning("⚠ IB **low already broken** today — the pure-retracement "
                           "short is no longer valid (breakout occurred). Shown for "
                           "reference / breakout-retest only.")
            else:
                st.success("✅ IB low intact — retracement short still valid.")
            bdt = pd.DataFrame([{
                "× range": e["level"],
                "Bounce to": round(e["dn_price"], 1),
                "reach %": None if e["dn_p"] is None else round(e["dn_p"] * 100, 1),
            } for e in retr])
            st.dataframe(bdt, use_container_width=True, hide_index=True)

    # ── today's day-wise retracement plan (from saved preset) ─────────────────
    plan = []
    if feat.get("ib_high"):
        presets = _load_presets()
        cfg_all = presets.get("last") or daywise.DEFAULT_CONFIG
        cfg = cfg_all.get(feat["dow"], daywise.DEFAULT_CONFIG[feat["dow"]])
        H, L, Rg = feat["ib_high"], feat["ib_low"], feat["ib_range"]
        for is_long, allow in [(True, cfg.get("allow_long")), (False, cfg.get("allow_short"))]:
            if not allow:
                continue
            e, s_, t1, t2 = daywise.dir_params(cfg, is_long)
            if is_long:
                plan.append(["LONG", H - e/100*Rg, H - s_/100*Rg, H + t1/100*Rg, H + t2/100*Rg])
            else:
                plan.append(["SHORT", L + e/100*Rg, L + s_/100*Rg, L - t1/100*Rg, L - t2/100*Rg])
        if plan:
            st.markdown(f"**Your day-wise plan for {feat['dow']} (from saved preset)**")
            st.dataframe(pd.DataFrame(plan, columns=["Dir", "Entry", "Stop", "TP1", "TP2"]).round(1),
                         use_container_width=True, hide_index=True)

    st.caption("Probabilities are historical frequencies for matching days (10-yr) — "
               "not guarantees. Treat thin samples and rare day-types with caution.")

    # ── PDF session report ────────────────────────────────────────────────────
    try:
        pdf_bytes = report.build_live_pdf(instrument, feat, probs, plan)
        st.download_button(
            "📄 Download PDF session report",
            pdf_bytes,
            file_name=(f"live_report_{instrument.replace(' ', '_')}_"
                       f"{pd.Timestamp.now():%Y%m%d_%H%M}.pdf"),
            mime="application/pdf", key="live_pdf_dl")
    except Exception as e:
        st.caption(f"PDF report unavailable: {e}")


def live_mode():
    st.title("📡 Live Market Statistics")
    st.caption("Classifies the day as it forms and shows live conditional probabilities "
               "from the 10-year history.")
    cfg = _load_live_cfg()

    LOOKBACKS = {"6 months": 6, "1 year": 12, "2 years": 24,
                 "3 years": 36, "5 years": 60, "All history": None}
    with st.sidebar:
        st.header("Live data")
        source_kind = st.radio("Source", ["Demo replay", "Kotak Neo (live)"], key="live_src")
        live_paths = data_store.discover()
        instrument = st.selectbox("Instrument", list(live_paths.keys()), key="live_inst")
        mpath = live_paths[instrument]
        mtime = os.path.getmtime(mpath)
        open_t = data_store.session_open(instrument, get_open_t(mpath, mtime))
        ib_min = int(st.number_input("IB duration (min)", 10, 240, 60, 10,
                                     key="live_ibmin",
                                     help="In 10-min steps (e.g. 40, 50, 60)."))
        lb_label = st.selectbox("Probability lookback", list(LOOKBACKS.keys()),
                                index=2, key="live_lookback",
                                help="Only days within this window feed the live "
                                     "probabilities — recent regimes are usually more "
                                     "relevant than the full 10 years.")

        if source_kind == "Kotak Neo (live)":
            auth_state = st.session_state.get("kotak_auth_state", "idle")

            if auth_state == "idle":
                st.markdown("**Kotak Neo credentials**")
                ckey = st.text_input("Consumer Key",
                                     value=cfg["kotak"].get("consumer_key", ""),
                                     key="kotak_ckey")
                csec = st.text_input("Consumer Secret",
                                     value=cfg["kotak"].get("consumer_secret", ""),
                                     type="password", key="kotak_csec")
                mob = st.text_input("Registered mobile",
                                    value=cfg["kotak"].get("mobile", ""),
                                    key="kotak_mob",
                                    help="The mobile number registered with your Kotak account.")
                pwd = st.text_input("Trading password",
                                    value=cfg["kotak"].get("password", ""),
                                    type="password", key="kotak_pwd")
                if st.button("💾 Save & Login (sends OTP)"):
                    cfg["kotak"] = {"consumer_key": ckey.strip(),
                                    "consumer_secret": csec.strip(),
                                    "mobile": mob.strip(),
                                    "password": pwd.strip()}
                    _save_live_cfg(cfg)
                    try:
                        from neo_api_client import NeoAPI
                        _client = NeoAPI(consumer_key=ckey.strip(),
                                         consumer_secret=csec.strip(),
                                         environment="prod")
                        resp = _client.login(mobilenumber=mob.strip(),
                                             password=pwd.strip())
                        st.session_state["kotak_client"] = _client
                        st.session_state["kotak_auth_state"] = "otp_sent"
                        msg = (resp.get("message") or resp.get("data", {}).get("message", "")
                               if isinstance(resp, dict) else str(resp))
                        st.success(f"OTP sent. {msg}")
                        st.rerun()
                    except ImportError:
                        st.error("neo-api-client not installed. "
                                 "Run:  pip install neo-api-client")
                    except Exception as ex:
                        st.error(f"Login failed: {ex}")

            elif auth_state == "otp_sent":
                st.info("OTP sent to your registered mobile.")
                otp = st.text_input("Enter OTP", key="kotak_otp", max_chars=6)
                col_a, col_b = st.columns(2)
                if col_a.button("✅ Verify OTP"):
                    _client = st.session_state.get("kotak_client")
                    if _client is None:
                        st.error("Session lost — start over.")
                        st.session_state.pop("kotak_auth_state", None)
                        st.rerun()
                    else:
                        try:
                            _client.session_2fa(OTP=otp.strip())
                            st.session_state["kotak_auth_state"] = "authenticated"
                            st.rerun()
                        except Exception as ex:
                            st.error(f"OTP verification failed: {ex}")
                if col_b.button("↺ Re-login"):
                    st.session_state.pop("kotak_auth_state", None)
                    st.session_state.pop("kotak_client", None)
                    st.rerun()

            else:  # authenticated
                st.success("✅ Connected to Kotak Neo")
                if st.button("🔓 Disconnect"):
                    st.session_state.pop("kotak_client", None)
                    st.session_state.pop("kotak_auth_state", None)
                    st.rerun()

            auto = st.toggle("Auto-refresh every 60s", value=True, key="live_auto")
            asof_t = None
            asof_date = None
        else:
            full = load_min(mpath, mtime)
            days = sorted(full["date_only"].unique())
            asof_date = st.date_input("Replay date", value=days[-1],
                                      min_value=days[0], max_value=days[-1], key="live_date")
            asof_t = st.slider("As-of time (minute of day)", open_t, 15 * 60 + 15,
                               11 * 60, 5, key="live_asof")
            st.caption(f"Replaying up to {asof_t//60:02d}:{asof_t%60:02d}")
            auto = False

    # facts for this instrument & IB duration (recomputed live when IB != 60)
    fn = get_facts(instrument, mpath, mtime, open_t, 15 * 60 + 15, ib_min).copy()

    # lookback window — anchor to the replay date in Demo mode, today otherwise
    months = LOOKBACKS[lb_label]
    ref_date = pd.Timestamp(asof_date) if asof_date else pd.Timestamp.today()
    if months is not None:
        fn = fn[fn["date"] >= ref_date - pd.DateOffset(months=months)]
    if asof_date:                                   # no lookahead into the replay day
        fn = fn[fn["date"].dt.date != asof_date]
    st.sidebar.caption(f"Probability sample: {len(fn):,} days "
                       f"({lb_label.lower()}, IB {ib_min} min)")

    # build the data source
    if source_kind == "Kotak Neo (live)":
        if st.session_state.get("kotak_auth_state") != "authenticated":
            st.info("Complete Kotak Neo authentication in the sidebar to start live data.")
            return
        _kotak_client = st.session_state.get("kotak_client")
        if _kotak_client is None:
            st.error("Authentication session lost — please reconnect in the sidebar.")
            st.session_state.pop("kotak_auth_state", None)
            return
        try:
            source = live_feed.KotakNeoSource(_kotak_client,
                                              instruments=cfg.get("instruments") or None)
        except Exception as e:
            st.error(f"Could not initialise Kotak Neo source: {e}")
            return
        now_t = None
    else:
        source = live_feed.DemoSource(load_min(mpath, mtime), asof_date, asof_t)
        now_t = asof_t

    run_every = "60s" if (source_kind == "Kotak Neo (live)" and auto) else None

    @st.fragment(run_every=run_every)
    def panel():
        try:
            today = source.today_minutes(instrument)
            prev = source.prev_day(instrument)
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            return
        nt = now_t
        if nt is None:
            ist = pd.Timestamp.now(tz="Asia/Kolkata")
            nt = min(ist.hour * 60 + ist.minute, 15 * 60 + 15)
        feat = live_stats.classify_live(today, prev, open_t, 15 * 60 + 15,
                                        now_t=nt, ib_min=ib_min)
        probs = live_stats.live_probabilities(fn, feat)
        _live_render(instrument, feat, probs, open_t)

    panel()
    if run_every is None and source_kind == "Kotak Neo (live)":
        if st.button("🔄 Refresh now"):
            st.rerun()


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    inject_style()
    if not os.path.exists(FACTS):
        # first boot (e.g. fresh clone / Streamlit Cloud): build the facts
        # table from whatever instruments are available, instead of erroring.
        st.title("📈 Backtester Pro")
        if not data_store.discover():
            st.error("No instrument data found. Add a `*_minute.csv` or "
                     "`*_minute.parquet` file to the `data/` folder (or upload "
                     "one once the app is running with data).")
            st.stop()
        with st.spinner("First run — building the facts table from the minute "
                        "data (one-time, a few minutes)…"):
            build_facts.main()
        st.rerun()

    mode = st.sidebar.radio(
        "Mode",
        ["Edge Backtester", "Day-wise IB Retracement", "Advanced Backtesting",
         "Probabilities", "Live Market Statistics"],
        key="app_mode")
    st.sidebar.divider()
    if mode == "Probabilities":
        prob_app.render()
        return
    if mode == "Live Market Statistics":
        live_mode()
        return
    if mode == "Advanced Backtesting":
        import advanced_mode
        advanced_mode.render()
        return

    st.title("📈 Backtester Pro")
    if mode == "Day-wise IB Retracement":
        daywise_mode()
        return

    st.caption("Trades the probability edges · targets & stops in × opening-range · "
               "permutation optimizer to find the best configuration")

    with st.sidebar:
        instrument, mpath, mtime, open_t, close_t, ib_min, (d0, d1) = data_sidebar("edge")
        setup = st.radio("Setup", [f"IB ({ib_min} min)", "ORB (15 min)"])
        tag = "ib" if setup.startswith("IB") else "orb"
        st.markdown(EDGE_NOTE)

    with st.spinner(f"Indexing {instrument} {setup} (first run is cached)…"):
        P = build_passage(instrument, tag, mpath, mtime, open_t, close_t, ib_min)

    P = P[(P["date"] >= pd.Timestamp(d0)) & (P["date"] <= pd.Timestamp(d1))].reset_index(drop=True)
    if P.empty:
        st.warning("No trading days in the selected date range.")
        st.stop()

    end_t = (open_t + 15 - 1) if tag == "orb" else (open_t + ib_min - 1)
    tab1, tab2, tab3 = st.tabs(["🎯 Single Strategy", "🔬 Optimizer (permutations)",
                                "🧩 Confluence (combine strategies)"])

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

            # ── full trade log + visual browser + confluence hook ────────────
            if st.button("➕ Add this strategy to Confluence", key="single_confl_add"):
                spec = dict(instrument=instrument, tag=tag, logic=logic,
                            t=float(t), s=float(s),
                            dow=None if len(dows) == 5 else (dows[0] if len(dows) == 1 else None),
                            gap=None if len(gaps) == 3 else (gaps[0] if len(gaps) == 1 else None),
                            kind=None if len(kinds) == 3 else (kinds[0] if len(kinds) == 1 else None),
                            first=first_f or None)
                n = _confl_add([spec])
                st.success("Added to the Confluence tab." if n else "Already in Confluence.")

            trades = edge_trades(P, logic, t, s, dows, gaps, kinds, first_f, close_t)
            if not trades.empty:
                with st.expander(f"📋 All trades ({len(trades):,}) — entry & exit times"):
                    st.dataframe(trades[EDGE_LOG_COLS].sort_values("date", ascending=False),
                                 use_container_width=True, hide_index=True, height=380)
                    st.download_button(
                        "⬇ Download trades (CSV)",
                        trades[EDGE_LOG_COLS].to_csv(index=False).encode(),
                        file_name=f"edge_trades_{instrument.replace(' ', '_')}_{tag}.csv",
                        mime="text/csv")
                with st.expander("🔍 Trade browser — any trade on its day's chart"):
                    _edge_trade_browser(trades, mpath, mtime, open_t, end_t, close_t,
                                        key="single")

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
                st.session_state.pop("edge_opt_res", None)
            else:
                res = pd.DataFrame(results)
                res = res.sort_values(sort_metric, ascending=False).reset_index(drop=True)
                st.session_state["edge_opt_res"] = res
                st.session_state["edge_opt_ctx"] = (instrument, tag)

        # results persist across reruns so rows can be ticked into the Confluence
        res = st.session_state.get("edge_opt_res")
        if res is not None and st.session_state.get("edge_opt_ctx") == (instrument, tag):
            st.success(f"{len(res):,} valid configurations · top 40 shown — tick "
                       f"**add** on any rows to combine them in the Confluence tab")
            disp = res.head(40).copy()
            disp.insert(0, "add", False)
            edited = st.data_editor(
                disp, use_container_width=True, hide_index=True, key="edge_opt_editor",
                disabled=[c for c in disp.columns if c != "add"],
                column_config={
                    "add": st.column_config.CheckboxColumn("add", help="Tick → Confluence"),
                    "win %": st.column_config.NumberColumn(format="%.1f"),
                    "expectancy": st.column_config.NumberColumn(format="%.2f"),
                })
            picked = edited[edited["add"]]
            if st.button(f"➕ Add ticked to Confluence ({len(picked)})",
                         disabled=picked.empty, key="opt_confl_add"):
                specs = [dict(instrument=instrument, tag=tag,
                              logic=SIDE_LOGICS[r["side logic"]],
                              t=float(r["target×"]), s=float(r["stop×"]),
                              dow=None if r["weekday"] == "all" else r["weekday"],
                              gap=None if r["gap"] == "any" else r["gap"],
                              kind=None if r["day kind"] == "any" else r["day kind"],
                              first=None if r["first"] == "any" else r["first"])
                         for _, r in picked.iterrows()]
                n = _confl_add(specs)
                st.success(f"Added {n} strategy(ies) — open the 🧩 Confluence tab.")
            st.download_button("⬇ Download all results",
                               res.to_csv(index=False).encode(),
                               file_name=f"optimizer_{instrument}_{tag}.csv",
                               mime="text/csv")
            st.caption("Expectancy = avg pts/trade. Prefer configs with high trades + "
                       "positive expectancy + shallow max-DD; treat extreme PF on few "
                       "trades with suspicion (overfitting).")

    # ============================ CONFLUENCE ============================
    with tab3:
        confluence_tab(P, instrument, tag, mpath, mtime, open_t, end_t, close_t)


if __name__ == "__main__":
    main()
