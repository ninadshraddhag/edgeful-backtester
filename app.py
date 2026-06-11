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
import live_feed
import live_stats
import report
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
def get_facts(instrument, mpath, mtime, open_t, close_t, ib_min=60):
    """
    Per-day facts for ANY instrument, session & IB duration. Uses the prebuilt
    facts.csv for the default indices at 09:15/15:15 with a 60-min IB; otherwise
    computes live (cached).
    """
    if (instrument in PATHS and open_t == DEFAULT_OPEN_T and ib_min == 60
            and close_t == DEFAULT_CLOSE_T and os.path.exists(FACTS)):
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
    ib_min = int(st.number_input("IB duration (min)", 15, 240, 60, 15, key=f"{key}_ibmin",
                                 help="Initial Balance window length from the session open."))
    st.caption(f"Session open auto-detected **{open_t//60:02d}:{open_t%60:02d}** · "
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
        trades = daywise.run(prepped, configs, entry_cond)
        with st.spinner("Building PDF report…"):
            try:
                pdf_bytes = report.build_daywise_pdf(
                    instrument, (d0, d1), open_t, close_t, configs, trades,
                    ib_min=ib_min, entry_cond=entry_cond)
            except Exception as e:
                pdf_bytes = None
                st.warning(f"PDF generation failed: {e}")
        st.session_state["dw_results"] = dict(
            instrument=instrument, trades=trades, pdf=pdf_bytes, period=(d0, d1))

    res = st.session_state.get("dw_results")
    if res is not None:
        _dw_show_results(res["instrument"], res["trades"])
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
                        opt_metric, int(opt_min), entry_cond=entry_cond)
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
                if st.session_state.get(f"dw_{od}_separate"):
                    # write into that direction's own params; leave the other side intact
                    for x in ("entry", "stop", "tp1", "tp2"):
                        st.session_state[f"dw_{od}_{bdir}_{x}"] = int(best[x])
                    st.session_state[f"dw_{od}_allow_{bdir}"] = True
                else:
                    for x in ("entry", "stop", "tp1", "tp2"):
                        st.session_state[f"dw_{od}_{x}"] = int(best[x])
                    st.session_state[f"dw_{od}_allow_long"] = bdir == "long"
                    st.session_state[f"dw_{od}_allow_short"] = bdir == "short"
                st.session_state[f"dw_{od}_trade"] = True
                del st.session_state["dw_opt_result"]
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
                        co_metric, int(co_min), entry_cond=entry_cond)
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
                for _, r in co.iterrows():
                    day = r["Day"]
                    for x in ("entry", "stop", "tp1", "tp2"):
                        st.session_state[f"dw_{day}_{x}"] = int(r[x])
                    st.session_state[f"dw_{day}_allow_long"] = r["direction"] == "long"
                    st.session_state[f"dw_{day}_allow_short"] = r["direction"] == "short"
                    st.session_state[f"dw_{day}_separate"] = False
                    st.session_state[f"dw_{day}_trade"] = True
                st.session_state.pop("dw_co_result", None)
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
        st.dataframe(trades.sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)
    st.download_button("⬇ Download trades (CSV)", trades.to_csv(index=False).encode(),
                       file_name=f"daywise_{instrument.replace(' ', '_')}.csv",
                       mime="text/csv")


# ─── live market statistics mode ──────────────────────────────────────────────

LIVE_CFG = os.path.join(HERE, "live_config.json")


def _load_live_cfg():
    if os.path.exists(LIVE_CFG):
        try:
            return json.load(open(LIVE_CFG, encoding="utf-8"))
        except Exception:
            pass
    return {"dhan": {"client_id": "", "access_token": ""}, "instruments": {}}


def _save_live_cfg(cfg):
    json.dump(cfg, open(LIVE_CFG, "w", encoding="utf-8"), indent=2)


def _badge(label, value, tone="#455A64"):
    return (f"<span style='background:{tone};color:#fff;padding:3px 10px;"
            f"border-radius:12px;font-size:0.85em;margin-right:6px;white-space:nowrap'>"
            f"{label}: <b>{value}</b></span>")


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
        source_kind = st.radio("Source", ["Demo replay", "Dhan (live)"], key="live_src")
        instrument = st.selectbox("Instrument", list(PATHS.keys()), key="live_inst")
        mpath = PATHS[instrument]
        mtime = os.path.getmtime(mpath)
        open_t = get_open_t(mpath, mtime)
        ib_min = int(st.number_input("IB duration (min)", 15, 240, 60, 15,
                                     key="live_ibmin"))
        lb_label = st.selectbox("Probability lookback", list(LOOKBACKS.keys()),
                                index=2, key="live_lookback",
                                help="Only days within this window feed the live "
                                     "probabilities — recent regimes are usually more "
                                     "relevant than the full 10 years.")

        if source_kind == "Dhan (live)":
            cid = st.text_input("Dhan client_id", value=cfg["dhan"].get("client_id", ""),
                                key="live_cid")
            tok = st.text_input("Dhan access_token", value=cfg["dhan"].get("access_token", ""),
                                type="password", key="live_tok")
            if st.button("💾 Save credentials"):
                cfg["dhan"] = {"client_id": cid.strip(), "access_token": tok.strip()}
                _save_live_cfg(cfg)
                st.success("Saved to live_config.json (gitignored).")
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
    if source_kind == "Dhan (live)":
        if not (cfg["dhan"].get("client_id") and cfg["dhan"].get("access_token")):
            st.warning("Enter your Dhan client_id and access_token in the sidebar, then Save. "
                       "Get them from web.dhan.co → Profile → Access DhanHQ APIs.  "
                       "Install the SDK with:  `pip install dhanhq`")
            return
        try:
            source = live_feed.DhanSource(cfg["dhan"]["client_id"], cfg["dhan"]["access_token"],
                                          instruments=cfg.get("instruments") or None)
        except Exception as e:
            st.error(f"Could not initialise Dhan: {e}")
            return
        now_t = None
    else:
        source = live_feed.DemoSource(load_min(mpath, mtime), asof_date, asof_t)
        now_t = asof_t

    run_every = "60s" if (source_kind == "Dhan (live)" and auto) else None

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
    if run_every is None and source_kind == "Dhan (live)":
        if st.button("🔄 Refresh now"):
            st.rerun()


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(FACTS):
        st.title("📈 ORB / IB Strategy Suite")
        st.error("analysis/facts.csv missing — run `python build_facts.py` first.")
        st.stop()

    mode = st.sidebar.radio(
        "Mode",
        ["Edge Backtester", "Day-wise IB Retracement", "Probabilities",
         "Live Market Statistics"],
        key="app_mode")
    st.sidebar.divider()
    if mode == "Probabilities":
        prob_app.render()
        return
    if mode == "Live Market Statistics":
        live_mode()
        return

    st.title("📈 ORB / IB Strategy Suite")
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
