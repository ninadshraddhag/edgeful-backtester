"""
report.py — PDF report builders for the ORB/IB Strategy Suite.

  build_daywise_pdf(...) — full backtest report: system definition (entry/exit
      criteria a newcomer can follow), performance summary incl. Avg RR, equity &
      drawdown chart, weekday/yearly/outcome breakdowns, and the complete trade log.
  build_live_pdf(...)    — live session snapshot: framework explainer, today's
      day-type classification, live probabilities, key levels, and today's plan.

No Streamlit imports — pure functions returning PDF bytes (unit-testable).
"""
import io
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos

import daywise

PAGE_W = 190          # usable width (A4, 10mm margins)

_REPL = {"×": "x", "—": "-", "–": "-", "·": "|", "≥": ">=", "≤": "<=",
         "→": "->", "₹": "Rs", "⚠": "!", "’": "'", "‘": "'", "“": '"', "”": '"'}


def _s(t):
    t = str(t)
    for k, v in _REPL.items():
        t = t.replace(k, v)
    return t.encode("latin-1", "replace").decode("latin-1")


class _PDF(FPDF):
    title_text = ""

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120)
            self.cell(0, 5, _s(self.title_text), align="R",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(0)
            self.ln(1)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}}", align="C")
        self.set_text_color(0)


def _new_pdf(title):
    pdf = _PDF("P", "mm", "A4")
    pdf.title_text = title
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_margins(10, 10, 10)
    pdf.add_page()
    return pdf


def _h1(pdf, t):
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 9, _s(t), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def _h2(pdf, t):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(232, 236, 241)
    pdf.cell(0, 7, _s(" " + t), fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)


def _para(pdf, t, size=9.5, style=""):
    pdf.set_font("Helvetica", style, size)
    pdf.multi_cell(0, 4.6, _s(t))
    pdf.ln(1)


def _table(pdf, df, widths, font=8, align=None, header_h=5.5, row_h=4.6):
    cols = list(df.columns)
    align = align or ["C"] * len(cols)

    def head():
        pdf.set_font("Helvetica", "B", font)
        pdf.set_fill_color(225, 229, 235)
        for i, c in enumerate(cols):
            pdf.cell(widths[i], header_h, _s(c), border=1, align="C", fill=True)
        pdf.ln(header_h)
        pdf.set_font("Helvetica", "", font)

    head()
    for _, row in df.iterrows():
        if pdf.get_y() > 278 - row_h:
            pdf.add_page()
            head()
        for i, c in enumerate(cols):
            v = row[c]
            if isinstance(v, float):
                v = f"{v:,.2f}" if abs(v) < 10000 else f"{v:,.0f}"
            pdf.cell(widths[i], row_h, _s(v), border=1, align=align[i])
        pdf.ln(row_h)
    pdf.ln(2)


def _kv_grid(pdf, pairs, per_row=3, font=9.5):
    """Metric grid: label over bold value, `per_row` cells per line."""
    w = PAGE_W / per_row
    for i in range(0, len(pairs), per_row):
        chunk = pairs[i:i + per_row]
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(110)
        for k, _v in chunk:
            pdf.cell(w, 4.2, _s(k), align="L")
        pdf.ln(4.2)
        pdf.set_text_color(0)
        pdf.set_font("Helvetica", "B", font + 1.5)
        for _k, v in chunk:
            pdf.cell(w, 6, _s(v), align="L")
        pdf.ln(7.5)
    pdf.ln(1)


def _metrics(pnl):
    n = len(pnl)
    if n == 0:
        return None
    wins = pnl > 0
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    cum = np.cumsum(pnl)
    dd = cum - np.maximum.accumulate(cum)
    return {
        "trades": n, "win_rate": wins.mean(), "net": pnl.sum(),
        "expectancy": pnl.mean(), "pf": (gp / gl) if gl > 0 else np.inf,
        "max_dd": dd.min(),
        "avg_win": pnl[wins].mean() if wins.any() else 0.0,
        "avg_loss": pnl[~wins].mean() if (~wins).any() else 0.0,
        "best": pnl.max(), "worst": pnl.min(), "cum": cum, "dd": dd,
    }


def _tmin(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def _equity_png(dates, cum, dd):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.6, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1]})
    ax1.plot(dates, cum, color="#2196F3", lw=1.4)
    ax1.fill_between(dates, cum, 0, color="#2196F3", alpha=0.10)
    ax1.set_ylabel("Cumulative PnL (pts)")
    ax1.set_title("Equity curve & drawdown", fontsize=11)
    ax1.grid(alpha=0.3)
    ax2.fill_between(dates, dd, 0, color="#F44336", alpha=0.35)
    ax2.plot(dates, dd, color="#F44336", lw=0.8)
    ax2.set_ylabel("Drawdown (pts)")
    ax2.grid(alpha=0.3)
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


DISCLAIMER = ("This report is generated from historical simulation/statistics for research "
              "purposes only. Past performance does not guarantee future results. PnL is in "
              "index points per 1 unit and excludes costs, slippage and taxes. This is not "
              "investment advice.")


# ────────────────────────────────────────────────────────────────────────────────
#  DAY-WISE BACKTEST REPORT
# ────────────────────────────────────────────────────────────────────────────────

def _system_definition(pdf, open_t, close_t, configs):
    ib_end = _tmin(open_t + 60)
    _h2(pdf, "1. The Trading System - Entry & Exit Criteria")
    _para(pdf,
          "INSTRUMENT SETUP - The Initial Balance (IB) is the price range formed in the "
          f"first 60 minutes of the session ({_tmin(open_t)} to {ib_end}). IB High / IB Low "
          "are the highest high and lowest low of that hour; IB Range = IB High - IB Low. "
          "All entry, stop and target levels below are expressed as a percentage of this "
          "IB Range. Each weekday has its own independent parameter set (table below).")
    _para(pdf,
          "ENTRY (Long) - After the IB completes, a limit order is placed at "
          "IB High - (Entry% x IB Range). An Entry% of 0 means entering exactly at the IB "
          "High (a breakout entry); 10 means waiting for a 10% retracement of the IB range "
          "below the IB High before entering. The order fills on the first 1-minute candle "
          "whose range spans the level, optionally only before the entry cutoff time. "
          "Short entries mirror this off the IB Low: entry = IB Low + (Entry% x IB Range). "
          "Long and short can use separate parameter sets on the same day.")
    _para(pdf,
          "INITIAL STOP-LOSS - For longs: IB High - (Stop% x IB Range), i.e. a deeper "
          "retracement than the entry (Stop% must exceed Entry%; 100% = the opposite IB "
          "boundary). Shorts mirror off the IB Low. Risk per trade = entry - stop distance.")
    _para(pdf,
          "PROFIT TARGETS & TRADE MANAGEMENT - TP1 = IB High + (TP1% x IB Range) and "
          "TP2 = IB High + (TP2% x IB Range) for longs (extensions beyond the IB). "
          "Half the position exits at TP1, after which the stop moves to breakeven "
          "(the entry price). The remaining half exits at TP2, at breakeven if price "
          "returns to entry, or at the end-of-day square-off. If neither the stop nor TP1 "
          f"is reached, the entire position exits at the {_tmin(close_t)} square-off close. "
          "No position is ever carried overnight. Conservative tie rule: if the stop and a "
          "target are both touched within the same 1-minute candle, the stop is assumed "
          "to have been hit first.")
    _para(pdf,
          "FILTERS - Per weekday: trading on/off, long/short direction toggles, an IB-size "
          "filter (IB Range as % of price must lie between Min and Max), and an optional "
          "latest-entry cutoff time. PnL convention: points per 1 unit; a full winner books "
          "0.5 x (TP1 - entry) + 0.5 x (TP2 - entry). R-multiple = PnL / initial risk.")

    rows = []
    for day in daywise.WEEKDAYS:
        cfg = configs.get(day)
        if not cfg:
            continue
        fmt = lambda p: f"{p[0]}/{p[1]} -> {p[2]}/{p[3]}"
        lp = fmt(daywise.dir_params(cfg, True)) if cfg.get("allow_long") else "-"
        sp = fmt(daywise.dir_params(cfg, False)) if cfg.get("allow_short") else "-"
        cut = _tmin(cfg["cutoff"]) if cfg.get("cutoff") else "-"
        rows.append({"Day": day, "On": "Yes" if cfg.get("trade") else "No",
                     "Long e/s -> t1/t2 (%)": lp, "Short e/s -> t1/t2 (%)": sp,
                     "IB size %": f"{cfg.get('min_size', 0)}-{cfg.get('max_size', 0)}",
                     "Cutoff": cut})
    _para(pdf, "Per-weekday configuration used in this backtest "
               "(e = entry retracement %, s = stop retracement %, t1/t2 = target extensions %):",
          style="I")
    _table(pdf, pd.DataFrame(rows), [22, 12, 56, 56, 26, 18], font=8)


def build_daywise_pdf(instrument, period, open_t, close_t, configs, trades):
    pdf = _new_pdf(f"Day-wise IB Retracement - {instrument}")
    _h1(pdf, "Day-wise IB Retracement - Backtest Report")
    _kv_grid(pdf, [
        ("Instrument", instrument),
        ("Period", f"{period[0]} to {period[1]}"),
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Session", f"{_tmin(open_t)} - {_tmin(close_t)} (square-off)"),
        ("IB window", f"{_tmin(open_t)} - {_tmin(open_t + 59)}"),
        ("Engine", "1-min candles, scale-out 50/50"),
    ])

    _system_definition(pdf, open_t, close_t, configs)

    if trades is None or len(trades) == 0:
        _h2(pdf, "2. Results")
        _para(pdf, "No trades were generated for this configuration and period.")
        _para(pdf, DISCLAIMER, size=7.5, style="I")
        return bytes(pdf.output())

    t = trades.sort_values("date").reset_index(drop=True)
    pnl = t["pnl"].values.astype(float)
    m = _metrics(pnl)
    rr = (m["avg_win"] / abs(m["avg_loss"])) if m["avg_loss"] else float("inf")
    avg_r = float(t["r_mult"].mean()) if "r_mult" in t.columns else float("nan")
    pf_s = f"{m['pf']:.2f}" if np.isfinite(m["pf"]) else "inf"

    pdf.add_page()
    _h2(pdf, "2. Performance Summary")
    _kv_grid(pdf, [
        ("Total trades", f"{m['trades']:,}"),
        ("Win rate", f"{m['win_rate'] * 100:.1f}%"),
        ("Net PnL (pts)", f"{m['net']:,.0f}"),
        ("Expectancy (pts/trade)", f"{m['expectancy']:.2f}"),
        ("Profit factor", pf_s),
        ("Max drawdown (pts)", f"{m['max_dd']:,.0f}"),
        ("Avg win (pts)", f"{m['avg_win']:.1f}"),
        ("Avg loss (pts)", f"{m['avg_loss']:.1f}"),
        ("Avg RR (avg win / avg loss)", f"{rr:.2f}"),
        ("Avg R per trade", f"{avg_r:+.2f}R"),
        ("Best trade (pts)", f"{m['best']:,.1f}"),
        ("Worst trade (pts)", f"{m['worst']:,.1f}"),
    ])
    _para(pdf,
          f"Reading the numbers: the system took {m['trades']:,} trades; "
          f"{m['win_rate'] * 100:.1f}% were profitable. Each trade earned "
          f"{m['expectancy']:.2f} pts on average ({avg_r:+.2f}R per unit of risk). The "
          f"deepest losing streak (max drawdown) was {abs(m['max_dd']):,.0f} pts from peak "
          f"equity. Avg RR of {rr:.2f} means the average winner paid {rr:.2f}x the average "
          "loser.", style="I", size=8.5)

    _h2(pdf, "3. Equity Curve & Drawdown")
    dates = pd.to_datetime(t["date"])
    pdf.image(_equity_png(dates, m["cum"], m["dd"]), w=PAGE_W)
    pdf.ln(2)

    pdf.add_page()
    _h2(pdf, "4. Breakdown by Weekday")
    wd_rows = []
    for day in daywise.WEEKDAYS:
        sub = t[t["dow"] == day]
        if sub.empty:
            continue
        sm = _metrics(sub["pnl"].values.astype(float))
        wd_rows.append({"Day": day, "Trades": sm["trades"],
                        "Win %": f"{sm['win_rate'] * 100:.1f}",
                        "Net (pts)": f"{sm['net']:,.0f}",
                        "Expectancy": f"{sm['expectancy']:.2f}",
                        "Avg R": f"{sub['r_mult'].mean():+.2f}" if "r_mult" in sub else "-",
                        "Max DD": f"{sm['max_dd']:,.0f}"})
    _table(pdf, pd.DataFrame(wd_rows), [26, 22, 24, 30, 30, 26, 30])

    _h2(pdf, "5. Breakdown by Year")
    t["_yr"] = pd.to_datetime(t["date"]).dt.year
    yr_rows = []
    for yr, sub in t.groupby("_yr"):
        sm = _metrics(sub["pnl"].values.astype(float))
        yr_rows.append({"Year": int(yr), "Trades": sm["trades"],
                        "Win %": f"{sm['win_rate'] * 100:.1f}",
                        "Net (pts)": f"{sm['net']:,.0f}",
                        "Expectancy": f"{sm['expectancy']:.2f}",
                        "Max DD": f"{sm['max_dd']:,.0f}"})
    _table(pdf, pd.DataFrame(yr_rows), [24, 24, 26, 34, 34, 32])

    _h2(pdf, "6. Exit Outcome Mix")
    oc_rows = []
    label = {"stop": "Stopped out (full loss)",
             "tp1+tp2": "TP1 + TP2 (full winner)",
             "tp1+be": "TP1 hit, runner at breakeven",
             "tp1+eod": "TP1 hit, runner to square-off",
             "eod": "Square-off exit (no stop/TP1)"}
    for oc, sub in t.groupby("outcome"):
        oc_rows.append({"Outcome": label.get(oc, oc), "Count": len(sub),
                        "% of trades": f"{len(sub) / len(t) * 100:.1f}",
                        "Avg PnL (pts)": f"{sub['pnl'].mean():.2f}"})
    _table(pdf, pd.DataFrame(oc_rows).sort_values("Count", ascending=False),
           [70, 28, 34, 40])

    pdf.add_page()
    _h2(pdf, f"7. Complete Trade Log ({len(t):,} trades)")
    log = pd.DataFrame({
        "#": range(1, len(t) + 1),
        "Date": pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d"),
        "Day": t["dow"].str[:3],
        "Dir": t["direction"].str.upper(),
        "Entry": t["entry"].round(1),
        "Risk": t.get("risk", pd.Series(["-"] * len(t))),
        "PnL": t["pnl"].round(1),
        "R": t.get("r_mult", pd.Series(["-"] * len(t))),
        "Outcome": t["outcome"],
    })
    _table(pdf, log, [14, 22, 14, 16, 26, 22, 24, 18, 28], font=7, row_h=4.2)

    _para(pdf, DISCLAIMER, size=7.5, style="I")
    return bytes(pdf.output())


# ────────────────────────────────────────────────────────────────────────────────
#  LIVE SESSION REPORT
# ────────────────────────────────────────────────────────────────────────────────

def build_live_pdf(instrument, feat, probs, plan_rows):
    pdf = _new_pdf(f"Live Market Statistics - {instrument}")
    _h1(pdf, "Live Market Statistics - Session Report")
    asof = _tmin(feat.get("now_t", 0))
    _kv_grid(pdf, [
        ("Instrument", instrument),
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("As-of time", asof),
        ("Session phase", feat.get("phase", "-")),
        ("Last price", f"{feat['price']:,.1f}" if feat.get("price") else "-"),
        ("History", "10-yr minute data (facts table)"),
    ])

    _h2(pdf, "1. The Framework - How to Read This Report")
    _para(pdf,
          "INITIAL BALANCE (IB) - the price range of the first 60 minutes of the session. "
          "OPENING RANGE (ORB) - the first 15 minutes. The IB High/Low act as the day's "
          "reference levels: statistically, whichever IB extreme forms FIRST tends to be "
          "faded - the opposite side breaks far more often (the 'first-move-fade' edge). "
          "EXTENSIONS are measured beyond the IB in units of IB Range (a 0.5x bull "
          "extension = IB High + half an IB range). PDH/PDL are the previous day's high "
          "and low. GAP is today's open vs yesterday's close (threshold +/-0.15%).")
    _para(pdf,
          "The probabilities below are historical frequencies among all past days that "
          "match today's profile - same weekday, same gap type, and (once the IB is "
          "complete) the same IB first-side - over the 10-year minute dataset. They are "
          "base rates, not predictions; sample sizes are shown throughout.")

    _h2(pdf, "2. Today's Day-Type Classification")
    cls = [("Weekday", feat.get("dow", "-")),
           ("Gap", f"{feat.get('gap_type', '-')} ({feat.get('gap_pct', 0):+.2f}%)"
            if feat.get("gap_type") else "-"),
           ("Session phase", feat.get("phase", "-"))]
    if feat.get("ib_high"):
        cls += [("IB High / Low", f"{feat['ib_high']:,.1f} / {feat['ib_low']:,.1f}"),
                ("IB Range", f"{feat['ib_range']:,.1f} pts ({feat.get('ib_size_pct', 0):.2f}% of price)"),
                ("IB first side", f"{feat['ib_first_side'].upper()} formed first")]
    if feat.get("orb_high"):
        cls += [("ORB High / Low", f"{feat['orb_high']:,.1f} / {feat['orb_low']:,.1f}")]
    flags = []
    for lab, key in [("PDH", "broke_pdh"), ("PDL", "broke_pdl"),
                     ("IB High", "broke_ib_high"), ("IB Low", "broke_ib_low")]:
        if key in feat:
            flags.append(f"{lab}: {'BROKEN' if feat[key] else 'intact'}")
    if flags:
        cls.append(("Levels so far", " | ".join(flags)))
    _kv_grid(pdf, cls, per_row=2)

    _h2(pdf, "3. Live Probabilities for This Day Type")
    s = probs.get("matched") or probs.get("gap_slice")
    n = probs.get("n_matched", probs.get("n_gap", 0))
    if s:
        _kv_grid(pdf, [
            ("P(IB High breaks)", f"{s['high'] * 100:.1f}%"),
            ("P(IB Low breaks)", f"{s['low'] * 100:.1f}%"),
            ("P(both sides)", f"{s['both'] * 100:.1f}%"),
            ("P(one side only)", f"{s['one'] * 100:.1f}%"),
            ("P(neither)", f"{s['none'] * 100:.1f}%"),
            ("Matching days (n)", f"{n:,}"),
        ])
        if "fade_opp_break" in probs:
            _para(pdf,
                  f"FIRST-MOVE-FADE - Today the IB {feat['ib_first_side'].upper()} formed "
                  f"first. Historically on matching days, the opposite "
                  f"{probs['fade_opp'].upper()} broke {probs['fade_opp_break'] * 100:.1f}% "
                  f"of the time (and broke FIRST {probs.get('fade_opp_first', 0) * 100:.1f}% "
                  f"of the time). Sample: {probs.get('n_matched', 0):,} days.", style="B")
        if n < 30:
            _para(pdf, f"CAUTION: only {n} matching historical days - low confidence.",
                  style="BI", size=8.5)
    pdt = probs.get("pd_table")
    if pdt is not None and not pdt.empty and feat.get("gap_type"):
        row = pdt[pdt["Gap"] == feat["gap_type"]]
        if not row.empty:
            r = row.iloc[0]
            _para(pdf,
                  f"PREVIOUS-DAY LEVELS - On {feat['gap_type']} {feat.get('dow', '')}s, price "
                  f"reached the PDH {r['reach PDH %']:.1f}% and the PDL {r['reach PDL %']:.1f}% "
                  f"of days (n={int(r['days'])}).")

    if feat.get("ib_high"):
        _h2(pdf, "4. Key Levels & Extension Targets")
        lv_rows = [{"Level": "IB High", "Price": f"{feat['ib_high']:,.1f}"},
                   {"Level": "IB Low", "Price": f"{feat['ib_low']:,.1f}"}]
        if feat.get("orb_high"):
            lv_rows += [{"Level": "ORB High", "Price": f"{feat['orb_high']:,.1f}"},
                        {"Level": "ORB Low", "Price": f"{feat['orb_low']:,.1f}"}]
        if feat.get("prev_high"):
            lv_rows += [{"Level": "PDH", "Price": f"{feat['prev_high']:,.1f}"},
                        {"Level": "PDL", "Price": f"{feat['prev_low']:,.1f}"},
                        {"Level": "Prev close", "Price": f"{feat['prev_close']:,.1f}"}]
        _table(pdf, pd.DataFrame(lv_rows), [60, 50])

        ext = probs.get("ext")
        if ext:
            _para(pdf, "Extension targets (price and the % of matching days that "
                       "historically reached them):", style="I")
            ext_rows = [{"x IB Range": e["level"],
                         "Bull target": f"{e['up_price']:,.1f}",
                         "Reached %": f"{e['up_p'] * 100:.1f}" if e["up_p"] is not None else "-",
                         "Bear target": f"{e['dn_price']:,.1f}",
                         "Reached % ": f"{e['dn_p'] * 100:.1f}" if e["dn_p"] is not None else "-"}
                        for e in ext]
            _table(pdf, pd.DataFrame(ext_rows), [28, 38, 28, 38, 28])

    if plan_rows:
        _h2(pdf, "5. Today's Day-wise Retracement Plan (from saved preset)")
        _para(pdf,
              "Levels computed from today's IB using the saved day-wise configuration for "
              f"{feat.get('dow', 'today')}. Management: half exits at TP1, stop moves to "
              "breakeven, runner to TP2; full square-off at session end - no overnight "
              "positions.", style="I")
        plan_df = pd.DataFrame(plan_rows, columns=["Dir", "Entry", "Stop", "TP1", "TP2"])
        for c in ["Entry", "Stop", "TP1", "TP2"]:
            plan_df[c] = plan_df[c].astype(float).round(1)
        _table(pdf, plan_df, [26, 38, 38, 38, 38])

    _para(pdf, DISCLAIMER, size=7.5, style="I")
    return bytes(pdf.output())
