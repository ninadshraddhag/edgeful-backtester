# Sidebar tooltip audit — Sharp backtesting suite

This lists every **sidebar** widget, per mode, that previously had **no `help=`
tooltip**, with the one-line plain-English tooltip suggested for each. Widgets that
already had a `help=` argument are noted as covered and are not repeated.

**STATUS: ✅ APPLIED.** Every suggested tooltip below has been copied into the
relevant widget call in `app.py`, `prob_app.py` and `advanced_mode.py`. The tables
are kept as the reference for what each tooltip says and where it lives.

---

## Shared data sidebar — `data_sidebar()` in `app.py`
Used by **Edge Backtester**, **Day-wise IB Retracement** and **IB50**. Fixing it
once covers all three modes.

| Widget (label) | Key | Has help? | Suggested help text |
|---|---|---|---|
| Instrument | `{key}_inst` | ❌ No | "Pick the market to test. NIFTY 50 / BANK NIFTY are Indian indices; NQ is Nasdaq futures; XAUUSD is gold." |
| Instrument name (upload) | `{key}_nm` | ❌ No | "Short name for the market you are adding, e.g. XAUUSD or NQ." |
| Minute CSV (upload) | `{key}_nf` | ❌ No (label only) | "Upload a minute-bar CSV with columns date, open, high, low, close. Use exchange-local timestamps (e.g. US Eastern for NQ)." |
| Session open (IB starts here) | `{wkey}_so` | ✅ Yes | — |
| Square-off (force exit) | `{wkey}_sq` | ✅ Yes | — |
| IB duration (min) | `{key}_ibmin` | ✅ Yes | — |
| Date range | `{key}_dates` | ❌ No | "Limit the backtest to these dates. Tip: tune on an earlier range, then confirm the edge still works on a later, untouched range." |

---

## Edge Backtester — extra sidebar widget (`main()` in `app.py`)

| Widget (label) | Key | Has help? | Suggested help text |
|---|---|---|---|
| Setup | (radio, no key) | ❌ No | "IB uses the first hour as the opening range; ORB uses just the first 15 minutes — faster and more frequent signals." |

(Plus everything in the shared data sidebar above.)

---

## Day-wise IB Retracement — sidebar (`daywise_mode()` in `app.py`)
All Day-wise-specific sidebar widgets already have tooltips. Only the shared
data-sidebar gaps above apply.

| Widget (label) | Key | Has help? | Suggested help text |
|---|---|---|---|
| Entry condition | `dw_entrycond` | ✅ Yes | — |
| First-candle window (min) | `dw_flt_fcmin` | ✅ Yes | — |
| LONG only if IB LOW formed first | `dw_flt_ll` | ✅ Yes | — |
| LONG only if IB closes ABOVE midpoint | `dw_flt_lc` | ✅ Yes | — |
| LONG only if first candle is GREEN | `dw_flt_lg` | ✅ Yes | — |
| SHORT only if IB HIGH formed first | `dw_flt_sh` | ✅ Yes | — |
| SHORT only if IB closes BELOW midpoint | `dw_flt_sc` | ✅ Yes | — |
| SHORT only if first candle is RED | `dw_flt_sr` | ✅ Yes | — |
| Exit on VWAP close | `dw_vwap_exit` | ✅ Yes | — |

---

## IB50 — sidebar (`ib50_mode()` in `app.py`)
IB50's own controls (direction filters, breach gate, entry/stop/target) live in
the **main area**, not the sidebar, and are mostly already tooltipped. The sidebar
only contains the shared data sidebar, so only those gaps above apply.

---

## Advanced Backtesting — sidebar (`_data_sidebar()` + `_settings_sidebar()` in `advanced_mode.py`)

| Widget (label) | Key | Has help? | Suggested help text |
|---|---|---|---|
| Instrument | `adv_inst` | ❌ No | "Pick the market to test. NQ is Nasdaq futures; XAUUSD is gold; NIFTY 50 / BANK NIFTY are Indian indices." |
| Instrument name (upload) | `adv_nm` | ❌ No | "Short name for the market you are adding, e.g. XAUUSD or NQ." |
| Minute CSV (upload) | `adv_nf` | ❌ No (label only) | "Upload a minute-bar CSV with columns date, open, high, low, close, in exchange-local time." |
| Session open | `adv_so_{instrument}` | ✅ Yes | — |
| Square-off (force exit) | `adv_sq_{instrument}` | ✅ Yes | — |
| Date range | `adv_dates` | ❌ No | "Limit the backtest to these dates. Features get a 90-day warm-up before the start so indicators are primed." |
| Execution timeframe | `adv_exec_tf` | ✅ Yes | — |
| Higher timeframe (HTF) | `adv_htf_tf` | ✅ Yes | — |
| NARROW < % | `adv_cprn` | ✅ Yes | — |
| WIDE > % | `adv_cprw` | ❌ No | "Central Pivot Range wider than this % of price counts as WIDE (a volatile, trending day type)." |
| CPR basis | `adv_cprb` | ❌ No | "Whether the Central Pivot Range is built from the previous Day or previous Week." |
| ATR period (days) | `adv_atrp` | ❌ No | "Number of days used to measure Average True Range (typical daily move). 14 is standard." |
| FVG min gap % | `adv_fvgm` | ✅ Yes | — |

---

## Probabilities — sidebar (`render()` in `prob_app.py`)

| Widget (label) | Key | Has help? | Suggested help text |
|---|---|---|---|
| Instrument | `f_instrument` | ❌ No | "Choose the market to analyse. NQ and XAUUSD also offer named sessions (New York, London, Asia, Globex)." |
| Session (24h only) | `f_session` | ✅ Yes | — |
| Setup | `f_setup` | ❌ No | "IB uses the first hour as the opening range; ORB uses only the first 15 minutes." |
| IB duration (min) | `f_ibmin` | ✅ Yes | — |
| Date range | `f_dates_{inst}_{session}` | ❌ No | "Only days in this range feed the probabilities. Compare an earlier range with a later one to see if an edge is stable." |
| Day of week | `f_dow` | ❌ No | "Keep only these weekdays. Deselect days to see whether the edge is specific to certain days." |
| Gap type | `f_gap` | ❌ No | "Keep only Gap Up, Flat or Gap Down opens (a gap is where today opens away from the previous close)." |
| Gap % bucket | `f_gapbucket_{inst}_{session}` | ✅ Yes | — |
| Day kind | `f_kind` | ❌ No | "Keep only Inside days (inside yesterday's range), Outside days (beyond both ends) or Normal days." |
| Which extreme formed first | `f_first` | ❌ No | "Keep only days where the opening-box high, or the low, printed first — the basis of the first-move-fade edge." |
| First candle | `f_fc_dur` | ❌ No | "Length of the session's first candle used by the Direction filter beside it (15, 30 or 60 minutes)." |
| Direction | `f_fc_dir` | ✅ Yes | — |
| IB/ORB size (% of price) | `f_sizepct_...` | ✅ Yes | — |
| Bull extension level | `f_bull_ext` | ❌ No | "How far above the box (in box-widths) to measure the up-move reach probability. 1.0 = one full range past the high." |
| Bear extension level | `f_bear_ext` | ❌ No | "How far below the box (in box-widths) to measure the down-move reach probability. 1.0 = one full range past the low." |
| Bull retracement level | `f_bull_retr` | ❌ No | "How deep a pull-back below the box high to measure (in box-widths), after the high breaks. 0.5 = halfway back down the box." |
| Bear retracement level | `f_bear_retr` | ❌ No | "How deep a bounce above the box low to measure (in box-widths), after the low breaks. 0.5 = halfway back up the box." |

---

## Live Market Statistics — sidebar (`live_mode()` in `app.py`)

| Widget (label) | Key | Has help? | Suggested help text |
|---|---|---|---|
| Source | `live_src` | ❌ No | "Demo replay steps through any past day; Kotak Neo streams a live feed once you log in." |
| Instrument | `live_inst` | ❌ No | "Market to classify live. Its 10-year history supplies the conditional probabilities." |
| IB duration (min) | `live_ibmin` | ✅ Yes | — |
| Probability lookback | `live_lookback` | ✅ Yes | — |
| Gap % band | `live_gap_band` | ✅ Yes | — |
| Consumer Key (Kotak) | `kotak_ckey` | ❌ No | "Your Kotak Neo API Consumer Key from the developer portal." |
| Consumer Secret (Kotak) | `kotak_csec` | ❌ No | "Your Kotak Neo API Consumer Secret. Kept only to log you in." |
| Registered mobile | `kotak_mob` | ✅ Yes | — |
| Trading password (Kotak) | `kotak_pwd` | ❌ No | "Your Kotak Neo account trading password. Used only to request a login OTP." |
| Enter OTP | `kotak_otp` | ❌ No | "The one-time passcode texted to your registered mobile." |
| Replay date (Demo) | `live_date` | ❌ No | "The past day to replay minute by minute." |
| As-of time (minute of day) | `live_asof` | ❌ No | "Pretend it is this time of day — the panel shows only data up to here, with no look-ahead." |

---

## Summary

| Mode | Sidebar widgets missing `help=` |
|---|---|
| Shared data sidebar (Edge / Day-wise / IB50) | 4 (Instrument, upload name, upload CSV, Date range) |
| Edge Backtester (extra) | 1 (Setup) |
| Day-wise IB Retracement (own) | 0 |
| IB50 (own sidebar) | 0 |
| Advanced Backtesting | 8 (Instrument, upload name, upload CSV, Date range, WIDE >%, CPR basis, ATR period) — 7 distinct + upload label |
| Probabilities | 12 |
| Live Market Statistics | 9 |

Priorities: fix the **shared data sidebar** first (covers three modes at once),
then **Probabilities** (most user-facing free mode with the most gaps), then
**Advanced Backtesting** and **Live Market Statistics**.
