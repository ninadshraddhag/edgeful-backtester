"""
guide_content.py — plain-English help content for the Sharp backtesting suite.
==============================================================================
Pure DATA only (no Streamlit, no imports). Any mode can read these dicts to
render a glossary, per-mode "how to use" guides, and a first-login welcome tour.

Everything here is written for a retail trader who has NEVER seen quant jargon.
Every abbreviation is spelled out the first time it appears inside each entry.

Exposes:
    GLOSSARY          dict[str, str]        — term -> plain-English definition
    MODE_GUIDES       dict[str, dict]       — mode name -> {tagline, steps, tips, example}
    WELCOME_TOUR      list[dict]            — first-login tour cards {title, body}
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1) GLOSSARY  —  25-40 terms, plain English, tiny examples where they help.
# ─────────────────────────────────────────────────────────────────────────────

GLOSSARY = {
    "IB (Initial Balance)":
        "The Initial Balance (IB) is the price range of the first hour after the "
        "market opens — the highest high and lowest low in those first 60 minutes. "
        "Traders watch it because the rest of the day often reacts to it. Example: "
        "if the first hour trades between 100 and 105, the IB is 100-105.",

    "ORB (Opening Range Breakout)":
        "ORB stands for Opening Range Breakout. It is the same idea as the Initial "
        "Balance but uses a shorter window — the first 15 minutes instead of the "
        "first hour. A shorter window means faster, more frequent signals.",

    "IB / ORB range":
        "The range is simply the height of the box: the high minus the low of the "
        "opening window. Example: an IB high of 105 and IB low of 100 gives a range "
        "of 5 points. Many targets and stops in this app are measured in multiples "
        "of this range, so 'one range' always means this width.",

    "Extension (x range)":
        "How far price travels BEYOND the opening box, measured in multiples of the "
        "range. A 1.0x extension means price moved one full box-width past the edge. "
        "Example: box is 100-105 (range 5). A 1.0x up-extension reaches 110 "
        "(105 + one range). A 0.5x extension reaches 107.5.",

    "Retracement (x range)":
        "How far price pulls BACK into the box after touching or breaking an edge, "
        "measured in multiples of the range. A retracement entry waits for a "
        "pull-back before joining the move. Example: box 100-105, price taps 105 "
        "then dips to 102.5 — that is a 0.5x retracement (halfway back down the box).",

    "PDH / PDL (Previous Day High / Low)":
        "PDH is the Previous Day's High and PDL is the Previous Day's Low — the "
        "highest and lowest prices of yesterday's session. They act as magnets and "
        "barriers today, so the app measures how often price reaches them.",

    "Touch (of a level)":
        "A 'touch' means price actually traded AT the level during the same session "
        "— not just that the level exists. Important: if the day OPENS beyond a level "
        "(say it gaps above the previous day high), it only counts as a touch once "
        "price pulls BACK to that level. Simply opening past it does not count.",

    "Gap %":
        "The gap is how far today opened away from the previous close, as a "
        "percentage. Example: previous close 100, today opens 100.5 → a +0.5% gap up. "
        "Opening at 99.5 → a -0.5% gap down. For 24-hour instruments the gap is "
        "measured against the SAME session's previous close (London vs London).",

    "Gap buckets":
        "Instead of just 'up' or 'down', gaps are sorted into clean size bands so you "
        "can see which gap SIZE carries the edge: 0-0.2%, 0.2-0.5%, 0.5-1%, and 1%+ "
        "(each mirrored for down-gaps). A tiny 0.1% gap behaves very differently from "
        "a big 1.5% gap.",

    "Gap fill":
        "A gap 'fills' when price trades back to the previous close, erasing the "
        "opening jump. Example: previous close 100, today gaps up to 100.5, then "
        "trades back down and touches 100 — the gap has filled.",

    "First-move fade":
        "A pattern where the FIRST extreme to form inside the opening box is usually "
        "the WRONG way — so you 'fade' it (bet on the opposite side). Example: if the "
        "high forms first, history says the low is more likely to break, so you lean "
        "short. 'Fade' means trade against the first move.",

    "Gap continuation":
        "The idea that a gap keeps going in its own direction: a gap UP tends to "
        "reach the previous day high, a gap DOWN tends to reach the previous day low. "
        "The Edge Backtester can trade this directly (gap up → long, gap down → short).",

    "Formed-first vs broke-first":
        "Two different moments. 'Formed first' = which edge of the opening box was "
        "printed first WHILE the box was still forming. 'Broke first' = which edge "
        "price pushed through first AFTER the box finished. The first-move-fade edge "
        "links the two: the side that formed first often is NOT the side that breaks.",

    "Single-break day":
        "A day where price breaks out of exactly ONE side of the opening box (only "
        "the high, or only the low) and never breaks the other. These are the "
        "cleanest directional days, so the app measures extensions and retracements "
        "on single-break days only, filtering out choppy both-sides days.",

    "Inside day / Outside day":
        "An INSIDE day trades entirely within yesterday's high-low range (quiet, "
        "compressed). An OUTSIDE day trades both above yesterday's high AND below its "
        "low (wide, volatile). Everything else is a 'Normal' day.",

    "Prev-inside breakout":
        "A setup where YESTERDAY was an inside day (quiet and compressed). "
        "Compression usually resolves with a move, so today often breaks the previous "
        "day high or low — the app shows how often either level gets touched.",

    "Midpoint confirmation":
        "The midpoint is the exact middle of the opening box: (high + low) / 2. If "
        "the box's final candle CLOSES back beyond the midpoint, it confirms the "
        "first move has already been faded — a stronger signal than the fade alone. "
        "Example: high formed first and the box closes below its midpoint → confirms "
        "the low is likely to break.",

    "First-candle direction filter":
        "Looks at the direction of the session's very first candle (its close versus "
        "its open over the first 15, 30, or 60 minutes). A green first candle (close "
        "above open) is a bullish vote; red is bearish. Use it to only take longs on "
        "bullish opens, for instance.",

    "Session views (NQ / XAUUSD)":
        "The 24-hour instruments (Nasdaq futures 'NQ' and gold 'XAUUSD') can be sliced "
        "into named trading sessions, each with its own opening box: New York "
        "(09:30-16:00 US Eastern Time), Globex/overnight (18:00 through 16:00, labeled "
        "by its close date), London (03:00-09:30), and Asia (21:30-03:00). The gap is "
        "always measured against the SAME session's previous close.",

    "IB size (% of price)":
        "The opening range width expressed as a percentage of price, so it can be "
        "compared fairly across instruments and across years. Example: a 50-point box "
        "on a 20,000 index is 0.25%. Use it to keep only quiet days or only wide days.",

    "R-multiple (R)":
        "R is your risk on one trade — the distance from entry to stop. Results in R "
        "tell you the payoff relative to what you risked. +2R means you made twice "
        "what you risked; -1R means you lost a full unit of risk. If you risk 1% of "
        "your account per trade, net R is roughly your percent return.",

    "Expectancy":
        "The average profit (or loss) per trade, across every trade. Positive "
        "expectancy means the strategy makes money on average. Example: +0.30 R per "
        "trade means each trade is worth, on average, 0.3 times your risk. This is the "
        "single most important number for judging an edge.",

    "Profit factor (PF)":
        "Total money won divided by total money lost. Above 1.0 means profitable. "
        "Example: PF of 1.5 means you won $1.50 for every $1.00 you lost. Be "
        "suspicious of a very high PF built on only a handful of trades — it usually "
        "will not last.",

    "Win rate":
        "The percentage of trades that finished in profit. Example: 55% win rate = "
        "just over half your trades won. On its own it means little — a 40% win rate "
        "can still be very profitable if the wins are much bigger than the losses.",

    "Max drawdown (max DD)":
        "The biggest peak-to-valley drop in your running profit — the worst losing "
        "stretch you would have had to sit through. Example: equity climbs to +50R, "
        "falls to +20R, then recovers → the max drawdown was -30R. Smaller is calmer.",

    "Scale-out (half at TP1, runner to TP2)":
        "An exit plan that banks profit in stages. You close HALF the position at the "
        "first target (TP1), then move the stop to breakeven so the trade can no "
        "longer lose, and let the other half run to a further target (TP2). This locks "
        "in gains while keeping upside on the 'runner'.",

    "Square-off":
        "The forced exit time at the end of the session — every open position is "
        "closed here so nothing is carried overnight. Example: square-off at 16:00 "
        "means any trade still open is exited at the 16:00 price, win or lose.",

    "Entry / stop as retracement %":
        "In the retracement modes, entry and stop are set as a percentage of the "
        "opening range measured back INTO the box. Entry 25% = wait for price to pull "
        "back a quarter of the box before entering. Stop 75% = the stop sits three "
        "quarters into the box (it must be deeper than the entry).",

    "TP1 / TP2 as extension %":
        "The two profit targets, set as a percentage of the opening range BEYOND the "
        "box edge. TP1 100% = one full range past the edge; TP2 150% = one and a half "
        "ranges past. Example: box 100-105 (range 5), a 100% up-target is at 110.",

    "Optimizer":
        "A tool that automatically tests many rule combinations at once (different "
        "targets, stops, filters) and ranks them by the metric you choose. It finds "
        "the best-looking settings on your data — but see the overfitting warning.",

    "Overfitting":
        "When a strategy is tuned so tightly to past data that it captures random "
        "luck instead of a real edge, and then fails on new data. Warning signs: "
        "sky-high profit factor, very few trades, or a result that falls apart on a "
        "different date range. Always sanity-check the optimizer's favourites.",

    "In-sample vs out-of-sample":
        "In-sample is the data you built and tuned the strategy on. Out-of-sample is "
        "fresh data the strategy has NEVER seen. A genuine edge keeps working "
        "out-of-sample; a curve-fit one collapses. The honest test is to tune on an "
        "earlier date range and confirm it still works on a later, untouched one.",

    "Session VWAP":
        "VWAP is the Volume-Weighted Average Price — the average price paid so far in "
        "the session, weighted by how much traded at each price. It acts as a "
        "fair-value line. Some strategies exit when price closes on the wrong side of "
        "it (a long exits on a close below VWAP).",

    "Breach gate":
        "An optional on/off rule in the IB50 strategy that only allows a trade "
        "depending on whether the opening-box edge has already been broken. "
        "'Require Not-Breached' = enter only from inside the box (a pure pull-back); "
        "'Require Breached' = enter only after a breakout, then a retest.",

    "Confluence (portfolio)":
        "Combining several separate strategies into ONE portfolio. When the "
        "strategies are unrelated, their good and bad patches cancel out, which "
        "smooths the overall equity curve — the safest way to grow returns without "
        "just risking more on a single bet.",

    "Point value ($ per point)":
        "How many dollars one point of price movement is worth for one contract. "
        "Example: the full-size Nasdaq future (NQ) is $20 per point; the micro (MNQ) "
        "is $2. It only affects the contract quantity shown; results in dollars and "
        "in R are unaffected.",

    "Expectancy vs sample size":
        "A number is only as trustworthy as the count of trades behind it. This app "
        "flags any result built on fewer than ~30 trades as 'low confidence'. Prefer "
        "edges with hundreds of trades — a great-looking result on 12 trades is "
        "usually noise, not a signal.",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2) MODE_GUIDES  —  keyed EXACTLY by the app's mode names.
# ─────────────────────────────────────────────────────────────────────────────

MODE_GUIDES = {

    "Edge Backtester": {
        "tagline":
            "Test the classic opening-range edges (fade, gap continuation, plain "
            "breakout) with targets and stops measured in box-widths.",
        "steps": [
            "In the sidebar, pick your Instrument and confirm Session open, "
            "Square-off, IB duration and Date range.",
            "Choose the Setup: IB (first hour) or ORB (first 15 minutes).",
            "On the Single Strategy tab, pick a Side logic (e.g. First-move fade or "
            "Gap continuation).",
            "Set Target (x range) and Stop (x range) with the sliders — this is your "
            "reward-to-risk.",
            "Optionally narrow Day of week, Gap type, Day kind and the First-side "
            "filter.",
            "Read the win rate, expectancy, profit factor and equity curve that "
            "appear below.",
            "Open the Optimizer tab to sweep many combinations, then tick the best "
            "rows into the Confluence tab to combine them.",
        ],
        "tips": [
            "Watch the trade count: the app warns below ~30 trades because small "
            "samples are unreliable.",
            "A high profit factor on very few trades is a red flag for overfitting — "
            "prefer many trades with steady expectancy.",
            "Before trusting an optimizer winner, re-run it on a different Date range "
            "to check the edge is real (out-of-sample).",
            "Combine only UNRELATED strategies in Confluence — that is what actually "
            "smooths the curve.",
        ],
        "example":
            "To test the fade on Nasdaq: set Instrument = NQ, Setup = IB, Side logic "
            "= First-move fade, Target = 1.0x, Stop = 0.5x, leave all day filters on, "
            "and read the expectancy.",
    },

    "Day-wise IB Retracement": {
        "tagline":
            "Build a different pull-back plan for each weekday, with staged exits "
            "(half at target one, runner to target two).",
        "steps": [
            "In the sidebar pick your Instrument, session times and Date range, and "
            "choose an Entry condition (any touch, pure retracement, or retest).",
            "Optionally switch on the IB directional filters (first-side, midpoint, "
            "green/red first candle) to be more selective.",
            "For each weekday panel, toggle Trade this day and Allow long / Allow "
            "short.",
            "Set Entry retr %, Stop retr %, Target 1 (ext %) and Target 2 (ext %) — "
            "the stop % must be larger than the entry %.",
            "Optionally set Min/Max IB size % and an Entry cutoff time to filter "
            "days.",
            "Click 'Run day-wise backtest' and review the win rate, expectancy, "
            "drawdown and per-weekday breakdown.",
            "Use the per-weekday or combined Optimizer to auto-find settings, then "
            "'Apply' them back into the panels.",
        ],
        "tips": [
            "The stop percentage must exceed the entry percentage, or that day is "
            "skipped — the app warns you.",
            "'Copy Monday to all days' is a fast way to start, then tweak each "
            "weekday from there.",
            "The optimizer result is an in-sample best — validate it on a date "
            "sub-range before trusting it.",
            "Set a realistic Min IB size % to skip flat, low-volatility days that add "
            "noise.",
        ],
        "example":
            "To test a Monday-only long pull-back: enable only Monday, Allow long on, "
            "Entry retr = 25%, Stop retr = 75%, TP1 = 100%, TP2 = 150%, then Run.",
    },

    "IB50": {
        "tagline":
            "A faithful port of the IB Retracement indicator: direction filters, an "
            "optional breakout gate, and dollar-risk position sizing reported in R.",
        "steps": [
            "In the sidebar choose your Instrument, session times, IB duration and "
            "Date range.",
            "Under Direction filters, combine Formation order, IB close location and "
            "VWAP with AND (all agree) or OR (any one).",
            "Optionally open the Breach gate to require the box to be Not-Breached "
            "(pull-back) or Breached (breakout retest).",
            "Set Entry %, Stop % and Target % (all as a percentage of the box), then "
            "Risk per trade ($) and Point value.",
            "On the Strategy tab, read Net R, win rate, profit factor and the equity "
            "curve.",
            "Use the Optimizer tab to sweep entry/stop/target and even the filters "
            "themselves.",
            "Add promising configs to the Confluence tab to build and stress-test a "
            "portfolio.",
        ],
        "tips": [
            "Keep 'Realistic fills' on — turning it off can roughly double the result "
            "with edge you cannot actually trade.",
            "You must enable at least one direction filter, or the strategy takes no "
            "trades (this matches the original indicator).",
            "In Confluence, turn on the cost model — commissions and slippage can "
            "quietly erase a thin edge.",
            "Use the Confluence train/holdout split: a real stack keeps most of its "
            "return out-of-sample.",
        ],
        "example":
            "To test gold's New York pull-back: set Instrument = XAUUSD, session "
            "09:30-16:00 ET, Formation + VWAP on with AND, Entry 25% / Stop 100% / "
            "Target 100%, Risk $500, and read Net R.",
    },

    "Advanced Backtesting": {
        "tagline":
            "Write a strategy the way you would on paper — a stack of entry rules "
            "plus separate exit rules — with shorts mirrored automatically.",
        "steps": [
            "In the sidebar pick your Instrument, session times, Date range, and set "
            "Execution timeframe, Higher timeframe, CPR and Detection settings.",
            "Under Entry Conditions, toggle LONG and SHORT and choose how the entry "
            "fills (signal-candle close or next open).",
            "Add condition rows (price vs EMA, sweeps PDL, RSI, FVG forms, day type, "
            "time window...). All rows must be true together.",
            "Under Exit Conditions set the Stop loss placement and trigger, the "
            "Target mode, and any optional trailing or indicator exits.",
            "Set Capital and Risk per trade (% of equity), then click 'Run "
            "backtest'.",
            "Review the metrics, then explore the Equity, Trade browser, Chart "
            "window, Trade log and Optimizer tabs.",
        ],
        "tips": [
            "Write every rule from the LONG point of view — shorts mirror "
            "automatically (above becomes below, PDH becomes PDL).",
            "Check the Day-type playbook first: it shows the historical odds for the "
            "filters you added.",
            "Pick 1-minute execution if you use 1-minute Fair Value Gap entries, or "
            "signals will not line up.",
            "The optimizer's best rows are in-sample — confirm them on a held-out "
            "date range before trusting them.",
        ],
        "example":
            "To test the seeded sample trade: keep the 'sweeps PDL' + 'FVG forms' "
            "entry rows, set Stop loss = FVG invalidation on candle close, Target = "
            "Strict Risk:Reward at 2R, then Run backtest.",
    },

    "Probabilities": {
        "tagline":
            "Browse the raw odds behind every edge across 10 years of data — no "
            "trading, just answers. This mode is free and costs no credits.",
        "steps": [
            "In the sidebar pick an Instrument (and, for NQ/XAUUSD, a Session), then "
            "the Setup (IB or ORB) and IB duration.",
            "Set the Date range and narrow Day of week, Gap type, Gap % bucket, Day "
            "kind and which extreme formed first.",
            "Optionally add a first-candle Direction filter and an IB size (% of "
            "price) band.",
            "Read the break probabilities, the first-move-fade card and the midpoint "
            "confirmation stats.",
            "Scroll to Extensions & Retracements and the Previous-Day High/Low cards "
            "for reach odds.",
            "Or just type a plain-English question in the 'Ask a question' box (e.g. "
            "'Nifty Mondays gap up') and press Ask.",
        ],
        "tips": [
            "Watch the sample size — the app warns below 30 matching days; treat thin "
            "slices as indicative only.",
            "Deltas are shown in percentage points versus the instrument's baseline, "
            "so you can see the real edge, not just the raw number.",
            "Extensions and retracements are measured on single-break days only, so "
            "they describe genuinely directional days.",
            "Use this mode to FIND an edge, then rebuild it in Day-wise or IB50 to "
            "backtest it with real entries and exits.",
        ],
        "example":
            "To check gap continuation: set Gap type = Gap Up, then read the "
            "'Gap Up → touch PDH' card for the historical percentage.",
    },

    "Live Market Statistics": {
        "tagline":
            "Classifies today as it forms and shows the live historical odds for "
            "this exact day type, with a demo replay or a live data feed.",
        "steps": [
            "In the sidebar choose the Source: Demo replay (any past day) or Kotak "
            "Neo for a live feed.",
            "Pick the Instrument and IB duration, and choose a Probability lookback "
            "window (e.g. 2 years).",
            "Optionally narrow the Gap % band so the odds match days with a similar "
            "gap to today.",
            "In Demo replay, pick a Replay date and drag the 'As-of time' slider to "
            "step through the day.",
            "Read the day-forming badges (gap, first side, levels) and the live "
            "probability cards.",
            "Check the extension targets, retracement depths and your saved day-wise "
            "plan for today, then download the PDF session report.",
        ],
        "tips": [
            "Recent regimes usually matter more than ancient history — a 1-to-2-year "
            "lookback is often more relevant than all 10 years.",
            "Full probabilities only unlock once the opening box has finished "
            "forming; before that you see gap-only stats.",
            "A narrow Gap % band gives more relevant odds but a smaller sample — "
            "watch for the low-confidence warning.",
            "These are historical frequencies, not guarantees — treat rare day types "
            "and thin samples with caution.",
        ],
        "example":
            "To rehearse a past day: set Source = Demo replay, pick a Replay date, "
            "set lookback = 2 years, and slide As-of time to 11:00 to see the odds "
            "mid-morning.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 3) WELCOME_TOUR  —  4-5 first-login cards (title + markdown body).
# ─────────────────────────────────────────────────────────────────────────────

WELCOME_TOUR = [
    {
        "title": "Welcome to Sharp",
        "body":
            "**Sharp is a probability-driven intraday backtesting suite.** It studies "
            "how the market behaves in the first part of the trading day and lets you "
            "test simple, rule-based strategies against **10 years of minute-by-minute "
            "data**.\n\n"
            "It covers four instruments: **NIFTY 50** and **BANK NIFTY** (Indian "
            "indices), **NQ** (Nasdaq-100 futures) and **XAUUSD** (gold). Everything is "
            "intraday — nothing is ever carried overnight.",
    },
    {
        "title": "How credits work",
        "body":
            "**Each backtest run costs 1 credit.** That includes running the Edge "
            "Backtester, Day-wise, IB50 and Advanced Backtesting engines, and their "
            "optimizers.\n\n"
            "**Browsing the Probabilities mode is completely free** — explore the odds "
            "as much as you like without spending a credit.\n\n"
            "Run low? **Email sharpbacktester@gmail.com to top up your credits.**",
    },
    {
        "title": "Where to start",
        "body":
            "A simple workflow that works well:\n\n"
            "1. **Open Probabilities** and hunt for an edge — a day type where the odds "
            "clearly lean one way (it is free to browse).\n"
            "2. **Rebuild that edge** as real entries and exits in **Day-wise IB "
            "Retracement** or **IB50**.\n"
            "3. **Backtest it**, then combine unrelated winners in a **Confluence** "
            "portfolio to smooth the ride.",
    },
    {
        "title": "Reading your results",
        "body":
            "Four numbers tell most of the story:\n\n"
            "- **Win rate** — the share of trades that made money.\n"
            "- **Expectancy** — the average profit per trade (the single most "
            "important number; it must be positive).\n"
            "- **Profit factor (PF)** — money won divided by money lost; above 1.0 is "
            "profitable.\n"
            "- **Max drawdown** — the worst losing stretch you would have sat "
            "through.\n\n"
            "**A caution:** these results are *in-sample* — measured on past data the "
            "strategy was tuned on. A real edge still works on fresh, unseen data, so "
            "always re-check winners on a different date range.",
    },
    {
        "title": "Please read: disclaimer",
        "body":
            "**Sharp is an educational and research tool only.** It is **not "
            "investment advice** and nothing here is a recommendation to buy or sell "
            "anything.\n\n"
            "**Past performance does not guarantee future results.** Backtests are "
            "simplified models of the real market and cannot capture every cost or "
            "risk. Markets can and do behave in ways history never showed.\n\n"
            "**You trade entirely at your own risk.** Do your own research and never "
            "risk money you cannot afford to lose.",
    },
]
