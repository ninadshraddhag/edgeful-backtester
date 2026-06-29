# Thinking Machine — ICT strategy research loop

A local, self-contained backtesting platform that turns a plain-English trading
idea into runnable code, backtests it on real minute data, and acts as a quant
researcher: it explains *why* the result happened and proposes one concrete
improvement you can accept or reject.

It lives inside `orb_ib_backtester/` as its own package so it can reuse the
shared parquet data in `../data/` **without touching** the existing Streamlit
`app.py` / `engine.py` at the repo root.

## Layout

```
thinking_machine/
├── app.py                 # the persistent idea → backtest → insight loop
├── core/
│   ├── indicators.py      # vectorised: ema, atr, fair_value_gaps,
│   │                      #             order_blocks, liquidity_sweeps
│   ├── strategy.py        # StrategySpec — the machine-readable idea
│   ├── backtester.py      # event-driven engine (one position at a time)
│   ├── metrics.py         # win rate, RR, streaks, MoM P/L, drawdown
│   ├── visualizer.py      # Plotly candles + trade markers + equity (HTML)
│   └── calendar_view.py   # monthly P/L heatmap (Edgeful-style)
├── ai/
│   └── agent.py           # parse_idea, write/load code, analyze, suggest
├── data/
│   ├── loader.py          # reads ../data/*.parquet, resamples, RTH filter
│   └── mock_generator.py  # synthetic OHLCV so it runs with zero data
├── strategies/
│   └── current_strategy.py  # AUTO-GENERATED; overwritten each iteration
└── reports/               # generated *_trades.html and *_calendar.html
```

## Run

```bash
cd thinking_machine
python app.py                                   # interactive, real NQ, 5min
python app.py --instrument XAUUSD --timeframe 15min --max-days 250
python app.py --mock                            # synthetic data, no parquet
python app.py --demo                            # non-interactive smoke test
```

Instruments: `NQ`, `XAUUSD` (US Eastern RTH 09:30–16:00), `NIFTY`, `BANKNIFTY`
(IST 09:15–15:30).

## The loop

1. You type an idea, e.g. *"fade liquidity sweeps at the 9:30 open, 1.5R target"*.
2. `ai/agent.py` translates it into a `StrategySpec` and writes
   `strategies/current_strategy.py` (real, importable code built on
   `core/indicators.py`).
3. `core/backtester.py` runs it; metrics print to the CLI and two HTML charts
   are written to `reports/`.
4. The agent prints a **Quant Researcher Insight** and **one** concrete
   improvement.
5. Answer `y` to apply — the strategy code is overwritten and re-run, and the
   agent reports whether expectancy improved or got worse.

## Notes

- The agent is intentionally **LLM-free** (deterministic keyword parsing + a
  metrics decision tree) so the platform is fully offline and reproducible.
- All return stats are in **R** (P/L ÷ per-trade risk) so they compare across
  instruments; cash P/L uses per-instrument point values and is also shown.
- Trade simulation is conservative: stop is assumed hit before target when both
  are touched in the same bar; open positions flatten at session close.
- Requires `pandas`, `numpy`, `plotly`, `pyarrow` (already in the parent env).
