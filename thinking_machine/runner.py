"""One-shot runner — the entry point Claude (the agent) calls per idea.

Given an instrument and a strategy spec (as JSON), it loads YOUR real parquet
data, backtests, prints a metrics report + quant insight, and writes the two
Plotly HTML charts. Designed to be invoked from a single shell command so the
chat-driven agent loop stays clean.

Examples
--------
# standard spec via JSON
python runner.py --instrument NQ --timeframe 5min --max-days 300 \
    --spec '{"name":"sweep_fade","long_entry":["sweep_low"],"short_entry":["sweep_high"],"stop_atr":1.0,"rr":2.5,"entry_after":"09:30","entry_before":"11:00"}'

# same idea across all four instruments
python runner.py --all --spec '{"long_entry":["ob_bull"],"short_entry":["ob_bear"],"rr":2.0}'
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai import agent
from core import backtester, metrics as metrics_mod
from core import visualizer, calendar_view
from core.strategy import StrategySpec
from data import loader

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def run_spec(instrument: str, spec: StrategySpec, timeframe: str = "5min",
             max_days: int = 300, tag: str | None = None, charts: bool = True,
             insight: bool = True) -> dict:
    """Backtest one spec on real data; print report; optionally write charts."""
    instrument = instrument.upper()
    df = loader.load_data(instrument, timeframe=timeframe, max_days=max_days)
    pv = loader.point_value(instrument)
    res = backtester.run(df, spec, instrument=instrument, point_value=pv)
    m = metrics_mod.compute_metrics(res.trades)

    print(metrics_mod.format_report(m, spec.describe(), instrument))
    if insight:
        print("\n" + agent.analyze_performance(m, spec))

    if charts:
        tag = tag or spec.name
        c = visualizer.plot_trades(df, res.trades, instrument,
                                   os.path.join(REPORTS, f"{instrument}_{tag}_trades.html"))
        h = calendar_view.monthly_pnl_heatmap(res.trades, instrument,
                                              os.path.join(REPORTS, f"{instrument}_{tag}_calendar.html"))
        print(f"\n  chart    -> {os.path.relpath(c)}")
        print(f"  calendar -> {os.path.relpath(h)}")
    return m


def spec_from_json(s: str) -> StrategySpec:
    """Build a StrategySpec from JSON, ignoring unknown keys."""
    d = json.loads(s) if s else {}
    fields = set(StrategySpec().__dict__.keys())
    clean = {k: v for k, v in d.items() if k in fields}
    return StrategySpec(**clean)


def main():
    ap = argparse.ArgumentParser(description="Thinking Machine one-shot runner")
    ap.add_argument("--instrument", default="NQ")
    ap.add_argument("--all", action="store_true", help="run across NQ, XAUUSD, NIFTY, BANKNIFTY")
    ap.add_argument("--timeframe", default="5min")
    ap.add_argument("--max-days", type=int, default=300)
    ap.add_argument("--spec", required=True, help="StrategySpec as JSON")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--idea", default=None, help="optional: parse English instead of --spec")
    args = ap.parse_args()

    spec = agent.parse_idea(args.idea) if args.idea else spec_from_json(args.spec)

    targets = loader.list_instruments() if args.all else [args.instrument]
    for inst in targets:
        print("\n" + "=" * 64)
        run_spec(inst, spec, timeframe=args.timeframe, max_days=args.max_days,
                 charts=not args.no_charts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
