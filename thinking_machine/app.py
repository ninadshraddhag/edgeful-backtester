"""Thinking Machine - interactive strategy research loop.

Flow per iteration:
  a) you type a trading idea in plain English
  b) the agent translates it into a StrategySpec and writes runnable code
     (strategies/current_strategy.py) built on core/indicators.py
  c) the backtester runs that strategy on real (or mock) data
  d) metrics print to the CLI + Plotly chart + monthly heatmap are written
  e) the agent analyses the result and proposes ONE concrete improvement
  f) you choose whether to apply it (which overwrites the strategy code & re-runs)

Usage:
  python app.py                         # interactive, real NQ data, 5min
  python app.py --instrument XAUUSD --timeframe 15min --max-days 250
  python app.py --mock                  # synthetic data, no parquet needed
  python app.py --demo                  # non-interactive smoke test

Run from inside the thinking_machine/ directory.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make `core`, `ai`, `data` importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai import agent
from core import backtester, metrics as metrics_mod
from core import visualizer, calendar_view
from data import loader

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def banner(instrument, timeframe, df):
    span = f"{df['date'].min()} -> {df['date'].max()}" if len(df) else "no data"
    print("\n" + "#" * 64)
    print("#  THINKING MACHINE - ICT strategy research loop")
    print(f"#  instrument={instrument}  timeframe={timeframe}  bars={len(df):,}")
    print(f"#  data span: {span}")
    print("#" * 64)
    print("Describe a strategy idea, e.g.:")
    print("  * 'fade liquidity sweeps at the 9:30 open, 1.5R target'")
    print("  * 'long bullish FVG with 20 EMA trend filter, 2R'")
    print("  * 'order block reversion both directions, 1 atr stop'")
    print("Type 'quit' to exit, 'show' to re-open the last charts.\n")


def run_and_report(df, spec, instrument, pv, tag):
    """Run the backtest, print metrics, write visuals, return (metrics, result)."""
    result = backtester.run(df, spec, instrument=instrument, point_value=pv)
    m = metrics_mod.compute_metrics(result.trades)
    print("\n" + metrics_mod.format_report(m, spec.describe(), instrument))

    chart = os.path.join(REPORTS, f"{instrument}_{tag}_trades.html")
    heat = os.path.join(REPORTS, f"{instrument}_{tag}_calendar.html")
    visualizer.plot_trades(df, result.trades, instrument, chart)
    calendar_view.monthly_pnl_heatmap(result.trades, instrument, heat)
    print(f"  charts -> {os.path.relpath(chart)}")
    print(f"          -> {os.path.relpath(heat)}")
    return m, result


def ask(prompt, demo_answers):
    if demo_answers is not None:
        ans = demo_answers.pop(0) if demo_answers else "quit"
        print(prompt + ans)
        return ans
    try:
        return input(prompt).strip()
    except EOFError:
        return "quit"


def main():
    ap = argparse.ArgumentParser(description="Thinking Machine backtester")
    ap.add_argument("--instrument", default="NQ", choices=loader.list_instruments())
    ap.add_argument("--timeframe", default="5min")
    ap.add_argument("--max-days", type=int, default=300)
    ap.add_argument("--mock", action="store_true", help="use synthetic data")
    ap.add_argument("--demo", action="store_true", help="non-interactive smoke test")
    args = ap.parse_args()

    print(f"Loading {args.instrument} ({args.timeframe}, last {args.max_days} days, "
          f"{'mock' if args.mock else 'real'})...")
    df = loader.load_data(args.instrument, timeframe=args.timeframe,
                          max_days=args.max_days, mock=args.mock)
    pv = loader.point_value(args.instrument)
    banner(args.instrument, args.timeframe, df)

    # Canned conversation for --demo.
    demo_answers = None
    if args.demo:
        demo_answers = [
            "fade liquidity sweeps at the 9:30 open with a 1.5R target",
            "y",      # apply suggested improvement
            "quit",
        ]

    iteration = 0
    while True:
        idea = ask("\n[idea]> ", demo_answers)
        if not idea:
            continue
        if idea.lower() in ("quit", "exit", "q"):
            print("Session ended. Generated strategy left in strategies/current_strategy.py")
            break
        if idea.lower() == "show":
            print("Charts are in the reports/ folder - open the latest *_trades.html.")
            continue

        iteration += 1
        # (b) translate idea -> code, then load it back (executes generated code).
        spec = agent.parse_idea(idea)
        path = agent.write_strategy(spec)
        spec = agent.load_spec(path)
        print(f"\n[agent] translated to code -> {os.path.relpath(path)}")
        print(f"[agent] {spec.describe()}")

        # (c+d) run + report
        m, _ = run_and_report(df, spec, args.instrument, pv, f"it{iteration}")

        # (e) analyse + suggest
        print("\n" + agent.analyze_performance(m, spec))
        explanation, new_spec = agent.suggest_improvement(m, spec)
        print("\nSUGGESTED IMPROVEMENT")
        print("-" * 60)
        print(explanation)

        # (f) apply?
        choice = ask("\nApply this improvement? [y/N] ", demo_answers)
        if choice.lower() in ("y", "yes"):
            path = agent.write_strategy(new_spec)        # overwrite strategy logic
            new_spec = agent.load_spec(path)
            print(f"[agent] overwrote strategy -> {os.path.relpath(path)}")
            print(f"[agent] re-running: {new_spec.describe()}")
            iteration += 1
            m2, _ = run_and_report(df, new_spec, args.instrument, pv, f"it{iteration}")
            print("\n" + agent.analyze_performance(m2, new_spec))
            delta = m2["expectancy_r"] - m["expectancy_r"]
            print(f"\n[agent] expectancy change: {delta:+.4f} R/trade "
                  f"({'improved' if delta > 0 else 'worse' if delta < 0 else 'flat'}).")
        else:
            print("[agent] kept the original strategy.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
