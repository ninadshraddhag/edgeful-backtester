"""
Headless smoke test: exercises the real app.py logic against the actual
NIFTY 50 CSV (a slice, for speed) to confirm it works on the installed
pandas / numpy versions before launching the Streamlit UI.
"""
import sys, io, pandas as pd

# import the real functions from the app
import app

CSV = r"C:\NIFTY 50_minute.csv"
ROWS = 120_000   # ~1.3 years of minute candles — enough to generate many trades

print(f"pandas {pd.__version__}")
print(f"Loading first {ROWS:,} rows of {CSV} ...")

raw = pd.read_csv(CSV, nrows=ROWS)
buf = io.BytesIO()
raw.to_csv(buf, index=False)
df = app.load_csv(buf.getvalue(), "slice.csv")
print(f"  loaded {len(df):,} clean candles over {df['date_only'].nunique()} trading days")
print(f"  range: {df['date_only'].min()} -> {df['date_only'].max()}")

print("\nComputing ORB(15) / IB(60) levels ...")
levels = app.compute_levels(df, orb_min=15, ib_min=60)
print(f"  levels for {len(levels)} days")
print(f"  avg ORB range: {levels['orb_range'].mean():.1f} pts | "
      f"avg IB range: {levels['ib_range'].mean():.1f} pts")

# run every strategy with RR=2, full-range SL, and report key metrics
print("\nBacktesting all 12 strategies (RR=2, SL=full range):")
print(f"{'strategy':<18}{'trades':>7}{'win%':>8}{'netPnL':>10}{'maxDD':>9}{'PF':>7}")
print("-" * 59)

for label, key in app.STRATEGIES.items():
    trades = app.run_backtest(df, levels, key, rr=2.0, orb_min=15, ib_min=60, sl_type="range")
    m = app.compute_metrics(trades)
    if not m:
        print(f"{key:<18}{'0':>7}{'  (no trades)':>26}")
        continue
    pf = m["profit_factor"]
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    print(f"{key:<18}{m['total_trades']:>7}{m['win_rate']*100:>7.1f}%"
          f"{m['net_pnl']:>10.0f}{m['max_drawdown']:>9.0f}{pf_s:>7}")

print("\nSMOKE TEST PASSED — app logic runs correctly on this environment.")
