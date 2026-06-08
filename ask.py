"""
ask.py — probability report over the facts table built by build_facts.py
Run:  python ask.py            (full report, both instruments)
"""
import os
import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

FACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "facts.csv")
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def load():
    df = pd.read_csv(FACTS, parse_dates=["date"])
    return df


def pct(x):
    return f"{x*100:5.1f}%"


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def overall(df, tag):
    n = len(df)
    print(f"  sample days                : {n:,}")
    print(f"  HIGH breaks (rest of day)  : {pct(df[f'{tag}_high_break'].mean())}")
    print(f"  LOW  breaks (rest of day)  : {pct(df[f'{tag}_low_break'].mean())}")
    print(f"  BOTH sides break           : {pct(df[f'{tag}_both_break'].mean())}")
    print(f"  ONE side only              : {pct(df[f'{tag}_one_side'].mean())}")
    print(f"  NEITHER side breaks        : {pct(df[f'{tag}_no_break'].mean())}")
    bf = df[f"{tag}_break_first"].value_counts(normalize=True)
    print(f"  breaks HIGH first          : {pct(bf.get('high', 0))}")
    print(f"  breaks LOW  first          : {pct(bf.get('low', 0))}")


def first_side_thesis(df, tag):
    """Given which extreme formed first, what breaks?"""
    print(f"\n  --- conditioned on which extreme formed FIRST inside the {tag.upper()} ---")
    for side in ("high", "low"):
        sub = df[df[f"{tag}_first_side"] == side]
        if sub.empty:
            continue
        opp = "low" if side == "high" else "high"
        print(f"  {side.upper()} formed first  (n={len(sub):,}, {pct(len(sub)/len(df))} of days):")
        print(f"       P(opposite {opp.upper()} breaks)      = {pct(sub[f'{tag}_{opp}_break'].mean())}")
        print(f"       P(same {side.upper()} breaks)          = {pct(sub[f'{tag}_{side}_break'].mean())}")
        bf = sub[f"{tag}_break_first"].value_counts(normalize=True)
        print(f"       P({opp.upper()} breaks FIRST)        = {pct(bf.get(opp, 0))}")


def by_group(df, tag, col, order=None):
    g = df.groupby(col).agg(
        days=(f"{tag}_high_break", "size"),
        high=(f"{tag}_high_break", "mean"),
        low=(f"{tag}_low_break", "mean"),
        both=(f"{tag}_both_break", "mean"),
        one=(f"{tag}_one_side", "mean"),
        none=(f"{tag}_no_break", "mean"),
    )
    if order:
        g = g.reindex([o for o in order if o in g.index])
    for c in ["high", "low", "both", "one", "none"]:
        g[c] = (g[c] * 100).round(1)
    g["days"] = g["days"].astype(int)
    print(g.to_string())


def report_instrument(df, inst):
    section(f"  {inst}   —   {len(df):,} trading days   ({df['date'].min().date()} → {df['date'].max().date()})")

    for tag, label in (("ib", "INITIAL BALANCE  (first 60 min, 09:15–10:14)"),
                       ("orb", "OPENING RANGE BREAKOUT  (first 15 min, 09:15–09:29)")):
        print("\n" + "-" * 78)
        print(f"  {label}")
        print("-" * 78)
        overall(df, tag)
        first_side_thesis(df, tag)

        print(f"\n  --- {tag.upper()} break rates by DAY OF WEEK (% of days) ---")
        by_group(df, tag, "dow", DOW_ORDER)

        print(f"\n  --- {tag.upper()} break rates by GAP TYPE (% of days) ---")
        by_group(df, tag, "gap_type", ["Gap Up", "Flat", "Gap Down"])

        print(f"\n  --- {tag.upper()} break rates: INSIDE DAY vs rest ---")
        tmp = df.copy()
        tmp["day_kind"] = np.where(tmp["inside_day"], "Inside Day",
                          np.where(tmp["outside_day"], "Outside Day", "Normal"))
        by_group(tmp, tag, "day_kind", ["Inside Day", "Normal", "Outside Day"])


def main():
    df = load()
    print(f"Loaded facts table: {len(df):,} rows, {df['date'].min().date()} → {df['date'].max().date()}")
    for inst in df["instrument"].unique():
        report_instrument(df[df["instrument"] == inst].copy(), inst)
    print("\n" + "=" * 78)
    print("  Done. Ask me any conditional question — I'll query analysis/facts.csv.")
    print("=" * 78)


if __name__ == "__main__":
    main()
