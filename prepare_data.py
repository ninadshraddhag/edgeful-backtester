"""
prepare_data.py — package instrument data for sharing / cloud deployment
========================================================================
Converts every discovered minute CSV (the legacy C:\\ files and anything in
data/) into compressed Parquet inside data/, which is what you commit to
GitHub and what Streamlit Cloud reads. Parquet is ~5x smaller than CSV and
loads faster.

Usage:
    python prepare_data.py            # convert all known instruments
    python prepare_data.py "D:\\NQ.csv" NQ        # add an external CSV as 'NQ'
"""
import os
import sys

import pandas as pd

import data_store
import build_facts


def convert(name: str, src: str):
    dst = os.path.join(data_store.DATA_DIR, f"{name}_minute.parquet")
    if (os.path.exists(dst)
            and os.path.getmtime(dst) >= os.path.getmtime(src)
            and not src.lower().endswith(".parquet")):
        print(f"[{name}] up to date -> {dst}")
        return
    if src.lower().endswith(".parquet"):
        print(f"[{name}] already parquet -> {src}")
        return
    print(f"[{name}] reading {src} ...")
    df = build_facts.clean_min(pd.read_csv(src))
    keep = df[["date", "open", "high", "low", "close"]]
    os.makedirs(data_store.DATA_DIR, exist_ok=True)
    keep.to_parquet(dst, index=False)
    mb_in = os.path.getsize(src) / 1e6
    mb_out = os.path.getsize(dst) / 1e6
    print(f"[{name}] {len(keep):,} rows  {mb_in:.0f} MB -> {mb_out:.1f} MB  ({dst})")
    # remove a redundant CSV twin inside data/ (the parquet replaces it)
    csv_twin = os.path.join(data_store.DATA_DIR, f"{name}_minute.csv")
    if os.path.abspath(src) == os.path.abspath(csv_twin):
        os.remove(csv_twin)
        print(f"[{name}] removed CSV twin (parquet kept)")


def main():
    if len(sys.argv) == 3:                       # external file + name
        convert(sys.argv[2], sys.argv[1])
        return
    files = data_store.discover()
    if not files:
        print("No instruments found (legacy C:\\ paths or data/).")
        return
    for name, path in files.items():
        convert(name, path)
    print("\nDone. Commit the data/ folder to share these instruments.")


if __name__ == "__main__":
    main()
