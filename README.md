# Edgeful — ORB / IB / Advanced Strategy Suite

Streamlit app for probability-driven intraday backtesting:

| Mode | What it does |
|---|---|
| **Edge Backtester** | ORB/IB edge strategies + permutation optimizer |
| **Day-wise IB Retracement** | Per-weekday IB retracement configs + optimizer |
| **Advanced Backtesting** | Composable entry/exit rule builder (EMA/RSI/CPR/ICT/FVG/liquidity sweeps), HTF confluence, trade browser, optimizer |
| **Probabilities** | Ask questions of the per-day facts table |
| **Live Market Statistics** | Day-type classification + live conditional odds |

## Run locally

```
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

or double-click **`run_localhost.bat`** (Windows). Opens http://localhost:8501.

## Data

Instruments are discovered automatically from the **`data/`** folder:
any `NAME_minute.csv` or `NAME_minute.parquet` with columns
`date, open, high, low, close` (1-minute bars) appears in **every mode**.

- **Timestamps must be exchange-local time** (naive, no timezone suffix).
  For NQ / US instruments export the data in **EST**; for XAUUSD use the
  session clock you want the IB measured against.
- **Session open**: auto-detected from the data. For 24-hour instruments
  (NQ Globex, XAUUSD) set the *Session open* override in the sidebar —
  e.g. **09:30** for the NQ cash open (EST). It is remembered per
  instrument (`data/instruments.json`). IB duration stays adjustable per
  mode.
- Add instruments either by dropping a file into `data/` or via the
  **➕ Upload new instrument** option in any data sidebar (uploads are
  saved into `data/` and persist locally).
- Big CSVs: run `python prepare_data.py` to convert them to Parquet
  (~5× smaller — what you should commit to GitHub).
- `analysis/facts.csv` (the probabilities table) builds automatically on
  first boot; rebuild after adding instruments with `python build_facts.py`.

## Share with a friend

**Option A — GitHub (recommended, keeps you in sync):**
1. Create a repo on github.com (private is fine) and push this folder.
2. Friend: `git clone <repo-url>`, then `run_localhost.bat`.
   The `data/` parquet files ship with the repo — zero setup.

**Option B — zip:** zip this folder (the `data/` parquets are inside;
`logs/`, `__pycache__/` and `live_config.json` are not needed) and send it.
They run `run_localhost.bat`.

> `live_config.json` holds broker credentials and is git-ignored — never
> share or commit it.

## Deploy live on Streamlit Community Cloud (free)

1. Push this folder to GitHub (see above). Keep data as **parquet** —
   each file must stay under GitHub's 100 MB limit.
2. Go to **share.streamlit.io** → *Create app* → pick your repo,
   branch `master`/`main`, main file **`app.py`** → Deploy.
3. First boot builds `analysis/facts.csv` automatically (a few minutes),
   then the app is live at `https://<your-app>.streamlit.app` — share the
   URL with anyone.

Cloud notes:
- The cloud filesystem is **ephemeral**: instruments uploaded through the
  UI disappear when the app restarts. For permanent instruments (NQ,
  XAUUSD), commit their parquet to `data/` instead.
- Don't use the Kotak live mode on a public deployment — credentials
  belong on your own machine.
- Free tier gives ~1 GB RAM; the 10-year minute parquets fit, but avoid
  loading many large instruments at once.
