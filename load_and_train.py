"""
load_and_train.py — Load all uploaded CSV datasets into the SQLite DB,
                    then retrain the IEEE-enhanced ensemble ML models.

Usage:
    python load_and_train.py

Steps:
  1. Auto-detect all CSV files in the trading-portal directory
  2. Normalise columns → date, open, high, low, close, volume
  3. Upsert into price_history with auto-detected ticker name
  4. Run run_all_predictions_v2() to retrain and save predictions
"""

import os
import re
import pandas as pd
import sqlite3
from datetime import datetime, timezone

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DB_PATH  = os.path.join(BASE_DIR, "trading.db")

# ── Ticker name overrides for known files ──────────────────────────────
FILE_TICKER_MAP = {
    "AAPL.csv":                                     "AAPL",
    "apple_stock.csv":                              "AAPL",
    "Amazon.csv":                                   "AMZN",
    "Microsoft_Stock.csv":                          "MSFT",
    "MSFT(2000-2023).csv":                          "MSFT",
    "MSFT_1986-03-13_2025-02-04.csv":               "MSFT",
    "MSFT_Stock_Data_1986_2026.csv":                "MSFT",
    "15 Years Stock Data of NVDA AAPL MSFT GOOGL and AMZN.csv": None,  # multi-ticker
    "Reliance Industries 1996 to 2020.csv":         "RELIANCE.NS",
    "reliance_data.csv":                            "RELIANCE.NS",
    "RILO - Copy.csv":                              "RELIANCE.NS",
    "RILO - Copy (1).csv":                          "RELIANCE.NS",
    "TCS_NSE_2004-2025.csv":                        "TCS.NS",
    "INFY_15m.csv":                                 "INFY.NS",
    "01-01-2021-TO-31-12-2021INFYALLN.csv":         "INFY.NS",
    "Training_data.csv":                            "NIFTY50",  # Indian index data
}

# CSV files to skip entirely (non-price data)
SKIP_FILES = {
    "Training_data.csv",  # index data — included separately as NIFTY50
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Map raw column names to: date, open, high, low, close, volume.
    Returns None if cannot find required columns.
    """
    col_map = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}

    # Date column
    for alias in ["date", "datetime", "timestamp", "ds", "time"]:
        if alias in cols_lower:
            col_map["date"] = cols_lower[alias]
            break

    # OHLCV
    for target, aliases in {
        "open":   ["open"],
        "high":   ["high"],
        "low":    ["low"],
        "close":  ["close", "adj close", "adjclose", "last"],
        "volume": ["volume", "vol"],
    }.items():
        for alias in aliases:
            if alias in cols_lower:
                col_map[target] = cols_lower[alias]
                break

    required = {"date", "close"}
    if not required.issubset(col_map):
        return None

    # Rename to standard names
    df = df.rename(columns={v: k for k, v in col_map.items()})
    final_cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[final_cols].copy()


def parse_date(series: pd.Series) -> pd.Series:
    """Try to parse dates as ISO strings first, then infer."""
    try:
        return pd.to_datetime(series, format="%Y-%m-%d")
    except Exception:
        return pd.to_datetime(series, infer_datetime_format=True)


def load_csv_to_db(filepath: str, ticker: str, conn: sqlite3.Connection) -> int:
    """
    Load a single CSV file into price_history for the given ticker.
    Returns number of rows inserted/updated.
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"    [ERROR] Cannot read {os.path.basename(filepath)}: {e}")
        return 0

    df = normalise_columns(df)
    if df is None:
        print(f"    [SKIP]  {os.path.basename(filepath)}: missing date/close columns")
        return 0

    try:
        df["date"] = parse_date(df["date"])
    except Exception as e:
        print(f"    [SKIP]  {os.path.basename(filepath)}: date parse failed → {e}")
        return 0

    df = df.dropna(subset=["date", "close"])
    df["date_str"]  = df["date"].dt.strftime("%Y-%m-%d")
    df["close"]     = pd.to_numeric(df["close"],  errors="coerce")
    df["open"]      = pd.to_numeric(df.get("open",  pd.Series(dtype=float)), errors="coerce").fillna(0)
    df["high"]      = pd.to_numeric(df.get("high",  pd.Series(dtype=float)), errors="coerce").fillna(0)
    df["low"]       = pd.to_numeric(df.get("low",   pd.Series(dtype=float)), errors="coerce").fillna(0)
    df["volume"]    = pd.to_numeric(df.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0)
    df = df.dropna(subset=["close"])

    rows = 0
    for _, row in df.iterrows():
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO price_history
                    (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    row["date_str"],
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    int(row["volume"]),
                ),
            )
            rows += 1
        except Exception:
            pass

    conn.commit()
    return rows


def load_multi_ticker_csv(filepath: str, conn: sqlite3.Connection) -> int:
    """
    Handle the '15 Years Stock Data…' file which contains columns like
    NVDA_Close, AAPL_Close, … or a multi-level structure.
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"    [ERROR] {e}")
        return 0

    # Detect per-ticker columns pattern: TICKER_Close etc.
    ticker_pattern = re.compile(r"^([A-Z]+)_(Open|High|Low|Close|Volume|Adj Close)$", re.IGNORECASE)
    tickers_found: dict[str, dict] = {}

    for col in df.columns:
        m = ticker_pattern.match(col.strip())
        if m:
            t, field = m.group(1).upper(), m.group(2).lower().replace(" ", "")
            tickers_found.setdefault(t, {})[field] = col

    # Detect date column
    date_col = None
    for c in df.columns:
        if c.lower() in ("date", "datetime", "timestamp", "ds"):
            date_col = c
            break

    if not date_col or not tickers_found:
        # Fall back: try treating as a wide OHLCV for one ticker
        return 0

    total = 0
    for ticker, cols in tickers_found.items():
        if "close" not in cols:
            continue
        sub = pd.DataFrame()
        sub["date"]   = df[date_col]
        sub["close"]  = pd.to_numeric(df[cols["close"]], errors="coerce")
        sub["open"]   = pd.to_numeric(df.get(cols.get("open",   ""), pd.Series(dtype=float)), errors="coerce").fillna(0)
        sub["high"]   = pd.to_numeric(df.get(cols.get("high",   ""), pd.Series(dtype=float)), errors="coerce").fillna(0)
        sub["low"]    = pd.to_numeric(df.get(cols.get("low",    ""), pd.Series(dtype=float)), errors="coerce").fillna(0)
        sub["volume"] = pd.to_numeric(df.get(cols.get("volume", ""), pd.Series(dtype=float)), errors="coerce").fillna(0)
        sub = sub.dropna(subset=["date", "close"])

        try:
            sub["date"] = parse_date(sub["date"])
        except Exception:
            continue

        sub["date_str"] = sub["date"].dt.strftime("%Y-%m-%d")
        rows = 0
        for _, row in sub.iterrows():
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO price_history
                        (ticker, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ticker, row["date_str"],
                     float(row["open"]), float(row["high"]),
                     float(row["low"]),  float(row["close"]),
                     int(row["volume"])),
                )
                rows += 1
            except Exception:
                pass
        conn.commit()
        total += rows
        print(f"      {ticker}: {rows} rows")

    return total


def load_training_data(filepath: str, conn: sqlite3.Connection) -> int:
    """
    Load Training_data.csv which is already feature-enriched (RSI, MACD…).
    We store only the OHLCV portion as NIFTY50.
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"    [ERROR] {e}")
        return 0

    cols_lower = {c.lower().strip(): c for c in df.columns}
    date_col  = cols_lower.get("date")
    close_col = cols_lower.get("close")
    if not date_col or not close_col:
        return 0

    df["date"]   = parse_date(df[date_col])
    df["close"]  = pd.to_numeric(df[close_col], errors="coerce")
    df["open"]   = pd.to_numeric(df.get(cols_lower.get("open",   ""), pd.Series(dtype=float)), errors="coerce").fillna(0)
    df["high"]   = pd.to_numeric(df.get(cols_lower.get("high",   ""), pd.Series(dtype=float)), errors="coerce").fillna(0)
    df["low"]    = pd.to_numeric(df.get(cols_lower.get("low",    ""), pd.Series(dtype=float)), errors="coerce").fillna(0)
    df["volume"] = pd.to_numeric(df.get(cols_lower.get("volume", ""), pd.Series(dtype=float)), errors="coerce").fillna(0)
    df = df.dropna(subset=["date", "close"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    rows = 0
    for _, row in df.iterrows():
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO price_history
                    (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("NIFTY50", row["date_str"],
                 float(row["open"]), float(row["high"]),
                 float(row["low"]),  float(row["close"]),
                 int(row["volume"])),
            )
            rows += 1
        except Exception:
            pass
    conn.commit()
    return rows


def main():
    print("=" * 65)
    print("  TRADEIQ — Dataset Loader & ML Trainer")
    print("=" * 65)

    # ── 1. Initialise DB ───────────────────────────────────────────────
    from database import init_db
    init_db()
    conn = sqlite3.connect(DB_PATH)

    # ── 2. Load all CSVs ───────────────────────────────────────────────
    print("\n[STEP 1] Loading CSV datasets into database...\n")
    total_rows = 0

    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))

    for fname in csv_files:
        fpath = os.path.join(DATA_DIR, fname)

        # Special handler for Training_data.csv
        if fname == "Training_data.csv":
            print(f"  >> {fname}  -> NIFTY50")
            rows = load_training_data(fpath, conn)
            print(f"     {rows} rows loaded")
            total_rows += rows
            continue

        # Special handler for multi-ticker file
        if fname == "15 Years Stock Data of NVDA AAPL MSFT GOOGL and AMZN.csv":
            print(f"  >> {fname}  -> (multi-ticker)")
            rows = load_multi_ticker_csv(fpath, conn)
            print(f"     {rows} total rows loaded")
            total_rows += rows
            continue

        ticker = FILE_TICKER_MAP.get(fname)
        if ticker is None:
            # Try to infer from filename
            stem = os.path.splitext(fname)[0].upper()
            ticker = re.split(r"[_\-\s]", stem)[0]
            if len(ticker) > 10:
                ticker = ticker[:10]

        print(f"  >> {fname}  -> {ticker}")
        rows = load_csv_to_db(fpath, ticker, conn)
        print(f"     {rows} rows loaded")
        total_rows += rows

    conn.close()
    print(f"\n  Total rows loaded: {total_rows:,}")

    # ── 3. Show DB summary ─────────────────────────────────────────────
    print("\n[STEP 2] Database summary by ticker:\n")
    conn2 = sqlite3.connect(DB_PATH)
    summary = pd.read_sql_query(
        """
        SELECT ticker,
               COUNT(*) AS rows,
               MIN(date) AS earliest,
               MAX(date) AS latest
        FROM   price_history
        GROUP  BY ticker
        ORDER  BY rows DESC
        """,
        conn2,
    )
    conn2.close()
    print(summary.to_string(index=False))

    # ── 4. Train ensemble ML models ────────────────────────────────────
    print("\n" + "=" * 65)
    print("[STEP 3] Training IEEE-enhanced Ensemble ML models...\n")

    try:
        from ml_model_v2 import run_all_predictions_v2
        results = run_all_predictions_v2(save=True)

        if not results.empty:
            print("\n" + "=" * 65)
            print("  ENSEMBLE PREDICTION RESULTS")
            print("=" * 65)
            for _, row in results.iterrows():
                icon = "[BUY] " if row["signal"] == "BUY" else ("[SELL]" if row["signal"] == "SELL" else "[HOLD]")
                print(
                    f"{icon}  {row['ticker']:15s}  "
                    f"Last: {row['last_price']:10.2f}  "
                    f"Pred: {row['predicted_price']:10.2f}  "
                    f"Chg: {row['change_pct']:+.2f}%  "
                    f"RSI: {row.get('rsi', 0):.0f}  "
                    f"Conf: {row['confidence']:.2f}  "
                    f"IND: {row.get('ind_signal','?')}"
                )
            print("=" * 65)
        else:
            print("[!] No predictions generated — check that tickers match price_history.")

    except Exception as e:
        print(f"[ERROR] ML training failed: {e}")
        import traceback; traceback.print_exc()

    # ── 5. Also retrain v1 baseline for completeness ───────────────────
    print("\n[STEP 4] Running v1 linear regression baseline...\n")
    try:
        from ml_model import run_all_predictions
        df_v1 = run_all_predictions(save=False)
        if not df_v1.empty:
            for _, row in df_v1.iterrows():
                icon = "[BUY] " if row["signal"] == "BUY" else ("[SELL]" if row["signal"] == "SELL" else "[HOLD]")
                print(f"  {icon}  {row['ticker']:15s}  Pred: {row['predicted_price']:,.2f}  Signal: {row['signal']}  MAPE: {row['mape']:.2f}%")
    except Exception as e:
        print(f"  [WARN] v1 baseline skipped: {e}")

    print("\n[DONE] All datasets loaded and models retrained successfully!")


if __name__ == "__main__":
    main()
