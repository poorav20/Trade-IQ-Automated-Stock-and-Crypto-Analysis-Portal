"""
train_agents.py — TradeIQ Agent Training Pipeline
Inspired by: sanketagarwal/hyperliquid-trading-agent (diary + training loop)

This script:
  1. Auto-discovers all *.csv files in the trading-portal directory
  2. Validates, normalizes and ingests them into trading.db
  3. Re-runs the IEEE-enhanced ensemble ML (ml_model_v2.py) per ticker
  4. Saves per-model metrics to training_report.csv
  5. Prints a final leaderboard of model accuracy

Usage:
    python train_agents.py              # train on all CSVs
    python train_agents.py --ticker AAPL  # train single ticker
    python train_agents.py --retrain    # force retrain (clears old predictions)
    python train_agents.py --report     # print existing training report
"""
import argparse
import os
import glob
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data", "raw")
REPORT_PATH  = os.path.join(BASE_DIR, "training_report.csv")
DB_PATH      = os.path.join(BASE_DIR, "trading.db")

# ── CSV Column Name Normaliser ────────────────────────────────────────────────
COLUMN_ALIASES = {
    # Date variants
    "Date": "date", "DATE": "date", "Datetime": "date", "datetime": "date",
    "timestamp": "date", "Timestamp": "date",
    # Open
    "Open": "open", "OPEN": "open",
    # High
    "High": "high", "HIGH": "high",
    # Low
    "Low": "low", "LOW": "low",
    # Close
    "Close": "close", "CLOSE": "close", "Adj Close": "close", "Adj_Close": "close",
    # Volume
    "Volume": "volume", "VOLUME": "volume", "Vol": "volume",
    # Ticker / Symbol
    "Ticker": "ticker", "Symbol": "symbol", "ticker": "ticker", "symbol": "symbol",
}

REQUIRED_COLS = {"date", "open", "high", "low", "close", "volume"}


def _normalise_df(df: pd.DataFrame, default_ticker: str) -> pd.DataFrame | None:
    """Rename columns to standard names, detect ticker, return clean df."""
    df = df.rename(columns=COLUMN_ALIASES)

    # Check required columns
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"    [SKIP] Missing columns: {missing}")
        return None

    # Parse date
    try:
        df["date"] = pd.to_datetime(df["date"], utc=False)
        df["date"] = df["date"].dt.tz_localize(None)  # strip tz
    except Exception as e:
        print(f"    [SKIP] Date parse error: {e}")
        return None

    # Numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN in core numeric cols
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)

    # Detect / assign ticker
    if "ticker" in df.columns:
        tickers = df["ticker"].unique()
        if len(tickers) == 1:
            df["ticker"] = tickers[0]
        else:
            # Multi-ticker CSV — split into separate groups (handled outside)
            df["_multi_ticker"] = True
    elif "symbol" in df.columns:
        df["ticker"] = df["symbol"]
    else:
        df["ticker"] = default_ticker

    df = df.sort_values("date").reset_index(drop=True)
    return df


def _infer_ticker_from_filename(path: str) -> str:
    """Try to extract a ticker-like string from a CSV filename."""
    name = os.path.splitext(os.path.basename(path))[0]
    # Common patterns: "AAPL", "MSFT(2000-2023)", "15 Years Stock Data of NVDA…"
    import re
    # Look for known tickers near start
    parts = re.split(r"[\s\-_\(\)\.]", name)
    for part in parts:
        if re.match(r"^[A-Z]{2,6}$", part):
            return part
    return name[:20].upper().replace(" ", "_")


# ── Database ingestion ────────────────────────────────────────────────────────
def _ensure_prices_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker  TEXT    NOT NULL,
            date    TEXT    NOT NULL,
            open    REAL,
            high    REAL,
            low     REAL,
            close   REAL    NOT NULL,
            volume  REAL,
            UNIQUE(ticker, date)
        )
    """)
    conn.commit()


def ingest_csv(path: str, conn: sqlite3.Connection) -> list[str]:
    """
    Load a CSV, normalise it, and insert rows into prices table.
    Returns list of tickers ingested.
    """
    print(f"\n  [CSV] {os.path.basename(path)}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"    [ERROR] Cannot read: {e}")
        return []

    default_ticker = _infer_ticker_from_filename(path)
    df_norm = _normalise_df(df, default_ticker)
    if df_norm is None:
        return []

    # Handle multi-ticker CSVs (e.g. "15 Years Stock Data of NVDA AAPL…")
    ingested_tickers = []
    if "_multi_ticker" in df_norm.columns:
        groups = df_norm.groupby("ticker")
        for tkr, grp in groups:
            inserted = _insert_rows(grp, str(tkr), conn)
            print(f"    [{tkr}] {inserted} rows inserted")
            ingested_tickers.append(str(tkr))
    else:
        ticker = df_norm["ticker"].iloc[0]
        inserted = _insert_rows(df_norm, str(ticker), conn)
        print(f"    [{ticker}] {inserted} rows inserted  "
              f"(range: {df_norm['date'].min().date()} → {df_norm['date'].max().date()})")
        ingested_tickers.append(str(ticker))

    return ingested_tickers


def _insert_rows(df: pd.DataFrame, ticker: str, conn: sqlite3.Connection) -> int:
    """Insert rows for a single ticker, skip duplicates."""
    count = 0
    cursor = conn.cursor()
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO prices (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"]),
                float(row.get("open", 0) or 0),
                float(row.get("high", 0) or 0),
                float(row.get("low", 0) or 0),
                float(row["close"]),
                float(row.get("volume", 0) or 0),
            ))
            if cursor.rowcount:
                count += 1
        except Exception:
            continue
    conn.commit()
    return count


# ── Training ──────────────────────────────────────────────────────────────────
def train_ticker(ticker: str) -> dict | None:
    """
    Train the IEEE ensemble on a single ticker and return metrics.
    """
    try:
        from ml_model_v2   import predict_ticker_v2
        from database      import init_db
        init_db()

        result = predict_ticker_v2(ticker, sentiment_score=0.0)
        if result is None:
            return None

        return {
            "ticker":            ticker,
            "signal":            result["signal"],
            "confidence":        result["confidence"],
            "change_pct":        result["change_pct"],
            "mape":              result["mape"],
            "lr_r2":             result["model_scores"].get("LinearRegression", 0),
            "gb_r2":             result["model_scores"].get("GradientBoosting", 0),
            "mlp_r2":            result["model_scores"].get("MLP_Neural", 0),
            "ind_signal":        result["ind_signal"],
            "ind_score":         result["ind_score"],
            "rsi":               result.get("rsi", 0),
            "trained_at":        datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        print(f"    [ERROR] Training {ticker}: {e}")
        return None


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(target_ticker: str = None, retrain: bool = False):
    """Full training pipeline: ingest CSVs → train ML → save report."""

    from database import init_db, get_connection
    init_db()
    conn = get_connection()
    _ensure_prices_table(conn)

    # ── Step 1: Discover and ingest CSVs ─────────────────────────────────────
    print("\n" + "="*70)
    print(" STEP 1 — Discovering and ingesting CSV files")
    print("="*70)

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    all_tickers = set()

    for csv_path in sorted(csv_files):
        tickers = ingest_csv(csv_path, conn)
        all_tickers.update(tickers)

    conn.close()
    print(f"\n  Total tickers in DB: {len(all_tickers)}")
    print(f"  Tickers: {sorted(all_tickers)}")

    # ── Step 2: Determine which tickers to train ──────────────────────────────
    print("\n" + "="*70)
    print(" STEP 2 — Training ML Ensemble on each ticker")
    print("="*70)

    if target_ticker:
        tickers_to_train = [target_ticker]
    else:
        # Also include any tickers already in DB (from previous sessions)
        try:
            conn2 = get_connection()
            rows = conn2.execute("SELECT DISTINCT ticker FROM prices").fetchall()
            conn2.close()
            db_tickers = {r[0] for r in rows}
            all_tickers.update(db_tickers)
        except Exception:
            pass
        tickers_to_train = sorted(all_tickers)

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    results = []
    for i, ticker in enumerate(tickers_to_train, 1):
        print(f"\n  [{i}/{len(tickers_to_train)}] Training {ticker}...")
        metrics = train_ticker(ticker)
        if metrics:
            results.append(metrics)
            signal_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(metrics["signal"], "⚪")
            print(f"    {signal_icon} {metrics['signal']:4s}  "
                  f"conf={metrics['confidence']:.2f}  "
                  f"mape={metrics['mape']:.2f}%  "
                  f"GB_R²={metrics['gb_r2']:.3f}  "
                  f"IND={metrics['ind_signal']}({metrics['ind_score']:+.1f})")
        else:
            print(f"    [SKIP] Insufficient data")

    # ── Step 4: Save report ───────────────────────────────────────────────────
    if results:
        print("\n" + "="*70)
        print(" STEP 3 — Saving training report")
        print("="*70)

        df_report = pd.DataFrame(results)
        df_report = df_report.sort_values("gb_r2", ascending=False)
        df_report.to_csv(REPORT_PATH, index=False)
        print(f"  Saved → {REPORT_PATH}")

        # ── Leaderboard ───────────────────────────────────────────────────────
        print("\n" + "="*70)
        print(" TRAINING LEADERBOARD (sorted by Gradient Boosting R²)")
        print("="*70)
        print(f"  {'Ticker':<18} {'Signal':<6} {'Conf':>5} {'MAPE':>6} {'LR_R²':>7} {'GB_R²':>7} {'MLP_R²':>7} {'IND':>5}")
        print("  " + "-"*65)
        for _, row in df_report.iterrows():
            icon = {"BUY": "BUY ", "SELL": "SELL", "HOLD": "HOLD"}.get(row["signal"], "???")
            print(f"  {row['ticker']:<18} {icon:<6} "
                  f"{row['confidence']:>5.2f} "
                  f"{row['mape']:>6.2f}% "
                  f"{row['lr_r2']:>7.3f} "
                  f"{row['gb_r2']:>7.3f} "
                  f"{row['mlp_r2']:>7.3f} "
                  f"{row['ind_signal']:>5}")
        print("="*70)
        print(f"\n  ✅ Trained {len(results)} tickers successfully")
        print(f"  Best model: {df_report.iloc[0]['ticker']} (GB R²={df_report.iloc[0]['gb_r2']:.3f})")


def print_report():
    """Print the existing training report."""
    if not os.path.exists(REPORT_PATH):
        print("No training report found. Run python train_agents.py first.")
        return
    df = pd.read_csv(REPORT_PATH)
    df = df.sort_values("gb_r2", ascending=False)
    print("\n" + "="*70)
    print(f" TRAINING REPORT (last run)")
    print("="*70)
    print(df.to_string(index=False))


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradeIQ Agent Training Pipeline")
    parser.add_argument("--ticker",  type=str, help="Train a single ticker only")
    parser.add_argument("--retrain", action="store_true", help="Force retrain all")
    parser.add_argument("--report",  action="store_true", help="Print existing report")
    args = parser.parse_args()

    if args.report:
        print_report()
    else:
        run_pipeline(target_ticker=args.ticker, retrain=args.retrain)
