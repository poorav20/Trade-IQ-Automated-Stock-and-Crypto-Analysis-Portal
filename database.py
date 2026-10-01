"""
database.py — SQLite database setup and helper functions
Creates and manages the local trading.db database
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "trading.db")


def get_connection():
    """Return a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)


def _ensure_column(cursor, table_name: str, column_name: str, column_sql: str):
    """Add a column to an existing SQLite table if it is missing."""
    columns = {row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})")}
    if column_name not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")


def init_db():
    """
    Create all required tables if they don't already exist.
    Run this ONCE before using any other module.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Table 1: Price history — stores OHLCV data per ticker per day
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      INTEGER,
            UNIQUE(ticker, date)          -- no duplicate rows
        )
    """)

    # Table 2: News sentiment — one row per headline
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            headline    TEXT    NOT NULL,
            source      TEXT,
            score       REAL,             -- VADER compound: -1.0 to +1.0
            fetched_at  TEXT    NOT NULL
        )
    """)

    # Table 3: ML predictions — stores model outputs for auditing
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT    NOT NULL,
            predicted_date  TEXT    NOT NULL,
            predicted_price REAL,
            signal          TEXT,         -- BUY / SELL / HOLD
            confidence      REAL,         -- 0.0 to 1.0
            created_at      TEXT    NOT NULL,
            UNIQUE(ticker, predicted_date)
        )
    """)

    # Table 4: Trade log — paper-trade history (auto_trader.py)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker         TEXT    NOT NULL,
            action         TEXT    NOT NULL,
            quantity       REAL,
            entry_price    REAL,
            exit_price     REAL,
            pnl            REAL,
            capital_before REAL,
            capital_after  REAL,
            signal_source  TEXT,
            confidence     REAL,
            reasoning      TEXT,
            sl_price       REAL,
            tp_price       REAL,
            timestamp      TEXT    NOT NULL
        )
    """)

    # Table 5: Open positions — current holdings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            ticker      TEXT  PRIMARY KEY,
            quantity    REAL  DEFAULT 0,
            entry_price REAL  DEFAULT 0,
            sl_price    REAL,
            tp_price    REAL,
            exit_plan   TEXT,
            entry_time  TEXT
        )
    """)

    # Table 6: Portfolio state — virtual cash balance
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id       INTEGER PRIMARY KEY CHECK (id = 1),
            cash     REAL    NOT NULL DEFAULT 100000.0,
            starting REAL    NOT NULL DEFAULT 100000.0
        )
    """)
    _ensure_column(cursor, "trades", "reasoning", "TEXT")
    _ensure_column(cursor, "trades", "sl_price", "REAL")
    _ensure_column(cursor, "trades", "tp_price", "REAL")
    _ensure_column(cursor, "portfolio", "sl_price", "REAL")
    _ensure_column(cursor, "portfolio", "tp_price", "REAL")
    _ensure_column(cursor, "portfolio", "exit_plan", "TEXT")
    _ensure_column(cursor, "portfolio_state", "starting", "REAL NOT NULL DEFAULT 100000.0")
    cursor.execute("""
        INSERT OR IGNORE INTO portfolio_state (id, cash, starting)
        VALUES (1, 100000.0, 100000.0)
    """)

    conn.commit()
    conn.close()
    print("[OK] Database initialised at:", DB_PATH)


def upsert_price(ticker: str, date: str, open_p: float, high: float,
                 low: float, close: float, volume: int):
    """Insert or replace a single OHLCV row."""
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO price_history
            (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (ticker, date, open_p, high, low, close, int(volume or 0)))
    conn.commit()
    conn.close()


def insert_sentiment(ticker: str, headline: str, source: str,
                     score: float, fetched_at: str):
    """Insert one news‑sentiment row."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO news_sentiment (ticker, headline, source, score, fetched_at)
        VALUES (?, ?, ?, ?, ?)
    """, (ticker, headline, source, score, fetched_at))
    conn.commit()
    conn.close()


def save_prediction(ticker: str, predicted_date: str, predicted_price: float,
                    signal: str, confidence: float, created_at: str):
    """Insert or replace an ML prediction row."""
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO ml_predictions
            (ticker, predicted_date, predicted_price, signal, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ticker, predicted_date, predicted_price, signal, confidence, created_at))
    conn.commit()
    conn.close()


def get_price_df(ticker: str, days: int = 90):
    """Return a pandas DataFrame of the last `days` rows for a ticker."""
    import pandas as pd
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT date, open, high, low, close, volume
        FROM   price_history
        WHERE  ticker = ?
        ORDER  BY date DESC
        LIMIT  ?
        """,
        conn, params=(ticker, days)
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_sentiment_df(ticker: str, limit: int = 20):
    """Return recent news sentiment rows for a ticker."""
    import pandas as pd
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT headline, source, score, fetched_at
        FROM   news_sentiment
        WHERE  ticker = ?
        ORDER  BY fetched_at DESC
        LIMIT  ?
        """,
        conn, params=(ticker, limit)
    )
    conn.close()
    return df


if __name__ == "__main__":
    init_db()
