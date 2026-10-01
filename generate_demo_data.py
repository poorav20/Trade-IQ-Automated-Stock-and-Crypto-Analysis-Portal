import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

DB_PATH = 'demo_trading.db'

# Remove if exists
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create Tables
cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_sentiment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        headline TEXT NOT NULL,
        source TEXT,
        score REAL,
        fetched_at TEXT NOT NULL
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS ml_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        predicted_date TEXT NOT NULL,
        predicted_price REAL,
        signal TEXT,
        confidence REAL,
        created_at TEXT NOT NULL,
        UNIQUE(ticker, predicted_date)
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        action TEXT NOT NULL,
        quantity REAL,
        entry_price REAL,
        exit_price REAL,
        pnl REAL,
        capital_before REAL,
        capital_after REAL,
        signal_source TEXT,
        confidence REAL,
        reasoning TEXT,
        sl_price REAL,
        tp_price REAL,
        timestamp TEXT NOT NULL
    )
""")

tickers = ['NVDA', 'AAPL', 'MSFT']
base_date = datetime(2026, 1, 1)

# Generate Sentiment (Positive trend)
for t in tickers:
    for i in range(90): # 90 days
        date = base_date + timedelta(days=i)
        # Fake positive sentiment between 0.2 and 0.9, drifting upward
        score = np.random.uniform(0.2, 0.6) + (i / 180) 
        score = min(score, 1.0)
        cursor.execute("INSERT INTO news_sentiment (ticker, headline, score, fetched_at) VALUES (?, ?, ?, ?)",
                       (t, "Great news for " + t, score, date.strftime('%Y-%m-%d %H:%M:%S')))

# Generate Predictions (High confidence)
for t in tickers:
    for i in range(90):
        date = base_date + timedelta(days=i)
        signal = np.random.choice(['BUY', 'BUY', 'SELL'], p=[0.7, 0.2, 0.1])
        # High confidence for BUY, moderate for SELL
        conf = np.random.uniform(0.75, 0.99) if signal == 'BUY' else np.random.uniform(0.5, 0.8)
        cursor.execute("INSERT INTO ml_predictions (ticker, predicted_date, predicted_price, signal, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                       (t, date.strftime('%Y-%m-%d'), 150 + i, signal, conf, date.strftime('%Y-%m-%d %H:%M:%S')))

# Generate Trades (Highly Profitable)
capital = 100000
for i in range(30):
    t = np.random.choice(tickers)
    date = base_date + timedelta(days=i*3)
    action = 'BUY'
    pnl = np.random.uniform(500, 2500) # Highly profitable trades
    # Occasional small loss to look somewhat realistic
    if np.random.random() > 0.9:
        pnl = -np.random.uniform(100, 500)
    
    capital_before = capital
    capital += pnl
    cursor.execute("INSERT INTO trades (ticker, action, pnl, capital_before, capital_after, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                   (t, action, pnl, capital_before, capital, date.strftime('%Y-%m-%d %H:%M:%S')))

conn.commit()
conn.close()

print(f"Generated demo data in {DB_PATH}")
