"""
webhook_server.py — TradeIQ FastAPI Webhook Bridge
====================================================
Acts as the bridge between n8n and the Python trading engine.
n8n calls these endpoints to trigger analysis and execute trades.

Endpoints:
  GET  /webhook/status    → portfolio cash, open positions, last 10 trades
  GET  /webhook/signals   → latest ML signals for all tickers (JSON)
  POST /webhook/sentiment → run VADER sentiment on all tickers + save to DB
  POST /webhook/predict   → run ML ensemble predictions for all tickers
  POST /webhook/run-cycle → run full auto-trading cycle (BUY/SELL agents)
  POST /webhook/price-update → receive raw price data from n8n (optional)

Usage:
  python webhook_server.py
  → Starts on http://0.0.0.0:8502

Install deps:
  pip install fastapi uvicorn
"""

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import traceback
import sys
import os
from datetime import datetime, timezone

# ── Ensure trading-portal directory is on path ───────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

app = FastAPI(
    title="TradeIQ Webhook Bridge",
    description="Bridges n8n automation with the TradeIQ Python trading engine",
    version="2.0.0",
)

# Allow n8n (localhost) to call this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class PriceRecord(BaseModel):
    ticker: str
    date: Optional[str] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    timestamp: Optional[str] = None


class PriceUpdatePayload(BaseModel):
    records: Optional[List[PriceRecord]] = []


# ── Helper: safe import + error response ─────────────────────────────────────
def _err(msg: str, exc: Exception):
    print(f"[ERROR] {msg}: {exc}")
    traceback.print_exc()
    raise HTTPException(status_code=500, detail=f"{msg}: {str(exc)}")


# ═══════════════════════════════════════════════════════════════════════════════
# GET /webhook/status
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/webhook/status")
def get_status():
    """
    Returns current portfolio status:
      - Available cash
      - Open positions (ticker, qty, entry_price, unrealised P&L)
      - Last 10 trades
    """
    try:
        from database import init_db, get_connection
        from data_fetcher import get_live_quotes
        import pandas as pd

        init_db()
        conn = get_connection()

        # Cash
        row = conn.execute("SELECT cash, starting FROM portfolio_state WHERE id=1").fetchone()
        cash = float(row[0]) if row else 100_000.0
        starting = float(row[1]) if row and len(row) > 1 else 100_000.0

        # Open positions
        positions_df = pd.read_sql_query(
            "SELECT ticker, quantity, entry_price, entry_time FROM portfolio",
            conn
        )

        # Enrich with live price for unrealised P&L
        positions = []
        if not positions_df.empty:
            try:
                quotes = get_live_quotes().set_index("ticker")
                for _, pos in positions_df.iterrows():
                    live_px = float(quotes.loc[pos["ticker"], "price"]) \
                        if pos["ticker"] in quotes.index else pos["entry_price"]
                    pnl = (live_px - pos["entry_price"]) * pos["quantity"]
                    pnl_pct = ((live_px - pos["entry_price"]) / pos["entry_price"] * 100) \
                        if pos["entry_price"] > 0 else 0
                    positions.append({
                        "ticker":       pos["ticker"],
                        "quantity":     round(pos["quantity"], 4),
                        "entry_price":  round(pos["entry_price"], 4),
                        "live_price":   round(live_px, 4),
                        "unrealised_pnl": round(pnl, 2),
                        "pnl_pct":      round(pnl_pct, 2),
                        "entry_time":   pos["entry_time"],
                    })
            except Exception:
                positions = positions_df.to_dict("records")

        # Last 10 trades
        trades_df = pd.read_sql_query(
            """SELECT ticker, action, quantity, entry_price, exit_price,
                      pnl, capital_after, signal_source, confidence, timestamp
               FROM   trades
               ORDER  BY timestamp DESC LIMIT 10""",
            conn
        )
        conn.close()

        return {
            "status": "ok",
            "cash": round(cash, 2),
            "starting": round(starting, 2),
            "open_positions": positions,
            "recent_trades":  trades_df.to_dict("records"),
        }

    except Exception as e:
        _err("status check failed", e)


# ═══════════════════════════════════════════════════════════════════════════════
# GET /webhook/signals
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/webhook/signals")
def get_signals():
    """
    Returns the latest ML predictions from the DB for all tickers.
    Used by n8n to route signals without re-running the model.
    """
    try:
        from database import init_db, get_connection
        import pandas as pd

        init_db()
        conn = get_connection()
        df = pd.read_sql_query(
            """SELECT ticker, predicted_date, predicted_price, signal, confidence, created_at
               FROM   ml_predictions
               ORDER  BY created_at DESC""",
            conn
        )
        conn.close()

        # Keep only the latest per ticker
        latest = df.drop_duplicates(subset=["ticker"], keep="first")
        return {
            "status": "ok",
            "count":   len(latest),
            "signals": latest.to_dict("records"),
        }
    except Exception as e:
        _err("get signals failed", e)


@app.post("/webhook/fetch-data")
def fetch_market_data():
    """
    Fetches OHLCV data through the Python data fetcher and saves it to SQLite.
    This is the preferred n8n entry point because it keeps DB writes in one place.
    """
    try:
        from database import init_db
        from data_fetcher import fetch_and_store

        init_db()
        saved_rows = fetch_and_store()
        return {
            "status": "ok",
            "message": "Market data fetched and saved",
            "saved_rows": saved_rows,
        }
    except Exception as e:
        _err("market data fetch failed", e)


# ═══════════════════════════════════════════════════════════════════════════════
# POST /webhook/sentiment
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/webhook/sentiment")
def run_sentiment(background_tasks: BackgroundTasks):
    """
    Triggers VADER sentiment analysis on all configured tickers.
    Fetches Google News RSS headlines, scores them, saves to DB.
    Returns a summary of average scores per ticker.
    """
    try:
        from database import init_db
        from sentiment import run_all_sentiment, average_sentiment
        from data_fetcher import ALL_TICKERS

        init_db()
        print("[Webhook] Running sentiment analysis...")
        run_all_sentiment(save_to_db=True)

        # Return summary
        summary = {}
        for ticker in ALL_TICKERS:
            score = average_sentiment(ticker)
            label = "Bullish" if score > 0.05 else ("Bearish" if score < -0.05 else "Neutral")
            summary[ticker] = {"score": score, "label": label}

        print(f"[Webhook] Sentiment done for {len(summary)} tickers")
        return {
            "status":  "ok",
            "message": f"Sentiment scored for {len(summary)} tickers",
            "summary": summary,
        }
    except Exception as e:
        _err("sentiment analysis failed", e)


# ═══════════════════════════════════════════════════════════════════════════════
# POST /webhook/predict
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/webhook/predict")
def run_predictions():
    """
    Runs the IEEE-enhanced ML ensemble prediction for all tickers.
    Models: LinearRegression + GradientBoosting + MLPRegressor.
    Fuses sentiment scores as a feature.
    Returns per-ticker BUY/SELL/HOLD signals with confidence.
    """
    try:
        from database import init_db
        from ml_model_v2 import run_all_predictions_v2

        init_db()
        print("[Webhook] Running ML ensemble predictions (v2)...")
        df = run_all_predictions_v2(save=True)

        if df.empty:
            return {
                "status":  "warning",
                "message": "No predictions generated. Run data_fetcher.py first to populate price history.",
                "signals": [],
            }

        # Convert output columns safely
        output_cols = [
            "ticker", "signal", "confidence", "change_pct",
            "last_price", "predicted_price",
            "rsi", "ind_signal", "ind_score"
        ]
        available = [c for c in output_cols if c in df.columns]
        signals = df[available].to_dict("records")

        buys  = sum(1 for s in signals if s.get("signal") == "BUY")
        sells = sum(1 for s in signals if s.get("signal") == "SELL")
        holds = sum(1 for s in signals if s.get("signal") == "HOLD")

        print(f"[Webhook] Predictions done: {buys} BUY, {sells} SELL, {holds} HOLD")
        return {
            "status":  "ok",
            "tickers": len(signals),
            "buys":    buys,
            "sells":   sells,
            "holds":   holds,
            "signals": signals,
        }
    except Exception as e:
        _err("ML prediction failed", e)


# ═══════════════════════════════════════════════════════════════════════════════
# POST /webhook/run-cycle
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/webhook/run-cycle")
def run_trade_cycle():
    """
    Runs a full auto-trading cycle:
      1. Fetch live prices
      2. Run technical indicators (RSI, MACD, Bollinger, Engulfing)
      3. Run ML v2 ensemble prediction + sentiment fusion
      4. BUY Agent: fires if ML=BUY, indicator score >= threshold, confidence >= MIN
      5. SELL Agent: fires on stop-loss, take-profit, or ML=SELL + indicator confirmation
    All trades are paper-mode only (LIVE_MODE=False).
    """
    try:
        from database import init_db
        from auto_trader import (
            run_trading_cycle,
            _ensure_trades_table,
            _ensure_portfolio_table,
        )

        init_db()
        _ensure_trades_table()
        _ensure_portfolio_table()

        print("[Webhook] Starting auto-trading cycle...")
        summary = run_trading_cycle(verbose=True)

        print(f"[Webhook] Cycle complete: {summary}")
        return {
            "status":  "ok",
            "message": "Trading cycle complete",
            "summary": summary,
        }
    except Exception as e:
        _err("trade cycle failed", e)


# ═══════════════════════════════════════════════════════════════════════════════
# POST /webhook/price-update   (optional — receive raw price data from n8n)
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/webhook/price-update")
def receive_price_update(payload: dict):
    """
    Optional: n8n can POST raw price records here and they will be stored.
    """
    try:
        from database import init_db, upsert_price

        init_db()
        records = payload if isinstance(payload, list) else payload.get("records", [])
        saved = 0
        skipped = 0
        today = datetime.now(timezone.utc).date().isoformat()

        for record in records:
            ticker = record.get("ticker")
            close = record.get("close") or record.get("price")
            if not ticker or close is None:
                skipped += 1
                continue

            close = float(close)
            upsert_price(
                ticker=ticker,
                date=record.get("date") or today,
                open_p=float(record.get("open") or close),
                high=float(record.get("high") or close),
                low=float(record.get("low") or close),
                close=close,
                volume=int(record.get("volume") or 0),
            )
            saved += 1

        print(f"[Webhook] Received {len(records)} price records from n8n; saved={saved}, skipped={skipped}")
        return {
            "status":  "ok",
            "received": len(records),
            "saved": saved,
            "skipped": skipped,
        }
    except Exception as e:
        _err("price update failed", e)


# ═══════════════════════════════════════════════════════════════════════════════
# Root health check
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "service": "TradeIQ Webhook Bridge",
        "version": "2.0.0",
        "endpoints": [
            "GET  /webhook/status",
            "GET  /webhook/signals",
            "POST /webhook/fetch-data",
            "POST /webhook/sentiment",
            "POST /webhook/predict",
            "POST /webhook/run-cycle",
            "POST /webhook/price-update",
        ],
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TradeIQ Webhook Bridge v2.0")
    print("  http://localhost:8502")
    print("  Docs: http://localhost:8502/docs")
    print("=" * 60)
    uvicorn.run(
        "webhook_server:app",
        host="0.0.0.0",
        port=8502,
        reload=True,
        log_level="info",
    )
