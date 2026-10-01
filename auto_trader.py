"""
auto_trader.py — TradeIQ Enhanced Auto-Trader (v2 — Hyperliquid Integration)
─────────────────────────────────────────────────────────────────────────────
Improvements over v1:
  ✅ RiskManager  — circuit breaker, drawdown, leverage caps, force-close
  ✅ LLM Agent    — Google Gemini decides BUY/SELL/HOLD with reasoning
  ✅ Cooldown     — prevents rapid re-trading the same ticker
  ✅ Diary        — every decision logged to llm_decisions.jsonl (Hyperliquid style)
  ✅ Force-close  — positions with >20% loss are auto-exited
  ✅ New indicators — ADX, OBV, VWAP, StochRSI used in decisions

Architecture (Q-Learning inspired, IEEE IPRIA 2025):
  BUY  Agent : LLM + ML + Indicators aligned → BUY
  SELL Agent : LLM/ML SELL OR stop-loss/take-profit hit OR force-close loss
  Portfolio  : Virtual cash + positions tracked in SQLite `trades` table

PAPER TRADE MODE = ON by default (no real money).

Usage:
    python auto_trader.py            # run one cycle
    python auto_trader.py --loop     # run every 5 min
    python auto_trader.py --history  # print trade history
    python auto_trader.py --reset    # reset portfolio
    python auto_trader.py --diary    # print LLM decision diary
"""
import argparse
import os
import sqlite3
import time
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field

import pandas as pd

from database      import get_connection, init_db
from indicators    import add_all_indicators, indicator_signal
from data_fetcher  import ALL_TICKERS, get_live_quotes
from risk_manager  import get_risk_manager
from llm_agent     import get_llm_agent

try:
    from zerodha import is_configured as zerodha_is_configured, place_zerodha_order
except ImportError:
    zerodha_is_configured = lambda: False
    place_zerodha_order = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Risk Configuration (overridable via .env) ─────────────────────────────────
DEFAULT_CAPITAL     = 100_000.0  # virtual starting capital (INR)
COOLDOWN_CYCLES     = 3          # cycles to skip after a trade on same ticker
LIVE_MODE           = os.getenv("LIVE_MODE", "False").lower() in ("1", "true", "yes")

# ── Cooldown tracker (in-memory) ──────────────────────────────────────────────
_cooldown: dict[str, int] = {}   # ticker → cycles remaining


def _check_cooldown(ticker: str) -> bool:
    """Returns True if ticker is in cooldown (should not trade)."""
    remaining = _cooldown.get(ticker, 0)
    if remaining > 0:
        _cooldown[ticker] = remaining - 1
        return True
    return False


def _set_cooldown(ticker: str):
    """Put ticker in cooldown after a trade."""
    _cooldown[ticker] = COOLDOWN_CYCLES


# ── Ensure DB tables ──────────────────────────────────────────────────────────
def _ensure_trades_table():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker         TEXT    NOT NULL,
            action         TEXT    NOT NULL,
            quantity       REAL    NOT NULL,
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
    conn.commit()
    conn.close()


def _ensure_portfolio_table():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            ticker      TEXT  PRIMARY KEY,
            quantity    REAL  DEFAULT 0,
            entry_price REAL  DEFAULT 0,
            sl_price    REAL,
            tp_price    REAL,
            entry_time  TEXT
        )
    """)
    try:
        conn.execute("ALTER TABLE portfolio ADD COLUMN exit_plan TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id       INTEGER PRIMARY KEY CHECK (id = 1),
            cash     REAL    NOT NULL DEFAULT 100000.0,
            starting REAL    NOT NULL DEFAULT 100000.0
        )
    """)
    try:
        conn.execute("ALTER TABLE portfolio_state ADD COLUMN starting REAL NOT NULL DEFAULT 100000.0")
    except sqlite3.OperationalError:
        pass
    conn.execute("""
        INSERT OR IGNORE INTO portfolio_state (id, cash, starting)
        VALUES (1, ?, ?)
    """, (DEFAULT_CAPITAL, DEFAULT_CAPITAL))
    conn.commit()
    conn.close()


# ── Portfolio helpers ─────────────────────────────────────────────────────────
def get_cash() -> float:
    conn = get_connection()
    row  = conn.execute("SELECT cash FROM portfolio_state WHERE id=1").fetchone()
    conn.close()
    return float(row[0]) if row else DEFAULT_CAPITAL


def get_starting_capital() -> float:
    conn = get_connection()
    row  = conn.execute("SELECT starting FROM portfolio_state WHERE id=1").fetchone()
    conn.close()
    return float(row[0]) if row else DEFAULT_CAPITAL


def set_cash(amount: float):
    conn = get_connection()
    conn.execute("UPDATE portfolio_state SET cash=? WHERE id=1", (amount,))
    conn.commit()
    conn.close()


def get_position(ticker: str) -> dict:
    conn = get_connection()
    row  = conn.execute(
        "SELECT quantity, entry_price, sl_price, tp_price, entry_time, exit_plan FROM portfolio WHERE ticker=?",
        (ticker,)
    ).fetchone()
    conn.close()
    if row:
        return {"quantity": row[0], "entry_price": row[1],
                "sl_price": row[2], "tp_price": row[3], "entry_time": row[4], "exit_plan": row[5]}
    return {"quantity": 0.0, "entry_price": 0.0, "sl_price": None, "tp_price": None, "entry_time": None, "exit_plan": None}


def get_all_positions() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT ticker, quantity, entry_price, sl_price, tp_price, entry_time, exit_plan FROM portfolio"
    ).fetchall()
    conn.close()
    return [{"ticker": r[0], "quantity": r[1], "entry_price": r[2],
             "sl_price": r[3], "tp_price": r[4], "entry_time": r[5], "exit_plan": r[6]} for r in rows]


def get_open_positions_count() -> int:
    conn = get_connection()
    row  = conn.execute("SELECT COUNT(*) FROM portfolio WHERE quantity > 0").fetchone()
    conn.close()
    return row[0] if row else 0


def get_open_positions() -> pd.DataFrame:
    """Return current open paper-trading positions for the dashboard."""
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT ticker, quantity, entry_price, sl_price, tp_price, entry_time, exit_plan
        FROM   portfolio
        WHERE  quantity > 0
        ORDER  BY entry_time DESC
        """,
        conn,
    )
    conn.close()
    return df


def get_total_positions_value(quotes: pd.DataFrame) -> float:
    """Current market value of all open positions."""
    positions = get_all_positions()
    total = 0.0
    for pos in positions:
        tkr = pos["ticker"]
        current_px = (
            float(quotes.loc[tkr, "price"])
            if tkr in quotes.index and pd.notna(quotes.loc[tkr, "price"])
            else pos["entry_price"]
        )
        total += pos["quantity"] * current_px
    return total


def set_position(ticker: str, quantity: float, entry_price: float,
                 sl_price: float = None, tp_price: float = None, exit_plan: str = None):
    conn = get_connection()
    if quantity == 0:
        conn.execute("DELETE FROM portfolio WHERE ticker=?", (ticker,))
    else:
        conn.execute("""
            INSERT OR REPLACE INTO portfolio (ticker, quantity, entry_price, sl_price, tp_price, entry_time, exit_plan)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ticker, quantity, entry_price, sl_price, tp_price,
              datetime.now(timezone.utc).isoformat(), exit_plan))
    conn.commit()
    conn.close()


def log_trade(ticker, action, qty, entry_price, exit_price,
              pnl, cap_before, cap_after, source, confidence,
              reasoning="", sl_price=None, tp_price=None):
    conn = get_connection()
    conn.execute("""
        INSERT INTO trades
          (ticker, action, quantity, entry_price, exit_price, pnl,
           capital_before, capital_after, signal_source, confidence,
           reasoning, sl_price, tp_price, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (ticker, action, qty, entry_price, exit_price,
          pnl, cap_before, cap_after, source, confidence,
          reasoning, sl_price, tp_price,
          datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()


# ── Force-close losing positions ──────────────────────────────────────────────
def force_close_losing_positions(quotes: pd.DataFrame, risk_mgr) -> int:
    """
    Check all open positions against RiskManager.positions_to_force_close().
    Returns count of force-closed positions.
    """
    positions = get_all_positions()
    enriched  = []
    for pos in positions:
        tkr = pos["ticker"]
        current_px = (
            float(quotes.loc[tkr, "price"])
            if tkr in quotes.index and pd.notna(quotes.loc[tkr, "price"])
            else pos["entry_price"]
        )
        enriched.append({**pos, "current_price": current_px})

    to_close = risk_mgr.positions_to_force_close(enriched)
    count = 0
    for pos in to_close:
        tkr        = pos["ticker"]
        qty        = pos["quantity"]
        entry      = pos["entry_price"]
        current_px = pos["current_price"]
        pnl        = pos["loss_amount"]
        cash       = get_cash()
        proceeds   = qty * current_px
        new_cash   = cash + proceeds

        set_cash(new_cash)
        set_position(tkr, 0, 0)
        log_trade(tkr, "FORCE_CLOSE", qty, entry, current_px,
                  pnl, cash, new_cash, "RISK_MANAGER", 0,
                  reasoning=f"Force-closed: {pos['loss_pct']:.1f}% loss")

        logger.warning(
            "[FORCE-CLOSE] %s qty=%.4f @ %.2f  pnl=%.2f  loss=%.1f%%",
            tkr, qty, current_px, pnl, pos["loss_pct"]
        )
        count += 1

    return count


# ── Check SL/TP for open positions ────────────────────────────────────────────
def check_sl_tp(ticker: str, current_price: float) -> str | None:
    """Check if a position's stop-loss or take-profit has been hit. Returns trigger or None."""
    pos = get_position(ticker)
    if pos["quantity"] <= 0:
        return None
    entry    = pos["entry_price"]
    sl_price = pos["sl_price"]
    tp_price = pos["tp_price"]

    pnl_pct  = (current_price - entry) / entry if entry > 0 else 0

    if sl_price and current_price <= sl_price:
        return "STOP_LOSS"
    if tp_price and current_price >= tp_price:
        return "TAKE_PROFIT"
    return None


# ── BUY Agent ─────────────────────────────────────────────────────────────────
def buy_agent(ticker: str, current_price: float,
              llm_decision: dict, ind_result: dict,
              risk_mgr, quotes: pd.DataFrame) -> bool:
    """
    Fires BUY order when:
      - LLM/ML signal is BUY AND
      - Not in cooldown AND
      - No existing position AND
      - RiskManager validates the trade
    """
    if _check_cooldown(ticker):
        return False
    pos = get_position(ticker)
    if pos["quantity"] > 0:
        return False

    action = llm_decision.get("action", "HOLD")
    if action != "BUY":
        return False

    cash            = get_cash()
    starting_capital = get_starting_capital()
    open_count      = get_open_positions_count()
    open_value      = get_total_positions_value(quotes)
    trade_value     = cash * 0.10  # start with 10%, risk manager may cap

    ok, capped_value, reason = risk_mgr.validate_buy(
        ticker               = ticker,
        trade_value          = trade_value,
        current_capital      = cash,
        starting_capital     = starting_capital,
        open_positions_count = open_count,
        open_positions_value = open_value,
    )

    if not ok:
        logger.info("[BLOCKED BUY] %s: %s", ticker, reason)
        return False

    trade_value = capped_value
    quantity    = trade_value / current_price if current_price > 0 else 0

    if quantity <= 0 or cash < trade_value:
        logger.info("[SKIP BUY] %s: insufficient cash (%.2f)", ticker, cash)
        return False

    # Auto SL/TP from risk manager
    sl_tp = risk_mgr.compute_sl_tp(current_price, is_buy=True)
    sl_price = llm_decision.get("sl_price") or sl_tp["sl_price"]
    tp_price = llm_decision.get("tp_price") or sl_tp["tp_price"]

    new_cash = cash - trade_value
    set_cash(new_cash)
    exit_plan = llm_decision.get("exit_plan", "")
    set_position(ticker, quantity, current_price, sl_price, tp_price, exit_plan)
    _set_cooldown(ticker)

    source    = llm_decision.get("source", "ML")
    reasoning = llm_decision.get("reasoning", "")
    confidence = llm_decision.get("confidence", 0)

    log_trade(ticker, "BUY", quantity, current_price, None,
              0, cash, new_cash, source, confidence, reasoning, sl_price, tp_price)

    if LIVE_MODE and zerodha_is_configured() and place_zerodha_order is not None:
        try:
            place_zerodha_order(
                symbol=ticker,
                transaction_type="BUY",
                quantity=int(quantity),
                order_type="MARKET",
                product="MIS",
            )
            logger.info("[LIVE BUY] sent Zerodha order for %s qty=%d", ticker, int(quantity))
        except Exception as err:
            logger.error("[LIVE BUY FAILED] %s: %s", ticker, err)

    mode = "PAPER" if not LIVE_MODE else "LIVE"
    logger.info(
        "[%s BUY] %s  qty=%.4f @ %.2f  SL=%.2f  TP=%.2f  cash=%.2f  [%s]",
        mode, ticker, quantity, current_price,
        sl_price or 0, tp_price or 0, new_cash, source
    )
    return True


# ── SELL Agent ────────────────────────────────────────────────────────────────
def sell_agent(ticker: str, current_price: float,
               llm_decision: dict, ind_result: dict) -> bool:
    """
    Fires SELL order when ANY of:
      - Stop-loss or Take-profit hit
      - LLM/ML signal = SELL AND indicator score < 0
    """
    pos = get_position(ticker)
    if pos["quantity"] <= 0:
        return False

    entry   = pos["entry_price"]
    qty     = pos["quantity"]
    action  = llm_decision.get("action", "HOLD")

    # Check SL/TP
    trigger = check_sl_tp(ticker, current_price)

    # ML/LLM SELL + negative indicator
    if trigger is None:
        if action == "SELL" and ind_result.get("score", 0) <= 0:
            trigger = "SELL"
        else:
            return False

    # Execute paper SELL
    trade_val = qty * current_price
    pnl       = (current_price - entry) * qty
    cash      = get_cash()
    new_cash  = cash + trade_val

    set_cash(new_cash)
    set_position(ticker, 0, 0)
    _set_cooldown(ticker)

    source    = llm_decision.get("source", "ML")
    confidence = llm_decision.get("confidence", 0)
    reasoning = llm_decision.get("reasoning", "")

    log_trade(ticker, trigger, qty, entry, current_price,
              pnl, cash, new_cash, trigger, confidence, reasoning)

    if LIVE_MODE and zerodha_is_configured() and place_zerodha_order is not None:
        try:
            place_zerodha_order(
                symbol=ticker,
                transaction_type="SELL",
                quantity=int(qty),
                order_type="MARKET",
                product="MIS",
            )
            logger.info("[LIVE SELL] sent Zerodha order for %s qty=%d", ticker, int(qty))
        except Exception as err:
            logger.error("[LIVE SELL FAILED] %s: %s", ticker, err)

    mode = "PAPER" if not LIVE_MODE else "LIVE"
    logger.info(
        "[%s %s] %s  qty=%.4f @ %.2f  pnl=%+.2f  cash=%.2f",
        mode, trigger, ticker, qty, current_price, pnl, new_cash
    )
    return True


# ── Main Cycle ────────────────────────────────────────────────────────────────
def run_trading_cycle(verbose: bool = True) -> dict:
    """
    One complete cycle:
      1. Fetch prices
      2. Force-close losing positions (RiskManager)
      3. Run ML v2 ensemble prediction
      4. Run technical indicators (9 signals)
      5. Call LLM Agent (Gemini) for final decision
      6. Execute BUY/SELL agents
    Returns summary dict.
    """
    from ml_model_v2 import predict_ticker_v2
    from sentiment   import average_sentiment
    from database    import get_price_df

    risk_mgr  = get_risk_manager()
    llm_agent = get_llm_agent()
    summary   = {"buys": 0, "sells": 0, "force_closed": 0, "skipped": 0, "cash": get_cash()}
    quotes    = get_live_quotes().set_index("ticker")

    logger.info(
        "[CYCLE] %s  Cash: %.2f  Circuit: %s",
        datetime.now().strftime("%H:%M:%S"),
        summary["cash"],
        "ACTIVE" if risk_mgr._circuit_breaker_on else "off"
    )

    # ── Force-close check (Hyperliquid-style hard safety) ─────────────────────
    closed = force_close_losing_positions(quotes, risk_mgr)
    summary["force_closed"] = closed

    # ── Collect active trades for LLM context ─────────────────────────────────
    active_trades = [p for p in get_all_positions() if p["quantity"] > 0]
    for p in active_trades:
        p["action"] = "BUY"

    # Prepare assets data for batch prediction
    assets_data = []
    predictions = {}
    indicators_map = {}

    for ticker in ALL_TICKERS:
        try:
            current_price = (
                float(quotes.loc[ticker, "price"])
                if ticker in quotes.index and pd.notna(quotes.loc[ticker, "price"])
                else None
            )
            if current_price is None or current_price <= 0:
                continue

            df = get_price_df(ticker, days=90)
            if df.empty:
                continue
            df  = add_all_indicators(df)
            ind = indicator_signal(df)

            sent = average_sentiment(ticker)
            pred = predict_ticker_v2(ticker, sentiment_score=sent)
            if pred is None:
                continue

            assets_data.append({
                "ticker": ticker,
                "current_price": current_price,
                "intraday": ind,
                "historical": {}, # Can be extended for dual timeframe
                "ml_pred": pred,
                "sentiment_score": sent,
            })
            predictions[ticker] = pred
            indicators_map[ticker] = ind

        except Exception as e:
            logger.error("[ERR] %s data prep: %s", ticker, e, exc_info=False)

    if not assets_data:
        summary["cash"] = get_cash()
        return summary

    # ── Batch LLM / ML decision ──────────────────────────────────────────────
    risk_summary = risk_mgr.get_summary()
    context = llm_agent.build_context(assets_data, active_trades, risk_summary)
    
    tickers_to_decide = [a["ticker"] for a in assets_data]
    batch_decisions = llm_agent.decide_batch(tickers_to_decide, context)

    # ── Execute Trades ────────────────────────────────────────────────────────
    for ticker_data in assets_data:
        ticker = ticker_data["ticker"]
        current_price = ticker_data["current_price"]
        ind = indicators_map[ticker]
        decision = batch_decisions.get(ticker, {})

        if not decision:
            summary["skipped"] += 1
            continue

        if verbose:
            src   = decision.get("source", "ML")
            act   = decision.get("action", "HOLD")
            conf  = decision.get("confidence", 0)
            score = ind.get("score", 0)
            logger.info(
                "  %-15s price=%10.2f  [%s] %s(%.2f)  IND=%s(%+.1f)",
                ticker, current_price, src, act, conf,
                ind.get("signal"), score
            )

        # BUY / SELL agents
        did_buy  = buy_agent(ticker, current_price, decision, ind, risk_mgr, quotes)
        did_sell = sell_agent(ticker, current_price, decision, ind)

        if did_buy:
            summary["buys"] += 1
        elif did_sell:
            summary["sells"] += 1

    summary["cash"] = get_cash()
    logger.info(
        "[DONE] Buys=%d  Sells=%d  ForceClose=%d  Skipped=%d  Cash=%.2f",
        summary["buys"], summary["sells"],
        summary["force_closed"], summary["skipped"], summary["cash"]
    )
    return summary


# ── History / Diary Queries ───────────────────────────────────────────────────
def get_trade_history(limit: int = 50) -> pd.DataFrame:
    conn = get_connection()
    df   = pd.read_sql_query("""
        SELECT ticker, action, quantity, entry_price, exit_price,
               pnl, capital_after, signal_source, confidence, reasoning, timestamp
        FROM   trades
        ORDER  BY timestamp DESC
        LIMIT  ?
    """, conn, params=(limit,))
    conn.close()
    return df


def print_diary(limit: int = 30):
    """Print LLM decision diary."""
    import os
    if not os.path.exists("llm_decisions.jsonl"):
        print("No LLM diary yet. Run a cycle first.")
        return
    print("\n" + "="*70)
    print(" LLM DECISION DIARY (last entries)")
    print("="*70)
    with open("llm_decisions.jsonl", "r") as f:
        lines = f.readlines()
    for line in lines[-limit:]:
        try:
            e = json.loads(line)
            ts  = e.get("timestamp", "?")[:19]
            act = e.get("action", "?")
            tkr = e.get("ticker", "?")
            src = e.get("source", "?")
            rsn = e.get("reasoning", "")[:80]
            print(f"  {ts}  [{src}] {act:<4} {tkr:<15}  {rsn}")
        except Exception:
            continue
    print("="*70)


# ── CLI Entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradeIQ Auto-Trader v2 (Hyperliquid Enhanced)")
    parser.add_argument("--loop",    action="store_true", help="Run every 5 minutes")
    parser.add_argument("--history", action="store_true", help="Print trade history")
    parser.add_argument("--diary",   action="store_true", help="Print LLM decision diary")
    parser.add_argument("--reset",   action="store_true", help="Reset portfolio to default capital")
    args = parser.parse_args()

    init_db()
    _ensure_trades_table()
    _ensure_portfolio_table()

    if args.reset:
        conn = get_connection()
        conn.execute("UPDATE portfolio_state SET cash=?, starting=? WHERE id=1",
                     (DEFAULT_CAPITAL, DEFAULT_CAPITAL))
        conn.execute("DELETE FROM portfolio")
        conn.commit()
        conn.close()
        print(f"[RESET] Portfolio reset to {DEFAULT_CAPITAL:,.2f}")

    elif args.history:
        df = get_trade_history()
        if df.empty:
            print("No trades yet. Run a cycle first.")
        else:
            print(df.to_string(index=False))

    elif args.diary:
        print_diary()

    elif args.loop:
        print("Starting auto-trader loop (every 5 min). Ctrl+C to stop.")
        while True:
            run_trading_cycle()
            time.sleep(300)

    else:
        run_trading_cycle()
