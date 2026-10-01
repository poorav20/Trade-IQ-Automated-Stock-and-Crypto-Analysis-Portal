import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from database import init_db, upsert_price

# ── Ticker Configuration ─────────────────────────────────────────────────────
TICKERS = {
    # Indian Stocks (NSE)
    "🇮🇳 Indian Stocks": [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
    ],
    # US Stocks
    "🇺🇸 US Stocks": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
    ],
    # Crypto
    "₿ Crypto": [
        "BTC-USD",
        "ETH-USD",
        "BNB-USD",
        "SOL-USD",
    ],
}

# Flat list for bulk download
ALL_TICKERS = [t for group in TICKERS.values() for t in group]

# How many days of history to fetch
FETCH_PERIOD = "90d"


def fetch_and_store(period: str = FETCH_PERIOD):
    """
    Download OHLCV data for all tickers and save each row into the DB.
    Uses period="90d" by default so the ML model has enough data.
    """
    print(f"[DATA] Fetching {period} of data for {len(ALL_TICKERS)} tickers...")

    # Fetch live USD-INR conversion rate
    try:
        usdinr_info = yf.Ticker("USDINR=X").fast_info
        usd_to_inr = float(getattr(usdinr_info, "last_price", 83.0))
    except Exception:
        usd_to_inr = 83.0
    print(f"[FX] Using USD/INR conversion rate: INR {usd_to_inr:.2f}")

    # yf.download returns a multi-level column DataFrame when multiple tickers
    raw = yf.download(
        ALL_TICKERS,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )

    saved_rows = 0

    for ticker in ALL_TICKERS:
        try:
            # Extract per-ticker slice
            if len(ALL_TICKERS) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy()

            df = df.dropna(subset=["Close"])
            is_usd = ticker not in TICKERS["🇮🇳 Indian Stocks"]
            mult = usd_to_inr if is_usd else 1.0

            for date_idx, row in df.iterrows():
                date_str = str(date_idx.date())
                upsert_price(
                    ticker   = ticker,
                    date     = date_str,
                    open_p   = float(row.get("Open",   0) or 0) * mult,
                    high     = float(row.get("High",   0) or 0) * mult,
                    low      = float(row.get("Low",    0) or 0) * mult,
                    close    = float(row.get("Close",  0) or 0) * mult,
                    volume   = int(row.get("Volume",   0) or 0),
                )
                saved_rows += 1

            print(f"  [OK] {ticker:15s} - {len(df)} rows saved")

        except Exception as exc:
            print(f"  [ERR] {ticker:15s} - Error: {exc}")

    print(f"\n[DONE] Total rows saved/updated: {saved_rows}")
    return saved_rows


def get_live_quotes() -> pd.DataFrame:
    """
    Fetch the most recent 1-day quote for all tickers.
    Returns a DataFrame with columns:
        ticker, price, change_pct, market_cap, currency
    """
    try:
        usdinr_info = yf.Ticker("USDINR=X").fast_info
        usd_to_inr = float(getattr(usdinr_info, "last_price", 83.0))
    except Exception:
        usd_to_inr = 83.0

    records = []
    for ticker in ALL_TICKERS:
        try:
            tkr   = yf.Ticker(ticker)
            info  = tkr.fast_info

            price      = getattr(info, "last_price",       None)
            prev_close = getattr(info, "previous_close",   None)
            mkt_cap    = getattr(info, "market_cap",        None)
            currency   = getattr(info, "currency",          "USD")

            # Convert if USD
            is_usd = ticker not in TICKERS["🇮🇳 Indian Stocks"]
            if is_usd:
                if price: price *= usd_to_inr
                if prev_close: prev_close *= usd_to_inr
                if mkt_cap: mkt_cap *= usd_to_inr
                currency = "INR"

            change_pct = None
            if price is not None and prev_close and prev_close != 0:
                change_pct = ((price - prev_close) / prev_close) * 100

            records.append({
                "ticker":     ticker,
                "price":      price,
                "change_pct": change_pct,
                "market_cap": mkt_cap,
                "currency":   currency,
            })
        except Exception as exc:
            print(f"  [WARN] Live quote failed for {ticker}: {exc}")
            records.append({
                "ticker":     ticker,
                "price":      None,
                "change_pct": None,
                "market_cap": None,
                "currency":   "USD",
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    init_db()                  # ensure tables exist
    fetch_and_store()          # pull history and save
    print("\n--- Live Quotes ---")
    quotes = get_live_quotes()
    print(quotes.to_string(index=False))
