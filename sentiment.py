"""
sentiment.py — News headline fetcher + VADER sentiment scoring

Fetches RSS headlines for each ticker from Google News and scores
them using VADER (Valence Aware Dictionary and sEntiment Reasoner).

Score guide:
    > +0.05  →  Positive (bullish signal)
    < -0.05  →  Negative (bearish signal)
    in between → Neutral

Usage:
    python sentiment.py
"""
import feedparser
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from database import init_db, insert_sentiment, get_sentiment_df

#ker → news search term mapping
TICKER_SEARCH = {
    "RELIANCE.NS":  "Reliance Industries stock",
    "TCS.NS":       "TCS Tata Consultancy stock",
    "INFY.NS":      "Infosys stock",
    "HDFCBANK.NS":  "HDFC Bank stock",
    "ICICIBANK.NS": "ICICI Bank stock",
    "AAPL":         "Apple stock AAPL",
    "MSFT":         "Microsoft stock MSFT",
    "GOOGL":        "Google Alphabet stock",
    "AMZN":         "Amazon stock",
    "NVDA":         "Nvidia stock",
    "BTC-USD":      "Bitcoin price crypto",
    "ETH-USD":      "Ethereum price crypto",
    "BNB-USD":      "Binance BNB crypto",
    "SOL-USD":      "Solana SOL crypto",
}

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

analyzer = SentimentIntensityAnalyzer()


def fetch_headlines(ticker: str, query: str, max_items: int = 10) -> list[dict]:
    """Fetch and score headlines from Google News RSS for a given query."""
    url  = GOOGLE_NEWS_RSS.format(query=query.replace(" ", "+"))
    feed = feedparser.parse(url)
    now  = datetime.now(timezone.utc).isoformat()

    results = []
    for entry in feed.entries[:max_items]:
        headline = entry.get("title", "")
        source   = entry.get("source", {}).get("title", "Google News")
        score    = analyzer.polarity_scores(headline)["compound"]

        results.append({
            "ticker":     ticker,
            "headline":   headline,
            "source":     source,
            "score":      score,
            "fetched_at": now,
        })

    return results


def run_all_sentiment(save_to_db: bool = True) -> dict:
    """
    Fetch + score headlines for every configured ticker.
    Returns a dict of  {ticker: [headline_records]}
    """
    all_results = {}

    for ticker, query in TICKER_SEARCH.items():
        print(f"  [NEWS] Fetching news for {ticker}...")
        try:
            items = fetch_headlines(ticker, query)

            if save_to_db:
                for item in items:
                    insert_sentiment(
                        ticker     = item["ticker"],
                        headline   = item["headline"],
                        source     = item["source"],
                        score      = item["score"],
                        fetched_at = item["fetched_at"],
                    )

            all_results[ticker] = items

            # Pretty print
            for item in items:
                bar = "[+]" if item["score"] > 0.05 else ("[-]" if item["score"] < -0.05 else "[=]")
                headline_preview = item["headline"][:80].encode("ascii", "replace").decode("ascii")
                print(f"    {bar} [{item['score']:+.2f}]  {headline_preview}")

        except Exception as exc:
            print(f"    [WARN] Failed for {ticker}: {exc}")
            all_results[ticker] = []

    return all_results


def average_sentiment(ticker: str, days: int = 7) -> float:
    """
    Returns the average VADER compound score for the last `days` days of
    stored news for a given ticker. Used by the ML model and dashboard.
    """
    df = get_sentiment_df(ticker, limit=50)
    if df.empty:
        return 0.0
    return round(float(df["score"].mean()), 4)


def sentiment_label(score: float) -> str:
    """Convert a compound score to a human-readable label."""
    if score >= 0.05:
        return "Bullish"
    elif score <= -0.05:
        return "Bearish"
    return "Neutral"


if __name__ == "__main__":
    init_db()
    print("[SENTIMENT] Running sentiment analysis on all tickers...\n")
    run_all_sentiment(save_to_db=True)
    print("\n[OK] Sentiment analysis complete.")
