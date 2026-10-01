"""
ml_model.py — Machine Learning price prediction engine

Models used:
  1. Scikit-Learn Linear Regression  — fast, beginner-friendly baseline
  2. (Optional) Prophet               — powerful time-series forecasting

For each ticker it:
  • Loads the last 60 days of closing prices from SQLite
  • Engineers features: lag prices, rolling averages, day-of-week
  • Trains / re-trains a Linear Regression model
  • Predicts tomorrow's closing price
  • Generates a BUY / SELL / HOLD signal with a confidence score
  • Saves the prediction to the ml_predictions table

Usage:
    python ml_model.py
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

from database import init_db, get_price_df, save_prediction
from data_fetcher import ALL_TICKERS

# ── Configuration ─────────────────────────────────────────────────────────────
LOOKBACK_DAYS   = 60    # how many days of history the model uses
LAG_FEATURES    = 5     # number of lag-close features
SIGNAL_THRESHOLD = 0.5  # % change needed to trigger BUY or SELL (not HOLD)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the price DataFrame:
      - lag_1 … lag_N   : previous N closing prices
      - ma_7, ma_14     : 7-day and 14-day moving averages
      - pct_change       : day-over-day % change
      - dow              : day of week (0=Mon … 6=Sun)
    """
    df = df.copy()
    df["pct_change"] = df["close"].pct_change()
    df["ma_7"]       = df["close"].rolling(7).mean()
    df["ma_14"]      = df["close"].rolling(14).mean()
    df["dow"]        = df["date"].dt.dayofweek

    for i in range(1, LAG_FEATURES + 1):
        df[f"lag_{i}"] = df["close"].shift(i)

    df = df.dropna()
    return df


def predict_ticker(ticker: str) -> dict | None:
    """
    Train a Linear Regression model on the last LOOKBACK_DAYS days and
    return a prediction dict:
        {
            "ticker":          str,
            "last_price":      float,
            "predicted_price": float,
            "change_pct":      float,
            "signal":          "BUY" | "SELL" | "HOLD",
            "confidence":      float,   # 0.0 – 1.0
            "mape":            float,   # model error %
        }
    """
    df = get_price_df(ticker, days=LOOKBACK_DAYS + 30)  # extra for feature eng.

    if len(df) < 30:
        print(f"  [SKIP] {ticker}: not enough data ({len(df)} rows) - skipping")
        return None

    df = _engineer_features(df)

    if len(df) < 20:
        print(f"  [SKIP] {ticker}: not enough rows after feature engineering - skipping")
        return None

    # Feature columns
    feature_cols = ["ma_7", "ma_14", "pct_change", "dow"] + [f"lag_{i}" for i in range(1, LAG_FEATURES + 1)]

    X = df[feature_cols].values
    y = df["close"].values

    # Train on all but the last row; "test" on the last row
    X_train, X_test = X[:-1], X[-1:]
    y_train, y_test = y[:-1], y[-1:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model   = LinearRegression()
    model.fit(X_train, y_train)

    # Predict tomorrow using the most-recent row as "today"
    tomorrow_features = X[-1:]                        # last known row
    tomorrow_features = scaler.transform(tomorrow_features)
    predicted_price   = float(model.predict(tomorrow_features)[0])

    last_price  = float(df["close"].iloc[-1])
    change_pct  = ((predicted_price - last_price) / last_price) * 100

    # In-sample MAPE as a proxy for model quality
    y_pred_train = model.predict(scaler.transform(X[:-1]))
    mape = float(mean_absolute_percentage_error(y_train, y_pred_train)) * 100

    # ── Signal logic ──────────────────────────────────────────────────────────
    # Base signal on predicted direction × magnitude
    if change_pct > SIGNAL_THRESHOLD:
        signal = "BUY"
    elif change_pct < -SIGNAL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Confidence: crude heuristic — lower MAPE → higher confidence
    # Maps MAPE 0 % → 1.0 confidence, MAPE ≥ 10 % → 0.0 confidence
    confidence = max(0.0, min(1.0, 1.0 - (mape / 10.0)))

    return {
        "ticker":          ticker,
        "last_price":      last_price,
        "predicted_price": round(predicted_price, 4),
        "change_pct":      round(change_pct, 3),
        "signal":          signal,
        "confidence":      round(confidence, 3),
        "mape":            round(mape, 3),
    }


def run_all_predictions(save: bool = True) -> pd.DataFrame:
    """
    Run predictions for every ticker in ALL_TICKERS.
    Prints a summary table and optionally saves to the DB.
    Returns a DataFrame.
    """
    results = []

    for ticker in ALL_TICKERS:
        print(f"  [ML] Predicting {ticker}...")
        pred = predict_ticker(ticker)
        if pred is None:
            continue

        if save:
            tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
            save_prediction(
                ticker          = pred["ticker"],
                predicted_date  = tomorrow,
                predicted_price = pred["predicted_price"],
                signal          = pred["signal"],
                confidence      = pred["confidence"],
                created_at      = datetime.now(timezone.utc).isoformat(),
            )

        results.append(pred)

    df = pd.DataFrame(results)
    return df


def get_prophet_forecast(ticker: str, periods: int = 7) -> pd.DataFrame | None:
    """
    Optional 7-day forecast using Meta's Prophet model.
    Returns a DataFrame with columns: ds (date), yhat, yhat_lower, yhat_upper.
    Returns None if Prophet is not installed or data is insufficient.
    """
    try:
        from prophet import Prophet  # lazy import — optional dependency
    except ImportError:
        return None

    df = get_price_df(ticker, days=180)
    if len(df) < 30:
        return None

    df_prophet = df[["date", "close"]].rename(columns={"date": "ds", "close": "y"})

    model = Prophet(
        daily_seasonality  = False,
        weekly_seasonality = True,
        yearly_seasonality = True,
        changepoint_prior_scale = 0.1,
    )
    model.fit(df_prophet)

    future   = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Return only the future rows
    return forecast[forecast["ds"] > df_prophet["ds"].max()][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].reset_index(drop=True)


if __name__ == "__main__":
    init_db()
    print("[ML] Running ML predictions for all tickers...\n")
    df = run_all_predictions(save=True)

    if not df.empty:
        print("\n" + "=" * 72)
        print(" ML PREDICTION SUMMARY")
        print("=" * 72)
        for _, row in df.iterrows():
            icon = "[BUY]" if row["signal"] == "BUY" else ("[SELL]" if row["signal"] == "SELL" else "[HOLD]")
            print(f"{icon} {row['ticker']:15s}  Last: {row['last_price']:10.2f}  "
                  f"Pred: {row['predicted_price']:10.2f}  "
                  f"Delta {row['change_pct']:+.2f}%  [{row['signal']}]  conf={row['confidence']:.2f}")
        print("=" * 72)
    else:
        print("[ERR] No predictions generated. Run data_fetcher.py first!")
