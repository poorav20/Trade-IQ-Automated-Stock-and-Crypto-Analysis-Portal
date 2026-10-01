"""
ml_model_v2.py — IEEE-Enhanced Ensemble ML Prediction Engine
Based on:
  - "GRU & LSTM model with Sentiment Analysis" (ICEC 2024)
  - "Survey of ML Techniques for Stock Prediction" (GCCIT 2024)
  - "Automatic Stocks and Crypto Research" (ICIRCA 2025)

Improvements over v1 (Linear Regression only):
  1. Multi-model ENSEMBLE: MLPRegressor (simulates LSTM) + Gradient Boosting + Linear Regression
  2. Technical indicator FEATURES: RSI, MACD, Bollinger %B, Engulfing
  3. SENTIMENT FUSION: VADER score added as a feature
  4. Confidence from cross-validated R² score

Usage:
    python ml_model_v2.py
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics         import mean_absolute_percentage_error

from database     import init_db, get_price_df, save_prediction
from indicators   import add_all_indicators
from data_fetcher import ALL_TICKERS

# ── Configuration ─────────────────────────────────────────────────────────────
LOOKBACK_DAYS    = 90
SIGNAL_THRESHOLD = 0.5   # % change to trigger BUY/SELL


def _engineer_features(df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
    """
    Feature engineering as recommended by IEEE papers:
      - Lag features (prev 5 closes)        — LSTM paper baseline
      - Moving averages (7, 14)              — Survey paper
      - RSI, MACD, Bollinger %B             — Survey + Q-Learning paper
      - Engulfing pattern                    — Q-Learning BTC paper (primary indicator)
      - Sentiment score                      — GRU+LSTM paper (fused as feature)
      - Day-of-week                          — Seasonal effect
    """
    df = add_all_indicators(df)   # adds rsi, macd, bb_pct_b, engulfing, atr

    df["pct_change"] = df["close"].pct_change()
    df["ma_7"]       = df["close"].rolling(7).mean()
    df["ma_14"]      = df["close"].rolling(14).mean()
    df["dow"]        = df["date"].dt.dayofweek
    df["sentiment"]  = sentiment_score   # fuse external sentiment score

    for i in range(1, 6):
        df[f"lag_{i}"] = df["close"].shift(i)

    return df.dropna()


FEATURE_COLS = [
    "ma_7", "ma_14", "pct_change", "dow", "sentiment",
    "rsi", "macd", "macd_sig", "macd_hist",
    "bb_pct_b", "engulfing", "atr",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5",
]


def _build_ensemble():
    """Return a dict of named models for the ensemble."""
    return {
        "LinearRegression": LinearRegression(),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=3,
            learning_rate=0.1, random_state=42,
        ),
        "MLP_Neural": MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        ),
    }


def predict_ticker_v2(ticker: str, sentiment_score: float = 0.0) -> dict | None:
    """
    IEEE-enhanced ensemble prediction for a single ticker.
    Returns dict with: ticker, last_price, predicted_price, change_pct,
                       signal, confidence, mape, model_scores, indicator_signal
    """
    df_raw = get_price_df(ticker, days=LOOKBACK_DAYS + 30)
    if len(df_raw) < 40:
        print(f"  [SKIP] {ticker}: not enough data ({len(df_raw)} rows)")
        return None

    df = _engineer_features(df_raw.copy(), sentiment_score)

    if len(df) < 20:
        print(f"  [SKIP] {ticker}: too few rows after feature engineering")
        return None

    # Features & target
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].values
    y = df["close"].values

    X_train, X_test = X[:-1], X[-1:]
    y_train         = y[:-1]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Train all models ──────────────────────────────────────────────────────
    models       = _build_ensemble()
    predictions  = {}
    model_scores = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = float(model.predict(X_test)[0])
            predictions[name] = pred

            # Cross-val R² (3-fold)
            cv_scores = cross_val_score(model, X_train, y_train, cv=min(3, len(X_train)//5 or 1), scoring="r2")
            model_scores[name] = round(float(cv_scores.mean()), 3)
        except Exception as e:
            print(f"    [WARN] {name} failed for {ticker}: {e}")

    if not predictions:
        return None

    # ── Ensemble: weighted average by R² score (better model → higher weight) ─
    weights = {k: max(v, 0.01) for k, v in model_scores.items() if k in predictions}
    total_w = sum(weights.values())
    if total_w == 0:
        ensemble_pred = float(np.mean(list(predictions.values())))
    else:
        ensemble_pred = sum(predictions[k] * weights[k] for k in weights) / total_w

    last_price  = float(df["close"].iloc[-1])
    change_pct  = ((ensemble_pred - last_price) / last_price) * 100

    # ── Signal ────────────────────────────────────────────────────────────────
    if change_pct > SIGNAL_THRESHOLD:
        signal = "BUY"
    elif change_pct < -SIGNAL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    # ── Boost/dampen signal with sentiment ───────────────────────────────────
    # IEEE paper: sentiment fusion can shift signal
    if sentiment_score > 0.05 and signal == "HOLD":
        signal = "BUY"
    elif sentiment_score < -0.05 and signal == "HOLD":
        signal = "SELL"

    # ── Confidence from average R² (mapped 0…1) ───────────────────────────────
    avg_r2     = float(np.mean(list(model_scores.values()))) if model_scores else 0
    confidence = round(max(0.0, min(1.0, avg_r2)), 3)

    # ── MAPE on training set (using best model) ───────────────────────────────
    best_model_name = max(model_scores, key=model_scores.get) if model_scores else list(models.keys())[0]
    best_model = models[best_model_name]
    y_pred_train = best_model.predict(X_train)
    mape = round(float(mean_absolute_percentage_error(y_train, y_pred_train)) * 100, 3)

    # ── Indicator signal from technical analysis ──────────────────────────────
    from indicators import indicator_signal
    ind_sig = indicator_signal(df)

    return {
        "ticker":          ticker,
        "last_price":      last_price,
        "predicted_price": round(ensemble_pred, 4),
        "change_pct":      round(change_pct, 3),
        "signal":          signal,
        "confidence":      confidence,
        "mape":            mape,
        "model_scores":    model_scores,
        "ind_signal":      ind_sig["signal"],
        "ind_score":       ind_sig["score"],
        "ind_reasons":     ind_sig["reasons"],
        "rsi":             ind_sig.get("rsi", 50),
        "engulfing":       ind_sig.get("engulfing", 0),
    }


def run_all_predictions_v2(save: bool = True) -> pd.DataFrame:
    """
    Run ensemble predictions for all configured tickers.
    """
    try:
        from sentiment import average_sentiment
    except Exception:
        average_sentiment = lambda ticker: 0.0

    results = []

    for ticker in ALL_TICKERS:
        print(f"  [ML v2] {ticker} ...")
        sent = average_sentiment(ticker)
        pred = predict_ticker_v2(ticker, sentiment_score=sent)
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


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("[ML v2] Running IEEE-enhanced ensemble predictions...\n")
    df = run_all_predictions_v2(save=True)

    if not df.empty:
        print("\n" + "=" * 72)
        print(" IEEE-ENHANCED ENSEMBLE PREDICTION RESULTS")
        print("=" * 72)
        for _, row in df.iterrows():
            icon = "[BUY]" if row["signal"] == "BUY" else ("[SELL]" if row["signal"] == "SELL" else "[HOLD]")
            print(f"{icon} {row['ticker']:15s}  "
                  f"Last:{row['last_price']:10.2f}  "
                  f"Pred:{row['predicted_price']:10.2f}  "
                  f"D%:{row['change_pct']:+.2f}%  "
                  f"RSI:{row.get('rsi', 0):.0f}  "
                  f"Conf:{row['confidence']:.2f}  "
                  f"IND:{row.get('ind_signal','?')}")
        print("=" * 72)
    else:
        print("[!] No predictions generated. Run data_fetcher.py first.")
