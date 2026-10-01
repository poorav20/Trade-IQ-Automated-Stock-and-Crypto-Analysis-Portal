"""
indicators.py — Technical Analysis Indicators (v2 — Hyperliquid Enhanced)
Based on IEEE papers + Hyperliquid local_indicators.py:
  - "A Survey of Stock Market Prediction Based on ML Techniques" (GCCIT 2024)
  - "Cooperative Multi-Agent RL for Bitcoin Trading" (IPRIA 2025)
  - "Harnessing Sentiment Analysis and Deep Learning" (ICEC 2024)
  - Hyperliquid trading agent (sanketagarwal/hyperliquid-trading-agent)

New in v2 (ported from Hyperliquid):
  - EMA (20 & 50)        — trend direction
  - ADX                  — trend strength (Wilder smoothing)
  - OBV                  — On-Balance Volume
  - VWAP                 — Volume-Weighted Average Price
  - StochRSI             — Stochastic RSI (%K and %D)
  - Enhanced indicator_signal() using all 9 signals
"""
import pandas as pd
import numpy as np


# ── RSI (Relative Strength Index) ────────────────────────────────────────────
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI = 100 - (100 / (1 + RS))  where RS = avg_gain / avg_loss
    > 70 → Overbought (potential SELL)
    < 30 → Oversold  (potential BUY)
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("rsi")


# ── MACD (Moving Average Convergence Divergence) ─────────────────────────────
def compute_macd(close: pd.Series,
                 fast: int = 12, slow: int = 26, signal: int = 9
                 ) -> pd.DataFrame:
    """
    MACD Line   = EMA(12) - EMA(26)
    Signal Line = EMA(9) of MACD
    Histogram   = MACD - Signal
    """
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line

    return pd.DataFrame({
        "macd":      macd_line,
        "macd_sig":  signal_line,
        "macd_hist": histogram,
    })


# ── Bollinger Bands ───────────────────────────────────────────────────────────
def compute_bollinger(close: pd.Series,
                      period: int = 20, std_dev: float = 2.0
                      ) -> pd.DataFrame:
    """
    Mid   = SMA(20)
    Upper = Mid + 2 × std
    Lower = Mid - 2 × std
    %B    = (close - lower) / (upper - lower)   [0=lower band, 1=upper band]
    """
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b = (close - lower) / (upper - lower + 1e-10)

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_mid":   mid,
        "bb_lower": lower,
        "bb_pct_b": pct_b,
    })


# ── Engulfing Candle Pattern ──────────────────────────────────────────────────
def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Bullish Engulfing (+1) / Bearish Engulfing (-1) / No pattern (0)
    Primary signal from: "Cooperative Multi-Agent RL for Bitcoin Trading" (IPRIA 2025)
    """
    o = df["open"]
    c = df["close"]
    po = o.shift(1)
    pc = c.shift(1)

    bullish = (
        (pc < po) &           # previous = red
        (c  > o)  &           # current  = green
        (o  <= pc) &          # current open  ≤ prev close
        (c  >= po)            # current close ≥ prev open
    )

    bearish = (
        (pc > po) &           # previous = green
        (c  < o)  &           # current  = red
        (o  >= pc) &          # current open  ≥ prev close
        (c  <= po)            # current close ≤ prev open
    )

    result = pd.Series(0, index=df.index, name="engulfing")
    result[bullish] = 1
    result[bearish] = -1
    return result


# ── EMA (Exponential Moving Average) ─────────────────────────────────────────
def compute_ema(close: pd.Series, period: int) -> pd.Series:
    """Returns EMA of given period. Ported from Hyperliquid local_indicators."""
    return close.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")


# ── ADX (Average Directional Index) ──────────────────────────────────────────
def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ADX measures trend strength (not direction):
    > 25 → Strong trend (follow direction)
    < 20 → Weak/no trend (signals less reliable)
    Ported from Hyperliquid local_indicators.py (Wilder smoothing method)
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    plus_dm  = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_w       = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di     = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, 1e-10)
    minus_di    = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, 1e-10)
    dx          = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx_series  = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx_series.rename("adx")


# ── OBV (On-Balance Volume) ───────────────────────────────────────────────────
def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    OBV accumulates volume on up-days, subtracts on down-days.
    Rising OBV + rising price → strong bullish confirmation.
    Ported from Hyperliquid local_indicators.py
    """
    direction = np.where(df["close"] > df["close"].shift(1), 1,
                np.where(df["close"] < df["close"].shift(1), -1, 0))
    obv_raw = (df["volume"] * direction).cumsum()
    return pd.Series(obv_raw, index=df.index, name="obv")


# ── VWAP (Volume-Weighted Average Price) ──────────────────────────────────────
def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP = cumulative(typical_price × volume) / cumulative(volume)
    Price > VWAP → bullish bias
    Price < VWAP → bearish bias
    Ported from Hyperliquid local_indicators.py
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol       = df["volume"].cumsum()
    cum_tp_vol    = (typical_price * df["volume"]).cumsum()
    vwap_series   = cum_tp_vol / cum_vol.replace(0, 1e-10)
    return vwap_series.rename("vwap")


# ── Stochastic RSI ────────────────────────────────────────────────────────────
def compute_stoch_rsi(close: pd.Series,
                      rsi_period: int = 14,
                      stoch_period: int = 14,
                      k_smooth: int = 3,
                      d_smooth: int = 3) -> pd.DataFrame:
    """
    Stochastic RSI — more sensitive momentum oscillator.
    %K < 20 → Oversold (BUY signal)
    %K > 80 → Overbought (SELL signal)
    Ported from Hyperliquid local_indicators.py
    """
    rsi_vals = compute_rsi(close, rsi_period)

    stoch_k = (rsi_vals - rsi_vals.rolling(stoch_period).min()) / (
        rsi_vals.rolling(stoch_period).max() - rsi_vals.rolling(stoch_period).min() + 1e-10
    ) * 100

    k_line = stoch_k.rolling(k_smooth).mean()
    d_line = k_line.rolling(d_smooth).mean()

    return pd.DataFrame({"stoch_k": k_line, "stoch_d": d_line})


# ── Add All Indicators to a DataFrame ────────────────────────────────────────
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with columns: open, high, low, close, volume
    Returns DataFrame with ALL indicator columns added (original + 9 new from Hyperliquid).
    """
    df = df.copy()

    # --- Original indicators ---
    df["rsi"] = compute_rsi(df["close"], 14)
    df["rsi_7"] = compute_rsi(df["close"], 7)

    macd_df = compute_macd(df["close"])
    df = pd.concat([df, macd_df], axis=1)

    bb_df = compute_bollinger(df["close"])
    df = pd.concat([df, bb_df], axis=1)

    df["engulfing"] = detect_engulfing(df)

    # ATR
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_3"] = tr.rolling(3).mean()

    # --- NEW (Hyperliquid) indicators ---
    df["ema_20"] = compute_ema(df["close"], 20)
    df["ema_50"] = compute_ema(df["close"], 50)
    df["adx"]   = compute_adx(df)
    df["obv"]   = compute_obv(df)
    df["vwap"]  = compute_vwap(df)

    stoch_df = compute_stoch_rsi(df["close"])
    df = pd.concat([df, stoch_df], axis=1)

    return df


# ── Enhanced Signal from All Indicators ──────────────────────────────────────
def indicator_signal(df: pd.DataFrame) -> dict:
    """
    Derives a composite signal from ALL indicators (original + Hyperliquid).
    Returns: {'signal': 'BUY'|'SELL'|'HOLD', 'score': float, 'reasons': list}

    Scoring weights:
      Engulfing   ±2.0  (primary — Q-Learning paper)
      RSI         ±1.5
      MACD        ±1.0
      Bollinger   ±1.0
      EMA cross   ±1.5  (NEW — trend direction)
      ADX boost   ×1.5  (NEW — multiplies score if strong trend)
      VWAP        ±0.75 (NEW — price vs VWAP)
      StochRSI    ±1.0  (NEW — momentum extremes)
      OBV slope   ±0.5  (NEW — volume confirmation)
    """
    row     = df.iloc[-1]
    score   = 0.0
    reasons = []

    # 1. RSI
    rsi_val = row.get("rsi", 50)
    if rsi_val < 30:
        score += 1.5
        reasons.append(f"RSI={rsi_val:.1f} (oversold → BUY)")
    elif rsi_val > 70:
        score -= 1.5
        reasons.append(f"RSI={rsi_val:.1f} (overbought → SELL)")

    # 2. MACD crossover
    macd_val = row.get("macd", 0)
    sig_val  = row.get("macd_sig", 0)
    if macd_val > sig_val:
        score += 1.0
        reasons.append("MACD > Signal (bullish)")
    elif macd_val < sig_val:
        score -= 1.0
        reasons.append("MACD < Signal (bearish)")

    # 3. Bollinger %B
    pct_b = row.get("bb_pct_b", 0.5)
    if pct_b < 0.1:
        score += 1.0
        reasons.append(f"%B={pct_b:.2f} (near lower band → BUY)")
    elif pct_b > 0.9:
        score -= 1.0
        reasons.append(f"%B={pct_b:.2f} (near upper band → SELL)")

    # 4. Engulfing pattern (primary Q-Learning signal)
    eng = row.get("engulfing", 0)
    if eng == 1:
        score += 2.0
        reasons.append("Bullish Engulfing candle pattern detected")
    elif eng == -1:
        score -= 2.0
        reasons.append("Bearish Engulfing candle pattern detected")

    # 5. EMA 20/50 crossover (NEW — Hyperliquid)
    ema20 = row.get("ema_20", 0)
    ema50 = row.get("ema_50", 0)
    close = row.get("close", 0)
    if ema20 > 0 and ema50 > 0:
        if ema20 > ema50:
            score += 1.5
            reasons.append(f"EMA20({ema20:.2f}) > EMA50({ema50:.2f}) → uptrend")
        else:
            score -= 1.5
            reasons.append(f"EMA20({ema20:.2f}) < EMA50({ema50:.2f}) → downtrend")

    # 6. VWAP position (NEW — Hyperliquid)
    vwap_val = row.get("vwap", 0)
    if vwap_val > 0 and close > 0:
        if close > vwap_val:
            score += 0.75
            reasons.append(f"Price({close:.2f}) > VWAP({vwap_val:.2f}) → bullish bias")
        else:
            score -= 0.75
            reasons.append(f"Price({close:.2f}) < VWAP({vwap_val:.2f}) → bearish bias")

    # 7. StochRSI extremes (NEW — Hyperliquid)
    stoch_k = row.get("stoch_k", 50)
    if pd.notna(stoch_k):
        if stoch_k < 20:
            score += 1.0
            reasons.append(f"StochRSI %K={stoch_k:.1f} (oversold → BUY)")
        elif stoch_k > 80:
            score -= 1.0
            reasons.append(f"StochRSI %K={stoch_k:.1f} (overbought → SELL)")

    # 8. OBV slope confirmation (NEW — Hyperliquid)
    if len(df) >= 5:
        obv_now  = df["obv"].iloc[-1]
        obv_prev = df["obv"].iloc[-5]
        if pd.notna(obv_now) and pd.notna(obv_prev):
            obv_change = obv_now - obv_prev
            if obv_change > 0 and score > 0:
                score += 0.5
                reasons.append("OBV rising → volume confirms uptrend")
            elif obv_change < 0 and score < 0:
                score -= 0.5
                reasons.append("OBV falling → volume confirms downtrend")

    # 9. ADX trend strength multiplier (NEW — Hyperliquid)
    adx_val = row.get("adx", 0)
    if pd.notna(adx_val) and adx_val > 25:
        score *= 1.5
        reasons.append(f"ADX={adx_val:.1f} (strong trend → signals amplified 1.5×)")
    elif pd.notna(adx_val) and adx_val < 15:
        score *= 0.5
        reasons.append(f"ADX={adx_val:.1f} (weak trend → signals dampened 0.5×)")

    # Final signal (score threshold ±2.0 due to more indicators)
    if score >= 2.0:
        signal = "BUY"
    elif score <= -2.0:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal":    signal,
        "score":     round(score, 2),
        "reasons":   reasons,
        "rsi":       round(float(rsi_val), 2),
        "rsi_7":     round(float(row.get("rsi_7", 50)), 2),
        "macd":      round(float(macd_val), 4),
        "macd_sig":  round(float(sig_val), 4),
        "bb_pct_b":  round(float(pct_b), 3),
        "engulfing": int(eng),
        "ema_20":    round(float(ema20), 2),
        "ema_50":    round(float(ema50), 2),
        "adx":       round(float(adx_val), 2) if pd.notna(adx_val) else 0,
        "stoch_k":   round(float(stoch_k), 2) if pd.notna(stoch_k) else 50,
        "vwap":      round(float(vwap_val), 2),
        "atr_3":     round(float(row.get("atr_3", 0)), 2),
        "atr_14":    round(float(row.get("atr", 0)), 2),
    }


# ── CLI Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from database import init_db, get_price_df

    init_db()
    for ticker in ["RELIANCE.NS", "AAPL", "BTC-USD"]:
        df = get_price_df(ticker, days=90)
        if df.empty:
            print(f"{ticker}: no data")
            continue

        df = add_all_indicators(df)
        sig = indicator_signal(df)

        print(f"\n{ticker}")
        print(f"  Signal   : {sig['signal']}  (score={sig['score']})")
        print(f"  RSI      : {sig['rsi']}")
        print(f"  MACD     : {sig['macd']}")
        print(f"  BB %B    : {sig['bb_pct_b']}")
        print(f"  Engulfing: {sig['engulfing']}")
        print(f"  EMA20/50 : {sig['ema_20']} / {sig['ema_50']}")
        print(f"  ADX      : {sig['adx']}")
        print(f"  StochRSI : {sig['stoch_k']}")
        print(f"  VWAP     : {sig['vwap']}")
        for r in sig["reasons"]:
            print(f"  >> {r}")
