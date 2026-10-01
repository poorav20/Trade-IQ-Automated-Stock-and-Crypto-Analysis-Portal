# TradeIQ Auto-Trading Portal Methodology

This methodology document outlines the core architecture and operational flow of the TradeIQ portal. The platform incorporates IEEE-inspired machine learning techniques, sentiment analysis, and multi-agent automated trading to provide a comprehensive market analysis and execution engine.

## 1. Data Acquisition & Preprocessing
The foundation of the portal relies on accurate, real-time, and historical data.
- **Market Data Engine (`data_fetcher.py`)**: Fetches OHLCV (Open, High, Low, Close, Volume) data for Indian stocks (e.g., RELIANCE.NS), US stocks, and Cryptocurrencies via Yahoo Finance (`yfinance`) and CoinGecko.
- **Storage Layer (`database.py`)**: Persists ingested data into a centralized, lightweight local SQLite database (`trading.db`).
- **Automation Pipeline (`n8n_workflow_FINAL_LOCAL.json`)**: The local n8n workflow uses both a manual trigger and a 5-minute schedule, calls the FastAPI bridge on `127.0.0.1:8502`, refreshes market data, runs sentiment plus ML predictions, and executes the paper-trading cycle against the freshest database state.

## 2. Technical and Sentiment Analysis
Market data is translated into actionable insights using composite technical indicators and natural language processing.
- **Technical Indicators (`indicators.py`)**: Automatically calculates essential momentum and trend metrics:
  - **Moving Averages** (MA-7 and MA-14)
  - **Relative Strength Index** (RSI)
  - **MACD** (Moving Average Convergence Divergence)
  - **Bollinger Bands** (%B)
  - **Engulfing Candlestick Patterns** (acts as a primary trend-reversal indicator)
- **News Sentiment (`sentiment.py`)**: Parses Google News RSS feeds for listed tickers, scoring headlines using the **VaderSentiment** NLP engine. This sentiment is mapped on a continuous score from `-1.0` (Bearish) to `+1.0` (Bullish).

## 3. IEEE-Enhanced Machine Learning Engine
The core intelligence stems from a multi-model ensemble predicting the future state by interpreting historical and technical state spaces (`ml_model_v2.py`). 
- **The Ensemble Architecture**:
  - **MLP Regressor (Neural Network)**: Simulates deep-learning sequence behaviours with complex hidden layers natively.
  - **Gradient Boosting**: Handles non-linear interactions efficiently.
  - **Linear Regression**: Acts as a robust, non-overfitting baseline.
- **Feature Engineering & Fusion**: Combines historical lag returns (t-1 through t-5), rolling averages, calculated technical indicators, and significantly, fuses the **external sentiment target score** as a core training feature.
- **Scoring & Confidence Generation**: Employs Cross-Validated R² scoring to evaluate model health in real-time. Models with higher R² dictate a heavier weight on the final target price prediction. The pipeline outputs directional signals (BUY/SELL/HOLD), an estimated future price, and an aggregated Confidence Metric.

## 4. Q-Learning Inspired Multi-Agent Auto-Trader
The execution framework simulates multi-agent reinforcement limits (`auto_trader.py`) with strict rulesets, primarily used for paper trading (virtual capital).
- **BUY Agent Mechanics**: Constantly monitors the asset pool. It explicitly executes entry orders when:
  1. The ML ensemble dictates a `BUY` signal.
  2. The combined technical indicator score surpasses a custom `COMPOSITE_THRESHOLD`.
  3. The model confidence strictly exceeds the required threshold.
- **SELL Agent & Risk Management**: Aggressively protects capital by evaluating open positions constantly against:
  - A strict **Stop-Loss** floor (e.g., -2.0% loss limit).
  - A definitive **Take-Profit** target (e.g., +4.0% profit limit).
  - Automatically liquidating positions if the technical state severely inverts (Composite score crosses the negative threshold alongside a `SELL` ML signal).
- **Portfolio Tracking Layer**: Manages a virtual ledger (starting at ₹100,000 default), tracking precise entry dates, quantities, resulting P&L (Profit & Loss), and recording historical transactions purely within SQLite.

## 5. Web Interface & Dashboarding
The overarching user experience bridges complex algorithms with an accessible, simplified interface (`app.py`):
- Engineered using **Streamlit**.
- Presents "Agent vs Manual" execution toggles and dynamically normalizes global asset prices into localized currency (INR).
- Replaces complex raw candlestick readings with high-level "Top Picks" rows, confidence meters, active position real-time tracking, and simplified alerts.
