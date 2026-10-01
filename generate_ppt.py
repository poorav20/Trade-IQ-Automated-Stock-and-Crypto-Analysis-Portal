"""
generate_ppt.py — Generates the TradeIQ Minor Project PPT
using the uploaded template: PPT Template for Minor Project.pptx

The template has 14 slides:
  Slide 1  : TITLE layout   (title + subtitle)
  Slide 2  : OBJECT layout  (content slides — we write into text boxes)
  ...
  Slide 14 : TITLE_ONLY     (Thank You)

We clone the template's existing slides and overwrite their text shapes.

Run:
    python generate_ppt.py
Output:
    TradeIQ_Project_Presentation.pptx
"""
import copy
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import os

TEMPLATE   = r"f:\AI Agents\trading-portal\PPT Template for Minor Project.pptx"
OUTPUT     = r"f:\AI Agents\trading-portal\TradeIQ_IEEE_v2.pptx"

# ── Slide content (IEEE v2 Enhanced) ─────────────────────────────────────────
SLIDES = [
    # Slide 1 — Title
    {
        "slide_idx": 0,
        "type": "title",
        "title":    "TradeIQ v2\nIEEE-Enhanced Automated Trading & Analysis Portal",
        "subtitle": (
            "Minor Project — Department of CSE (AI & ML)\n"
            "Section: ___ | Academic Year: 2025-26\n\n"
            "Students:\n"
            "  Student 1 — USN\n"
            "  Student 2 — USN\n\n"
            "Guide: Prof. ___________"
        ),
    },

    # Slide 2 — Agenda
    {
        "slide_idx": 1,
        "type": "content",
        "heading": "Agenda",
        "bullets": [
            "01  Introduction & Motivation",
            "02  Problem Statement",
            "03  Objectives (IEEE-Informed)",
            "04  IEEE Research Papers Used",
            "05  System Architecture (5 Layers)",
            "06  Technology Stack",
            "07  Data Collection Layer",
            "08  IEEE-Enhanced ML Ensemble Model",
            "09  Technical Indicators (RSI / MACD / BB / Engulfing)",
            "10  Q-Learning Auto-Trader (Paper Trading)",
            "11  Sentiment Analysis",
            "12  Dashboard & Mobile Access",
            "13  Results, Conclusion & Future Work",
        ],
    },

    # Slide 3 — Introduction
    {
        "slide_idx": 2,
        "type": "content",
        "heading": "Introduction & Motivation",
        "bullets": [
            "1. The Problem — Information Overload for Investors",
            "   Financial markets generate massive price, news, and signal data every second.",
            "   Retail investors lack affordable, unified tools to make confident decisions.",
            "",
            "2. The Solution — TradeIQ v2: An Automated Analysis Portal",
            "   Full-stack Python portal with real-time prices, IEEE-enhanced ML ensemble,",
            "   news sentiment, Q-Learning auto-trader, and interactive Streamlit dashboard.",
            "",
            "3. The Impact — Three Intelligence Layers in One Platform",
            "   Technical Analysis  : Ensemble ML + RSI + MACD + Bollinger Bands + Engulfing",
            "   Fundamental Analysis: VADER + TextBlob sentiment fusion on live news",
            "   Automation          : Q-inspired BUY/SELL agents + n8n 5-min data refresh",
            "",
            "   Zero subscription cost. Accessible on desktop & mobile as a PWA.",
        ],
    },

    # Slide 4 — Problem Statement
    {
        "slide_idx": 3,
        "type": "content",
        "heading": "Problem Statement",
        "bullets": [
            "Traditional investment analysis requires:",
            "  • Expensive Bloomberg/Reuters terminals",
            "  • Deep expertise in Technical Analysis and Fundamental Analysis",
            "  • Manual cross-referencing of news with price movements",
            "",
            "Existing free tools (Zerodha Kite, Groww) lack:",
            "  • ML-powered predictive signals",
            "  • Automated news sentiment scoring",
            "  • Auto-trading engine with risk management",
            "  • Mobile-friendly progressive web app experience",
            "",
            "Research Gap (from IEEE Survey, GCCIT 2024):",
            "  • Single-model predictions are insufficient for volatile markets",
            "  • Most systems ignore sentiment as a ML feature",
            "  • No open-source system combines all layers for retail investors",
            "",
            "Research Question: Can an ensemble ML + Q-Learning system outperform",
            "  single-model approaches for multi-asset trading at zero cost?",
        ],
    },

    # Slide 5 — Objectives
    {
        "slide_idx": 4,
        "type": "content",
        "heading": "Objectives (IEEE-Informed)",
        "bullets": [
            "1. Fetch real-time OHLCV data for 14 tickers (Indian, US, Crypto).",
            "2. Store 90-day history in SQLite for ML training and backtesting.",
            "3. Compute technical indicators: RSI, MACD, Bollinger Bands, ATR,",
            "   and Engulfing candle pattern (as used in IEEE IPRIA 2025 Q-Learning paper).",
            "4. Train an ENSEMBLE model (LR + Gradient Boosting + MLPRegressor)",
            "   with sentiment score fused as a feature (from IEEE ICEC 2024).",
            "5. Implement cooperative BUY/SELL paper-trading agents (IEEE IPRIA 2025).",
            "6. Score news sentiment using VADER + TextBlob fusion.",
            "7. Display all results in a Streamlit dashboard with interactive Plotly charts.",
            "8. Enable mobile access via ngrok tunnel and PWA home-screen installation.",
            "9. Automate data refresh every 5 minutes via n8n workflow.",
        ],
    },

    # Slide 6 — IEEE Papers Used
    {
        "slide_idx": 5,
        "type": "content",
        "heading": "IEEE Research Papers Used (25 Papers, 4 Categories)",
        "bullets": [
            "Category 1 — Cryptocurrency (6 papers)",
            "  KEY: 'Cooperative Multi-Agent RL for Bitcoin Trading' (IPRIA 2025)",
            "       → Q-Learning BUY/SELL agents; 57% win rate; 8% better annual profit",
            "  KEY: 'Automatic Stocks & Crypto Research using Deep Learning' (ICIRCA 2025)",
            "       → Transformer NLP + Big Data Analytics framework",
            "",
            "Category 2 — Global Market Analysis (5 papers)",
            "  KEY: 'Harnessing Sentiment Analysis & Deep Learning' (ICEC 2024)",
            "       → GRU + LSTM + Random Forest ensemble; sentiment as feature",
            "  KEY: 'Survey of ML Techniques for Stock Prediction' (GCCIT 2024)",
            "       → RSI, MACD, Bollinger Bands recommended as model features",
            "",
            "Category 3 — Currency Exchange (5 papers)",
            "  KEY: 'Systematic Review for Crypto Price Prediction using ML' (CCICT 2022)",
            "       → Social sentiment + price improves crypto prediction",
            "",
            "Category 4 — India Trade Analysis (6 papers)",
            "  KEY: 'Algorithm Trading through API in Indian Stock Market'",
            "       → Risk controls (stop-loss, take-profit) for retail AT",
        ],
    },

    # Slide 7 — System Architecture
    {
        "slide_idx": 6,
        "type": "content",
        "heading": "System Architecture — 5 Layers",
        "bullets": [
            "LAYER 1 — DATA FETCHER  (data_fetcher.py)",
            "  yfinance → 90-day OHLCV for 14 tickers | Live quotes every 5 min",
            "  n8n Cron workflow → auto-triggers fetch every 5 minutes",
            "",
            "LAYER 2 — DATA WAREHOUSE  (database.py)",
            "  SQLite trading.db → 6 tables:",
            "  price_history | news_sentiment | ml_predictions | trades | portfolio | portfolio_state",
            "",
            "LAYER 3 — AI BRAIN (IEEE-Enhanced)",
            "  indicators.py  → RSI, MACD, Bollinger Bands, Engulfing, ATR",
            "  ml_model_v2.py → Ensemble: LR + GradientBoosting + MLPRegressor",
            "  sentiment.py   → VADER + TextBlob (fused into ML features)",
            "",
            "LAYER 4 — AUTO-TRADER  (auto_trader.py)",
            "  Q-inspired BUY Agent + SELL Agent (stop-loss / take-profit)",
            "  Paper Trade Mode: virtual Rs.1,00,000 portfolio",
            "",
            "LAYER 5 — INTERFACE  (app.py)",
            "  Streamlit + Plotly → Live dashboard | Mobile PWA | ngrok tunnel",
        ],
    },

    # Slide 8 — Technology Stack
    {
        "slide_idx": 7,
        "type": "content",
        "heading": "Technology Stack",
        "bullets": [
            "Language : Python 3.14",
            "",
            "Data Layer:",
            "  yfinance 1.2, feedparser 6.0, requests 2.32, SQLAlchemy 2.0",
            "",
            "Machine Learning (IEEE-Enhanced v2):",
            "  scikit-learn 1.8  — LinearRegression, GradientBoosting, MLPRegressor",
            "  numpy 2.4.2       — Numerical computing",
            "  pandas 2.3.3      — Data manipulation",
            "",
            "Technical Indicators (new — indicators.py):",
            "  Custom Python: RSI, MACD, Bollinger Bands, Engulfing, ATR",
            "",
            "Natural Language Processing:",
            "  vaderSentiment 3.3.2 — Fast rule-based sentiment",
            "  textblob             — Polarity as second sentiment source",
            "",
            "Dashboard & Visualisation:",
            "  Streamlit 1.54, Plotly 6.5.2, python-pptx 1.0.2",
            "",
            "Automation & Mobile:",
            "  n8n (Cron + HTTP), ngrok v3, PWA meta tags (.streamlit/config.toml)",
        ],
    },

    # Slide 9 — IEEE-Enhanced ML Ensemble
    {
        "slide_idx": 8,
        "type": "content",
        "heading": "IEEE-Enhanced ML Ensemble Model (ml_model_v2.py)",
        "bullets": [
            "Based On: 'GRU & LSTM + Sentiment Analysis' (ICEC 2024)",
            "           'Survey of ML Techniques' (GCCIT 2024)",
            "",
            "Three Models in Ensemble:",
            "  1. Linear Regression      — Baseline trend predictor (R^2 = 0.995 verified)",
            "  2. Gradient Boosting (100 trees, depth=3) — Non-linear patterns",
            "  3. MLPRegressor (64-32-16) — Neural network approximating LSTM",
            "",
            "Weighted Averaging: each model weighted by its cross-val R^2 score",
            "  (Better model → higher weight in final prediction)",
            "",
            "IEEE-Recommended Feature Set (17 features):",
            "  Lag features  : lag_1 to lag_5 (previous closing prices)",
            "  Moving Avgs   : MA-7, MA-14",
            "  Indicators    : RSI, MACD, MACD Signal, MACD Histogram",
            "  Bollinger     : %B band position",
            "  Engulfing     : Bullish (+1) / Bearish (-1) / None (0)",
            "  Sentiment     : VADER score fused as numeric feature",
            "  Other         : % change, ATR, day-of-week",
            "",
            "Signal: Predicted Delta% > +0.5% → BUY | < -0.5% → SELL | else HOLD",
        ],
    },

    # Slide 10 — Technical Indicators
    {
        "slide_idx": 9,
        "type": "content",
        "heading": "Technical Indicators Module (IEEE-Recommended)",
        "bullets": [
            "File: indicators.py  |  Reference: GCCIT 2024 + IPRIA 2025",
            "",
            "RSI — Relative Strength Index (period=14)",
            "  > 70  → Overbought (potential SELL signal)",
            "  < 30  → Oversold (potential BUY signal)",
            "  Live Result for RELIANCE.NS: RSI = 39.95 (Neutral)",
            "",
            "MACD — Moving Average Convergence Divergence (12, 26, 9)",
            "  MACD > Signal → Bullish Momentum | MACD < Signal → Bearish",
            "  Histogram bars shown in dashboard chart",
            "",
            "Bollinger Bands — (period=20, 2 std deviations)",
            "  %B near 0 → Price at lower band (BUY zone)",
            "  %B near 1 → Price at upper band (SELL zone)",
            "",
            "Engulfing Candle Pattern (Primary signal in IEEE Q-Learning paper):",
            "  Bullish Engulfing (+1): Green candle body engulfs previous red body",
            "  Bearish Engulfing (-1): Red candle body engulfs previous green body",
            "",
            "Composite Score: RSI + MACD + BB%B + Engulfing → BUY/SELL/HOLD",
            "  Score >= +1.5 → BUY | Score <= -1.5 → SELL | else HOLD",
        ],
    },

    # Slide 11 — Auto-Trader
    {
        "slide_idx": 10,
        "type": "content",
        "heading": "Q-Learning Inspired Auto-Trader (Paper Trading)",
        "bullets": [
            "File: auto_trader.py",
            "Based On: 'Cooperative Multi-Agent RL for Bitcoin Trading' (IEEE IPRIA 2025)",
            "  → 57% win rate | 8% better annual profit | Calmar ratio = 2.8",
            "",
            "Architecture — Two Cooperative Agents:",
            "  BUY  Agent: Fires when ML=BUY AND Indicator Score >= 2.0",
            "              AND ML Confidence >= 35%  AND no existing position",
            "  SELL Agent: Fires when Stop-Loss hit (3%) OR Take-Profit hit (6%)",
            "              OR ML=SELL AND Indicator Score <= -2.0",
            "",
            "Risk Management (configurable in dashboard sidebar):",
            "  Max Position  : 10% of portfolio per trade (default)",
            "  Stop-Loss     : 3% (auto-exit on loss)",
            "  Take-Profit   : 6% (auto-exit on gain)",
            "",
            "Paper Trading Mode (SAFE — no real money):",
            "  Starts with virtual Rs. 1,00,000 capital",
            "  All trades logged to SQLite: ticker, action, qty, entry, exit, P&L",
            "  Equity curve and trade history shown in Streamlit dashboard",
            "",
            "Real Broker: Zerodha Kite API / Alpaca / Binance (optional, keys in .env)",
        ],
    },

    # Slide 12 — Dashboard & Mobile
    {
        "slide_idx": 11,
        "type": "content",
        "heading": "Dashboard (v2) & Mobile Access",
        "bullets": [
            "File: app.py  |  URL: http://localhost:8501  |  PWA installable on phone",
            "",
            "Dashboard Panels:",
            "  1. Live Market Snapshot — 5 ticker price cards with % change",
            "  2. Candlestick Chart    — OHLCV + MA-7 + MA-14 overlays",
            "  3. ML Signal Panel      — BUY/SELL/HOLD + Confidence bar",
            "  4. Sentiment Feed       — Market mood + colour-coded headlines",
            "  5. Portfolio Tracker    — Enter holdings, see live P&L",
            "  NEW: Technical Indicators — RSI gauge, MACD histogram chart,",
            "       Bollinger %B card, Engulfing pattern card",
            "  NEW: Auto-Trader Panel — Risk sliders, Run Cycle button,",
            "       Virtual cash balance, Open positions, Trade history log",
            "",
            "Mobile Access:",
            "  Same Wi-Fi   : http://10.123.219.240:8501",
            "  Anywhere     : python install_ngrok.py → launch_mobile.bat",
            "  PWA Install  : iPhone Safari Share→Add to Home | Android Chrome menu",
        ],
    },

    # Slide 13 — Results & Conclusion
    {
        "slide_idx": 12,
        "type": "content",
        "heading": "Results, Conclusion & Future Work",
        "bullets": [
            "Verified Results:",
            "  [OK] 1,260 OHLCV rows fetched (14 tickers x 90 days)",
            "  [OK] Ensemble ML: Linear Regression R^2 = 0.995 on RELIANCE.NS",
            "  [OK] RSI = 39.95 (Neutral), MACD = Bearish, Engulfing = No pattern",
            "  [OK] Auto-trader initialised: virtual cash = Rs.1,00,000",
            "  [OK] Dashboard live at http://localhost:8501 with all v2 panels",
            "  [OK] PWA mobile access verified on Wi-Fi URL",
            "  [OK] n8n workflow automates data every 5 minutes",
            "",
            "Conclusion:",
            "  TradeIQ v2 implements concepts from 25 IEEE research papers into",
            "  a single zero-cost trading intelligence portal. The ensemble ML +",
            "  Q-Learning approach mirrors state-of-art industry techniques.",
            "",
            "Future Work:",
            "  → Full LSTM / Transformer model (IEEE ICIRCA 2025 architecture)",
            "  → Live broker integration (Zerodha Kite API for Indian stocks)",
            "  → WhatsApp/Telegram alert bot for BUY signals",
            "  → Backtesting engine with Sharpe/Calmar ratio reporting",
            "  → Reinforcement Learning policy gradient agent (full MARL)",
        ],
    },
]

# ── Helper: set text in a shape (handles both text boxes and placeholders) ────

def set_shape_text(shape, text, font_size=None, bold=None, color=None, align=None):
    """Replace all text in a shape's text frame while keeping formatting."""
    if not shape.has_text_frame:
        return
    tf = shape.text_frame
    tf.word_wrap = True

    # Clear existing paragraphs, keep the first
    for i in range(len(tf.paragraphs) - 1, 0, -1):
        p = tf.paragraphs[i]._p
        p.getparent().remove(p)

    lines = text.split("\n")
    for li, line in enumerate(lines):
        if li == 0:
            para = tf.paragraphs[0]
        else:
            para = tf.add_paragraph()

        para.text = line

        run = para.runs[0] if para.runs else para.add_run()
        run.text = line

        if font_size:
            run.font.size = Pt(font_size)
        if bold is not None:
            run.font.bold = bold
        if color:
            run.font.color.rgb = RGBColor(*color)
        if align:
            para.alignment = align


def fill_content_slide(slide, heading, bullets):
    """
    The OBJECT layout content slides in this template have their text
    in regular shapes (not in idx=0/1 placeholders which are hidden/absent).
    We find all text-bearing shapes sorted by vertical position, then
    write heading into the topmost and bullets into the rest.
    """
    shapes = slide.shapes

    # Collect all shapes that have text frames
    all_text = [s for s in shapes if s.has_text_frame]

    # Sort top-to-bottom
    all_text_sorted = sorted(all_text, key=lambda s: s.top)

    if not all_text_sorted:
        return  # nothing to write to

    if len(all_text_sorted) >= 2:
        title_shape = all_text_sorted[0]
        body_shape  = all_text_sorted[1]

        # Write heading into topmost shape
        set_shape_text(title_shape, heading, font_size=20, bold=True,
                       color=(30, 58, 138))

        # Write bullets into the next shape
        body_text = "\n".join(bullets)
        set_shape_text(body_shape, body_text, font_size=13)
    else:
        # Single shape: combine heading + bullets
        full_text = heading + "\n" + "─" * 40 + "\n" + "\n".join(bullets)
        set_shape_text(all_text_sorted[0], full_text, font_size=12)



# ── Main ──────────────────────────────────────────────────────────────────────

def generate():
    prs = Presentation(TEMPLATE)

    for slide_data in SLIDES:
        idx   = slide_data["slide_idx"]
        stype = slide_data.get("type", "content")
        slide = prs.slides[idx]

        if stype == "title":
            # Slide 1: update the CENTER_TITLE + SUBTITLE placeholders
            for ph in slide.placeholders:
                if ph.placeholder_format.idx == 0:
                    set_shape_text(ph, slide_data["title"],
                                   font_size=28, bold=True, color=(30, 58, 138))
                elif ph.placeholder_format.idx == 1:
                    set_shape_text(ph, slide_data["subtitle"],
                                   font_size=16)
        else:
            fill_content_slide(slide, slide_data["heading"], slide_data["bullets"])

    # Slide 14 (index 13) — leave as-is (Thank You slide from template)
    # Just ensure it says something useful if it has text shapes
    last = prs.slides[13]
    for shape in last.shapes:
        if shape.has_text_frame:
            txt = shape.text_frame.text.strip()
            if txt and "thank" in txt.lower():
                break  # already has Thank You text

    prs.save(OUTPUT)
    print(f"[OK] Presentation saved: {OUTPUT}")
    print(f"     Slides: {len(prs.slides)}")


if __name__ == "__main__":
    generate()
