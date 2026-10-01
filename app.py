"""
app.py — Streamlit Trading & Analysis Dashboard

Launch with:
    streamlit run app.py

Features:
  • Live price cards with % change
  • Interactive Plotly candlestick chart + ML prediction overlay
  • BUY / SELL / HOLD signal badge with confidence meter
  • News sentiment feed with color coding
  • 7-day Prophet forecast (if prophet is installed)
  • Portfolio tracker — enter your holdings, see P&L
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import os
import time

DEFAULT_WALLET_AMOUNT = 100000.0

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title  = "TradeIQ — Market Portal",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "collapsed",   # collapsed by default = better on mobile
)

# ── PWA + Mobile support ──────────────────────────────────────────────────────
try:
    from pwa_config import inject_pwa
    inject_pwa()
except Exception:
    pass  # graceful fallback if pwa_config.py is missing

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%);
    color: #e2e8f0;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, rgba(30,41,59,0.9), rgba(15,22,40,0.9));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, border-color 0.2s;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.metric-card:hover { transform: translateY(-2px); border-color: rgba(99,102,241,0.5); }
.metric-label  { font-size: 12px; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
.metric-value  { font-size: 26px; font-weight: 700; color: #f1f5f9; }
.metric-change { font-size: 14px; font-weight: 500; margin-top: 4px; }
.up   { color: #22c55e; }
.down { color: #ef4444; }
.flat { color: #94a3b8; }

/* Signal badges */
.signal-buy  { background:#14532d; color:#4ade80; border:1px solid #16a34a; padding:6px 18px; border-radius:50px; font-weight:700; font-size:18px; }
.signal-sell { background:#450a0a; color:#f87171; border:1px solid #dc2626; padding:6px 18px; border-radius:50px; font-weight:700; font-size:18px; }
.signal-hold { background:#1c1917; color:#fbbf24; border:1px solid #d97706; padding:6px 18px; border-radius:50px; font-weight:700; font-size:18px; }

/* News feed */
.news-item { border-left: 3px solid; padding: 8px 14px; margin: 6px 0; border-radius: 0 8px 8px 0; background: rgba(15,23,42,0.5); }
.news-bull  { border-color: #22c55e; }
.news-bear  { border-color: #ef4444; }
.news-neut  { border-color: #94a3b8; }

/* Portfolio table */
.dataframe { background: rgba(15,23,42,0.8) !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: rgba(10,14,26,0.95); }

/* Section headers */
.section-header {
    font-size: 18px; font-weight: 600; color: #818cf8;
    border-bottom: 1px solid rgba(99,102,241,0.3);
    padding-bottom: 8px; margin: 24px 0 16px;
    letter-spacing: 0.04em;
}

/* Divider */
hr { border-color: rgba(99,102,241,0.15) !important; }

/* ── Wallet ── */
.wallet-card {
    background: linear-gradient(135deg, #0f2027, #1a3a4a, #0f2027);
    border: 1px solid rgba(34,197,94,0.35);
    border-radius: 18px;
    padding: 18px 24px;
    text-align: center;
    box-shadow: 0 0 30px rgba(34,197,94,0.12);
    position: relative;
    overflow: hidden;
}
.wallet-card::before {
    content: '';
    position: absolute; top: -40%; left: -40%;
    width: 180%; height: 180%;
    background: radial-gradient(circle, rgba(34,197,94,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.wallet-label  { font-size: 11px; color: #64748b; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px; }
.wallet-amount { font-size: 36px; font-weight: 800; color: #22c55e; letter-spacing: -0.02em; }
.wallet-sub    { font-size: 12px; color: #475569; margin-top: 4px; }
.wallet-pill   { display:inline-block; background:rgba(34,197,94,0.1); border:1px solid rgba(34,197,94,0.25);
                 border-radius:50px; padding:3px 12px; font-size:11px; color:#4ade80; margin-top:8px; }
</style>
""", unsafe_allow_html=True)


# ── Lazy imports (only after pip install) ────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_modules():
    """Cache heavy imports so we only pay the cost once."""
    from database     import init_db, get_price_df, get_sentiment_df
    from data_fetcher import get_live_quotes, TICKERS
    from ml_model     import predict_ticker, get_prophet_forecast
    from sentiment    import average_sentiment, sentiment_label
    init_db()
    return {
        "get_price_df":       get_price_df,
        "get_sentiment_df":   get_sentiment_df,
        "get_live_quotes":    get_live_quotes,
        "TICKERS":            TICKERS,
        "predict_ticker":     predict_ticker,
        "get_prophet_forecast": get_prophet_forecast,
        "average_sentiment":  average_sentiment,
        "sentiment_label":    sentiment_label,
    }


try:
    M = _load_modules()
except Exception as e:
    st.error(f"⚠️ Failed to load modules: {e}\n\nRun `python data_fetcher.py` first.")
    st.stop()

TICKERS = M["TICKERS"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:36px;'>📈</div>
        <div style='font-size:22px; font-weight:700; color:#818cf8;'>TradeIQ</div>
        <div style='font-size:11px; color:#64748b; letter-spacing:0.1em;'>MARKET PORTAL</div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    # Build ticker groups for selectbox
    ticker_groups = M["TICKERS"]
    ticker_options = []
    ticker_labels  = {}
    for group, tickers in ticker_groups.items():
        for t in tickers:
            ticker_options.append(t)
            ticker_labels[t] = f"{group}  ›  {t}"

    selected_ticker = st.selectbox(
        "🔍 Select Ticker",
        options   = ticker_options,
        format_func = lambda x: ticker_labels.get(x, x),
        index     = 0,
    )

    chart_days = st.slider("📅 Chart History (days)", 14, 90, 60)

    st.markdown("<hr/>", unsafe_allow_html=True)

    auto_refresh = st.checkbox("🔄 Auto-refresh every 5 min", value=False)

    if st.button("⚡ Fetch Latest Data Now", use_container_width=True):
        with st.spinner("Fetching data …"):
            from data_fetcher import fetch_and_store
            fetch_and_store()
        st.success("Data updated!")
        st.cache_data.clear()

    if st.button("🤖 Run ML Predictions", use_container_width=True):
        with st.spinner("Running ML model …"):
            from ml_model import run_all_predictions
            run_all_predictions(save=True)
        st.success("Predictions saved!")
        st.cache_data.clear()

    if st.button("📰 Refresh News Sentiment", use_container_width=True):
        with st.spinner("Fetching headlines …"):
            from sentiment import run_all_sentiment
            run_all_sentiment(save_to_db=True)
        st.success("Sentiment updated!")
        st.cache_data.clear()

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Wallet Section in Sidebar ────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:13px; font-weight:600; color:#22c55e; margin-bottom:8px;'>💰 Wallet</div>",
        unsafe_allow_html=True,
    )

    # Read current cash balance and starting amount
    try:
        from auto_trader import get_cash, get_starting_capital
        _wallet_cash = get_cash()
        _starting_cash = get_starting_capital()
    except Exception:
        _wallet_cash = DEFAULT_WALLET_AMOUNT
        _starting_cash = DEFAULT_WALLET_AMOUNT

    st.markdown(f"""
    <div style='background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.2);
         border-radius:12px; padding:14px; text-align:center; margin-bottom:10px;'>
        <div style='font-size:11px; color:#64748b; letter-spacing:0.1em;'>VIRTUAL BALANCE</div>
        <div style='font-size:26px; font-weight:800; color:#22c55e;'>
            ₹&nbsp;{_wallet_cash:,.2f}
        </div>
        <div style='font-size:11px; color:#475569; margin-top:4px;'>Paper Trading Account · Start: ₹{_starting_cash:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    _new_default = st.number_input(
        "Set Default Wallet Starting Amount",
        min_value=1000.0, max_value=5_000_000.0,
        value=_starting_cash, step=1000.0, format="%.2f",
        key="wallet_default_input",
    )

    if st.button("Set default starting cash", use_container_width=True, key="wallet_set_default_btn"):
        try:
            from database import get_connection
            _conn = get_connection()
            _conn.execute("UPDATE portfolio_state SET cash=?, starting=? WHERE id=1", (_new_default, _new_default))
            _conn.commit(); _conn.close()
            st.success(f"Default wallet amount set to ₹{_new_default:,.2f}")
            st.cache_data.clear()
            st.rerun()
        except Exception as _e:
            st.error(f"Could not update default amount: {_e}")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Zerodha Kite Broker Integration ────────────────────────────────────────
    st.markdown("<div class='section-header'>🧾 Zerodha Kite Broker</div>", unsafe_allow_html=True)
    zerodha_api_key = st.text_input("ZERODHA API Key", value=os.getenv("ZERODHA_API_KEY", ""), type="password")
    zerodha_api_secret = st.text_input("ZERODHA API Secret", value=os.getenv("ZERODHA_API_SECRET", ""), type="password")
    zerodha_access_token = st.text_input("ZERODHA Access Token", value=os.getenv("ZERODHA_ACCESS_TOKEN", ""), type="password")

    zerodha_message = None
    try:
        from zerodha import (
            is_configured as zerodha_is_configured,
            build_login_url as zerodha_login_url,
            save_zerodha_credentials as zerodha_save_credentials,
            get_positions as zerodha_get_positions,
            get_holdings as zerodha_get_holdings,
        )
        zerodha_enabled = True
        zerodha_connected = zerodha_is_configured()
    except Exception as _e:
        zerodha_enabled = False
        zerodha_connected = False
        zerodha_message = str(_e)

    if st.button("Save Zerodha credentials", use_container_width=True, key="wallet_save_zerodha"):
        if not zerodha_enabled:
            st.error("Zerodha support is not available. Install kiteconnect in your environment.")
        else:
            try:
                zerodha_save_credentials(
                    api_key=zerodha_api_key,
                    api_secret=zerodha_api_secret,
                    access_token=zerodha_access_token,
                )
                st.success("Zerodha credentials saved to .env")
                st.experimental_rerun()
            except Exception as _e:
                st.error(f"Could not save Zerodha credentials: {_e}")

    if not zerodha_enabled:
        st.warning("Zerodha Kite Connect is not available. Install `kiteconnect` and restart the app.")
    elif zerodha_connected:
        st.success("Zerodha is configured and ready for live orders.")
        if st.button("Refresh Zerodha positions", use_container_width=True, key="wallet_refresh_zerodha"):
            try:
                positions = zerodha_get_positions()
                if positions:
                    st.write(positions[:3])
                else:
                    st.info("No open Zerodha positions found.")
            except Exception as _e:
                st.error(f"Could not fetch Zerodha positions: {_e}")
    else:
        st.info("Enter your Zerodha API Key, Secret, and Access Token to enable live ordering.")
        if zerodha_api_key and not zerodha_access_token:
            try:
                login_url = zerodha_login_url()
                st.markdown(f"[Open Zerodha login URL to create a request token]({login_url})")
            except Exception:
                pass

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Deposit / Withdraw
    with st.expander("➕ Deposit / ➖ Withdraw", expanded=False):
        _txn_type = st.radio("Type", ["Deposit", "Withdraw"], horizontal=True, key="wallet_txn_type")
        _txn_amt  = st.number_input(
            "Amount", min_value=100.0, max_value=10_000_000.0,
            value=10_000.0, step=1000.0, key="wallet_txn_amt",
            format="%.2f",
        )
        if st.button("✅ Confirm", use_container_width=True, key="wallet_confirm_btn"):
            try:
                from database import get_connection
                _conn = get_connection()
                if _txn_type == "Deposit":
                    _conn.execute("UPDATE portfolio_state SET cash = cash + ? WHERE id=1", (_txn_amt,))
                    st.success(f"Deposited ₹{_txn_amt:,.2f}")
                else:
                    _cur = get_cash()
                    if _txn_amt > _cur:
                        st.error(f"Insufficient balance (₹{_cur:,.2f})")
                    else:
                        _conn.execute("UPDATE portfolio_state SET cash = cash - ? WHERE id=1", (_txn_amt,))
                        st.success(f"Withdrawn ₹{_txn_amt:,.2f}")
                _conn.commit(); _conn.close()
                st.cache_data.clear()
                st.rerun()
            except Exception as _e:
                st.error(f"Transaction failed: {_e}")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px; color:#475569; text-align:center;'>"
        f"Last loaded: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}</div>",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
# Wallet balance for header bar
_header_cash = DEFAULT_WALLET_AMOUNT
_header_starting_cash = DEFAULT_WALLET_AMOUNT
try:
    from auto_trader import get_cash as _get_cash_hdr, get_starting_capital as _get_starting_cash_hdr
    _header_cash = _get_cash_hdr()
    _header_starting_cash = _get_starting_cash_hdr()
except Exception:
    pass

_wallet_color = "#22c55e" if _header_cash >= 50000 else ("#f59e0b" if _header_cash >= 10000 else "#ef4444")

hdr_left, hdr_mid, hdr_right = st.columns([1, 3, 1])

with hdr_mid:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 6px;'>
        <h1 style='font-size:36px; font-weight:800; color:#f1f5f9; letter-spacing:-0.02em; margin:0;'>
            📊 TradeIQ Market Portal
        </h1>
        <p style='color:#94a3b8; margin:4px 0 0; font-size:14px;'>
            Real-time prices · ML predictions · Sentiment analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

with hdr_right:
    st.markdown("<div style='margin-top: 10px; margin-bottom: 2px; font-size: 12px; font-weight: bold; color: #818cf8; letter-spacing: 0.1em;'>⚙️ TRADING MODE</div>", unsafe_allow_html=True)
    mode = st.radio("Trading Mode", ["🤖 Agent", "👤 Manual"], horizontal=True, label_visibility="collapsed")
    st.session_state["trading_mode"] = mode

    st.markdown(f"""
    <div class="wallet-card" style="margin-top:6px;">
        <div class="wallet-label">💰 Wallet</div>
        <div class="wallet-amount" style="font-size:22px; color:{_wallet_color};">
            &#8377;&nbsp;{_header_cash:,.2f}
        </div>        <div class="wallet-sub">Start: ₹{_starting_cash:,.2f} · Paper Trading</div>        <div class="wallet-pill">Paper Trading</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)


# ── Live Quotes Bar ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _live_quotes():
    return M["get_live_quotes"]()


st.markdown('<div class="section-header">⚡ Live Market Snapshot</div>', unsafe_allow_html=True)

try:
    quotes_df = _live_quotes()
    cols = st.columns(min(len(quotes_df), 5))
    for i, (_, row) in enumerate(quotes_df.head(5).iterrows()):
        price  = row["price"]
        chg    = row["change_pct"]
        ticker = row["ticker"]

        price_str  = f"{price:,.2f}"  if price  is not None else "N/A"
        chg_str    = f"{chg:+.2f}%"  if chg    is not None else "—"
        chg_class  = "up" if (chg or 0) > 0 else ("down" if (chg or 0) < 0 else "flat")
        chg_icon   = "▲" if (chg or 0) > 0 else ("▼" if (chg or 0) < 0 else "—")

        with cols[i % 5]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{ticker}</div>
                <div class="metric-value">{price_str}</div>
                <div class="metric-change {chg_class}">{chg_icon} {chg_str}</div>
            </div>
            """, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Could not load live quotes: {e}. Run data_fetcher.py first.")

st.markdown("---")

# ── Main Content ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 1], gap="large")

# ── LEFT: Chart ───────────────────────────────────────────────────────────────
with left_col:
    st.markdown(f'<div class="section-header">📉 {selected_ticker} — Price Chart</div>',
                unsafe_allow_html=True)

    @st.cache_data(ttl=300)
    def _price_df(ticker, days):
        return M["get_price_df"](ticker, days=days)

    price_df = _price_df(selected_ticker, chart_days)

    if price_df.empty:
        st.info("No data yet. Click '⚡ Fetch Latest Data Now' in the sidebar.")
    else:
        # ── Candlestick + Volume ──────────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x     = price_df["date"],
            open  = price_df["open"],
            high  = price_df["high"],
            low   = price_df["low"],
            close = price_df["close"],
            name  = selected_ticker,
            increasing_line_color = "#22c55e",
            decreasing_line_color = "#ef4444",
        ), row=1, col=1)

        # 7-day MA
        price_df["ma7"] = price_df["close"].rolling(7).mean()
        fig.add_trace(go.Scatter(
            x=price_df["date"], y=price_df["ma7"],
            name="MA 7", line=dict(color="#818cf8", width=1.5, dash="dot"),
        ), row=1, col=1)

        # 14-day MA
        price_df["ma14"] = price_df["close"].rolling(14).mean()
        fig.add_trace(go.Scatter(
            x=price_df["date"], y=price_df["ma14"],
            name="MA 14", line=dict(color="#f59e0b", width=1.5, dash="dot"),
        ), row=1, col=1)

        # Prophet forecast overlay
        prophet_df = None
        try:
            prophet_df = M["get_prophet_forecast"](selected_ticker, periods=7)
        except Exception:
            pass

        if prophet_df is not None and not prophet_df.empty:
            fig.add_trace(go.Scatter(
                x    = prophet_df["ds"],
                y    = prophet_df["yhat"],
                name = "7-day Forecast",
                line = dict(color="#a78bfa", width=2, dash="dash"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x         = pd.concat([prophet_df["ds"], prophet_df["ds"].iloc[::-1]]),
                y         = pd.concat([prophet_df["yhat_upper"], prophet_df["yhat_lower"].iloc[::-1]]),
                fill      = "toself",
                fillcolor = "rgba(167,139,250,0.12)",
                line      = dict(color="rgba(255,255,255,0)"),
                name      = "Forecast Range",
            ), row=1, col=1)

        # Volume bars
        vol_colors = [
            "#22c55e" if c >= o else "#ef4444"
            for c, o in zip(price_df["close"], price_df["open"])
        ]
        fig.add_trace(go.Bar(
            x=price_df["date"], y=price_df["volume"],
            name="Volume", marker_color=vol_colors, opacity=0.6,
        ), row=2, col=1)

        fig.update_layout(
            template        = "plotly_dark",
            paper_bgcolor   = "rgba(0,0,0,0)",
            plot_bgcolor    = "rgba(10,14,26,0.5)",
            height          = 520,
            margin          = dict(l=0, r=0, t=10, b=0),
            legend          = dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            xaxis_rangeslider_visible = False,
            font            = dict(family="Inter"),
        )
        fig.update_yaxes(
            gridcolor="rgba(99,102,241,0.08)",
            zeroline=False,
        )
        fig.update_xaxes(gridcolor="rgba(99,102,241,0.08)")

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── RIGHT: Signal + Sentiment ─────────────────────────────────────────────────
with right_col:

    # ── ML Signal ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🤖 ML Signal</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=600)
    def _predict(ticker):
        return M["predict_ticker"](ticker)

    pred = _predict(selected_ticker)

    if pred:
        signal = pred["signal"]
        sig_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[signal]
        sig_icon  = {"BUY": "🚀", "SELL": "🔻", "HOLD": "⏸"}[signal]

        st.markdown(
            f"<div style='text-align:center; padding:16px 0;'>"
            f"<span class='{sig_class}'>{sig_icon}  {signal}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        action_label = {"BUY": "BUY NOW", "SELL": "SELL NOW", "HOLD": "HOLD"}[signal]
        action_text  = {
            "BUY":  "The model currently favors buying this ticker. Consider increasing exposure if you are paper trading.",
            "SELL": "The model currently favors selling or trimming exposure. Review open positions for this ticker.",
            "HOLD": "No strong action now. Wait for a clearer buy or sell signal before trading."
        }[signal]
        action_color = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#fbbf24"}[signal]

        st.markdown(f"""
        <div style='background:rgba(15,23,42,0.75); border:1px solid {action_color}; border-radius:16px; padding:16px; margin-top:12px;'>
            <div style='font-size:13px; color:#94a3b8; margin-bottom:8px;'>QUICK TRADE ACTION</div>
            <div style='font-size:28px; font-weight:800; color:{action_color};'>{action_label}</div>
            <div style='font-size:13px; color:#cbd5e1; margin-top:8px;'>{action_text}</div>
            <div style='font-size:12px; color:#94a3b8; margin-top:10px;'>Predicted target: ₹{pred["predicted_price"]:,.2f} · Last close: ₹{pred["last_price"]:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        conf_pct = int(pred["confidence"] * 100)
        conf_color = "#22c55e" if conf_pct >= 70 else ("#f59e0b" if conf_pct >= 40 else "#ef4444")

        st.markdown(f"""
        <div style='background:rgba(15,23,42,0.6); border-radius:12px; padding:16px; margin-top:8px;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                <span style='color:#94a3b8; font-size:12px;'>CONFIDENCE</span>
                <span style='color:{conf_color}; font-weight:600;'>{conf_pct}%</span>
            </div>
            <div style='background:#1e293b; border-radius:99px; height:8px;'>
                <div style='background:{conf_color}; width:{conf_pct}%; height:8px; border-radius:99px; transition:width 0.5s;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:rgba(15,23,42,0.6); border-radius:12px; padding:16px; margin-top:12px; font-size:13px;'>
            <div style='display:flex; justify-content:space-between; padding:4px 0;'>
                <span style='color:#94a3b8;'>Last Close</span>
                <span style='color:#f1f5f9; font-weight:600;'>{pred["last_price"]:,.4f}</span>
            </div>
            <hr style='border-color:rgba(99,102,241,0.15); margin:6px 0;'/>
            <div style='display:flex; justify-content:space-between; padding:4px 0;'>
                <span style='color:#94a3b8;'>Predicted</span>
                <span style='color:#818cf8; font-weight:600;'>{pred["predicted_price"]:,.4f}</span>
            </div>
            <hr style='border-color:rgba(99,102,241,0.15); margin:6px 0;'/>
            <div style='display:flex; justify-content:space-between; padding:4px 0;'>
                <span style='color:#94a3b8;'>Δ Expected</span>
                <span class='{"up" if pred["change_pct"] > 0 else "down"}'>{pred["change_pct"]:+.2f}%</span>
            </div>
            <hr style='border-color:rgba(99,102,241,0.15); margin:6px 0;'/>
            <div style='display:flex; justify-content:space-between; padding:4px 0;'>
                <span style='color:#94a3b8;'>Model Error</span>
                <span style='color:#94a3b8;'>{pred["mape"]:.2f}% MAPE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No prediction yet. Fetch data then run ML Predictions.")

    # ── Sentiment ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📰 Sentiment</div>', unsafe_allow_html=True)

    avg_score = M["average_sentiment"](selected_ticker)
    label     = M["sentiment_label"](avg_score)
    bar_color = "#22c55e" if avg_score > 0.05 else ("#ef4444" if avg_score < -0.05 else "#94a3b8")
    # Normalise -1…+1 to 0…100 for the bar
    bar_pct   = int((avg_score + 1) / 2 * 100)

    st.markdown(f"""
    <div style='background:rgba(15,23,42,0.6); border-radius:12px; padding:16px; margin-top:8px;'>
        <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
            <span style='color:#94a3b8; font-size:12px;'>MARKET MOOD</span>
            <span style='color:{bar_color}; font-weight:600;'>{label}</span>
        </div>
        <div style='background:#1e293b; border-radius:99px; height:8px;'>
            <div style='background:{bar_color}; width:{bar_pct}%; height:8px; border-radius:99px;'></div>
        </div>
        <div style='text-align:center; color:#94a3b8; font-size:12px; margin-top:8px;'>
            Score: {avg_score:+.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Recent headlines
    sent_df = M["get_sentiment_df"](selected_ticker, limit=8)
    if not sent_df.empty:
        st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
        for _, row in sent_df.iterrows():
            s = row["score"]
            css_cls  = "news-bull" if s > 0.05 else ("news-bear" if s < -0.05 else "news-neut")
            icon     = "🟢" if s > 0.05 else ("🔴" if s < -0.05 else "🟡")
            headline = str(row["headline"])[:90] + ("…" if len(str(row["headline"])) > 90 else "")
            st.markdown(f"""
            <div class="news-item {css_cls}">
                <div style='font-size:11px;'>{icon} <span style='color:#94a3b8;'>{row["source"]}</span></div>
                <div style='font-size:12px; color:#e2e8f0; margin-top:2px;'>{headline}</div>
                <div style='font-size:11px; color:#64748b; margin-top:2px;'>Score: {s:+.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No news loaded yet. Click 'Refresh News Sentiment'.")


# ── Portfolio Tracker ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">💼 Portfolio Tracker</div>', unsafe_allow_html=True)
st.caption("Enter your holdings below. P&L is calculated from the latest fetched close price.")

all_tickers = [t for group in TICKERS.values() for t in group]

default_portfolio = pd.DataFrame({
    "Ticker": ["RELIANCE.NS", "AAPL", "BTC-USD"],
    "Qty":    [10, 5, 0.05],
    "Avg Buy Price": [2800.0, 175.0, 42000.0],
})

portfolio = st.data_editor(
    default_portfolio,
    num_rows   = "dynamic",
    use_container_width = True,
    column_config = {
        "Ticker": st.column_config.SelectboxColumn(
            "Ticker", options=all_tickers, required=True,
        ),
        "Qty": st.column_config.NumberColumn("Qty", min_value=0.0, format="%.4f"),
        "Avg Buy Price": st.column_config.NumberColumn("Avg Buy Price", format="%.2f"),
    },
    key="portfolio_editor",
)

if not portfolio.empty:
    live_q = _live_quotes().set_index("ticker")

    rows = []
    for _, row in portfolio.iterrows():
        tkr   = row["Ticker"]
        qty   = float(row["Qty"] or 0)
        avg   = float(row["Avg Buy Price"] or 0)
        price = float(live_q.loc[tkr, "price"]) if tkr in live_q.index and live_q.loc[tkr, "price"] else avg

        current_val  = price * qty
        cost_val     = avg * qty
        pnl          = current_val - cost_val
        pnl_pct      = ((price - avg) / avg * 100) if avg else 0

        rows.append({
            "Ticker":        tkr,
            "Qty":           qty,
            "Avg Buy":       avg,
            "Current Price": round(price, 4),
            "Value (₹/$)":   round(current_val, 2),
            "P&L":           round(pnl, 2),
            "P&L %":         round(pnl_pct, 2),
        })

    pnl_df = pd.DataFrame(rows)

    total_value = pnl_df["Value (₹/$)"].sum()
    total_pnl   = pnl_df["P&L"].sum()

    def _style_pnl(val):
        color = "#22c55e" if val > 0 else ("#ef4444" if val < 0 else "#94a3b8")
        return f"color: {color}; font-weight: 600;"

    styled = pnl_df.style
    if hasattr(styled, "applymap"):
        styled = styled.applymap(_style_pnl, subset=["P&L", "P&L %"])

    styled = styled.format({"P&L": "{:+.2f}", "P&L %": "{:+.2f}%", "Current Price": "{:,.4f}"})

    st.dataframe(styled, width="stretch", hide_index=True)

    p1, p2, p3 = st.columns(3)
    pnl_color = "#22c55e" if total_pnl >= 0 else "#ef4444"
    p1.metric("💰 Total Value",  f"{total_value:,.2f}")
    p2.metric("📊 Total P&L",   f"{total_pnl:+,.2f}")
    p3.metric("📈 Tickers",     len(pnl_df))


# ── Agent's Top Picks Analysis ──────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-header'>🤖 Agent's Top Picks Analysis</div>",
            unsafe_allow_html=True)

try:
    from database import get_connection
    conn = get_connection()
    # Get the latest predictions where signal is BUY, ordered by confidence
    top_picks_df = pd.read_sql_query("""
        SELECT ticker, predicted_price, confidence
        FROM ml_predictions
        WHERE signal = 'BUY'
        ORDER BY created_at DESC, confidence DESC
        LIMIT 3
    """, conn)
    conn.close()

    if not top_picks_df.empty:
        cols = st.columns(len(top_picks_df))
        for i, row in top_picks_df.iterrows():
            tkr = row["ticker"]
            price_pred = row["predicted_price"]
            conf = min(100, int(row["confidence"] * 100))
            
            # Simple English summary based on confidence
            if conf > 60:
                summary = f"Strong BUY recommended. The AI predicts strong upward momentum and historically favorable risk/reward."
            elif conf > 30:
                summary = f"BUY recommended. The Agent sees steady trends confirming an upcoming price increase."
            else:
                summary = f"Speculative BUY. The price is currently low and our agent detects a potential early breakout."

            with cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #22c55e;">
                    <div style="font-size:20px; font-weight:800; color:#f1f5f9; margin-bottom: 8px;">{tkr}</div>
                    <div style="font-size:14px; color:#22c55e; font-weight: 600; margin-bottom: 12px;">✅ AI Confident ({conf}%)</div>
                    <div style="font-size:13px; color:#94a3b8; line-height: 1.5; text-align: left;">{summary}</div>
                    <div style="margin-top:14px; padding-top:10px; border-top: 1px solid rgba(255,255,255,0.05); font-size:12px; color:#64748b;">
                        Expected Target: <span style="color:#f1f5f9; font-weight: 600;">₹{price_pred:,.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No active Agent recommendations right now. Please run ML Predictions.")
except Exception as e:
    st.info(f"Could not load Top Picks: {e}")

# ── Active LLM Trades & Exit Plans ──────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-header'>🔥 Active Agent Trades & Exit Plans</div>", unsafe_allow_html=True)

try:
    from auto_trader import get_all_positions
    active_positions = [p for p in get_all_positions() if p["quantity"] > 0]
    
    if active_positions:
        cols = st.columns(min(3, len(active_positions)))
        for i, pos in enumerate(active_positions):
            tkr = pos["ticker"]
            qty = pos["quantity"]
            entry = pos["entry_price"]
            exit_plan = pos.get("exit_plan", "No specific exit plan recorded.") or "No specific exit plan recorded."
            
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #818cf8;">
                    <div style="font-size:20px; font-weight:800; color:#f1f5f9; margin-bottom: 8px;">{tkr}</div>
                    <div style="font-size:13px; color:#818cf8; font-weight: 600; margin-bottom: 12px;">📈 Qty: {qty:.4f} @ ₹{entry:,.2f}</div>
                    <div style="font-size:12px; color:#cbd5e1; line-height: 1.5; text-align: left; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 6px;">
                        <strong>Exit Plan:</strong><br/>{exit_plan}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No active Agent trades at the moment.")
except Exception as e:
    st.info(f"Could not load Active Agent Trades: {e}")

# ── IEEE v2: Auto-Trader Panel ────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🤖 Auto-Trader (Paper Trading — IEEE Q-Learning Model)</div>',
            unsafe_allow_html=True)

# ── Wallet Summary Banner ────────────────────────────────────────────────────
try:
    from auto_trader import get_cash as _get_cash_at
    _at_cash = _get_cash_at()
except Exception:
    _at_cash = 100000.0

_at_color  = "#22c55e" if _at_cash >= 50000 else ("#f59e0b" if _at_cash >= 10000 else "#ef4444")
_at_profit = _at_cash - 100000.0
_at_pct    = (_at_profit / 100000.0) * 100

wb1, wb2, wb3, wb4 = st.columns(4)
wb1.markdown(f"""
<div class="metric-card" style="border-color:rgba(34,197,94,0.3);">
    <div class="metric-label">💰 Wallet Balance</div>
    <div class="metric-value" style="color:{_at_color}; font-size:22px;">&#8377; {_at_cash:,.2f}</div>
    <div class="metric-change" style="color:{_at_color};">Virtual Cash</div>
</div>
""", unsafe_allow_html=True)

wb2.markdown(f"""
<div class="metric-card" style="border-color:rgba({'34,197,94' if _at_profit >= 0 else '239,68,68'},0.3);">
    <div class="metric-label">📈 Total P&L</div>
    <div class="metric-value" style="color:{'#22c55e' if _at_profit >= 0 else '#ef4444'}; font-size:22px;">{_at_profit:+,.2f}</div>
    <div class="metric-change {'up' if _at_profit >= 0 else 'down'}">{_at_pct:+.2f}% vs. start</div>
</div>
""", unsafe_allow_html=True)

wb3.markdown(f"""
<div class="metric-card">
    <div class="metric-label">🏦 Initial Capital</div>
    <div class="metric-value" style="font-size:22px;">&#8377; {_starting_cash:,.2f}</div>
    <div class="metric-change flat">Starting balance</div>
</div>
""", unsafe_allow_html=True)

_at_status = "Active" if _at_cash > 0 else "Depleted"
_at_status_color = "#22c55e" if _at_cash > 50000 else ("#f59e0b" if _at_cash > 0 else "#ef4444")
wb4.markdown(f"""
<div class="metric-card">
    <div class="metric-label">⚡ Account Status</div>
    <div class="metric-value" style="color:{_at_status_color}; font-size:22px;">{_at_status}</div>
    <div class="metric-change" style="color:{_at_status_color};">Paper Mode</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3);
     border-radius:10px; padding:12px; margin: 12px 0; font-size:13px; color:#fbbf24;'>
    ⚠️ <b>Paper Trading Mode</b> — No real money is used. All trades are simulated.
    Inspired by: "Cooperative Multi-Agent RL for Bitcoin Trading" (IEEE IPRIA 2025)
</div>
""", unsafe_allow_html=True)

# Mode Check
is_agent = "Agent" in st.session_state.get("trading_mode", "🤖 Agent")

if not is_agent:
    st.info("👤 Manual Trading Mode Active: Here you can integrate your manual broker buy/sell endpoints.")
else:
    # Risk controls
    rc1, rc2, rc3 = st.columns(3)
    stop_loss   = rc1.slider("Stop-Loss %",   1, 10, 3, key="sl")
    take_profit = rc2.slider("Take-Profit %", 2, 20, 6, key="tp")
    max_pos     = rc3.slider("Max Position %", 5, 30, 10, key="mp")
    
    at1, at2 = st.columns([1, 2])
    
    with at1:
        if st.button("▶ Run One Trading Cycle", use_container_width=True, type="primary"):
            with st.spinner("Running BUY/SELL agents on all tickers..."):
                try:
                    import auto_trader as _at
                    _at.STOP_LOSS_PCT    = stop_loss / 100
                    _at.TAKE_PROFIT_PCT  = take_profit / 100
                    _at.MAX_POSITION_PCT = max_pos / 100
                    _at._ensure_trades_table()
                    _at._ensure_portfolio_table()
                    summary = _at.run_trading_cycle(verbose=False)
                    st.success(
                        f"Cycle done! Buys: {summary['buys']} | "
                        f"Sells: {summary['sells']} | "
                        f"Cash: {summary['cash']:,.2f}"
                    )
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Error: {e}")
    
        if st.button("🔄 Reset Portfolio", use_container_width=True):
            try:
                import auto_trader as _at
                from database import get_connection
                conn = get_connection()
                conn.execute("UPDATE portfolio_state SET cash=100000.0, starting=100000.0 WHERE id=1")
                conn.execute("DELETE FROM portfolio")
                conn.commit(); conn.close()
                st.success("Portfolio reset to 100,000")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Reset failed: {e}")

    if not is_agent:
        # Zerodha manual order entry
        st.markdown('<div class="section-header">🧾 Zerodha Manual Order</div>', unsafe_allow_html=True)
        try:
            from zerodha import is_configured as zerodha_is_configured, place_zerodha_order
            zerodha_manual_ready = zerodha_is_configured()
        except Exception as _e:
            zerodha_manual_ready = False
            zerodha_error = str(_e)

        order_symbol = st.text_input("Order Ticker", value=selected_ticker)
        order_side = st.radio("Order Side", ["BUY", "SELL"], horizontal=True)
        order_qty = st.number_input("Quantity", min_value=1.0, value=1.0, step=1.0)
        order_product = st.selectbox("Product", ["MIS", "CNC"], index=0)
        order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"], index=0)
        order_price = st.number_input("Limit Price (if LIMIT)", min_value=0.0, value=0.0, step=0.05)

        if zerodha_manual_ready:
            if st.button("Place Zerodha Order", use_container_width=True, key="zerodha_place_order"):
                try:
                    order_kwargs = {
                        "symbol": order_symbol,
                        "transaction_type": order_side,
                        "quantity": int(order_qty),
                        "order_type": order_type,
                        "product": order_product,
                    }
                    if order_type == "LIMIT":
                        order_kwargs["price"] = float(order_price)
                    response = place_zerodha_order(**order_kwargs)
                    st.success("Zerodha order placed successfully.")
                    st.json(response)
                except Exception as _e:
                    st.error(f"Order failed: {_e}")

            st.markdown("<div style='margin-top:16px; color:#818cf8; font-weight:600; font-size:14px;'>Zerodha Broker Positions</div>", unsafe_allow_html=True)
            try:
                holdings = zerodha_get_holdings()
                positions = zerodha_get_positions()

                if holdings:
                    holdings_df = pd.DataFrame(holdings)
                    holdings_df = holdings_df.rename(columns={
                        "tradingsymbol": "Symbol",
                        "quantity": "Qty",
                        "m2m": "M2M PnL",
                        "avg_price": "Avg Price",
                        "last_price": "LTP",
                        "pnl": "P&L"
                    })
                    st.markdown("<div style='margin-top:12px; font-size:13px; color:#94a3b8;'>Current Zerodha holdings from your account.</div>", unsafe_allow_html=True)
                    st.dataframe(holdings_df[[c for c in ["Symbol", "Qty", "Avg Price", "LTP", "M2M PnL", "P&L"] if c in holdings_df.columns]], use_container_width=True)
                else:
                    st.info("No Zerodha holdings found.")

                if positions:
                    positions_df = pd.DataFrame(positions)
                    positions_df = positions_df.rename(columns={
                        "tradingsymbol": "Symbol",
                        "quantity": "Qty",
                        "last_price": "LTP",
                        "pnl": "P&L",
                        "m2m": "M2M PnL"
                    })
                    st.markdown("<div style='margin-top:12px; font-size:13px; color:#94a3b8;'>Current Zerodha position details.</div>", unsafe_allow_html=True)
                    st.dataframe(positions_df[[c for c in ["Symbol", "Qty", "LTP", "P&L", "M2M PnL"] if c in positions_df.columns]], use_container_width=True)
                else:
                    st.info("No open Zerodha positions found.")
            except Exception as _e:
                st.error(f"Could not fetch Zerodha position data: {_e}")
        else:
            st.warning(
                "Zerodha is not yet configured. Enter API credentials and access token in the sidebar, then save them."
            )
            if not zerodha_enabled:
                st.caption(f"Broker setup error: {zerodha_message}")

        st.markdown("<hr/>", unsafe_allow_html=True)
    
    with at2:
        # Current cash + open positions
        try:
            from auto_trader import get_cash, get_open_positions
            cash = get_cash()
            pos_df = get_open_positions()
            _cash_color = "#22c55e" if cash >= 50000 else ("#f59e0b" if cash >= 10000 else "#ef4444")
    
            st.markdown(f"""
            <div style='background:rgba(15,23,42,0.7); border-radius:10px; padding:14px; margin-bottom: 12px;'>
                <div style='color:#94a3b8; font-size:12px; letter-spacing:0.1em;'>💰 WALLET BALANCE</div>
                <div style='font-size:32px; font-weight:800; color:{_cash_color}; letter-spacing:-0.02em;'>
                    &#8377;&nbsp;{cash:,.2f}
                </div>
                <div style='font-size:11px; color:#475569; margin-top:4px;'>Virtual paper-trading account</div>
            </div>
            """, unsafe_allow_html=True)
    
            if not pos_df.empty:
                st.markdown("<div style='color:#818cf8; font-size:14px; font-weight: 600; margin-bottom: 8px;'>Live Held Positions Performance</div>", unsafe_allow_html=True)
                quotes_now = _live_quotes().set_index("ticker")
                
                # Render mini row graphs for each position
                for i, row in pos_df.iterrows():
                    tkr = row["ticker"]
                    cp = float(quotes_now.loc[tkr, "price"]) if tkr in quotes_now.index else row["entry_price"]
                    unr = (cp - row["entry_price"]) * row["quantity"]
                    pnl_pct = (cp - row["entry_price"]) / row["entry_price"] * 100
                    graph_color = "#22c55e" if unr >= 0 else "#ef4444"
                    
                    bg_col = "rgba(34,197,94,0.1)" if unr >= 0 else "rgba(239,68,68,0.1)"
                    st.markdown(f"""
                    <div style='border:1px solid {graph_color}; border-radius:8px; padding:12px; margin-bottom: 8px; background: {bg_col}; display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='font-size:16px; font-weight:bold;'>{tkr}</span> <span style='font-size:12px; color:#94a3b8;'>({row["quantity"]:.4f} shares)</span><br>
                            <span style='color:#94a3b8; font-size:12px;'>Entry: ₹{row["entry_price"]:,.2f} → Now: ₹{cp:,.2f}</span>
                        </div>
                        <div style='text-align: right;'>
                            <div style='font-size:18px; font-weight:bold; color:{graph_color};'>{unr:+,.2f} ₹</div>
                            <div style='font-size:12px; color:{graph_color};'>{pnl_pct:+.2f}% Output</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mini sparkline using plotly
                    try:
                        import plotly.graph_objects as go
                        _hist = M["get_price_df"](tkr, days=15)
                        if not _hist.empty:
                            fig_spark = go.Figure()
                            # actual performance line
                            fig_spark.add_trace(go.Scatter(x=_hist["date"], y=_hist["close"], line=dict(color=graph_color, width=2)))
                            # entry price line
                            fig_spark.add_hline(y=row["entry_price"], line_dash="dash", line_color="#94a3b8", annotation_text="Entry", annotation_position="bottom right")
                            fig_spark.update_layout(height=80, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
                            fig_spark.update_xaxes(visible=False)
                            fig_spark.update_yaxes(visible=False)
                            st.plotly_chart(fig_spark, use_container_width=True, config={"displayModeBar": False})
                    except Exception as e:
                        pass
            else:
                st.caption("No open positions")
        except Exception as msg:
            st.info(f"Run a trading cycle to see positions here. ({msg})")
    
    # Trade history
    st.markdown('<div style="margin-top:16px; color:#818cf8; font-weight:600; font-size:14px;">Trade History Log</div>',
                unsafe_allow_html=True)
    try:
        from auto_trader import get_trade_history
        th = get_trade_history(limit=20)
        if not th.empty:
            th["timestamp"] = pd.to_datetime(th["timestamp"]).dt.strftime("%d %b %H:%M")
            th["pnl"] = th["pnl"].fillna(0)
    
            def _color_action(val):
                colors = {"BUY": "#22c55e", "SELL": "#f59e0b",
                          "STOP_LOSS": "#ef4444", "TAKE_PROFIT": "#22c55e"}
                return f"color: {colors.get(val, '#94a3b8')}; font-weight: 600;"
    
            def _color_pnl(val):
                return f"color: {'#22c55e' if val >= 0 else '#ef4444'}; font-weight: 600;"
    
            styled_th = th.style \
                .applymap(_color_action, subset=["action"]) \
                .applymap(_color_pnl, subset=["pnl"]) \
                .format({"pnl": "{:+.2f}", "entry_price": "{:,.2f}",
                         "exit_price": lambda x: f"{x:,.2f}" if x else "—",
                         "confidence": "{:.2f}"})
            st.dataframe(styled_th, use_container_width=True, hide_index=True)
        else:
            st.info("No trades logged yet. Click 'Run One Trading Cycle' above.")
    except Exception as e:
        st.info(f"Auto-trader not initialised yet. ({e})")


# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(300)
    st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#475569; font-size:11px; padding:8px 0;'>
    TradeIQ Market Portal · Built with Streamlit, yfinance, Scikit-Learn & VADER ·
    <span style='color:#4f46e5;'>Not financial advice</span>
</div>
""", unsafe_allow_html=True)
