"""
pwa_config.py — Injects Progressive Web App (PWA) support into the Streamlit app

Add `from pwa_config import inject_pwa` and call `inject_pwa()` at the top of app.py
to make the dashboard installable as a home-screen app on iOS and Android.
"""
import streamlit as st


PWA_MANIFEST = """
{
  "name": "TradeIQ - Market Portal",
  "short_name": "TradeIQ",
  "description": "Real-time stock and crypto trading analysis with ML signals",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0a0e1a",
  "theme_color": "#6366f1",
  "orientation": "portrait-primary",
  "icons": [
    {
      "src": "https://img.icons8.com/fluency/192/increase.png",
      "sizes": "192x192",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "https://img.icons8.com/fluency/512/increase.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ],
  "categories": ["finance", "utilities"],
  "shortcuts": [
    {
      "name": "Live Prices",
      "short_name": "Prices",
      "url": "/?view=prices",
      "description": "View live market prices"
    }
  ]
}
"""

MOBILE_CSS = """
<style>
/* ── Mobile-first responsive tweaks ───────────────────────────── */

/* Ensure viewport is set for mobile */
@media (max-width: 768px) {
    /* Stack columns to full width on mobile */
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* Shrink metric cards slightly */
    .metric-card {
        padding: 12px !important;
        border-radius: 12px !important;
    }
    .metric-value  { font-size: 20px !important; }
    .metric-label  { font-size: 10px !important; }
    .metric-change { font-size: 12px !important; }

    /* Full-width charts on mobile */
    [data-testid="stPlotlyChart"] {
        width: 100% !important;
    }

    /* Larger tap targets for sidebar items */
    .stSelectbox > div { min-height: 44px; }
    .stSlider         { touch-action: pan-x; }

    /* Compact sidebar */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
    }

    /* Section headers smaller */
    .section-header { font-size: 15px !important; }

    /* News items more readable */
    .news-item { padding: 6px 10px !important; }
}

/* ── Touch-friendly button improvements ─────────────────────── */
.stButton > button {
    min-height: 44px;
    font-size: 15px;
    touch-action: manipulation;
}

/* ── Prevent horizontal scroll on mobile ────────────────────── */
body { overflow-x: hidden; }

/* ── Smooth scrolling on mobile ─────────────────────────────── */
html { scroll-behavior: smooth; -webkit-overflow-scrolling: touch; }
</style>
"""

PWA_INJECTION = """
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="TradeIQ">
<meta name="theme-color" content="#6366f1">
<link rel="apple-touch-icon" href="https://img.icons8.com/fluency/192/increase.png">
</head>
"""


def inject_pwa():
    """Call this once at the top of app.py (after st.set_page_config)."""
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    st.markdown(PWA_INJECTION, unsafe_allow_html=True)
