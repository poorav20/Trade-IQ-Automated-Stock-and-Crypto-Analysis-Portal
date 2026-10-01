"""verify_v2.py — Verification script for IEEE v2 upgrades"""
from database import init_db
init_db()
print("DB: OK")

from indicators import add_all_indicators, indicator_signal
from database import get_price_df

df = get_price_df("RELIANCE.NS", days=90)
if df.empty:
    print("WARN: No RELIANCE data - run data_fetcher.py first")
else:
    df2 = add_all_indicators(df)
    sig = indicator_signal(df2)
    print(f"Indicators: OK  signal={sig['signal']}  rsi={sig['rsi']}  score={sig['score']}")
    print(f"  Reasons: {sig['reasons']}")

# Test ML v2
print("\nTesting ml_model_v2 ...")
from ml_model_v2 import predict_ticker_v2
pred = predict_ticker_v2("RELIANCE.NS", sentiment_score=0.1)
if pred:
    print(f"ML v2: OK  signal={pred['signal']}  conf={pred['confidence']}  pred={pred['predicted_price']}")
    print(f"  Models: {pred['model_scores']}")
else:
    print("ML v2: SKIP (not enough data)")

# Test auto-trader init
print("\nTesting auto_trader ...")
from auto_trader import _ensure_trades_table, _ensure_portfolio_table, get_cash, get_open_positions
_ensure_trades_table()
_ensure_portfolio_table()
cash = get_cash()
pos  = get_open_positions()
print(f"AutoTrader: OK  cash={cash:,.2f}  open_positions={len(pos)}")

print("\n=== ALL MODULES VERIFIED ===")
