import sqlite3
import pandas as pd

pd.set_option("display.width", 200)
conn = sqlite3.connect("trading.db")

print("=== PRICE HISTORY BY TICKER ===")
df = pd.read_sql_query(
    "SELECT ticker, COUNT(*) AS rows, MIN(date) AS earliest, MAX(date) AS latest "
    "FROM price_history GROUP BY ticker ORDER BY rows DESC",
    conn
)
for _, r in df.iterrows():
    print(f"  {r['ticker']:20s}  rows={r['rows']:8d}  from={r['earliest']}  to={r['latest']}")

print()
print("=== LATEST ML PREDICTIONS ===")
pred = pd.read_sql_query(
    "SELECT ticker, predicted_date, ROUND(predicted_price,2) AS price, "
    "signal, ROUND(confidence,3) AS conf FROM ml_predictions ORDER BY created_at DESC LIMIT 20",
    conn
)
for _, r in pred.iterrows():
    print(f"  {r['ticker']:20s}  pred={r['price']:12.2f}  signal={str(r['signal']):4s}  conf={r['conf']}  for={r['predicted_date']}")

conn.close()
