import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta


GRAMS_PER_TROY_OUNCE = 31.1034768
GST_RATE = 0.03

#FETCH DATA 
gold = yf.download("GC=F", start="2020-01-01")
usd_inr = yf.download("USDINR=X", start="2020-01-01")

# Fix multi-index columns
if isinstance(gold.columns, pd.MultiIndex):
    gold.columns = gold.columns.get_level_values(0)
if isinstance(usd_inr.columns, pd.MultiIndex):
    usd_inr.columns = usd_inr.columns.get_level_values(0)

gold.reset_index(inplace=True)
usd_inr.reset_index(inplace=True)

gold.rename(columns={'Close': 'Gold_Close_USD'}, inplace=True)
usd_inr.rename(columns={'Close': 'USD_INR'}, inplace=True)

# Merge
data = pd.merge(
    gold[['Date', 'Gold_Close_USD']],
    usd_inr[['Date', 'USD_INR']],
    on='Date',
    how='inner'
)

# FEATURE ENGINEERING 
data['Gold_INR_per_gram'] = (
    data['Gold_Close_USD'] * data['USD_INR'] / GRAMS_PER_TROY_OUNCE
)

data['Gold_INR_GST'] = data['Gold_INR_per_gram'] * (1 + GST_RATE)

data['5_day_avg'] = data['Gold_INR_GST'].rolling(5).mean()
data['20_day_avg'] = data['Gold_INR_GST'].rolling(20).mean()
data.dropna(inplace=True)

#  MODEL 
features = ['Gold_INR_GST', '5_day_avg', '20_day_avg']
X = data[features]
y = data['Gold_INR_GST'].shift(-1)
X, y = X[:-1], y.dropna()

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

#  TODAY PRICE 
today_price = data.iloc[-1]['Gold_INR_GST']
print(f"\nToday's Gold Price (₹/gram incl GST): ₹{today_price:.2f}")

#  NEXT 7 DAYS PREDICTION 
last_row = data.iloc[-1][features]
future_prices = []

for _ in range(7):
    next_price = model.predict([last_row])[0]
    future_prices.append(next_price)

    last_row['5_day_avg'] = (last_row['5_day_avg'] * 4 + next_price) / 5
    last_row['20_day_avg'] = (last_row['20_day_avg'] * 19 + next_price) / 20
    last_row['Gold_INR_GST'] = next_price

#  OUTPUT 
print("\nNext 7 Days Gold Price Prediction (₹/gram incl GST):")
for i, price in enumerate(future_prices, 1):
    date = (datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d')
    print(f"Day {i} ({date}): ₹{price:.2f}")
