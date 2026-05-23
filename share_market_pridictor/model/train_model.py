import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Download Stock Data (Reliance Industries on NSE)
print("Downloading stock data for RELIANCE.NS...")
df = yf.download('RELIANCE.NS', start='2015-01-01', end='2025-01-01')

# Safeguard against yfinance MultiIndex columns (introduced in recent updates)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Create Features
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

# Create Target: Predict if tomorrow's close price is higher than today's
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop rows with NaN values (from moving averages and target shift)
df.dropna(inplace=True)

# Features and Target
x = df[['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'MA200']]
y = df['Target']

# Split Data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train Model
print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Save Model
joblib.dump(model, 'model/stock_model.pkl')
print("Model Trained and Saved Successfully under model/stock_model.pkl!")
