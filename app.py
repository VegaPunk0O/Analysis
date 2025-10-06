
# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="📊 Stock Dashboard", layout="wide")

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("Stock Analysis Settings")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL, TSLA, INFY):", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
seq_length = st.sidebar.slider("Sequence Length (Days)", 30, 120, 60)
epochs = st.sidebar.slider("Training Epochs", 5, 50, 20)
forecast_days = st.sidebar.slider("Forecast Future Days", 1, 30, 7)

# ===============================
# Download Stock Data
# ===============================
st.title("📊 Stock Analysis Dashboard with LSTM & Indicators")
st.write(f"Fetching data for **{ticker}**...")

data = yf.download(ticker, start=start_date, end=end_date)
if data.empty:
    st.error("No data found for this ticker.")
    st.stop()

# ===============================
# Calculate Technical Indicators
# ===============================
data['SMA50'] = data['Close'].rolling(50).mean()
data['SMA200'] = data['Close'].rolling(200).mean()
data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()

delta = data['Close'].diff()
up = delta.clip(lower=0)
down = -1*delta.clip(upper=0)
roll_up = up.rolling(14).mean()
roll_down = down.rolling(14).mean()
RS = roll_up / roll_down
data['RSI'] = 100 - (100 / (1 + RS))

EMA12 = data['Close'].ewm(span=12, adjust=False).mean()
EMA26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = EMA12 - EMA26
data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

st.subheader("Recent Stock Data with Indicators")
st.dataframe(data.tail(10))

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(data['Close'], label="Close Price")
ax.plot(data['SMA50'], label="SMA50")
ax.plot(data['SMA200'], label="SMA200")
ax.plot(data['EMA20'], label="EMA20")
ax.set_title(f"{ticker} Price with SMA/EMA")
ax.legend()
st.pyplot(fig)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data[['Close']])

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, 0])
        y.append(data[i,0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
with st.spinner("Training LSTM model..."):
    model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1,1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

def forecast_future(model, data, seq_len, days):
    forecast_input = data[-seq_len:].reshape(1, seq_len, 1)
    future_prices = []
    for _ in range(days):
        pred = model.predict(forecast_input)[0,0]
        future_prices.append(pred)
        forecast_input = np.append(forecast_input[:,1:,:], [[[pred]]], axis=1)
    return scaler.inverse_transform(np.array(future_prices).reshape(-1,1))

future_prices = forecast_future(model, scaled_data, seq_length, forecast_days)

fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(y_test_actual, label="Actual Price", color="blue")
ax2.plot(predictions, label="Predicted Price", color="red")
future_index = np.arange(len(y_test_actual), len(y_test_actual)+forecast_days)
ax2.plot(future_index, future_prices, label=f"Forecast Next {forecast_days} Days", color="green", linestyle="--")
ax2.set_title(f"{ticker} LSTM Prediction & Forecast")
ax2.set_xlabel("Days")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

st.subheader(f"Forecasted Prices for Next {forecast_days} Days")
st.dataframe(pd.DataFrame(future_prices, columns=["Forecasted Price"]))

fig3, ax3 = plt.subplots(2,1, figsize=(12,8), sharex=True)
ax3[0].plot(data['RSI'], label="RSI", color="purple")
ax3[0].axhline(70, color="red", linestyle="--")
ax3[0].axhline(30, color="green", linestyle="--")
ax3[0].set_title("RSI Indicator")
ax3[0].legend()
ax3[1].plot(data['MACD'], label="MACD", color="orange")
ax3[1].plot(data['Signal'], label="Signal Line", color="blue")
ax3[1].set_title("MACD Indicator")
ax3[1].legend()
st.pyplot(fig3)

st.success("✅ Dashboard Ready!")
