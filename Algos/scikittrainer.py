import os
import pickle
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def get_historical_data(coin_id="bitcoin", vs_currency="usd", days=180, interval="daily"):
    """
    Fetches historical market data (price and volume) from CoinGecko.
    
    Parameters:
        coin_id (str): The CoinGecko coin id (e.g., "bitcoin").
        vs_currency (str): The currency (e.g., "usd").
        days (int): Number of days to retrieve.
        interval (str): Data interval ("daily" is recommended for multi-day data).
        
    Returns:
        pd.DataFrame: A DataFrame with datetime index, price, and volume.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": interval
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        
        # Create DataFrames from the returned lists
        df_prices = pd.DataFrame(prices, columns=["timestamp", "price"])
        df_volumes = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        
        # Merge price and volume data on the timestamp
        df = pd.merge(df_prices, df_volumes, on="timestamp")
        
        # Convert timestamp (ms) to datetime and set as index
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("date", inplace=True)
        df.drop("timestamp", axis=1, inplace=True)
        return df
    else:
        print("Error fetching data:", response.status_code)
        return None

def prepare_buy_model_dataset(df):
    """
    Prepares a dataset for training the buy potential model.
    
    For each day t (starting from the second row) where the daily return is negative (a drop day),
    we compute:
      - Feature 1: Absolute price drop (%) from day t-1 to day t.
      - Feature 2: Volume ratio = volume[t] / volume[t-1]
      - Feature 3: Architecture potential (set to constant 0.5 as a normalized value).
    
    The target is the next day's percentage return (from day t to day t+1).
    
    Returns:
        tuple: (X, y) where X is a 2D array of features and y is a 1D array of target returns.
    """
    # Compute daily return percentage
    df['return'] = df['price'].pct_change() * 100
    # Compute volume ratio: today's volume divided by previous day's volume
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    # Set architecture potential as constant (normalized value)
    df['arch_potential'] = 0.5
    
    # Drop rows with NaN values (first row will be NaN due to pct_change)
    df = df.dropna()

    X, y = [], []
    # Iterate over rows (except the last row, which has no "next day" target)
    for i in range(len(df) - 1):
        # Only use days where there was a drop
        if df.iloc[i]['return'] < 0:
            features = [
                abs(df.iloc[i]['return']),        # Absolute drop in %
                df.iloc[i]['volume_ratio'],         # Volume ratio
                df.iloc[i]['arch_potential']        # Architecture potential (constant)
            ]
            target = df.iloc[i+1]['return']  # Next day's return percentage
            X.append(features)
            y.append(target)
    return np.array(X), np.array(y)

def train_buy_model():
    """
    Trains a linear regression model using historical Bitcoin data to predict the next-day return
    (as a proxy for buy potential) on days following a price drop.
    Saves the trained model to 'buy_model.pkl'.
    """
    df = get_historical_data(coin_id="bitcoin", days=180, interval="daily")
    if df is None or df.empty:
        print("No historical data available for training.")
        return None
    
    X, y = prepare_buy_model_dataset(df)
    if len(X) == 0:
        print("No drop days found in the dataset to train the model.")
        return None
    
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print("Training Mean Squared Error:", mse)
    
    # Save the model to a pickle file
    with open("buy_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved as buy_model.pkl")
    return model

def main():
    model = train_buy_model()
    if model is not None:
        print("Buy model trained and saved successfully.")

if __name__ == "__main__":
    main()
