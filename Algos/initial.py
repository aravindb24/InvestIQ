import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =============================================================================
# 1. Data Ingestion (Scraping) - Placeholder Functions
# =============================================================================
def scrape_historical_price_data(stock):
    """
    Placeholder function for scraping historical price data for a given stock.
    Expected output: A DataFrame with columns: Date (datetime), Open, High, Low, Close (floats), Volume (int)
    """
    dates = pd.date_range(end=datetime.today(), periods=250)
    data = {
        'Date': dates,
        'Open': np.random.uniform(100, 200, size=len(dates)),
        'High': np.random.uniform(100, 200, size=len(dates)),
        'Low': np.random.uniform(100, 200, size=len(dates)),
        'Close': np.random.uniform(100, 200, size=len(dates)),
        'Volume': np.random.randint(1000000, 5000000, size=len(dates))
    }
    df = pd.DataFrame(data)
    df.sort_values("Date", inplace=True)
    return df

def scrape_historical_news_data(stock):
    """
    Placeholder function for scraping historical news data for a given stock.
    Expected output: A DataFrame with columns: Date (datetime), Sentiment (float; daily aggregated sentiment)
    """
    dates = pd.date_range(end=datetime.today(), periods=250)
    data = {
        'Date': dates,
        'Sentiment': np.random.uniform(-1, 1, size=len(dates))
    }
    df = pd.DataFrame(data)
    return df

# =============================================================================
# 2. Feature Engineering and Data Preparation
# =============================================================================
def prepare_data(stock):
    """
    - Ingests historical price and news data.
    - Merges the two DataFrames on Date.
    - Computes daily returns and technical indicators (SMA_50, SMA_200, RSI, Volatility).
    
    Expected columns in the resulting DataFrame include:
      Date, Open, High, Low, Close, Volume, Sentiment, Return, SMA_50, SMA_200, RSI, Volatility.
    """
    price_df = scrape_historical_price_data(stock)
    news_df = scrape_historical_news_data(stock)
    
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    
    # Merge on Date (inner join)
    df = pd.merge(price_df, news_df, on='Date', how='inner')
    
    # Calculate daily return
    df['Return'] = df['Close'].pct_change()
    
    # Compute technical indicators
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
    df['Volatility'] = df['Return'].rolling(window=14).std()
    
    df.dropna(inplace=True)
    return df

# =============================================================================
# 3. Labeling the Data for Supervised Learning
# =============================================================================
def label_data(df, future_window=5, threshold=0.03):
    """
    Adds a 'Future_Return' column calculated over a given future window, then creates a 'Signal' column:
      - 1 (Buy) if Future_Return > threshold
      - -1 (Sell) if Future_Return < -threshold
      - 0 (Hold) otherwise.
    
    Parameters:
      future_window (int): Number of days into the future to calculate the return.
      threshold (float): Return threshold for generating a buy/sell signal.
    """
    df = df.copy()
    df['Future_Return'] = df['Close'].pct_change(periods=future_window).shift(-future_window)
    
    def signal(x):
        if x > threshold:
            return 1
        elif x < -threshold:
            return -1
        else:
            return 0
    
    df['Signal'] = df['Future_Return'].apply(signal)
    df.dropna(inplace=True)
    return df

# =============================================================================
# 4. Model Training
# =============================================================================
def train_model(stock, future_window, threshold):
    """
    Trains a model (RandomForestClassifier) using historical features and generated signals.
    
    Parameters:
      stock (str): Stock ticker.
      future_window (int): Future window (in days) for labeling.
      threshold (float): Return threshold for labeling.
      
    Features include:
      Open, High, Low, Close, Volume, Sentiment, Return, SMA_50, SMA_200, RSI, Volatility.
      
    Returns:
      model: Trained classifier.
      features: List of feature names.
    """
    df = prepare_data(stock)
    df = label_data(df, future_window=future_window, threshold=threshold)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 'Return', 'SMA_50', 'SMA_200', 'RSI', 'Volatility']
    X = df[features]
    y = df['Signal']
    
    # For time series data, use chronological split (e.g., 80% training, 20% testing)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy (Window: {future_window} days, Threshold: {threshold}): {accuracy:.2f}")
    
    return model, features

# =============================================================================
# 5. Live Signal Generation for Different Horizons
# =============================================================================
def generate_live_signal(stock, model, features, rsi_overbought=70, rsi_oversold=30):
    """
    Generates a live trading signal using the latest available data.
    
    Process:
      - Prepare the latest data (with same features as training).
      - Use the model to predict a signal (1, -1, or 0).
      - Check RSI for additional technical confirmation:
          If RSI > rsi_overbought, force a sell (-1).
          If RSI < rsi_oversold, force a buy (1).
      - Combine the model prediction and technical signal.
    
    Returns:
      final_signal (str): "Buy", "Sell", or "Hold".
      prediction (int): Model's raw prediction.
      technical_signal (int): RSI-based adjustment.
      rsi (float): Latest RSI value.
    """
    df = prepare_data(stock)
    latest = df.iloc[-1]
    X_live = latest[features].values.reshape(1, -1)
    prediction = model.predict(X_live)[0]
    
    rsi = latest['RSI']
    if rsi > rsi_overbought:
        technical_signal = -1  # Overbought: sell signal
    elif rsi < rsi_oversold:
        technical_signal = 1   # Oversold: buy signal
    else:
        technical_signal = 0
    
    # Combine model and technical signals (simple summation; adjust weighting as needed)
    combined_signal = prediction + technical_signal
    if combined_signal > 0:
        final_signal = "Buy"
    elif combined_signal < 0:
        final_signal = "Sell"
    else:
        final_signal = "Hold"
    
    return final_signal, prediction, technical_signal, rsi

# =============================================================================
# 6. Running the Models and Generating Live Signals
# =============================================================================
if __name__ == "__main__":
    stock = "AAPL"  # Example stock ticker
    
    # Train separate models for medium-term and long-term trades:
    # For medium-term trades (e.g., 10-day future return, threshold 5%)
    model_medium, features_medium = train_model(stock, future_window=10, threshold=0.05)
    
    # For long-term trades (e.g., 60-day future return, threshold 10%)
    model_long, features_long = train_model(stock, future_window=60, threshold=0.10)
    
    # Generate live signals for each horizon
    med_signal, med_pred, med_tech, med_rsi = generate_live_signal(stock, model_medium, features_medium)
    long_signal, long_pred, long_tech, long_rsi = generate_live_signal(stock, model_long, features_long)
    
    print("\nLive Trading Signals:")
    print(f"Medium-Term Signal: {med_signal} (Model Prediction: {med_pred}, Technical Signal: {med_tech}, RSI: {med_rsi:.2f})")
    print(f"Long-Term Signal: {long_signal} (Model Prediction: {long_pred}, Technical Signal: {long_tech}, RSI: {long_rsi:.2f})")
