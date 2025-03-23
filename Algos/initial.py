import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =============================================================================
# 1. Multi-Aspect & Stock-Specific News Analysis
# =============================================================================
# Initialize FinBERT pipeline (ensure you have installed transformers: pip install transformers)
finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")

def analyze_article_with_stock_specific_aspects(article_text, stock_keywords, finbert_pipeline=finbert):
    """
    Analyzes a news article to extract overall sentiment, general market aspect sentiment,
    and stock-specific sentiment.
    
    Parameters:
      article_text (str): Full text of the news article.
      stock_keywords (list): List of keywords relevant to the stock (e.g., company name, ticker).
      finbert_pipeline: A Hugging Face sentiment-analysis pipeline (default: FinBERT).
    
    Process:
      - Splits the article into sentences using nltk's sentence tokenizer.
      - For each sentence, obtains a sentiment score from FinBERT:
           positive -> +confidence,
           negative -> -confidence,
           neutral  -> 0.
      - Checks each sentence against pre-defined aspects (Fed, Earnings, Inflation, Geopolitics, Commodity, Economy, Tech)
        using keyword matching, aggregating sentiment scores and counts.
      - Separately, checks for stock-specific keywords and aggregates their sentiment.
    
    Returns:
      A dictionary with:
        "Overall_Sentiment": Average sentiment of all sentences.
        For each general aspect:
            "{Aspect}_Sentiment" and "{Aspect}_Mention_Count".
        "Stock_Overall_Sentiment" and "Stock_Mention_Count" for stock-specific sentiment.
    """
    # Define general aspects and keywords (expand as needed)
    aspects = {
        "Fed": ["Fed", "Federal Reserve", "interest rate", "rate hike", "rate cut"],
        "Earnings": ["earnings", "revenue", "profit", "loss", "guidance"],
        "Inflation": ["inflation", "CPI", "prices", "cost"],
        "Geopolitics": ["war", "geopolitical", "trade war", "sanctions", "diplomacy"],
        "Commodity": ["oil", "gold", "commodity", "crude"],
        "Economy": ["GDP", "unemployment", "jobs", "economic growth", "economy"],
        "Tech": ["technology", "innovation", "tech", "software", "hardware"]
    }
    # Precompile regex patterns for aspects (case-insensitive)
    aspect_regex = {aspect: re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)
                    for aspect, keywords in aspects.items()}
    # Compile regex for stock-specific keywords
    stock_pattern = r'\b(' + '|'.join(stock_keywords) + r')\b'
    stock_regex = re.compile(stock_pattern, re.IGNORECASE)
    
    sentences = sent_tokenize(article_text)
    overall_scores = []
    aspect_scores = {aspect: [] for aspect in aspects}
    aspect_counts = {aspect: 0 for aspect in aspects}
    stock_specific_scores = []
    stock_specific_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Get sentiment from FinBERT
        result = finbert_pipeline(sentence)[0]
        label = result['label'].lower()
        confidence = result['score']
        if label == 'positive':
            score = confidence
        elif label == 'negative':
            score = -confidence
        else:
            score = 0
        
        overall_scores.append(score)
        
        # Check for general aspects
        for aspect, regex in aspect_regex.items():
            if regex.search(sentence):
                aspect_scores[aspect].append(score)
                aspect_counts[aspect] += 1
        
        # Check for stock-specific keywords
        if stock_regex.search(sentence):
            stock_specific_scores.append(score)
            stock_specific_count += 1

    overall_sentiment = np.mean(overall_scores) if overall_scores else 0
    results = {"Overall_Sentiment": overall_sentiment}
    
    for aspect in aspects:
        aspect_avg = np.mean(aspect_scores[aspect]) if aspect_scores[aspect] else 0
        results[f"{aspect}_Sentiment"] = aspect_avg
        results[f"{aspect}_Mention_Count"] = aspect_counts[aspect]
    
    stock_overall_sentiment = np.mean(stock_specific_scores) if stock_specific_scores else 0
    results["Stock_Overall_Sentiment"] = stock_overall_sentiment
    results["Stock_Mention_Count"] = stock_specific_count
    
    return results

# =============================================================================
# 2. Data Ingestion (Scraping) - Placeholder Functions
# =============================================================================
def scrape_historical_price_data(stock):
    """
    Placeholder for scraping historical price data.
    Expected output: DataFrame with columns:
      Date (datetime), Open, High, Low, Close (floats), Volume (int)
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
    Placeholder for scraping historical news data.
    Expected output: DataFrame with columns:
      Date (datetime) and aggregated news features:
        Overall_Sentiment, Fed_Sentiment, Fed_Mention_Count,
        Earnings_Sentiment, Earnings_Mention_Count, Inflation_Sentiment, Inflation_Mention_Count,
        Geopolitics_Sentiment, Geopolitics_Mention_Count, Commodity_Sentiment, Commodity_Mention_Count,
        Economy_Sentiment, Economy_Mention_Count, Tech_Sentiment, Tech_Mention_Count,
        Stock_Overall_Sentiment, Stock_Mention_Count.
    Here we simulate one “article” per day using a sample text.
    """
    dates = pd.date_range(end=datetime.today(), periods=250)
    news_data = []
    # Define stock-specific keywords; for example, for Apple use ["Apple", "AAPL", "iPhone"]
    stock_keywords = [stock, stock.upper()]
    sample_article = (
        f"{stock} reports mixed earnings today. The Federal Reserve's decisions continue to impact markets. "
        "Inflation and geopolitical tensions are affecting commodity prices. "
        "Technology and innovation remain key drivers, while overall economic growth stays resilient. "
        f"{stock} in particular is showing promising signs with its new products."
    )
    for date in dates:
        analysis = analyze_article_with_stock_specific_aspects(sample_article, stock_keywords)
        row = {
            'Date': date,
            'Overall_Sentiment': analysis["Overall_Sentiment"],
            'Fed_Sentiment': analysis["Fed_Sentiment"],
            'Fed_Mention_Count': analysis["Fed_Mention_Count"],
            'Earnings_Sentiment': analysis["Earnings_Sentiment"],
            'Earnings_Mention_Count': analysis["Earnings_Mention_Count"],
            'Inflation_Sentiment': analysis["Inflation_Sentiment"],
            'Inflation_Mention_Count': analysis["Inflation_Mention_Count"],
            'Geopolitics_Sentiment': analysis["Geopolitics_Sentiment"],
            'Geopolitics_Mention_Count': analysis["Geopolitics_Mention_Count"],
            'Commodity_Sentiment': analysis["Commodity_Sentiment"],
            'Commodity_Mention_Count': analysis["Commodity_Mention_Count"],
            'Economy_Sentiment': analysis["Economy_Sentiment"],
            'Economy_Mention_Count': analysis["Economy_Mention_Count"],
            'Tech_Sentiment': analysis["Tech_Sentiment"],
            'Tech_Mention_Count': analysis["Tech_Mention_Count"],
            'Stock_Overall_Sentiment': analysis["Stock_Overall_Sentiment"],
            'Stock_Mention_Count': analysis["Stock_Mention_Count"]
        }
        news_data.append(row)
    news_df = pd.DataFrame(news_data)
    return news_df

# =============================================================================
# 3. Feature Engineering and Data Preparation
# =============================================================================
def prepare_data(stock):
    """
    - Ingests historical price and news data.
    - Merges them on Date.
    - Computes daily returns and technical indicators (SMA_50, SMA_200, RSI, Volatility).
    
    Expected columns include:
      Date, Open, High, Low, Close, Volume,
      Return, SMA_50, SMA_200, RSI, Volatility,
      and the aggregated news features.
    """
    price_df = scrape_historical_price_data(stock)
    news_df = scrape_historical_news_data(stock)
    
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    
    # Merge on Date (inner join)
    df = pd.merge(price_df, news_df, on='Date', how='inner')
    df['Return'] = df['Close'].pct_change()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
    df['Volatility'] = df['Return'].rolling(window=14).std()
    
    df.dropna(inplace=True)
    return df

# =============================================================================
# 4. Labeling the Data for Supervised Learning
# =============================================================================
def label_data(df, future_window=5, threshold=0.03):
    """
    Adds a 'Future_Return' column (pct change over future_window) and a 'Signal' column:
      1 (Buy) if Future_Return > threshold,
     -1 (Sell) if Future_Return < -threshold,
      0 (Hold) otherwise.
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
# 5. Neural Network Model Training
# =============================================================================
def train_nn_model(stock, future_window, threshold):
    """
    Trains a neural network to predict trading signals using combined price, technical, and news features.
    
    Expected features include:
      Price/Technical: Open, High, Low, Close, Volume, Return, SMA_50, SMA_200, RSI, Volatility
      News: Overall_Sentiment, Fed_Sentiment, Earnings_Sentiment, Inflation_Sentiment,
            Geopolitics_Sentiment, Commodity_Sentiment, Economy_Sentiment, Tech_Sentiment,
            Stock_Overall_Sentiment
    Maps signals -1, 0, 1 to classes 0, 1, 2 respectively.
    """
    df = prepare_data(stock)
    df = label_data(df, future_window=future_window, threshold=threshold)
    
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 
        'SMA_50', 'SMA_200', 'RSI', 'Volatility',
        'Overall_Sentiment', 'Fed_Sentiment', 'Earnings_Sentiment',
        'Inflation_Sentiment', 'Geopolitics_Sentiment', 'Commodity_Sentiment',
        'Economy_Sentiment', 'Tech_Sentiment', 'Stock_Overall_Sentiment'
    ]
    
    X = df[features].values
    y = df['Signal'].values
    # Map: -1 -> 0, 0 -> 1, 1 -> 2
    y_class = y + 1
    
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y_class[:split_index], y_class[split_index:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Three classes: Sell, Hold, Buy
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[es], verbose=0)
    
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Neural Network Accuracy (Window: {future_window} days, Threshold: {threshold}): {accuracy:.2f}")
    
    return model, scaler, features

# =============================================================================
# 6. Live Signal Generation using Neural Network
# =============================================================================
def generate_live_signal_nn(stock, model, scaler, features, rsi_overbought=70, rsi_oversold=30):
    """
    Generates a live trading signal for the given stock.
    
    Process:
      - Prepares the latest data with the same features.
      - Scales the features and uses the neural network to predict a class.
      - Converts the class (0,1,2) back to signal (-1,0,1).
      - Applies an RSI-based adjustment (force sell if RSI > rsi_overbought, buy if RSI < rsi_oversold).
      - Combines the NN prediction and technical signal to return a final decision.
    """
    df = prepare_data(stock)
    latest = df.iloc[-1]
    X_live = latest[features].values.reshape(1, -1)
    X_live_scaled = scaler.transform(X_live)
    
    probs = model.predict(X_live_scaled)
    pred_class = np.argmax(probs, axis=1)[0]
    nn_prediction = pred_class - 1  # Convert back: 0->-1, 1->0, 2->1
    
    rsi = latest['RSI']
    if rsi > rsi_overbought:
        technical_signal = -1  # Overbought: sell
    elif rsi < rsi_oversold:
        technical_signal = 1   # Oversold: buy
    else:
        technical_signal = 0
    
    combined_signal = nn_prediction + technical_signal
    if combined_signal > 0:
        final_signal = "Buy"
    elif combined_signal < 0:
        final_signal = "Sell"
    else:
        final_signal = "Hold"
    
    print(f"Live Signal for {stock}: {final_signal}")
    print(f"NN Prediction: {nn_prediction}, Technical Signal: {technical_signal}, RSI: {rsi:.2f}")
    
    return final_signal, nn_prediction, technical_signal, rsi

# =============================================================================
# 7. Running the Models and Generating Live Signals
# =============================================================================
if __name__ == "__main__":
    stock = "AAPL"  # Example ticker
    
    # Train separate neural network models for medium-term and long-term trades:
    # For medium-term trades (e.g., 10-day future return, threshold 5%)
    model_medium, scaler_medium, features_medium = train_nn_model(stock, future_window=10, threshold=0.05)
    
    # For long-term trades (e.g., 60-day future return, threshold 10%)
    model_long, scaler_long, features_long = train_nn_model(stock, future_window=60, threshold=0.10)
    
    # Generate live signals for each horizon
    med_signal, med_nn, med_tech, med_rsi = generate_live_signal_nn(stock, model_medium, scaler_medium, features_medium)
    long_signal, long_nn, long_tech, long_rsi = generate_live_signal_nn(stock, model_long, scaler_long, features_long)
    
    print("\nLive Trading Signals:")
    print(f"Medium-Term Signal: {med_signal} (NN: {med_nn}, Tech: {med_tech}, RSI: {med_rsi:.2f})")
    print(f"Long-Term Signal: {long_signal} (NN: {long_nn}, Tech: {long_tech}, RSI: {long_rsi:.2f})")
