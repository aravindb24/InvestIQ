import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib
import re
import asyncio
from pyppeteer import launch
import requests
from bs4 import BeautifulSoup
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
# Initialize FinBERT pipeline (ensure you have installed transformers via pip)
finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")

def analyze_article_with_stock_specific_aspects(article_text, stock_keywords, finbert_pipeline=finbert):
    """
    Analyzes a news article to extract overall sentiment, general market aspect sentiment,
    and stock-specific sentiment.
    """
    aspects = {
        "Fed": ["Fed", "Federal Reserve", "interest rate", "rate hike", "rate cut"],
        "Earnings": ["earnings", "revenue", "profit", "loss", "guidance"],
        "Inflation": ["inflation", "CPI", "prices", "cost"],
        "Geopolitics": ["war", "geopolitical", "trade war", "sanctions", "diplomacy"],
        "Commodity": ["oil", "gold", "commodity", "crude"],
        "Economy": ["GDP", "unemployment", "jobs", "economic growth", "economy"],
        "Tech": ["technology", "innovation", "tech", "software", "hardware"]
    }
    aspect_regex = {aspect: re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)
                    for aspect, keywords in aspects.items()}
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
        for aspect, regex in aspect_regex.items():
            if regex.search(sentence):
                aspect_scores[aspect].append(score)
                aspect_counts[aspect] += 1
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
# 2. Data Ingestion (Webscraping) using Pyppeteer for Historical Prices
# =============================================================================
async def scrape_historical_price_data_pyppeteer(stock):
    """
    Uses Pyppeteer to scrape Yahoo Finance's historical prices for the given stock.
    It simulates clicking the date picker, setting the start date to 20 years ago (keeping day/month),
    and then extracts the HTML of the loaded table.
    """
    url = f"https://finance.yahoo.com/quote/{stock}/history?p={stock}"
    browser = await launch(headless=True, args=['--no-sandbox'])
    page = await browser.newPage()
    await page.goto(url, {"waitUntil": "networkidle2"})
    
    # Wait for and click the date picker button
    date_picker_selector = "#nimbus-app > section > section > section > article > div.container > div.container.yf-e8ilep > div.menuContainer.yf-jef819 > button"
    await page.waitForSelector(date_picker_selector)
    await page.click(date_picker_selector)
    
    # Wait for the date input to appear and modify its value
    start_date_input_selector = "#menu-49 > div > section > div:nth-child(2) > input:nth-child(2)"
    await page.waitForSelector(start_date_input_selector)
    
    # Get the current value from the input (format assumed MM/DD/YYYY)
    current_value = await page.evaluate(f'document.querySelector("{start_date_input_selector}").value')
    try:
        current_date = datetime.strptime(current_value, "%m/%d/%Y")
    except Exception:
        current_date = datetime.today()
    # Subtract 20 years (keeping day and month)
    new_year = current_date.year - 20
    new_date = current_date.replace(year=new_year)
    new_date_str = new_date.strftime("%m/%d/%Y")
    
    # Set the input's value to the new date
    await page.evaluate(f'document.querySelector("{start_date_input_selector}").value = "{new_date_str}"')
    
    # Click the "Apply" button (this selector might need adjustment)
    apply_button_selector = "#menu-49 > div > section > div.controls.yf-1th5n0r > button.primary-btn.fin-size-small.rounded.yf-1bk9lim"
    try:
        await page.click(apply_button_selector)
    except Exception:
        # If no apply button is found, assume the change is auto-applied.
        pass

    # Wait for the table to reload (adjust time as needed)
    await page.waitFor(5000)
    table_selector = "#nimbus-app > section > section > section > article > div.container > div.table-container.yf-1jecxey > table > thead"
    await page.waitForSelector(table_selector)
    table_html = await page.evaluate(f'document.querySelector("{table_selector}").outerHTML')
    await browser.close()
    
    # Parse the table HTML using BeautifulSoup
    soup = BeautifulSoup(table_html, "html.parser")
    rows = soup.find_all("tr")
    data = []
    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) < 6:
            continue  # Skip non-data rows (e.g., dividends)
        try:
            date_str = cols[0].get_text(strip=True)
            date_obj = datetime.strptime(date_str, "%b %d, %Y")
            open_val = float(cols[1].get_text(strip=True).replace(',', ''))
            high_val = float(cols[2].get_text(strip=True).replace(',', ''))
            low_val = float(cols[3].get_text(strip=True).replace(',', ''))
            close_val = float(cols[4].get_text(strip=True).replace(',', ''))
            vol_str = cols[6].get_text(strip=True).replace(',', '')
            volume_val = int(vol_str) if vol_str.isdigit() else 0
            data.append({
                "Date": date_obj,
                "Open": open_val,
                "High": high_val,
                "Low": low_val,
                "Close": close_val,
                "Volume": volume_val
            })
        except Exception:
            continue
        print(data)
    return pd.DataFrame(data)

def scrape_historical_price_data(stock):
    """
    Synchronous wrapper for the asynchronous Pyppeteer scraper.
    """
    return asyncio.get_event_loop().run_until_complete(scrape_historical_price_data_pyppeteer(stock))

def scrape_historical_news_data(stock):
    """
    Scrapes Yahoo Finance's news page for the given stock.
    Extracts headlines and summaries, groups articles by date, and
    aggregates multi-aspect and stock-specific sentiment using FinBERT.
    """
    url = f"https://finance.yahoo.com/quote/{stock}/news?p={stock}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    articles = soup.find_all("li", class_="js-stream-content")
    news_data = []
    for article in articles:
        try:
            headline = article.find("h3").get_text(strip=True)
            summary_tag = article.find("p")
            summary = summary_tag.get_text(strip=True) if summary_tag else ""
            pub_date = datetime.today().date()  # For demonstration, assign today's date
            full_text = headline + ". " + summary
            news_data.append({"Date": pub_date, "Article": full_text})
        except Exception:
            continue
    if not news_data:
        return pd.DataFrame()
    
    news_df = pd.DataFrame(news_data)
    aggregated = news_df.groupby("Date")["Article"].apply(lambda texts: " ".join(texts)).reset_index()
    
    rows = []
    stock_keywords = [stock, stock.upper()]
    for idx, row in aggregated.iterrows():
        analysis = analyze_article_with_stock_specific_aspects(row["Article"], stock_keywords)
        analysis["Date"] = row["Date"]
        rows.append(analysis)
    aggregated_features = pd.DataFrame(rows)
    return aggregated_features

# =============================================================================
# 3. Feature Engineering and Data Preparation
# =============================================================================
def prepare_data(stock):
    """
    Ingests historical price data (via Pyppeteer) and news data,
    merges them on Date, and computes additional technical indicators.
    """
    price_df = scrape_historical_price_data(stock)
    news_df = scrape_historical_news_data(stock)
    
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    
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
    Adds a Future_Return column (pct change over future_window days) and a Signal column:
      1 if Future_Return > threshold,
     -1 if Future_Return < -threshold,
      0 otherwise.
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
    Trains a neural network to predict trading signals using combined price/technical and news features.
    Signals (-1, 0, 1) are mapped to classes (0, 1, 2).
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
    y_class = y + 1  # Map -1->0, 0->1, 1->2
    
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
    model.add(Dense(3, activation='softmax'))
    
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
        technical_signal = -1
    elif rsi < rsi_oversold:
        technical_signal = 1
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
    prepare_data('AAPL')
    model_medium, scaler_medium, features_medium = train_nn_model(stock, future_window=10, threshold=0.05)
    model_long, scaler_long, features_long = train_nn_model(stock, future_window=60, threshold=0.10)
    
    # Generate live signals for each horizon
    med_signal, med_nn, med_tech, med_rsi = generate_live_signal_nn(stock, model_medium, scaler_medium, features_medium)
    long_signal, long_nn, long_tech, long_rsi = generate_live_signal_nn(stock, model_long, scaler_long, features_long)
    
    print("\nLive Trading Signals:")
    print(f"Medium-Term Signal: {med_signal} (NN: {med_nn}, Tech: {med_tech}, RSI: {med_rsi:.2f})")
    print(f"Long-Term Signal: {long_signal} (NN: {long_nn}, Tech: {long_tech}, RSI: {long_rsi:.2f})")
