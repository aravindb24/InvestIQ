import pandas as pd
import numpy as np
import requests
import yfinance as yf
from textblob import TextBlob
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.news import News
import talib

# API Keys (replace with your own)
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_KEY"

# Stock Universe (Modify as needed)
STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]

# Function to get fundamental data (P/E ratio, Revenue Growth)
def get_fundamentals(stock):
    fd = FundamentalData(ALPHA_VANTAGE_API_KEY, output_format='pandas')
    try:
        income_stmt, _ = fd.get_income_statement_annual(stock)
        balance_sheet, _ = fd.get_balance_sheet_annual(stock)
        
        latest_income = income_stmt.iloc[0]
        prev_income = income_stmt.iloc[1]

        revenue_growth = (latest_income["totalRevenue"] - prev_income["totalRevenue"]) / prev_income["totalRevenue"]

        pe_ratio = latest_income["netIncome"] / balance_sheet.iloc[0]["commonStockSharesOutstanding"]

        return {"PE_Ratio": pe_ratio, "Revenue_Growth": revenue_growth}
    except:
        return {"PE_Ratio": None, "Revenue_Growth": None}

# Function to get technical indicators (Moving Averages, RSI)
def get_technical_indicators(stock):
    data = yf.download(stock, period="6mo")
    
    if data.empty:
        return {"SMA_50": None, "SMA_200": None, "RSI": None}
    
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    
    data["RSI"] = talib.RSI(data["Close"], timeperiod=14)
    
    latest = data.iloc[-1]
    return {
        "SMA_50": latest["SMA_50"],
        "SMA_200": latest["SMA_200"],
        "RSI": latest["RSI"]
    }

# Function to get sentiment analysis from news
def get_news_sentiment(stock):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url).json()
    
    if "feed" not in response:
        return 0  # Neutral sentiment
    
    news_articles = response["feed"]
    sentiment_score = 0
    for article in news_articles:
        sentiment_score += TextBlob(article["summary"]).sentiment.polarity
    
    return sentiment_score / len(news_articles) if news_articles else 0

# Stock Scoring and Ranking Function
def rank_stocks():
    results = []
    
    for stock in STOCKS:
        print(f"Analyzing {stock}...")
        fundamentals = get_fundamentals(stock)
        technicals = get_technical_indicators(stock)
        sentiment = get_news_sentiment(stock)
        
        if fundamentals["PE_Ratio"] is None or technicals["SMA_50"] is None:
            continue  # Skip if data is missing
        
        # Scoring System (Higher is Better)
        score = (
            (1 / fundamentals["PE_Ratio"]) * 10 +  # Lower P/E Ratio is better
            fundamentals["Revenue_Growth"] * 20 +  # Higher revenue growth is better
            (technicals["SMA_50"] > technicals["SMA_200"]) * 15 +  # Bullish crossover
            (50 < technicals["RSI"] < 70) * 10 +  # RSI between 50 and 70 is good
            sentiment * 10  # Positive news sentiment boosts score
        )
        
        results.append({"Stock": stock, "Score": score})
    
    # Sort stocks by score
    ranked_stocks = sorted(results, key=lambda x: x["Score"], reverse=True)
    return pd.DataFrame(ranked_stocks)

# Run the Stock-Picking Algorithm
top_stocks = rank_stocks()
import ace_tools as tools
tools.display_dataframe_to_user(name="Stock Picking Algorithm Results", dataframe=top_stocks)
