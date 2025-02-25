import pandas as pd
import numpy as np
import requests
import yfinance as yf
from textblob import TextBlob
from alpha_vantage.fundamentaldata import FundamentalData
import talib
import ace_tools as tools
from bs4 import BeautifulSoup

# API Key (replace with your own)
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_KEY"

# Function to get emerging sectors based on performance
def get_emerging_sectors():
    url = f"https://www.alphavantage.co/query?function=SECTOR&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url).json()

    sector_performance = {}
    if "Rank A: Real-Time Performance" in response:
        sector_performance = response["Rank A: Real-Time Performance"]

    # Sort sectors by performance (descending)
    emerging_sectors = sorted(sector_performance.items(), key=lambda x: float(x[1].strip('%')), reverse=True)[:3]
    return [sector[0] for sector in emerging_sectors]

# Function to dynamically get top stocks in an emerging sector
def get_stocks_in_sector(sector):
    sector = sector.replace(" ", "-").lower()
    url = f"https://finance.yahoo.com/sector/ms_{sector}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    stocks = []
    for link in soup.find_all("a", {"class": "Fw(600)"}):
        stock_symbol = link.get_text()
        if stock_symbol.isalpha():
            stocks.append(stock_symbol)
    
    return stocks[:10]  # Limit to top 10 stocks per sector

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

# Function to score and rank stocks within emerging sectors
def rank_stocks_by_sector():
    emerging_sectors = get_emerging_sectors()
    sector_results = []
    stock_results = []

    for sector in emerging_sectors:
        print(f"Analyzing stocks in the emerging sector: {sector}")
        stocks = get_stocks_in_sector(sector)
        sector_results.append({"Sector": sector, "Top Stocks": stocks})

        for stock in stocks:
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
            
            stock_results.append({"Sector": sector, "Stock": stock, "Score": score})
    
    # Sort stocks by score
    ranked_stocks = sorted(stock_results, key=lambda x: x["Score"], reverse=True)

    # Display both sector and stock results
    sector_df = pd.DataFrame(sector_results)
    stock_df = pd.DataFrame(ranked_stocks)

    tools.display_dataframe_to_user(name="Emerging Sectors", dataframe=sector_df)
    tools.display_dataframe_to_user(name="Top Stock Picks in Emerging Sectors", dataframe=stock_df)

# Run the updated Stock-Picking Algorithm
rank_stocks_by_sector()
