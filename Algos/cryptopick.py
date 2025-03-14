import requests
import time
import pandas as pd

# Constants for scoring
LIQUIDITY_THRESHOLD = 100000  # Minimum liquidity in USD
VOLUME_THRESHOLD = 50000  # Minimum 24h trading volume
AGE_THRESHOLD = 7  # Maximum days since launch to consider

def fetch_new_tokens():
    """Fetches new tokens from a crypto API (e.g., CoinGecko or DexTools)."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_asc",
        "per_page": 100,
        "page": 1,
        "sparkline": False
    }
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else []

def get_token_data(token_id):
    """Fetches individual token details for deeper analysis."""
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else {}

def analyze_token(token):
    """Scores tokens based on various factors."""
    score = 0
    volume = token.get("total_volume", 0)
    market_cap = token.get("market_cap", 0)
    launch_date = token.get("atl_date", None)  # Approximate launch date
    age_days = (time.time() - pd.to_datetime(launch_date).timestamp()) / 86400 if launch_date else 999

    # Apply scoring
    if market_cap > LIQUIDITY_THRESHOLD:
        score += 2
    if volume > VOLUME_THRESHOLD:
        score += 2
    if age_days <= AGE_THRESHOLD:
        score += 1

    return {
        "name": token.get("name"),
        "symbol": token.get("symbol"),
        "score": score,
        "market_cap": market_cap,
        "volume": volume,
        "age_days": age_days,
    }

def recommend_tokens():
    """Fetches, analyzes, and recommends new crypto tokens."""
    tokens = fetch_new_tokens()
    analyzed_tokens = [analyze_token(token) for token in tokens]
    
    # Filter and sort based on score
    top_picks = sorted(
        [t for t in analyzed_tokens if t["score"] > 3], 
        key=lambda x: x["score"], 
        reverse=True
    )
    
    return top_picks

# Run the recommendation
if __name__ == "__main__":
    recommended = recommend_tokens()
    for token in recommended[:10]:  # Show top 10 picks
        print(token)
