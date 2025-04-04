import os
import pickle
import requests
import numpy as np
from sklearn.linear_model import LinearRegression

API_KEY = "CG-z9dFsALqjPqj9ndKhKotVnfW"
BASE_URL = "https://api.coingecko.com/api/v3"

def get_crypto_data(vs_currency="usd", per_page=100, page=1):
    """
    Fetches market data for cryptocurrencies from CoinGecko.
    """
    url = f"{BASE_URL}/coins/markets"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": "false",
        "price_change_percentage": "24h"
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching data:", response.status_code)
        return None

def calculate_architecture_potential(coin):
    """
    Computes a simple architecture potential score based on market cap rank.
    For demonstration, coins with rank 1 score near 0, while those around rank 200 score near 100.
    """
    market_cap_rank = coin.get("market_cap_rank")
    if market_cap_rank is None:
        return 0
    potential = max(0, (200 - market_cap_rank) / 200) * 100
    return potential

def report_stats(coins, top_n=10):
    """
    Prints a formatted report of the top performing coins with additional details.
    """
    header = (f"{'Rank':<5}{'Name':<15}{'Symbol':<8}{'Price':<12}"
              f"{'24h Change (%)':<15}{'Volume':<15}{'Market Cap':<15}"
              f"{'Arch. Potential':<18}")
    print(header)
    print("-" * len(header))
    for coin in coins[:top_n]:
        rank = coin.get("market_cap_rank", "N/A")
        name = coin.get("name", "N/A")
        symbol = coin.get("symbol", "N/A").upper()
        price = coin.get("current_price", 0)
        change = coin.get("price_change_percentage_24h", 0)
        volume = coin.get("total_volume", 0)
        market_cap = coin.get("market_cap", 0)
        arch_potential = calculate_architecture_potential(coin)
        print(f"{rank:<5}{name:<15}{symbol:<8}${price:<11.2f}{change:<15.2f}"
              f"{volume:<15}{market_cap:<15}{arch_potential:<18.2f}")

def get_top_performing_coins(coins, threshold=0):
    """
    Filters and sorts coins with a positive 24h change.
    """
    top_coins = [
        coin for coin in coins 
        if coin.get("price_change_percentage_24h") is not None and coin.get("price_change_percentage_24h") > threshold
    ]
    top_coins.sort(key=lambda coin: coin.get("price_change_percentage_24h"), reverse=True)
    return top_coins

def load_buy_model():
    """
    Loads a pre-trained ML model for predicting buy potential.
    If no model file exists, trains a dummy linear regression model on synthetic data and saves it.
    Replace this with your own model if available.
    """
    model_path = "buy_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        # For demonstration, create synthetic training data.
        # Features: [abs_price_drop, volume_ratio, normalized_architecture]
        # Target: weighted sum using weights [0.5, 0.3, 0.2]
        X = np.random.rand(100, 3)
        y = X[:, 0]*0.5 + X[:, 1]*0.3 + X[:, 2]*0.2
        model = LinearRegression().fit(X, y)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    return model

def calculate_buy_potential_ai_ml(coin, model):
    """
    Uses an ML model to calculate a buy potential score.
    Features used:
      - Absolute 24h price drop (only if negative)
      - Volume-to-Market Cap ratio
      - Normalized architecture potential (dividing by 100)
    """
    price_change = coin.get("price_change_percentage_24h", 0)
    abs_price_drop = abs(price_change) if price_change < 0 else 0
    market_cap = coin.get("market_cap", 0)
    volume = coin.get("total_volume", 0)
    volume_ratio = volume / market_cap if market_cap > 0 else 0
    arch_potential = calculate_architecture_potential(coin) / 100.0  # normalize to 0-1
    
    features = np.array([[abs_price_drop, volume_ratio, arch_potential]])
    ai_score = model.predict(features)[0]
    return ai_score

def get_good_buy_coins(coins, drop_threshold=-5, model=None):
    """
    Filters coins that have dropped in price beyond a specified threshold,
    calculates an AI-based buy potential score using the ML model,
    and sorts them in descending order.
    """
    buy_candidates = [
        coin for coin in coins 
        if coin.get("price_change_percentage_24h") is not None and coin.get("price_change_percentage_24h") <= drop_threshold
    ]
    # Load the model if it's not provided
    if model is None:
        model = load_buy_model()
    # Calculate the AI-based buy potential score for each candidate.
    for coin in buy_candidates:
        coin["buy_potential_score"] = calculate_buy_potential_ai_ml(coin, model)
    buy_candidates.sort(key=lambda coin: coin.get("buy_potential_score", 0), reverse=True)
    return buy_candidates

def report_buy_potential(coins, top_n=10):
    """
    Prints a formatted report for coins that are potential buy candidates.
    """
    header = (f"{'Rank':<5}{'Name':<15}{'Symbol':<8}{'Price':<12}"
              f"{'24h Change (%)':<15}{'Volume':<15}{'Market Cap':<15}"
              f"{'Buy Potential':<15}")
    print(header)
    print("-" * len(header))
    for coin in coins[:top_n]:
        rank = coin.get("market_cap_rank", "N/A")
        name = coin.get("name", "N/A")
        symbol = coin.get("symbol", "N/A").upper()
        price = coin.get("current_price", 0)
        change = coin.get("price_change_percentage_24h", 0)
        volume = coin.get("total_volume", 0)
        market_cap = coin.get("market_cap", 0)
        buy_potential = coin.get("buy_potential_score", 0)
        print(f"{rank:<5}{name:<15}{symbol:<8}${price:<11.2f}{change:<15.2f}"
              f"{volume:<15}{market_cap:<15}{buy_potential:<15.2f}")

def main():
    # Retrieve cryptocurrency data from CoinGecko
    coins_data = get_crypto_data()
    if coins_data:
        print("=== Top Performing Coins (24h gain) ===")
        top_coins = get_top_performing_coins(coins_data, threshold=0)
        report_stats(top_coins, top_n=10)

        print("\n=== Coins at a Good Buy Point (Price Drop) ===")
        # For buy candidates, consider coins that dropped at least 5% in the last 24h.
        model = load_buy_model()  # Load your trained model (or dummy model)
        buy_candidates = get_good_buy_coins(coins_data, drop_threshold=-5, model=model)
        report_buy_potential(buy_candidates, top_n=10)
    else:
        print("No data retrieved.")

if __name__ == "__main__":
    main()
