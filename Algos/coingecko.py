import requests

API_KEY = "CG-z9dFsALqjPqj9ndKhKotVnfW"
BASE_URL = "https://api.coingecko.com/api/v3"

def get_crypto_data(vs_currency="usd", per_page=100, page=1):
    """
    Fetches market data for cryptocurrencies from CoinGecko.
    
    Parameters:
        vs_currency (str): The target currency (e.g., "usd").
        per_page (int): Number of coins per page.
        page (int): Which page of data to fetch.
    
    Returns:
        list: A list of coin data dictionaries if successful, otherwise None.
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

def get_top_performing_coins(coins, threshold=0):
    """
    Filters and sorts coins with a positive 24h change.
    
    Parameters:
        coins (list): List of coin data dictionaries.
        threshold (float): Minimum 24h percentage change (default is 0, so only positive changes).
    
    Returns:
        list: Sorted list of coins (from highest to lowest change) meeting the threshold.
    """
    top_coins = [
        coin for coin in coins 
        if coin.get("price_change_percentage_24h") is not None and coin.get("price_change_percentage_24h") > threshold
    ]
    top_coins.sort(key=lambda coin: coin.get("price_change_percentage_24h"), reverse=True)
    return top_coins

def report_stats(coins, top_n=10):
    """
    Prints out a formatted report of the top performing coins.
    
    Parameters:
        coins (list): Sorted list of coin data dictionaries.
        top_n (int): Number of top coins to display.
    """
    print(f"{'Rank':<5}{'Name':<15}{'Symbol':<8}{'Price':<12}{'24h Change (%)':<15}{'Market Cap':<15}")
    print("-" * 70)
    for coin in coins[:top_n]:
        rank = coin.get("market_cap_rank", "N/A")
        name = coin.get("name", "N/A")
        symbol = coin.get("symbol", "N/A").upper()
        price = coin.get("current_price", 0)
        change = coin.get("price_change_percentage_24h", 0)
        market_cap = coin.get("market_cap", 0)
        print(f"{rank:<5}{name:<15}{symbol:<8}${price:<11.2f}{change:<15.2f}{market_cap:<15}")

def main():
    # Retrieve cryptocurrency data
    coins_data = get_crypto_data()
    if coins_data:
        # Filter for coins that are performing well (positive change)
        top_coins = get_top_performing_coins(coins_data, threshold=0)
        # Report stats on the top 10 performing coins
        report_stats(top_coins, top_n=10)
    else:
        print("No data retrieved.")

if __name__ == "__main__":
    main()