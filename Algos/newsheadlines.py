import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK dependencies if not already installed
nltk.download("punkt")
nltk.download("stopwords")

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Define urgency keywords related to financial crises
URGENT_WORDS = {
    "crash", "collapse", "plummet", "default", "bankruptcy", "crisis",
    "recession", "inflation", "panic", "downturn", "sell-off", "meltdown",
    "volatility", "rate hikes", "interest rates", "bear market"
}

# Define keywords for market-moving impact
MARKET_MOVERS = {"fed", "rate", "inflation", "stocks", "nasdaq", "dow", "s&p", "bond", "oil", "OPEC"}

def fetch_financial_headlines():
    """Scrapes live financial headlines from Yahoo Finance."""
    url = "https://finance.yahoo.com/"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Failed to retrieve Yahoo Finance headlines")
        return []

    soup = BeautifulSoup(response.text, "lxml")
    headlines = [a.text for a in soup.find_all("a", class_="js-content-viewer") if a.text]
    
    return headlines[:10]  # Return top 10 headlines

def preprocess_text(text):
    """Tokenize and clean text"""
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return tokens

def compute_sentiment_score(text):
    """Compute financial sentiment score using VADER"""
    sentiment = analyzer.polarity_scores(text)
    return sentiment["compound"]  # VADER compound sentiment score

def compute_urgency_score(text):
    """Compute urgency score based on the presence of urgent words"""
    tokens = preprocess_text(text)
    urgent_count = sum(1 for word in tokens if word in URGENT_WORDS)
    return urgent_count / max(1, len(tokens))  # Normalize by total words

def compute_market_impact(text):
    """Estimate market impact score based on major financial keywords"""
    tokens = preprocess_text(text)
    impact_count = sum(1 for word in tokens if word in MARKET_MOVERS)
    return impact_count / max(1, len(tokens))

def rank_headlines(headlines):
    """Rank financial headlines based on sentiment, urgency, and market impact"""
    ranked_headlines = []
    
    for headline in headlines:
        sentiment_score = compute_sentiment_score(headline)
        urgency_score = compute_urgency_score(headline)
        impact_score = compute_market_impact(headline)
        
        # Calculate pressing score (weights: Sentiment 40%, Urgency 30%, Impact 30%)
        pressing_score = (0.4 * sentiment_score) + (0.3 * urgency_score) + (0.3 * impact_score)
        
        ranked_headlines.append((headline, pressing_score))
    
    # Sort headlines by pressing score in descending order
    ranked_headlines.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_headlines

# Fetch and rank live financial headlines
financial_headlines = fetch_financial_headlines()
if financial_headlines:
    ranked = rank_headlines(financial_headlines)

    # Display the results
    df = pd.DataFrame(ranked, columns=["Headline", "Pressing Score"])
    import ace_tools as tools
    tools.display_dataframe_to_user(name="Live Financial Headlines Ranking", dataframe=df)
