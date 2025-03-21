import nltk
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Initialize Hugging Face pipelines
# Urgency: zero-shot classification with facebook/bart-large-mnli
urgency_classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli"
)
# FinBERT for sentiment analysis (using a model fine-tuned for financial tone)
finbert_classifier = pipeline(
    "sentiment-analysis", model="yiyanghkust/finbert-tone"
)

def fetch_google_news_data(query="financial news"):
    """
    Scrapes Google News search results for financial headlines and authors.
    
    Follows the JavaScript logic:
      - Select the main element with class 'IKXQhd'
      - Within it, select all anchor tags with class 'JtKRv' as headlines
      - Also select all span elements as authors
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://news.google.com/search?q={query.replace(' ', '%20')}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    main_element = soup.select_one("main.IKXQhd")
    headlines = []
    authors = []
    if main_element:
        headline_elements = main_element.select("a.JtKRv")
        author_elements = main_element.select("span")
        headlines = [el.get_text(strip=True) for el in headline_elements if el.get_text(strip=True)]
        authors = [el.get_text(strip=True) for el in author_elements if el.get_text(strip=True)]
    return headlines, authors

def compute_urgency_score(text):
    """
    Computes urgency using zero-shot classification.
    
    It classifies the text into 'urgent' or 'non-urgent' and returns
    the probability for the 'urgent' label.
    """
    candidate_labels = ["urgent", "non-urgent"]
    result = urgency_classifier(text, candidate_labels)
    label_scores = dict(zip(result["labels"], result["scores"]))
    return label_scores.get("urgent", 0)

def compute_finbert_score(text):
    """
    Uses FinBERT for sentiment analysis.
    
    Returns the sum of the positive and negative sentiment probabilities,
    representing how strongly the headline deviates from a neutral tone.
    """
    results = finbert_classifier(text, top_k=None)[0]
    pos_score = 0
    neg_score = 0
    for r in results:
        label = r["label"].lower()
        if label == "positive":
            pos_score = r["score"]
        elif label == "negative":
            neg_score = r["score"]
    return pos_score + neg_score

def rank_headlines(headlines):
    """
    Ranks headlines by combining:
      - FinBERT non-neutral sentiment score
      - Urgency score from zero-shot classification
    
    The final pressing score is the average (50/50 split) of these two scores.
    """
    ranked = []
    for headline in headlines:
        urgency = compute_urgency_score(headline)
        finbert_score = compute_finbert_score(headline)
        pressing_score = 0.5 * finbert_score + 0.5 * urgency
        ranked.append((headline, round(pressing_score, 4)))
    return sorted(ranked, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    headlines, authors = fetch_google_news_data()
    if not headlines:
        print("No headlines found.")
    else:
        ranked = rank_headlines(headlines)
        print("\nTop Financial Headlines (Ranked by 50/50 FinBERT & Urgency):\n")
        for i, (headline, score) in enumerate(ranked, 1):
            print(f"{i}. {headline} — Pressing Score: {score}")
        # Optionally, print all scraped headlines and authors
        print("\nAll Scraped Headlines:")
        print(headlines)
        print("\nAll Scraped Authors:")
        print(authors)
