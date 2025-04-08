import nltk
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify
from flask_cors import CORS
from transformers import pipeline

# Download required NLTK resources.
nltk.download("punkt")
nltk.download("stopwords")

# Initialize Hugging Face pipelines:
# Urgency classification (zero-shot using facebook/bart-large-mnli)
urgency_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# Sentiment analysis for a financial tone using FinBERT.
finbert_classifier = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

def fetch_google_news_data(query="financial news"):
    """
    Scrapes Google News search results for financial headlines, links, and authors.
    
    The function:
      - Constructs the URL using the query
      - Selects the main element with class 'IKXQhd'
      - Extracts all anchor tags with class 'JtKRv' as headlines and retrieves their href attribute
      - Converts any relative URLs to absolute URLs
      - Extracts authors from span elements.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://news.google.com/search?q={query.replace(' ', '%20')}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    main_element = soup.select_one("main.IKXQhd")
    headlines = []
    links = []
    authors = []
    if main_element:
        headline_elements = main_element.select("a.JtKRv")
        author_elements = main_element.select("span")
        for el in headline_elements:
            headline_text = el.get_text(strip=True)
            link = el.get("href")
            # Convert relative URLs to absolute URLs.
            if link.startswith("./"):
                link = "https://news.google.com" + link[1:]
            headlines.append(headline_text)
            links.append(link)
        authors = [el.get_text(strip=True) for el in author_elements if el.get_text(strip=True)]
    return headlines, links, authors

def compute_urgency_score(text):
    """
    Computes an urgency score using zero-shot classification.
    Classifies the text as 'urgent' or 'non-urgent' and returns
    the probability for the 'urgent' label.
    """
    candidate_labels = ["urgent", "non-urgent"]
    result = urgency_classifier(text, candidate_labels)
    label_scores = dict(zip(result["labels"], result["scores"]))
    return label_scores.get("urgent", 0)

def compute_finbert_score(text):
    """
    Computes a score from FinBERT's sentiment analysis.
    
    It returns the sum of positive and negative sentiment scores,
    representing how strongly the headline deviates from a neutral tone.
    """
    results = finbert_classifier(text, top_k=None)
    pos_score = 0
    neg_score = 0
    for r in results:
        label = r["label"].lower()
        if label == "positive":
            pos_score = r["score"]
        elif label == "negative":
            neg_score = r["score"]
    return pos_score + neg_score

def rank_headlines_with_links(headline_link_pairs):
    """
    Ranks headlines by combining:
      - FinBERT non-neutral sentiment score
      - Urgency score from zero-shot classification
      
    The final pressing score is the average (50/50 split) of these two scores.
    """
    ranked = []
    for headline, link in headline_link_pairs:
        urgency = compute_urgency_score(headline)
        finbert_score = compute_finbert_score(headline)
        pressing_score = 0.5 * finbert_score + 0.5 * urgency
        ranked.append((headline, link, round(pressing_score, 4)))
    return sorted(ranked, key=lambda x: x[2], reverse=True)

# Create Flask application.
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing to allow requests from your Chrome extension.
CORS(app)

@app.route('/api/news', methods=['GET'])
def get_news():
    """
    Endpoint that returns a JSON list of headlines with links and pressing scores.
    Each item in the output list contains:
      - headline: The news headline.
      - link: The URL to the full news article.
      - pressing_score: The computed score based on sentiment and urgency.
    """
    headlines, links, authors = fetch_google_news_data()
    headline_link_pairs = list(zip(headlines, links))
    ranked = rank_headlines_with_links(headline_link_pairs)
    data = [{
        "headline": headline,
        "link": link,
        "pressing_score": score
    } for headline, link, score in ranked]
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
