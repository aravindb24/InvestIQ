import nltk
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Initialize Hugging Face pipelines
# Zero-shot classification with facebook/bart-large-mnli for urgency
urgency_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# FinBERT for sentiment analysis (using a model fine-tuned for financial tone)
finbert_classifier = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

def fetch_google_news_data(query="financial news"):
    """
    Scrapes Google News search results for financial headlines, links, and authors.
    
    Follows the JavaScript logic:
      - Select the main element with class 'IKXQhd'
      - Within it, select all anchor tags with class 'JtKRv' as headlines (and their links)
      - Also select all span elements as authors
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
            # Convert relative URLs to absolute links if needed.
            if link.startswith("./"):
                link = "https://news.google.com" + link[1:]
            headlines.append(headline_text)
            links.append(link)
        authors = [el.get_text(strip=True) for el in author_elements if el.get_text(strip=True)]
    return headlines, links, authors

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
    Each pair in headline_link_pairs is of the form (headline, link).
    """
    ranked = []
    for headline, link in headline_link_pairs:
        urgency = compute_urgency_score(headline)
        finbert_score = compute_finbert_score(headline)
        pressing_score = 0.5 * finbert_score + 0.5 * urgency
        ranked.append((headline, link, round(pressing_score, 4)))
    return sorted(ranked, key=lambda x: x[2], reverse=True)

if __name__ == "__main__":
    headlines, links, authors = fetch_google_news_data()
    if not headlines:
        print("No headlines found.")
    else:
        # Combine headlines and links into a list of tuples for ranking.
        headline_link_pairs = list(zip(headlines, links))
        ranked = rank_headlines_with_links(headline_link_pairs)
        
        print("\nTop Financial Headlines (Ranked by 50/50 FinBERT & Urgency):\n")
        for i, (headline, link, score) in enumerate(ranked, 1):
            print(f"{i}. {headline} — Pressing Score: {score}")
            print(f"   Link: {link}")
            
        # Optionally, print all scraped headlines, links, and authors.
        print("\nAll Scraped Headlines:")
        print(headlines)
        print("\nAll Scraped Links:")
        print(links)
        print("\nAll Scraped Authors:")
        print(authors)
