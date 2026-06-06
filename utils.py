import re
from textblob import TextBlob
from collections import Counter

# Common stopwords
STOPWORDS = {
    "the", "is", "was", "and", "a", "an", "to", "of", "it", "this", "that", "for", "on", "in",
    "with", "very", "my", "i", "me", "we", "our", "you", "your", "they", "their", "at", "as",
    "are", "be", "been", "have", "has", "had", "but", "or", "if", "so", "too", "not", "from",
    "by", "about", "just", "all", "can", "will", "would", "should", "could", "product", "item",
    "amazon", "flipkart", "buy", "bought", "using", "use", "used", "one", "get", "got",
    "dress", "top", "shirt", "wear", "wore", "clothing"
}

# Clothing-focused negative keywords
NEGATIVE_HINT_WORDS = {
    "bad", "poor", "worst", "cheap", "problem", "issue", "small", "large", "tight", "loose",
    "size", "fit", "itchy", "rough", "uncomfortable", "thin", "transparent", "see through",
    "faded", "tear", "torn", "stitch", "stitching", "damaged", "color", "colour", "different",
    "late", "delay", "delivery", "refund", "return", "quality", "fabric", "material"
}

# Clothing-focused positive keywords
POSITIVE_HINT_WORDS = {
    "good", "great", "excellent", "amazing", "best", "nice", "love", "perfect", "comfortable",
    "soft", "beautiful", "pretty", "stylish", "quality", "worth", "fit", "fitting", "awesome",
    "premium", "durable", "cute", "flattering"
}

def clean_text(text):
    """Clean review text"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_columns(df):
    """
    Auto-detect columns in uploaded e-commerce dataset.
    Specially optimized for clothing/fashion datasets.
    """
    columns = [c.lower().strip() for c in df.columns]
    original_cols = list(df.columns)

    mapping = {
        "product": None,
        "review": None,
        "rating": None,
        "seller": None,
        "brand": None,
        "title": None
    }

    # Better support for your clothing dataset
    product_candidates = [
        "class name", "class_name", "product", "productname", "product_name", "name", "item"
    ]
    review_candidates = [
        "review text", "review_text", "review", "reviewtext", "text", "summary", "comment", "description"
    ]
    rating_candidates = ["rating", "score", "stars", "rate"]
    seller_candidates = ["seller", "sellername", "seller_name", "vendor", "division name", "division_name"]
    brand_candidates = ["brand", "company", "manufacturer", "department name", "department_name"]

    def find_col(candidates):
        for cand in candidates:
            for i, col in enumerate(columns):
                if cand == col or cand in col:
                    return original_cols[i]
        return None

    mapping["product"] = find_col(product_candidates)
    mapping["review"] = find_col(review_candidates)
    mapping["rating"] = find_col(rating_candidates)
    mapping["seller"] = find_col(seller_candidates)
    mapping["brand"] = find_col(brand_candidates)
    mapping["title"] = find_col(["title", "summary", "headline"])

    return mapping

def sentiment_from_rating(rating):
    """Convert numeric rating to sentiment"""
    try:
        rating = float(rating)
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"
    except:
        return None

def sentiment_from_text(text):
    """Fallback sentiment using TextBlob"""
    try:
        polarity = TextBlob(str(text)).sentiment.polarity
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"
    except:
        return "Neutral"

def extract_keywords(text_series, top_n=15):
    """Extract frequent useful words from a list/series of reviews"""
    words = []
    for text in text_series.dropna():
        cleaned = clean_text(text)
        for word in cleaned.split():
            if len(word) > 2 and word not in STOPWORDS:
                words.append(word)

    counter = Counter(words)
    return counter.most_common(top_n)

def extract_issue_keywords(text_series, top_n=10):
    """Extract issue words from negative reviews"""
    words = []
    for text in text_series.dropna():
        cleaned = clean_text(text)
        for word in cleaned.split():
            if len(word) > 2 and word not in STOPWORDS:
                if word in NEGATIVE_HINT_WORDS or len(word) > 4:
                    words.append(word)

    counter = Counter(words)
    return counter.most_common(top_n)

def extract_positive_keywords(text_series, top_n=10):
    """Extract praise words from positive reviews"""
    words = []
    for text in text_series.dropna():
        cleaned = clean_text(text)
        for word in cleaned.split():
            if len(word) > 2 and word not in STOPWORDS:
                if word in POSITIVE_HINT_WORDS or len(word) > 4:
                    words.append(word)

    counter = Counter(words)
    return counter.most_common(top_n)

def generate_improvement_suggestion(issue_keywords):
    """Generate business suggestions based on clothing-related issue keywords"""
    issues = [w for w, _ in issue_keywords]
    suggestions = []

    if any(word in issues for word in ["size", "small", "large", "tight", "loose", "fit"]):
        suggestions.append("Improve size chart accuracy and provide clearer fit guidance for customers.")

    if any(word in issues for word in ["fabric", "material", "cheap", "quality", "thin", "transparent"]):
        suggestions.append("Enhance fabric/material quality and ensure product descriptions match actual material standards.")

    if any(word in issues for word in ["color", "colour", "faded", "different"]):
        suggestions.append("Improve product image accuracy and color consistency between online listing and delivered item.")

    if any(word in issues for word in ["stitch", "stitching", "tear", "torn", "damaged"]):
        suggestions.append("Strengthen stitching quality control and inspect garments before dispatch.")

    if any(word in issues for word in ["itchy", "rough", "uncomfortable"]):
        suggestions.append("Focus on customer comfort by improving fabric softness and wearable design.")

    if any(word in issues for word in ["delivery", "late", "delay"]):
        suggestions.append("Optimize delivery logistics to reduce shipping delays and improve customer satisfaction.")

    if any(word in issues for word in ["refund", "return"]):
        suggestions.append("Improve return and refund support to build trust and improve post-purchase experience.")

    if not suggestions:
        suggestions.append("Review negative feedback themes and improve product quality, fit accuracy, comfort, and service support.")

    return suggestions

def create_sentiment_from_rating(rating):
    try:
        rating = float(rating)
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
    except:
        return "neutral"