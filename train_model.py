import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("Loading dataset...")

df = pd.read_csv("data_amazon.xlsx - Sheet1.csv")

# Use your columns
df = df[['Review', 'Cons_rating']]
df.dropna(inplace=True)

# Convert rating to sentiment
def sentiment(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"

df['Sentiment'] = df['Cons_rating'].apply(sentiment)

print("Vectorizing...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

print("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training model...")
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

print("Saving model...")
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Training completed successfully!")