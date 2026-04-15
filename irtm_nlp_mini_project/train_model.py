import os
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_and_save_models():
    # Sentiment dataset
    sentiment_data = pd.DataFrame({
        "text": [
            "I love this product, it is amazing and useful",
            "This is the best experience ever",
            "I am very happy with the service",
            "The movie was fantastic and inspiring",
            "The results are excellent and satisfying",
            "I hate this item, it is terrible",
            "This is the worst thing I bought",
            "I am disappointed and unhappy",
            "The service was awful and slow",
            "The experience was bad and frustrating"
        ],
        "label": [
            "Positive", "Positive", "Positive", "Positive", "Positive",
            "Negative", "Negative", "Negative", "Negative", "Negative"
        ]
    })

    # Topic classification dataset
    topic_data = pd.DataFrame({
        "text": [
            "The stock market gained points in business trading",
            "Company profits increased this quarter",
            "The startup raised funds from investors",
            "New smartphone launched with AI features",
            "Technology is changing software development",
            "The computer processor performance improved",
            "The cricket team won the tournament",
            "Football players trained for the match",
            "The athlete broke a sports record",
            "Election campaigns are active in politics",
            "The government announced a new policy",
            "Parliament discussed national issues",
            "Doctors found a new healthcare treatment",
            "Hospitals are improving medical services",
            "The patient recovered after proper medicine"
        ],
        "label": [
            "Business", "Business", "Business",
            "Technology", "Technology", "Technology",
            "Sports", "Sports", "Sports",
            "Politics", "Politics", "Politics",
            "Health", "Health", "Health"
        ]
    })

    sentiment_data["clean_text"] = sentiment_data["text"].apply(clean_text)
    topic_data["clean_text"] = topic_data["text"].apply(clean_text)

    sentiment_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    topic_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    X_sentiment = sentiment_vectorizer.fit_transform(sentiment_data["clean_text"])
    X_topic = topic_vectorizer.fit_transform(topic_data["clean_text"])

    sentiment_model = MultinomialNB()
    topic_model = MultinomialNB()

    sentiment_model.fit(X_sentiment, sentiment_data["label"])
    topic_model.fit(X_topic, topic_data["label"])

    with open(os.path.join(MODEL_DIR, "sentiment_model.pkl"), "wb") as f:
        pickle.dump(sentiment_model, f)

    with open(os.path.join(MODEL_DIR, "topic_model.pkl"), "wb") as f:
        pickle.dump(topic_model, f)

    with open(os.path.join(MODEL_DIR, "sentiment_vectorizer.pkl"), "wb") as f:
        pickle.dump(sentiment_vectorizer, f)

    with open(os.path.join(MODEL_DIR, "topic_vectorizer.pkl"), "wb") as f:
        pickle.dump(topic_vectorizer, f)

    print("Models and vectorizers saved successfully.")


if __name__ == "__main__":
    train_and_save_models()
