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

    # Normalize common words
    text = text.replace("smart phone", "smartphone")
    text = text.replace("mobile phone", "smartphone")

    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_and_save_models():
    sentiment_data = pd.DataFrame({
        "text": [
            # Positive
            "I love this product",
            "This is amazing and useful",
            "I am very happy",
            "The service was excellent",
            "The result is satisfying",
            "This app is helpful",
            "The phone is very good",
            "I like this smartphone",
            "The experience was fantastic",
            "The quality is great",

            # Negative
            "I hate this product",
            "This is very bad",
            "I am disappointed",
            "The service was awful",
            "The result is frustrating",
            "This app is useless",
            "The phone is very poor",
            "I dislike this smartphone",
            "The experience was terrible",
            "The quality is worst",

            # Neutral
            "Akshay has a smartphone",
            "The user has a smartphone",
            "He owns a smartphone",
            "The smartphone is available",
            "The product was delivered today",
            "The meeting is scheduled tomorrow",
            "The report contains basic details",
            "The user logged into the system",
            "The order status was updated",
            "The application form is submitted",
            "The phone is on the table",
            "This is a mobile device",
            "The system stores user information",
            "The file was uploaded successfully",
            "The message was received"
        ],
        "label": [
            "Positive", "Positive", "Positive", "Positive", "Positive",
            "Positive", "Positive", "Positive", "Positive", "Positive",

            "Negative", "Negative", "Negative", "Negative", "Negative",
            "Negative", "Negative", "Negative", "Negative", "Negative",

            "Neutral", "Neutral", "Neutral", "Neutral", "Neutral",
            "Neutral", "Neutral", "Neutral", "Neutral", "Neutral",
            "Neutral", "Neutral", "Neutral", "Neutral", "Neutral"
        ]
    })

    topic_data = pd.DataFrame({
        "text": [
            # Business
            "The stock market gained points",
            "Company profits increased this quarter",
            "The startup raised funds from investors",
            "Business trading increased rapidly",
            "The company announced financial results",

            # Technology
            "New smartphone launched with AI features",
            "Technology is changing software development",
            "The computer processor performance improved",
            "Smartphone has advanced technology",
            "Akshay has a smartphone",
            "The user has a smart phone",
            "Mobile phone uses modern technology",
            "Artificial intelligence is used in mobile apps",
            "Software engineers developed an application",
            "The android phone has advanced features",
            "Laptop and computer are technology devices",
            "The device has new software update",

            # Sports
            "The cricket team won the tournament",
            "Football players trained for the match",
            "The athlete broke a sports record",
            "The team practiced before the game",
            "The player scored a goal",

            # Politics
            "Election campaigns are active",
            "The government announced a new policy",
            "Parliament discussed national issues",
            "The minister gave a speech",
            "Political leaders attended the meeting",

            # Health
            "Doctors found a new healthcare treatment",
            "Hospitals are improving medical services",
            "The patient recovered after medicine",
            "The doctor checked the patient",
            "Healthcare services are important"
        ],
        "label": [
            "Business", "Business", "Business", "Business", "Business",

            "Technology", "Technology", "Technology", "Technology",
            "Technology", "Technology", "Technology", "Technology",
            "Technology", "Technology", "Technology", "Technology",

            "Sports", "Sports", "Sports", "Sports", "Sports",

            "Politics", "Politics", "Politics", "Politics", "Politics",

            "Health", "Health", "Health", "Health", "Health"
        ]
    })

    sentiment_data["clean_text"] = sentiment_data["text"].apply(clean_text)
    topic_data["clean_text"] = topic_data["text"].apply(clean_text)

    sentiment_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    topic_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_sentiment = sentiment_vectorizer.fit_transform(sentiment_data["clean_text"])
    X_topic = topic_vectorizer.fit_transform(topic_data["clean_text"])

    sentiment_model = MultinomialNB()
    topic_model = MultinomialNB()

    sentiment_model.fit(X_sentiment, sentiment_data["label"])
    topic_model.fit(X_topic, topic_data["label"])

    with open(os.path.join(MODEL_DIR, "sentiment_model.pkl"), "wb") as f:
        pickle.dump(sentiment_model, f)

    with open(os.path.join(MODEL_DIR, "sentiment_vectorizer.pkl"), "wb") as f:
        pickle.dump(sentiment_vectorizer, f)

    with open(os.path.join(MODEL_DIR, "topic_model.pkl"), "wb") as f:
        pickle.dump(topic_model, f)

    with open(os.path.join(MODEL_DIR, "topic_vectorizer.pkl"), "wb") as f:
        pickle.dump(topic_vectorizer, f)

    print("Models and vectorizers saved successfully.")
    print("Sentiment labels: Positive, Negative, Neutral")
    print("Topic labels: Business, Technology, Sports, Politics, Health")


if __name__ == "__main__":
    train_and_save_models()