from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import pickle
import os
from datetime import datetime

app = Flask(__name__)

DB_NAME = "history.db"
MODEL_DIR = "models"
SENTIMENT_MODEL = os.path.join(MODEL_DIR, "sentiment_model.pkl")
TOPIC_MODEL = os.path.join(MODEL_DIR, "topic_model.pkl")
SENTIMENT_VECTORIZER = os.path.join(MODEL_DIR, "sentiment_vectorizer.pkl")
TOPIC_VECTORIZER = os.path.join(MODEL_DIR, "topic_vectorizer.pkl")


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            category TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_models():
    needed = [
        SENTIMENT_MODEL,
        TOPIC_MODEL,
        SENTIMENT_VECTORIZER,
        TOPIC_VECTORIZER
    ]
    if all(os.path.exists(p) for p in needed):
        return

    try:
        import train_model
        train_model.train_and_save_models()
    except Exception as e:
        raise RuntimeError(f"Could not create model files automatically: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if not text:
            error = "Please enter some text."
            return render_template("index.html", result=result, error=error)

        try:
            ensure_models()
            sentiment_model = load_pickle(SENTIMENT_MODEL)
            topic_model = load_pickle(TOPIC_MODEL)
            sentiment_vectorizer = load_pickle(SENTIMENT_VECTORIZER)
            topic_vectorizer = load_pickle(TOPIC_VECTORIZER)

            sentiment_features = sentiment_vectorizer.transform([text])
            topic_features = topic_vectorizer.transform([text])

            sentiment = sentiment_model.predict(sentiment_features)[0]
            category = topic_model.predict(topic_features)[0]

            result = {
                "text": text,
                "sentiment": sentiment,
                "category": category
            }

            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO analysis_history (input_text, sentiment, category, created_at)
                VALUES (?, ?, ?, ?)
            """, (text, sentiment, category, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()

        except Exception as e:
            error = f"Error while analyzing text: {e}"

    return render_template("index.html", result=result, error=error)


@app.route("/history")
def history():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, input_text, sentiment, category, created_at
        FROM analysis_history
        ORDER BY id DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return render_template("history.html", rows=rows)


if __name__ == "__main__":
    init_db()
    ensure_models()
    app.run(debug=True)
