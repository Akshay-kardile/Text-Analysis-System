IRTM MINI PROJECT
=================

Project Title:
Text Analysis System using IRTM Concepts

Features:
1. Sentiment Analysis
   - Detects Positive or Negative sentiment from input text.

2. Text Classification
   - Categorizes text into topics such as Business, Technology, Sports, Politics, and Health.

3. History Tracking
   - Saves all analysis results in SQLite database and shows them on a history page.

IRTM Concepts Used:
- Text preprocessing
- Tokenization / cleaning
- Stop-word removal
- TF-IDF feature extraction
- Naive Bayes classification
- Information storage and retrieval

Project Structure:
irtm_nlp_mini_project/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.txt
├── history.db               (auto-created)
├── models/                  (auto-created)
│   ├── sentiment_model.pkl
│   ├── topic_model.pkl
│   ├── sentiment_vectorizer.pkl
│   └── topic_vectorizer.pkl
│
└── templates/
    ├── index.html
    └── history.html

How to Run:
1. Open terminal in project folder
2. Install dependencies:
   pip install -r requirements.txt

3. Run:
   python app.py

4. Open browser:
   http://127.0.0.1:5000

Sample Inputs:
- "I really enjoyed this amazing service"
- "The cricket team played a fantastic match"
- "The government introduced a new policy"
- "This product is terrible and disappointing"

Notes:
- Models are trained automatically on first run if they are missing.
- This is a mini project for learning/demo use.
