# Text Analysis System using IRTM Concepts

## Overview
This project is a web-based Text Analysis System built using Python and Flask. It analyzes user-input text and predicts sentiment (Positive/Negative) and category (Business, Technology, Sports, Politics, Health).

## Features
- Sentiment Analysis
- Text Classification
- History Tracking
- Automatic Model Training
- Simple Web Interface

## Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- SQLite

## Project Structure
```
irtm_nlp_mini_project/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── history.db
├── models/
│   ├── sentiment_model.pkl
│   ├── topic_model.pkl
│   ├── sentiment_vectorizer.pkl
│   └── topic_vectorizer.pkl
│
└── templates/
    ├── index.html
    └── history.html
```

## How It Works
1. User enters text in the web app
2. Text is processed using TF-IDF
3. Model predicts sentiment and category
4. Result is stored in SQLite database
5. History page shows previous results

## Installation

### Clone repository
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Install dependencies
```
pip install -r requirements.txt
```

### Run the app
```
python app.py
```

### Open in browser
http://127.0.0.1:5000

## Dependencies
- Flask
- scikit-learn
- pandas

## Sample Inputs
- I love this product
- The match was amazing
- This is a bad experience

## Future Improvements
- Add more categories
- Improve accuracy
- Better UI
- Deploy online

## Conclusion
This project demonstrates NLP and IRTM concepts using a simple and practical Flask application.
