from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)

# Load sentiment classification model
with open('model/log_classifier_bow.pkl', 'rb') as f:
    log_classifier_bow = joblib.load(f)

with open('model/vocab.pkl', 'rb') as f:
    vocab = joblib.load(f)


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Define preprocessing function
def preprocess(text):
    sentence = re.sub(r'[^a-zA-Z]', ' ', text)
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)  # Tokenize using word_tokenize
    stop_words_excluded = set(stopwords.words("english")) - {'not', 'bad'}
    clean_tokens = [token for token in tokens if token not in stop_words_excluded]
    clean_tokens = [lemmatizer.lemmatize(token) for token in clean_tokens]
    return ' '.join(clean_tokens)

# Function to get embeddings for a given text
def get_embeddings(review):
    processed_review = preprocess(review)
    embeddings = vocab.transform([processed_review])
    return embeddings


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    processed_review = get_embeddings(review)
    sentiment = log_classifier_bow.predict(processed_review)  
    predictions = [(review, "Positive" if sentiment == "Positive" else "Negative")]
    return render_template("index.html", review=review, predictions=predictions)


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
