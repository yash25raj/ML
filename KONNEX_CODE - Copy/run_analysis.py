import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from wordcloud import WordCloud
from scipy.stats import f_oneway

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('all')  # This will download all NLTK data

# Load data
print("Loading data...")
data = pd.read_csv('Task1.csv')

# Basic inspection
print("\nDataset Shape:", data.shape)
print("Columns:", data.columns)
print("\nSample Data:\n", data.head())
print("\nMissing Values:\n", data.isnull().sum())

# Data Cleaning
print("\nCleaning data...")
data = data.dropna(subset=['review', 'rating'])
data = data.drop_duplicates()

# Text Preprocessing
print("\nPreprocessing text...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['cleaned_review'] = data['review'].apply(preprocess_text)

# Feature Engineering
def infer_sentiment(rating):
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positive'

data['sentiment'] = data['rating'].apply(infer_sentiment)

# Save preprocessed data
data.to_csv('preprocessed_data.csv', index=False)
print("\nPreprocessed Data Sample:\n", data[['cleaned_review', 'sentiment']].head())

# Sentiment Analysis
print("\nPerforming sentiment analysis...")
X = data['cleaned_review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test_tfidf)
f1 = f1_score(y_test, y_pred, average='weighted')
print("\nF1 Score:", f1)

# Save predictions
test_data = pd.DataFrame({'review': X_test, 'true_sentiment': y_test, 'predicted_sentiment': y_pred})
test_data.to_csv('sentiment_predictions.csv', index=False)

print("\nAnalysis complete! Check the output files:")
print("1. preprocessed_data.csv - Contains cleaned and preprocessed data")
print("2. sentiment_predictions.csv - Contains sentiment predictions") 