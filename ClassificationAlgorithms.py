#TF-IDF vectorizer, Naive bayes, logistic regression

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

# Download stopwords for text processing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset (Generated dataset with 1 million entries)
df = pd.read_csv("large_instagram_sentiment_dataset.csv")

# Display dataset structure
print(df.head())

# Data Preprocessing Function
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'@\w+|#\w+|http\S+', '', text)  # Remove mentions, hashtags, URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply preprocessing
df['Cleaned_Comment'] = df['Comment'].apply(preprocess_text)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Comment'])

# Convert Sentiment Labels to Numerical Format
df['Sentiment'] = df['Sentiment'].map({"Positive": 1, "Negative": -1, "Neutral": 0})

# Split dataset into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, df['Sentiment'], test_size=0.2, random_state=42)

# Train a Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Train a Logistic Regression Model
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# Model Evaluation
nb_pred = nb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

print("\nClassification Report (Naive Bayes):\n", classification_report(y_test, nb_pred))
print("\nClassification Report (Logistic Regression):\n", classification_report(y_test, lr_pred))

# Function to Predict Sentiment for New Comments
def predict_sentiment(comment):
    processed_comment = preprocess_text(comment)
    vectorized_comment = vectorizer.transform([processed_comment])

    nb_prediction = nb_model.predict(vectorized_comment)[0]
    lr_prediction = lr_model.predict(vectorized_comment)[0]

    sentiment_map = {1: "Positive", -1: "Negative", 0: "Neutral"}

    print(f"\nComment: {comment}")
    print(f"Naive Bayes Prediction: {sentiment_map[nb_prediction]}")
    print(f"Logistic Regression Prediction: {sentiment_map[lr_prediction]}")

# Example Predictions
predict_sentiment("I love this product! It's amazing 🔥😍")
predict_sentiment("This update is the worst! I hate it 😡")
predict_sentiment("It's okay, not too bad I guess 🤔")
