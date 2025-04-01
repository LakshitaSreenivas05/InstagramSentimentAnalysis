import numpy as np
import re
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import pickle

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load Pre-Trained Model
model = tf.keras.models.load_model("sentiment_lstm_model.h5")  # Load your trained LSTM model

# Load Tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Improved Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'@\w+|#\w+|http\S+', '', text)  # Remove mentions, hashtags, URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

    # Keep important words like "not", "no", "isn't" to preserve sentiment
    custom_stopwords = stop_words - {"not", "no", "isnt", "wasnt", "werent", "couldnt", "shouldnt", "dont", "didnt"}
    filtered_words = [word for word in text.split() if word not in custom_stopwords]

    return " ".join(filtered_words)

# Function to Predict Sentiment with Threshold Correction
def predict_sentiment(comment):
    processed_comment = preprocess_text(comment)
    seq = tokenizer.texts_to_sequences([processed_comment])
    padded_seq = pad_sequences(seq, maxlen=50, padding='post')

    prediction = model.predict(padded_seq)
    raw_probabilities = prediction[0]  # Extracting the single prediction output

    # Print Debugging Information
    print(f"\nRaw Model Output: {raw_probabilities}")

    # Apply temperature scaling (adjust confidence in predictions)
    temperature = 0.7
    prediction_scaled = np.exp(raw_probabilities / temperature) / np.sum(np.exp(raw_probabilities / temperature))
    print(f"Normalized Probabilities: {prediction_scaled}")

    # Apply threshold to avoid incorrect classifications when all values are too close
    max_prob = np.max(prediction_scaled)
    sentiment_label = np.argmax(prediction_scaled)

    if max_prob < 0.4:  # If no class has strong confidence, set to neutral
        sentiment_label = 1  # Neutral 😐

    sentiment_map = {2: "Positive 😊", 1: "Neutral 😐", 0: "Negative 😡"}
    return sentiment_map[sentiment_label]

# Real-Time User Input
user_input = input("\nType a comment: ")

# Debugging tokenization process
processed_comment = preprocess_text(user_input)
seq = tokenizer.texts_to_sequences([processed_comment])
padded_seq = pad_sequences(seq, maxlen=50, padding='post')

print(f"\nTokenizer Vocabulary Size: {len(tokenizer.word_index)}")
print(f"Processed Comment: {processed_comment}")
print(f"Tokenized Sequence: {seq}")
print(f"Padded Sequence: {padded_seq}")

# Predict and Print Sentiment
sentiment = predict_sentiment(user_input)
print(f"\nPredicted Sentiment: {sentiment}")
