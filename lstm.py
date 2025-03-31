#LSTM MODEL

import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv("large_instagram_sentiment_dataset.csv")

# Display dataset structure
print(df.head())

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'@\w+|#\w+|http\S+', '', text)  # Remove mentions, hashtags, URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply preprocessing
df['Cleaned_Comment'] = df['Comment'].apply(preprocess_text)

# Convert Sentiment Labels to Numerical Values
label_encoder = LabelEncoder()
df['Sentiment_Label'] = label_encoder.fit_transform(df['Sentiment'])  # Positive=2, Neutral=1, Negative=0

# Tokenization
max_words = 5000  # Max words in tokenizer
max_len = 50  # Max length of each sequence

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Cleaned_Comment'])
X = tokenizer.texts_to_sequences(df['Cleaned_Comment'])
X = pad_sequences(X, maxlen=max_len, padding='post')

# Split dataset into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, df['Sentiment_Label'], test_size=0.2, random_state=42)

# Define LSTM Model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 output classes (Positive, Negative, Neutral)
])

# Compile Model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
epochs = 5
batch_size = 64

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Function to Predict Sentiment for New Comments
def predict_sentiment(comment):
    processed_comment = preprocess_text(comment)
    seq = tokenizer.texts_to_sequences([processed_comment])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')

    prediction = model.predict(padded_seq)
    sentiment_label = np.argmax(prediction, axis=1)[0]
    sentiment_map = {2: "Posit ive", 1: "Neutral", 0: "Negative"}

    print(f"\nComment: {comment}")
    print(f"Predicted Sentiment: {sentiment_map[sentiment_label]}")

# Example Predictions
predict_sentiment("I love this product! It's amazing 🔥😍")
predict_sentiment("This update is the worst! I hate it 😡")
predict_sentiment("It's okay, not too bad I guess 🤔")
