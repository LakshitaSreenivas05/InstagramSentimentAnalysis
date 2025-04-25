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

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv("large_instagram_sentiment_dataset.csv")
print(df.head())

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'@\w+|#\w+|http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df['Cleaned_Comment'] = df['Comment'].apply(preprocess_text)

label_encoder = LabelEncoder()
df['Sentiment_Label'] = label_encoder.fit_transform(df['Sentiment'])

max_words = 5000
max_len = 50

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Cleaned_Comment'])
X = tokenizer.texts_to_sequences(df['Cleaned_Comment'])
X = pad_sequences(X, maxlen=max_len, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X, df['Sentiment_Label'], test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

def predict_sentiment(comment):
    processed_comment = preprocess_text(comment)
    seq = tokenizer.texts_to_sequences([processed_comment])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(padded_seq)
    sentiment_label = np.argmax(prediction, axis=1)[0]
    sentiment_map = {2: "Positive", 1: "Neutral", 0: "Negative"}
    print(f"\nComment: {comment}")
    print(f"Predicted Sentiment: {sentiment_map[sentiment_label]}")

predict_sentiment("I love this product! It's amazing 🔥😍")
predict_sentiment("This update is the worst! I hate it 😡")
predict_sentiment("It's okay, not too bad I guess 🤔")

