# bar chart for sentiment distribution
# wordclouds for positive, negetive and neutral comments
# boxplot for likes and replies distribution by sentiment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv("large_instagram_sentiment_dataset.csv")

# Count sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=df["Sentiment"], palette="coolwarm")
plt.title("Sentiment Distribution", fontsize=14)
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Function to generate WordCloud
def generate_wordcloud(sentiment):
    text = " ".join(df[df["Sentiment"] == sentiment]["Comment"].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud for {sentiment} Comments", fontsize=14)
    plt.show()

# Generate WordClouds
generate_wordcloud("Positive")
generate_wordcloud("Negative")
generate_wordcloud("Neutral")

# Sentiment & Engagement Analysis
plt.figure(figsize=(8, 5))
sns.boxplot(x="Sentiment", y="Likes", data=df, palette="coolwarm")
plt.title("Likes Distribution by Sentiment", fontsize=14)
plt.xlabel("Sentiment")
plt.ylabel("Number of Likes")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="Sentiment", y="Replies", data=df, palette="coolwarm")
plt.title("Replies Distribution by Sentiment", fontsize=14)
plt.xlabel("Sentiment")
plt.ylabel("Number of Replies")
plt.show()
