# DATASET GENERATION
import pandas as pd
import random
import emoji
from faker import Faker
from tqdm import tqdm  # For progress bar

# Initialize Faker
fake = Faker()

# Predefined sentiment words
positive_words = ["amazing", "love", "awesome", "fantastic", "great", "😍", "🔥", "💖", "LOL", "LMAO"]
negative_words = ["worst", "terrible", "hate", "awful", "bad", "😡", "😢", "💔", "WTF", "IDK"]
neutral_words = ["okay", "fine", "alright", "not bad", "hmm", "🤔", "🙂", "😐", "IDK", "meh"]

# Common abbreviations & slang
abbreviations = ["OMG", "LOL", "IDK", "TTYL", "BRB", "SMH", "FOMO", "ICYMI", "GTG", "FYI"]

# Hashtags and entities
hashtags = ["#fun", "#awesome", "#sad", "#trending", "#viral", "#instagram", "#mood", "#OOTD", "#FYP"]
brands = ["Nike", "Adidas", "Apple", "Samsung", "Tesla", "McDonald's", "Coca-Cola"]
celebrities = ["Elon Musk", "Taylor Swift", "Cristiano Ronaldo", "Billie Eilish", "Drake"]
locations = ["NYC", "LA", "Paris", "Tokyo", "London", "Dubai"]

# Generate a random comment with abbreviations and entities
def generate_comment(sentiment):
    comment = fake.sentence()

    if sentiment == "Positive":
        comment += f" {random.choice(positive_words)} {random.choice(positive_words)}"
    elif sentiment == "Negative":
        comment += f" {random.choice(negative_words)} {random.choice(negative_words)}"
    else:
        comment += f" {random.choice(neutral_words)} {random.choice(neutral_words)}"

    # Add abbreviations
    if random.random() > 0.5:
        comment += f" {random.choice(abbreviations)}"

    # Add brand, celebrity, or location
    if random.random() > 0.5:
        comment += f" @{random.choice(celebrities)}" if random.random() > 0.5 else f" #{random.choice(hashtags)}"

    return comment

# Define number of records
num_samples = 1_000_000  # 1 Million Records
batch_size = 100_000  # Save in batches to avoid memory issues

# Open CSV file
csv_filename = "large_instagram_sentiment_dataset.csv"
with open(csv_filename, "w", encoding="utf-8") as file:
    file.write("Comment,Sentiment,Likes,Replies\n")  # Write header

# Generate data in batches
for batch in tqdm(range(num_samples // batch_size), desc="Generating Data"):
    data = []
    for _ in range(batch_size):
        sentiment = random.choice(["Positive", "Negative", "Neutral"])
        comment = generate_comment(sentiment)
        likes = random.randint(0, 5000)  # Simulating viral posts
        replies = random.randint(0, 200)  # Simulating engagement

        data.append([comment, sentiment, likes, replies])

    # Save batch to CSV
    df = pd.DataFrame(data, columns=["Comment", "Sentiment", "Likes", "Replies"])
    df.to_csv(csv_filename, mode="a", index=False, header=False)  # Append data

print(f"✅ Dataset saved as '{csv_filename}' with 1 million entries!")
