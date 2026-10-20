
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Project folders
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "FusionTech Online Reviews.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original dataset
df = pd.read_csv(RAW_FILE)

print(f"Original reviews: {len(df)}")

# Keep the fields needed for evaluation
columns = [
    "rating",
    "title_x",
    "text",
    "asin",
    "user_id",
    "timestamp",
    "main_category",
    "title_y",
]

df = df[columns].copy()

# Standardize review text and ratings
df["text"] = df["text"].astype("string").str.strip()
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Remove records with unusable review text or ratings
df = df.dropna(subset=["text", "rating"])
df = df[df["text"].str.len() > 0].copy()
df = df[df["rating"].between(1, 5)].copy()

# Remove repeated reviews from the same reviewer for the same product
before_duplicates = len(df)
df = df.drop_duplicates(
    subset=["asin", "user_id", "text"]
).copy()

print(f"Duplicates removed: {before_duplicates - len(df)}")
print(f"Cleaned reviews: {len(df)}")

# Create deterministic sentiment proxy labels from star ratings
def rating_to_sentiment(rating):
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"

df["rating_sentiment"] = df["rating"].apply(rating_to_sentiment)

# Assign stable IDs based on the original CSV row index
df["review_id"] = df.index.map(lambda i: f"FT-{i:05d}")

# Save cleaned reviews
cleaned_path = OUTPUT_DIR / "cleaned_reviews.csv"
df.to_csv(cleaned_path, index=False)

# Stratified sampling to represent each sentiment group
sample_size = 150

if len(df) < sample_size:
    raise ValueError("Not enough reviews for the requested sample.")

sample, _ = train_test_split(
    df,
    train_size=sample_size,
    random_state=42,
    stratify=df["rating_sentiment"],
)

sample = sample.sort_values("review_id")

sample_path = OUTPUT_DIR / "evaluation_sample.csv"
sample.to_csv(sample_path, index=False)

print(f"Evaluation sample: {len(sample)}")
print("\nSentiment distribution:")
print(sample["rating_sentiment"].value_counts())
print(f"\nSaved: {cleaned_path}")
print(f"Saved: {sample_path}")
