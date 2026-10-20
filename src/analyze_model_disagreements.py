
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "results" / "model_comparison.csv"
OUTPUT_FILE = ROOT / "results" / "disagreement_review_sample.csv"

print("Loading model comparison results...")

df = pd.read_csv(INPUT_FILE)

required = {
    "review_id",
    "text",
    "rating_sentiment",
    "vader_prediction",
    "transformer_prediction",
    "comparison_category",
    "models_agree",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Review examples from each comparison category
categories = [
    "only_transformer_matches",
    "only_vader_matches",
    "both_differ_from_rating",
]

samples = []

for category in categories:
    subset = df[df["comparison_category"] == category]

    sample = subset.sample(
        n=min(10, len(subset)),
        random_state=42,
    )

    samples.append(sample)

if not samples:
    raise ValueError("No review samples found")

review_sample = pd.concat(samples, ignore_index=True)

review_sample.to_csv(OUTPUT_FILE, index=False)

print("\nDELL LLM EVALUATION FRAMEWORK")
print("=" * 50)
print("MODEL DISAGREEMENT ANALYSIS")
print("=" * 50)

print(f"\nReviews sampled: {len(review_sample)}")

for category in categories:
    print(f"\n{'=' * 50}")
    print(f"CATEGORY: {category}")
    print("=" * 50)

    subset = review_sample[
        review_sample["comparison_category"] == category
    ]

    for _, row in subset.iterrows():
        text = str(row["text"]).replace("\n", " ")

        print(f"\nReview ID: {row['review_id']}")
        print(f"Rating label: {row['rating_sentiment']}")
        print(f"VADER: {row['vader_prediction']}")
        print(f"RoBERTa: {row['transformer_prediction']}")
        print(f"Text: {text[:350]}")
        print("-" * 50)

print(f"\nReview sample saved to: {OUTPUT_FILE}")
