
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "results" / "vader_predictions.csv"
)

# Negative star ratings classified as positive
missed = df[
    (df["rating_sentiment"] == "negative")
    & (df["predicted_sentiment"] == "positive")
].copy()

missed = missed.sort_values(
    "vader_compound", ascending=False
)

print("\nVADER ERROR ANALYSIS")
print("=" * 50)
print(f"Missed negative reviews: {len(missed)}")

for _, row in missed.head(10).iterrows():
    print("\n" + "-" * 50)
    print(f"Review ID: {row['review_id']}")
    print(f"Rating: {row['rating']}")
    print(f"VADER Score: {row['vader_compound']:.4f}")
    print(f"Review: {str(row['text'])[:500]}")

output = ROOT / "results" / "vader_missed_negative.csv"
missed.to_csv(output, index=False)

print(f"\nSaved error analysis to: {output}")
