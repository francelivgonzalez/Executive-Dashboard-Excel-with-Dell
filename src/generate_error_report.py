
from pathlib import Path
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

INPUT_FILE = RESULTS / "model_comparison.csv"

print("Loading model comparison data...")
df = pd.read_csv(INPUT_FILE)

required = {
    "review_id", "text", "rating_sentiment",
    "vader_prediction", "transformer_prediction",
    "comparison_category", "models_agree"
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["text"] = df["text"].fillna("").astype(str)
df["review_length_chars"] = df["text"].str.len()
df["review_length_words"] = df["text"].str.split().str.len()

# Potential linguistic and formatting challenges
df["has_update_marker"] = df["text"].str.contains(
    r"\b(?:update|updated|edit)\s*(?:\d+|:|-)?",
    case=False, regex=True, na=False
)

df["has_html_markup"] = df["text"].str.contains(
    r"<[^>]+>|&#\d+;",
    regex=True, na=False
)

df["has_contrast_marker"] = df["text"].str.contains(
    r"\b(?:but|however|although|despite|unfortunately)\b",
    case=False, regex=True, na=False
)

df["predictions_opposite"] = (
    (
        (df["vader_prediction"] == "negative") &
        (df["transformer_prediction"] == "positive")
    ) |
    (
        (df["vader_prediction"] == "positive") &
        (df["transformer_prediction"] == "negative")
    )
)

df["both_differ_from_rating"] = (
    df["comparison_category"] == "both_differ_from_rating"
)

# Aggregate descriptive statistics
flag_columns = [
    "has_update_marker",
    "has_html_markup",
    "has_contrast_marker",
    "predictions_opposite",
    "both_differ_from_rating",
]

summary = {
    "total_reviews": int(len(df)),
    "model_disagreements": int((~df["models_agree"]).sum()),
    "average_review_words": float(
        df["review_length_words"].mean()
    ),
    "median_review_words": float(
        df["review_length_words"].median()
    ),
    "flags": {
        column: {
            "count": int(df[column].sum()),
            "percentage": float(df[column].mean() * 100)
        }
        for column in flag_columns
    },
}

by_category = df.groupby("comparison_category").agg(
    count=("review_id", "size"),
    mean_words=("review_length_words", "mean"),
    median_words=("review_length_words", "median"),
    update_markers=("has_update_marker", "sum"),
    html_markup=("has_html_markup", "sum"),
    contrast_markers=("has_contrast_marker", "sum"),
    opposite_predictions=("predictions_opposite", "sum"),
)

# Export full report
df.to_csv(
    RESULTS / "error_analysis_full.csv",
    index=False
)

by_category.to_csv(
    RESULTS / "error_analysis_by_category.csv"
)

# Prioritize examples for manual investigation
priority = df[
    df["both_differ_from_rating"] |
    df["predictions_opposite"]
].copy()

priority = priority.sort_values(
    ["predictions_opposite", "review_length_words"],
    ascending=[False, False]
)

priority.head(100).to_csv(
    RESULTS / "error_analysis_priority_sample.csv",
    index=False
)

with open(
    RESULTS / "error_analysis_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=2)

print("\nDELL LLM EVALUATION FRAMEWORK")
print("=" * 50)
print("STRUCTURED ERROR ANALYSIS")
print("=" * 50)

print(f"Total reviews: {summary['total_reviews']}")
print(f"Model disagreements: {summary['model_disagreements']}")
print(
    f"Average review length: "
    f"{summary['average_review_words']:.1f} words"
)
print(
    f"Median review length: "
    f"{summary['median_review_words']:.1f} words"
)

print("\nPotential challenge indicators:")
for name, values in summary["flags"].items():
    print(
        f"{name}: {values['count']} "
        f"({values['percentage']:.2f}%)"
    )

print("\nSummary by comparison category:")
print(by_category.round(2).to_string())

print("\nError analysis files saved successfully.")
