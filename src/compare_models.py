
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

VADER_FILE = RESULTS / "vader_full_predictions.csv"
TRANSFORMER_FILE = RESULTS / "transformer_full_predictions.csv"

RESULTS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load model predictions
# --------------------------------------------------

print("Loading VADER and RoBERTa predictions...")

vader = pd.read_csv(VADER_FILE)
transformer = pd.read_csv(TRANSFORMER_FILE)

required_vader = {
    "review_id", "text", "rating_sentiment",
    "predicted_sentiment", "vader_compound"
}
required_transformer = {
    "review_id", "rating_sentiment",
    "predicted_sentiment", "transformer_confidence"
}

if not required_vader.issubset(vader.columns):
    raise ValueError(
        f"Missing VADER columns: {required_vader - set(vader.columns)}"
    )

if not required_transformer.issubset(transformer.columns):
    raise ValueError(
        "Missing RoBERTa columns: "
        f"{required_transformer - set(transformer.columns)}"
    )

# Ensure IDs uniquely identify reviews
for name, frame in [("VADER", vader), ("RoBERTa", transformer)]:
    if frame["review_id"].isna().any():
        raise ValueError(f"{name} contains missing review IDs")

    if frame["review_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate review IDs")

# --------------------------------------------------
# 3. Merge predictions by review ID
# --------------------------------------------------

vader = vader[
    [
        "review_id",
        "text",
        "rating_sentiment",
        "predicted_sentiment",
        "vader_compound",
    ]
].rename(columns={
    "rating_sentiment": "vader_rating_sentiment",
    "predicted_sentiment": "vader_prediction",
})

transformer = transformer[
    [
        "review_id",
        "rating_sentiment",
        "predicted_sentiment",
        "transformer_confidence",
    ]
].rename(columns={
    "rating_sentiment": "transformer_rating_sentiment",
    "predicted_sentiment": "transformer_prediction",
})

df = vader.merge(
    transformer,
    on="review_id",
    how="outer",
    validate="one_to_one",
    indicator=True,
)

if not (df["_merge"] == "both").all():
    raise ValueError("Review IDs do not match across both models")

df = df.drop(columns=["_merge"])

if not (
    df["vader_rating_sentiment"]
    == df["transformer_rating_sentiment"]
).all():
    raise ValueError("Rating labels differ between model datasets")

df = df.rename(columns={
    "vader_rating_sentiment": "rating_sentiment"
}).drop(columns=["transformer_rating_sentiment"])

for column in [
    "rating_sentiment",
    "vader_prediction",
    "transformer_prediction",
]:
    df[column] = df[column].astype(str).str.lower().str.strip()

labels = {"negative", "neutral", "positive"}

for column in [
    "rating_sentiment",
    "vader_prediction",
    "transformer_prediction",
]:
    if not df[column].isin(labels).all():
        raise ValueError(f"Unexpected sentiment label in {column}")

print(f"Matched reviews: {len(df)}")

# --------------------------------------------------
# 4. Compare model predictions
# --------------------------------------------------

df["models_agree"] = (
    df["vader_prediction"] == df["transformer_prediction"]
)

df["vader_matches_rating"] = (
    df["vader_prediction"] == df["rating_sentiment"]
)

df["transformer_matches_rating"] = (
    df["transformer_prediction"] == df["rating_sentiment"]
)

df["comparison_category"] = "both_differ_from_rating"

df.loc[
    df["vader_matches_rating"] &
    df["transformer_matches_rating"],
    "comparison_category"
] = "both_match_rating"

df.loc[
    ~df["vader_matches_rating"] &
    df["transformer_matches_rating"],
    "comparison_category"
] = "only_transformer_matches"

df.loc[
    df["vader_matches_rating"] &
    ~df["transformer_matches_rating"],
    "comparison_category"
] = "only_vader_matches"

# --------------------------------------------------
# 5. Calculate metrics
# --------------------------------------------------

total = len(df)

if total == 0:
    raise ValueError("No matching reviews available")

agreement_count = int(df["models_agree"].sum())
disagreement_count = total - agreement_count

vader_accuracy = accuracy_score(
    df["rating_sentiment"],
    df["vader_prediction"]
)

transformer_accuracy = accuracy_score(
    df["rating_sentiment"],
    df["transformer_prediction"]
)

kappa = cohen_kappa_score(
    df["vader_prediction"],
    df["transformer_prediction"],
    labels=["negative", "neutral", "positive"],
)

category_counts = df["comparison_category"].value_counts()

negative = df[df["rating_sentiment"] == "negative"].copy()

vader_negative_to_positive = int(
    (negative["vader_prediction"] == "positive").sum()
)

transformer_negative_to_positive = int(
    (negative["transformer_prediction"] == "positive").sum()
)

# --------------------------------------------------
# 6. Print comparison report
# --------------------------------------------------

print("\nDELL LLM EVALUATION FRAMEWORK")
print("=" * 50)
print("MODEL COMPARISON: VADER vs RoBERTa")
print("=" * 50)

print(f"\nTotal reviews: {total}")
print(f"Models agree: {agreement_count}")
print(f"Models disagree: {disagreement_count}")
print(f"Model agreement rate: {agreement_count / total:.2%}")
print(f"Cohen's kappa: {kappa:.4f}")

print("\nAgreement with star-rating sentiment labels:")
print(f"VADER: {vader_accuracy:.2%}")
print(f"RoBERTa: {transformer_accuracy:.2%}")

print("\nComparison categories:")

for category in [
    "both_match_rating",
    "only_transformer_matches",
    "only_vader_matches",
    "both_differ_from_rating",
]:
    count = int(category_counts.get(category, 0))
    print(f"{category}: {count} ({count / total:.2%})")

print("\nNegative-rated reviews predicted positive:")
print(f"VADER: {vader_negative_to_positive}")
print(f"RoBERTa: {transformer_negative_to_positive}")

# --------------------------------------------------
# 7. Save analysis files
# --------------------------------------------------

df.to_csv(
    RESULTS / "model_comparison.csv",
    index=False
)

df[~df["models_agree"]].to_csv(
    RESULTS / "model_disagreements.csv",
    index=False
)

df[df["comparison_category"] == "only_transformer_matches"].to_csv(
    RESULTS / "transformer_only_matches.csv",
    index=False
)

df[df["comparison_category"] == "only_vader_matches"].to_csv(
    RESULTS / "vader_only_matches.csv",
    index=False
)

df[
    (df["rating_sentiment"] == "negative") &
    (df["vader_prediction"] == "positive")
].to_csv(
    RESULTS / "vader_negative_as_positive.csv",
    index=False
)

df[
    (df["rating_sentiment"] == "negative") &
    (df["transformer_prediction"] == "positive")
].to_csv(
    RESULTS / "transformer_negative_as_positive.csv",
    index=False
)

# --------------------------------------------------
# 8. Save summary JSON
# --------------------------------------------------

summary = {
    "total_reviews": total,
    "label_source": "star_rating_proxy",
    "model_agreement_count": agreement_count,
    "model_disagreement_count": disagreement_count,
    "model_agreement_rate": agreement_count / total,
    "cohens_kappa": float(kappa),
    "vader_rating_agreement": float(vader_accuracy),
    "transformer_rating_agreement": float(transformer_accuracy),
    "comparison_categories": {
        category: int(category_counts.get(category, 0))
        for category in [
            "both_match_rating",
            "only_transformer_matches",
            "only_vader_matches",
            "both_differ_from_rating",
        ]
    },
    "negative_rated_predicted_positive": {
        "vader": vader_negative_to_positive,
        "transformer": transformer_negative_to_positive,
    },
}

with open(
    RESULTS / "model_comparison_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=2)

print("\nComparison files saved successfully!")
