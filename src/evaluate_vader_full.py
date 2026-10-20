
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "cleaned_reviews.csv"
OUTPUT_DIR = ROOT / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

print("Loading FusionTech customer reviews...")

df = pd.read_csv(INPUT_FILE)

required = {"review_id", "text", "rating_sentiment"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["text", "rating_sentiment"]).copy()
df["text"] = df["text"].astype(str)
df["rating_sentiment"] = (
    df["rating_sentiment"].astype(str).str.lower().str.strip()
)

labels = ["negative", "neutral", "positive"]

df = df[df["rating_sentiment"].isin(labels)].copy()

print(f"Reviews loaded: {len(df)}")

# --------------------------------------------------
# 3. Initialize VADER
# --------------------------------------------------

print("\nInitializing VADER sentiment analyzer...")

analyzer = SentimentIntensityAnalyzer()

# --------------------------------------------------
# 4. Perform sentiment analysis
# --------------------------------------------------

print("\nEvaluating reviews...")

def predict_sentiment(text):
    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return pd.Series({
        "predicted_sentiment": label,
        "vader_compound": compound,
    })

predictions = df["text"].apply(predict_sentiment)

df = pd.concat([df, predictions], axis=1)

print(f"Processed {len(df)}/{len(df)} reviews")

# --------------------------------------------------
# 5. Calculate evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(
    df["rating_sentiment"],
    df["predicted_sentiment"],
)

report = classification_report(
    df["rating_sentiment"],
    df["predicted_sentiment"],
    labels=labels,
    output_dict=True,
    zero_division=0,
)

matrix = confusion_matrix(
    df["rating_sentiment"],
    df["predicted_sentiment"],
    labels=labels,
)

# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\nDELL LLM EVALUATION FRAMEWORK")
print("=" * 45)

print("Model: VADER")
print(f"Reviews evaluated: {len(df)}")
print(f"Agreement with rating labels: {accuracy:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        df["rating_sentiment"],
        df["predicted_sentiment"],
        labels=labels,
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print(
    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )
)

# --------------------------------------------------
# 7. Save predictions
# --------------------------------------------------

predictions_file = OUTPUT_DIR / "vader_full_predictions.csv"

df.to_csv(
    predictions_file,
    index=False,
)

# --------------------------------------------------
# 8. Save evaluation metrics
# --------------------------------------------------

metrics = {
    "model": "VADER",
    "dataset": "cleaned_reviews.csv",
    "sample_size": len(df),
    "label_source": "star_rating_proxy",
    "accuracy": float(accuracy),
    "macro_f1": float(report["macro avg"]["f1-score"]),
    "weighted_f1": float(report["weighted avg"]["f1-score"]),
    "negative_recall": float(report["negative"]["recall"]),
    "neutral_recall": float(report["neutral"]["recall"]),
    "positive_recall": float(report["positive"]["recall"]),
    "classification_report": report,
    "labels": labels,
    "confusion_matrix": matrix.tolist(),
}

metrics_file = OUTPUT_DIR / "vader_full_metrics.json"

with open(metrics_file, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

# --------------------------------------------------
# 9. Completion message
# --------------------------------------------------

print("\nFull VADER evaluation completed successfully!")
print(f"Predictions saved to: {predictions_file}")
print(f"Metrics saved to: {metrics_file}")
