
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from transformers import pipeline

# Project paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "cleaned_reviews.csv"
OUTPUT_DIR = ROOT / "results"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
print("Loading FusionTech customer reviews...")

df = pd.read_csv(INPUT_FILE)

required = {"review_id", "text", "rating_sentiment"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=["text", "rating_sentiment"]).copy()
df["text"] = df["text"].astype(str)

print(f"Reviews loaded: {len(df)}")

# Initialize transformer
print("\nLoading RoBERTa transformer...")

classifier = pipeline(
    task="text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0,
)

print(f"Model device: {classifier.device}")

# Run batch inference
print("\nEvaluating reviews...")

BATCH_SIZE = 4
predicted_labels = []
confidence_scores = []

texts = df["text"].tolist()

for start in range(0, len(texts), BATCH_SIZE):
    batch = texts[start:start + BATCH_SIZE]

    results = classifier(
        batch,
        truncation=True,
        max_length=512,
        batch_size=BATCH_SIZE,
    )

    for result in results:
        predicted_labels.append(result["label"].lower())
        confidence_scores.append(float(result["score"]))

    processed = min(start + BATCH_SIZE, len(texts))

    if processed % 100 < BATCH_SIZE or processed == len(texts):
        print(f"Processed {processed}/{len(texts)} reviews")

df["predicted_sentiment"] = predicted_labels
df["transformer_confidence"] = confidence_scores

# Evaluate
labels = ["negative", "neutral", "positive"]

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

print("\nDELL LLM EVALUATION FRAMEWORK")
print("=" * 45)
print("Model: RoBERTa Transformer")
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

# Save predictions
df.to_csv(
OUTPUT_DIR / "transformer_full_predictions.csv",
    index=False,
)

# Save evaluation metrics
metrics = {
    "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "sample_size": len(df),
    "label_source": "star_rating_proxy",
    "accuracy": float(accuracy),
    "classification_report": report,
    "labels": labels,
    "confusion_matrix": matrix.tolist(),
}

with open(
    OUTPUT_DIR / "transformer_full_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=2)

print("\nTransformer evaluation saved successfully.")
