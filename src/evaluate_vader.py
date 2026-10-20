
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "evaluation_sample.csv"
OUTPUT_DIR = ROOT / "results"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required = {"review_id", "text", "rating_sentiment"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["text", "rating_sentiment"]).copy()

analyzer = SentimentIntensityAnalyzer()

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

labels = ["negative", "neutral", "positive"]

accuracy = accuracy_score(
    df["rating_sentiment"],
    df["predicted_sentiment"]
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
print("Model: VADER")
print(f"Reviews evaluated: {len(df)}")
print(f"Agreement with star-rating labels: {accuracy:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        df["rating_sentiment"],
        df["predicted_sentiment"],
        labels=labels,
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(pd.DataFrame(matrix, index=labels, columns=labels))

df.to_csv(
    OUTPUT_DIR / "vader_predictions.csv",
    index=False,
)

metrics = {
    "model": "VADER",
    "sample_size": len(df),
    "label_source": "star_rating_proxy",
    "accuracy": float(accuracy),
    "classification_report": report,
    "labels": labels,
    "confusion_matrix": matrix.tolist(),
}

with open(OUTPUT_DIR / "vader_metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\nEvaluation results saved successfully.")
