# Executive Dashboard - Excel with Dell
> Turning thousands of customer reviews into decision-ready themes- with NLP sentiment evaluation, TF-IDF + NMF topic modeling, and an interactive Streamlit dashboard.

An interactive customer intelligence dashboard built as an extension of the Excel with Dell case competition (Catalyst Consulting), combining sentiment classification, topic modeling, and review exploration to turn customer feedback into actionable insights.

## What This Project Does

Manually reading thousands of customer reviews is slow and easy to miss patterns in. This project builds an automated evaluation pipeline and interactive Streamlit app tailored to two audiences:
1. **Executive Dashboard:** Visualizes recurring customer complaint themes using TF-IDF and Non-negative Matrix Factorization (NMF), displaying topic frequency through differently sized red ovals so leaders can spot high-frequency issues at a glance.
2. **Analysts & Engineers:** Compares sentiment models, shows evaluation metrics, confusion matrices, and model disagreements, and lets users inspect individual customer reviews.

## What I'm Focusing On

- **Sentiment Model Evaluation:** Comparing lexicon-based **VADER** against transformer-based **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) across 4,087 cleaned FusionTech reviews.
- **Disagreement & Error Analysis:** Inspecting the 1,032 reviews where models diverged—including technical complaints, mixed sentiment, long reviews, historical updates, and rating-text mismatches.
- **Exploratory Topic Modeling:** Surfacing complaint themes in negative-rated reviews with TF-IDF + NMF and red-oval frequency sizing without conflating mention count with business severity.
- **Separation of Views:** Keeping high-level executive summaries distinct from deep technical diagnostics and raw review exploration.

## How the Pipeline & App Are Set Up

- **Dataset & Preprocessing (`src/prepare_data.py`):** Cleans the 4,132-review FusionTech competition dataset down to 4,087 reviews after removing 43 duplicates, using star ratings as proxy sentiment labels (raw datasets excluded from the repo).
- **Model Evaluation & Comparison (`src/`):** Evaluates VADER and RoBERTa (run on an NVIDIA Quadro T2000 with CUDA PyTorch), computes rating-label agreement, macro F1, recall, and Cohen's kappa (0.4903), and generates disagreement/error reports into `results/`.
- **Streamlit Dashboard (`dashboard/`):** Loads generated artifacts from `results/` to serve the main executive overview and the `1_Analysts_and_Engineers.py` deep-dive view.

## How to Run

1. Create and activate a virtual environment, then install dependencies:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```
2. Place the dataset under `data/raw/`, run the preprocessing and evaluation scripts in `src/` to populate `results/`, and launch the Streamlit app:
   ```cmd
   python -m streamlit run dashboard\app.py
   ```

## Tech Stack & Key Results

- **Stack:** Python, PyTorch, Hugging Face Transformers (`cardiffnlp/twitter-roberta-base-sentiment-latest`), VADER, Scikit-learn (TF-IDF, NMF), Pandas, Streamlit, Git/GitHub
- **4,087-Review Benchmark:** RoBERTa reached **77.64%** rating-label agreement and **0.60** macro F1 (83% negative recall) vs. VADER at **73.23%** agreement and **0.51** macro F1 (89% positive recall), with both models struggling on neutral reviews.
- **Limitations & Scope:** Star ratings are proxy labels, models target English text, long reviews may truncate in transformer inference, topics are exploratory, and this is an independent extension rather than an official Dell Technologies product.

## Project Structure

```text
executive-dashboard-excel-with-dell/
├── README.md
├── requirements.txt
├── .gitignore
├── dashboard/
│   ├── app.py
│   └── pages/
│       └── 1_Analysts_and_Engineers.py
├── src/
│   ├── prepare_data.py
│   ├── evaluate_vader.py
│   ├── evaluate_vader_full.py
│   ├── evaluate_transformer.py
│   ├── analyze_errors.py
│   ├── compare_models.py
│   ├── analyze_model_disagreements.py
│   └── generate_error_report.py
├── data/
│   ├── raw/
│   └── processed/
├── results/
└── docs/
```
