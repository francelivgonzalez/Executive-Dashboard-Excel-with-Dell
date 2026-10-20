

\# Executive Dashboard — Excel with Dell



\### Catalyst Consulting | Customer Intelligence Platform



An interactive customer intelligence dashboard developed as an extension of the Excel with Dell case competition.



The application combines sentiment analysis, natural language processing, topic modeling, and interactive visualizations to transform customer feedback into actionable business insights.



\## Project Overview



Customer reviews contain valuable information about product performance, customer satisfaction, and recurring issues. However, manually interpreting thousands of reviews can be time-consuming.



This project develops an automated evaluation framework that analyzes customer feedback and presents recurring concerns through an executive-facing dashboard.



The application is organized around two audiences:



\*\*Executive Dashboard\*\*

\- Visualizes the most frequently discussed customer issues.

\- Uses topic modeling to identify recurring complaint themes.

\- Represents issue frequency through differently sized red ovals.

\- Highlights areas requiring further investigation.

\- Prioritizes concise, decision-oriented insights.



\*\*Analysts \& Engineers\*\*

\- Compares sentiment classification models.

\- Displays evaluation metrics and confusion matrices.

\- Investigates disagreements between models.

\- Explores individual customer reviews.

\- Supports more detailed technical analysis.



\## Dataset



The project uses a FusionTech customer-review dataset provided for the Excel with Dell case competition.



| Description | Reviews |

|---|---:|

| Original dataset | 4,132 |

| Duplicate reviews removed | 43 |

| Cleaned dataset | 4,087 |

| Initial evaluation sample | 150 |

| Full evaluation | 4,087 |



Sentiment reference labels were derived from customer star ratings.



These labels are proxies for customer sentiment rather than independently verified sentiment annotations.



The original datasets are not included in this repository.



\## Sentiment Analysis Models



\### VADER



VADER is a lexicon-based sentiment classifier that assigns sentiment scores using predefined word associations and linguistic heuristics.



\### RoBERTa



The framework uses:



`cardiffnlp/twitter-roberta-base-sentiment-latest`



RoBERTa is a transformer-based classifier that uses contextual language representations to predict negative, neutral, or positive sentiment.



GPU inference was performed using an NVIDIA Quadro T2000 with CUDA-enabled PyTorch.



\## Full-Dataset Evaluation Results



Both models were evaluated using the same 4,087 cleaned customer reviews.



| Metric | VADER | RoBERTa |

|---|---:|---:|

| Rating-label agreement | 73.23% | 77.64% |

| Macro F1 | 0.51 | 0.60 |

| Negative recall | 56% | 83% |

| Neutral recall | 6% | 17% |

| Positive recall | 89% | 83% |



RoBERTa demonstrated stronger overall agreement with the rating-derived labels and higher negative recall.



However, VADER achieved higher positive recall.



Both models struggled with neutral sentiment.



\## Model Comparison



Across 4,087 reviews:



\- Identical model predictions: 3,055

\- Different model predictions: 1,032

\- Model agreement rate: 74.75%

\- Cohen's kappa: 0.4903

\- Only RoBERTa matched the rating: 517 reviews

\- Only VADER matched the rating: 337 reviews

\- Neither model matched the rating: 577 reviews



The disagreement analysis identified several areas for further investigation, including technical complaints, mixed sentiment, long reviews, historical review updates, and rating-text mismatches.



\## Topic Modeling



The executive dashboard incorporates unsupervised topic modeling using TF-IDF and Non-negative Matrix Factorization (NMF).



Negative-rated customer reviews are analyzed to identify recurring discussion themes.



The dashboard visualizes these themes using red ovals:



\- Larger ovals indicate more frequently discussed topics.

\- Darker red indicates greater topic frequency.

\- Smaller ovals indicate less frequently discussed topics.



Oval size represents discussion frequency, not necessarily business severity.



The generated topics are exploratory and should be validated before being used to support business decisions.



\## Technology Stack



\*\*Programming\*\*

\- Python



\*\*Machine Learning and NLP\*\*

\- PyTorch

\- Hugging Face Transformers

\- RoBERTa

\- VADER

\- Scikit-learn

\- TF-IDF

\- NMF



\*\*Analysis\*\*

\- Pandas

\- Classification metrics

\- Confusion matrices

\- Model disagreement analysis



\*\*Application\*\*

\- Streamlit

\- Interactive dashboards

\- Review exploration

\- Executive visualizations



\*\*Development\*\*

\- Git

\- GitHub

\- Windows CMD

\- Python virtual environments



\## Project Structure



```text

dashboard/

&#x20;   app.py

&#x20;   pages/

&#x20;       1\_Analysts\_and\_Engineers.py



src/

&#x20;   prepare\_data.py

&#x20;   evaluate\_vader.py

&#x20;   evaluate\_vader\_full.py

&#x20;   evaluate\_transformer.py

&#x20;   analyze\_errors.py

&#x20;   compare\_models.py

&#x20;   analyze\_model\_disagreements.py

&#x20;   generate\_error\_report.py



data/

&#x20;   raw/

&#x20;   processed/



results/

docs/

README.md

requirements.txt

.gitignore

```



\## Running the Application



Clone the repository and navigate to the project directory.



Create a Python virtual environment:



```cmd

python -m venv .venv

```



Activate it on Windows:



```cmd

.venv\\Scripts\\activate

```



Install the required packages:



```cmd

python -m pip install -r requirements.txt

```



Supply the required review data and generate the evaluation outputs using the scripts in `src/`.



The dashboard requires generated analysis results in the `results/` directory.



Start the dashboard:



```cmd

python -m streamlit run dashboard\\app.py

```



The application runs locally in a web browser.



\## Evaluation Limitations



\- Star ratings may not accurately represent written sentiment.

\- The two sentiment models are primarily designed for English text.

\- Long customer reviews may be truncated during transformer inference.

\- Mixed sentiment and historical review updates can complicate classification.

\- Automatically discovered topics may overlap.

\- Topic frequency does not necessarily indicate business impact or severity.

\- The evaluation does not establish model accuracy against independently annotated sentiment labels.



\## Project Purpose



The primary objective is to demonstrate how machine learning and natural language processing can turn large volumes of customer feedback into understandable, decision-oriented information.



The project emphasizes the separation of executive decision support from detailed technical model evaluation.



\## Project Background



Developed as a portfolio extension of the Excel with Dell case competition.



The application and analysis represent independent project development and should not be interpreted as an official Dell Technologies product or endorsement.

