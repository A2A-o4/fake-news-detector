# Fake News Detector

A small machine-learning project that classifies news headlines or articles as **fake** or **real**, using TF-IDF features and logistic regression (scikit-learn).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --data data/sample_news.csv
```

This prints a precision/recall report and saves the model to `model.joblib`.

## Predict

```bash
python src/predict.py "SHOCKING: miracle pill cures everything overnight"
```

## Tests

```bash
pytest
```

## Better data

`data/sample_news.csv` is a tiny starter set so the code runs out of the box. For real results, swap in a larger labeled dataset (for example the public "Fake and Real News" or LIAR datasets on Kaggle) with the same `text` and `label` columns.
