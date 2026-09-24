# Fake News Detector

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)

A lightweight text classifier that labels a news headline or article as **fake** or **real**. It uses a
classic, interpretable NLP pipeline: **TF-IDF features** (words and word pairs) fed into a
**logistic regression** model, built with scikit-learn. It comes with command-line tools to train and
predict, and a unit test.

## How it works

```
text ──► TF-IDF vectoriser ──► logistic regression ──► "fake" / "real" + confidence
         (lowercase, English stop words removed,
          unigrams + bigrams)
```

1. `src/train.py` reads a CSV of labelled texts and holds out 25% for testing (stratified split).
2. It fits the pipeline, prints a precision / recall / F1 report on the held-out set and saves the model
   to `model.joblib`.
3. `src/predict.py` loads the model and prints the predicted label with its probability.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Train**

```bash
python src/train.py --data data/sample_news.csv
```

Options: `--out` (model path, default `model.joblib`) and `--test-size` (default `0.25`).

**Predict**

```bash
python src/predict.py "SHOCKING: miracle pill cures everything overnight"
# FAKE (xx% confidence)
```

**Test**

```bash
pytest
```

## Data

The model expects a CSV with two columns:

| Column | Content |
|---|---|
| `text` | headline or article text |
| `label` | `fake` or `real` |

`data/sample_news.csv` is a tiny starter set (24 examples) so that everything runs out of the box. It is
too small to give meaningful scores. For real results, train on a larger public dataset with the same
two columns, for example **Fake and Real News** or **LIAR** (both on Kaggle).

## Project structure

```
src/train.py          training script (TF-IDF + logistic regression pipeline)
src/predict.py        command-line prediction
data/sample_news.csv  starter dataset
tests/                unit test of the pipeline
```

## Possible next steps

- Train and report scores on a full public dataset
- Compare with a fine-tuned transformer (e.g. DistilBERT)
- Show which words drive each prediction, using the model's coefficients

## Author

**Assim Ayoub**, [@A2A-o4](https://github.com/A2A-o4)
