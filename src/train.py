"""Train a fake-news classifier (TF-IDF + Logistic Regression).

Usage:
    python src/train.py --data data/sample_news.csv --out model.joblib

The CSV needs two columns: `text` (the article or headline) and
`label` ("fake" or "real").
"""
import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sample_news.csv")
    parser.add_argument("--out", default="model.joblib")
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["text", "label"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size, random_state=42, stratify=df["label"]
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test), zero_division=0))
    joblib.dump(model, args.out)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
