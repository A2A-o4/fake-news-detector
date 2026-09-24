"""Classify a piece of text as fake or real news.

Usage:
    python src/predict.py "Scientists confirm the moon is made of cheese"
"""
import argparse

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Headline or article text to classify")
    parser.add_argument("--model", default="model.joblib")
    args = parser.parse_args()

    model = joblib.load(args.model)
    label = model.predict([args.text])[0]
    confidence = max(model.predict_proba([args.text])[0])
    print(f"{label.upper()} ({confidence:.0%} confidence)")


if __name__ == "__main__":
    main()
