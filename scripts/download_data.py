"""Download the two public datasets and convert them to `text,label` CSV files.

Usage:
    python scripts/download_data.py

Outputs (in data/):
    fake_or_real_news.csv   6,335 full news articles (McIntire), cleaned and de-duplicated
    liar_train.csv / liar_valid.csv / liar_test.csv
                            12,836 short political statements (LIAR, Wang 2017),
                            official splits, mapped to a binary fake / real label
"""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
RAW = DATA / "raw"
FAKE_OR_REAL_URL = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
LIAR_URL = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/{split}.tsv"

# LIAR has 6 truthfulness levels; the usual binary mapping groups the three lowest as fake.
LIAR_BINARY = {
    "pants-fire": "fake", "false": "fake", "barely-true": "fake",
    "half-true": "real", "mostly-true": "real", "true": "real",
}


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"Downloading {url}")
        urlretrieve(url, dest)
    return dest


def build_fake_or_real() -> None:
    df = pd.read_csv(fetch(FAKE_OR_REAL_URL, RAW / "fake_or_real_news.csv"))
    body_empty = df["text"].fillna("").str.strip().str.len() < 40
    df = df[~body_empty]
    n_before = len(df)
    # same article body can appear under several titles: de-duplicate on the body so that
    # no article ends up on both sides of the train/test split
    df = df.drop_duplicates(subset="text")
    df["text"] = (df["title"].fillna("") + ". " + df["text"]).str.strip()
    df["label"] = df["label"].str.lower()
    df[["text", "label"]].to_csv(DATA / "fake_or_real_news.csv", index=False)
    print(f"fake_or_real_news.csv: {len(df)} articles "
          f"({body_empty.sum()} near-empty and {n_before - len(df)} duplicates removed)")


def build_liar() -> None:
    for split in ("train", "valid", "test"):
        raw = pd.read_csv(fetch(LIAR_URL.format(split=split), RAW / f"liar_{split}.tsv"),
                          sep="\t", header=None, quoting=3)
        out = pd.DataFrame({"text": raw[2], "label": raw[1].map(LIAR_BINARY), "label_6": raw[1]})
        out.dropna().to_csv(DATA / f"liar_{split}.csv", index=False)
        print(f"liar_{split}.csv: {len(out)} statements")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    build_fake_or_real()
    build_liar()
