"""Benchmark several TF-IDF models on the two datasets and write the report.

Usage:
    python scripts/download_data.py
    python src/experiments.py

Writes reports/results.json, reports/results.md and figures in reports/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS = ROOT / "data", ROOT / "reports"
FIG = REPORTS / "figures"


def tfidf(**kw) -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                           min_df=2, max_df=0.9, sublinear_tf=True, **kw)


MODELS = {
    "Naive Bayes": lambda: Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
    "Logistic Regression": lambda: Pipeline([("tfidf", tfidf()),
                                             ("clf", LogisticRegression(C=10, max_iter=2000))]),
    "Linear SVM": lambda: Pipeline([("tfidf", tfidf()), ("clf", LinearSVC(C=1.0))]),
}


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, pos_label="fake"),
        "recall_fake": recall_score(y_true, y_pred, pos_label="fake"),
        "f1_fake": f1_score(y_true, y_pred, pos_label="fake"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def plot_confusion(y_true, y_pred, title: str, path: Path) -> list:
    labels = ["fake", "real"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Fake", "Real"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return cm.tolist()


def plot_top_words(model: Pipeline, path: Path, n: int = 15) -> dict:
    vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
    coef = model.named_steps["clf"].coef_[0]            # positive = "real" (classes sorted: fake, real)
    fake_idx, real_idx = np.argsort(coef)[:n], np.argsort(coef)[-n:]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].barh(vocab[fake_idx][::-1], -coef[fake_idx][::-1], color="#C0392B"); axes[0].set_title("Words pushing towards FAKE")
    axes[1].barh(vocab[real_idx], coef[real_idx], color="#1A7A8A"); axes[1].set_title("Words pushing towards REAL")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return {"fake": vocab[fake_idx].tolist(), "real": vocab[real_idx][::-1].tolist()}


def run(name, X_tr, y_tr, X_te, y_te, key) -> dict:
    print(f"\n=== {name} ===  train={len(X_tr)}  test={len(X_te)}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out = {"n_train": len(X_tr), "n_test": len(X_te), "models": {}}
    for m_name, make in MODELS.items():
        cv_f1 = cross_val_score(make(), X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
        model = make().fit(X_tr, y_tr)
        test = scores(y_te, model.predict(X_te))
        out["models"][m_name] = {"cv_f1_macro_mean": cv_f1.mean(), "cv_f1_macro_std": cv_f1.std(), "test": test}
        print(f"{m_name:20s} CV F1-macro {cv_f1.mean():.3f} ± {cv_f1.std():.3f} | "
              f"test acc {test['accuracy']:.3f}  F1(fake) {test['f1_fake']:.3f}  F1-macro {test['f1_macro']:.3f}")
    best = max(out["models"], key=lambda m: out["models"][m]["cv_f1_macro_mean"])  # chosen on CV, not on test
    best_model = MODELS[best]().fit(X_tr, y_tr)
    out["best_model"] = best
    out["confusion_matrix"] = plot_confusion(y_te, best_model.predict(X_te), f"{name} — {best} (test set)",
                                             FIG / f"confusion_{key}.png")
    lr = MODELS["Logistic Regression"]().fit(X_tr, y_tr)   # linear model -> interpretable weights
    out["top_words"] = plot_top_words(lr, FIG / f"top_words_{key}.png")
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    results = {}

    news = pd.read_csv(DATA / "fake_or_real_news.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(news["text"], news["label"], test_size=0.2,
                                              stratify=news["label"], random_state=SEED)
    results["fake_or_real_news"] = run("Fake or Real News (articles)", X_tr, y_tr, X_te, y_te, "articles")

    tr = pd.concat([pd.read_csv(DATA / "liar_train.csv"), pd.read_csv(DATA / "liar_valid.csv")])
    te = pd.read_csv(DATA / "liar_test.csv")
    results["liar"] = run("LIAR (short statements)", tr["text"], tr["label"], te["text"], te["label"], "liar")
    results["liar"]["majority_baseline_accuracy"] = float((te["label"] == tr["label"].mode()[0]).mean())

    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    lines = []
    for key, title in [("fake_or_real_news", "Fake or Real News"), ("liar", "LIAR")]:
        r = results[key]
        lines += [f"### {title} (train {r['n_train']:,} / test {r['n_test']:,})", "",
                  "| Model | CV F1-macro (5-fold) | Test accuracy | Test F1 (fake) | Test F1-macro |",
                  "|---|---|---|---|---|"]
        for m, v in r["models"].items():
            b = "**" if m == r["best_model"] else ""
            t = v["test"]
            lines.append(f"| {b}{m}{b} | {v['cv_f1_macro_mean']:.3f} ± {v['cv_f1_macro_std']:.3f} | "
                         f"{t['accuracy']:.3f} | {t['f1_fake']:.3f} | {t['f1_macro']:.3f} |")
        lines.append("")
    (REPORTS / "results.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
