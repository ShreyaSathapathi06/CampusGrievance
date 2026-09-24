"""
Trains a category classifier on SBERT embeddings of your 114 hand-labeled
complaints (evaluation_70.csv + holdout_set.csv combined), replacing the
keyword-based classify_category() with a trained model that generalizes
to paraphrased language the keyword approach structurally could not.

WHY THIS APPROACH:
The keyword classifier scored 43.2% on genuinely new phrasing (the
holdout set) after scoring 100% on data it had been repeatedly patched
against — proof the keyword approach was memorizing specific words, not
learning what a category actually means. SBERT embeddings place semantically
similar sentences near each other in vector space regardless of shared
vocabulary ("ants on the serving counter" lands near "insects in the food"
even sharing zero words), so a classifier trained on top of those
embeddings inherits that generalization instead of needing every possible
phrasing spelled out as a keyword.

WHY CROSS-VALIDATION, NOT A SINGLE TRAIN/TEST SPLIT:
114 examples across 9 categories is roughly 12-14 per category. A single
80/20 split would put only 2-3 examples of some categories in the test
set — one lucky or unlucky split could make the reported accuracy swing
by 10+ points for reasons that have nothing to do with real model quality.
5-fold stratified cross-validation trains and tests 5 times on different
splits and averages the result, which is a far more trustworthy estimate
at this data size. This is standard practice for small datasets, not a
shortcut.

USAGE (run from Backend/ folder):
    python ml/train_classifier.py

This will:
  1. Load and combine evaluation/evaluation_70.csv and
     evaluation/holdout_set.csv
  2. Embed every complaint with SBERT (all-MiniLM-L6-v2)
  3. Run 5-fold stratified cross-validation, printing the REAL,
     trustworthy accuracy estimate plus a per-category report
  4. Retrain on ALL 114 examples (best use of limited data for the
     model you'll actually deploy) and save it to ml/model/

IMPORTANT: the cross-validation accuracy printed here is your honest,
reportable number. Do not evaluate this model against evaluation_70.csv
or holdout_set.csv again afterward and call that a real test — both
were used to TRAIN the final saved model, so testing against them again
would repeat the exact contamination mistake made with the keyword
classifier. If you want a truly fresh number later, write a THIRD set
of complaints you've never shown this model.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

BACKEND_DIR = Path(__file__).parent.parent
EVAL_DIR = BACKEND_DIR / "evaluation"
MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

DATA_FILES = [
    EVAL_DIR / "evaluation_70.csv",
    EVAL_DIR / "holdout_set.csv",
]

MODEL_NAME = "all-MiniLM-L6-v2"


def load_data() -> pd.DataFrame:
    frames = []
    for path in DATA_FILES:
        if not path.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        print("ERROR: no data files found. Expected files at:")
        for p in DATA_FILES:
            print(f"  {p}")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset="complaint")
    after = len(combined)
    if before != after:
        print(f"Dropped {before - after} exact-duplicate complaint(s) "
              f"found in both files.")
    return combined


def main():
    df = load_data()
    print(f"Loaded {len(df)} labeled complaints across "
          f"{df['actual_category'].nunique()} categories.\n")

    counts = df["actual_category"].value_counts()
    print("--- Examples per category ---")
    for cat, n in counts.items():
        flag = "  <- fewer than 5, CV results for this category will be noisy" if n < 5 else ""
        print(f"  {cat:20s} {n}{flag}")
    print()

    print(f"Loading {MODEL_NAME} and embedding {len(df)} complaints...")
    model = SentenceTransformer(MODEL_NAME)
    X = model.encode(df["complaint"].tolist(), convert_to_numpy=True,
                      show_progress_bar=True)

    le = LabelEncoder()
    y = le.fit_transform(df["actual_category"])

    # 5-fold stratified CV — the honest accuracy estimate
    n_splits = min(5, counts.min())  # can't have more folds than the
                                       # smallest class has examples
    if n_splits < 5:
        print(f"\nNOTE: smallest category has only {counts.min()} "
              f"example(s), so using {n_splits}-fold CV instead of the "
              f"usual 5-fold. Fewer folds = less reliable estimate — "
              f"more labeled data for small categories would help.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    print(f"\nRunning {n_splits}-fold stratified cross-validation...")
    y_pred = cross_val_predict(clf, X, y, cv=skf)

    accuracy = (y_pred == y).mean()
    print(f"\n=== Cross-validated accuracy: {accuracy:.1%} "
          f"(this is your real, reportable number) ===\n")

    print("--- Per-category precision / recall / f1 ---")
    print(classification_report(y, y_pred, target_names=le.classes_,
                                 zero_division=0))

    print("--- Confusion matrix (rows=actual, columns=predicted) ---")
    cm = confusion_matrix(y, y_pred)
    labels = le.classes_
    header = "                      " + "".join(f"{l[:8]:>10s}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"{labels[i]:20s}  " + "".join(f"{v:>10d}" for v in row))

    # Retrain on ALL data for the model that actually gets deployed —
    # cross-validation already told us how good this approach is; now
    # use every labeled example available for the real thing.
    print("\nRetraining final model on all data...")
    final_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    final_clf.fit(X, y)

    joblib.dump(final_clf, MODEL_DIR / "category_classifier.joblib")
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
    with open(MODEL_DIR / "model_name.txt", "w") as f:
        f.write(MODEL_NAME)

    print(f"Saved trained model to {MODEL_DIR}/")
    print("\nDo not re-evaluate this model against evaluation_70.csv or "
          "holdout_set.csv — both were used to train it. The "
          f"cross-validated {accuracy:.1%} above is your honest number.")


if __name__ == "__main__":
    main()
