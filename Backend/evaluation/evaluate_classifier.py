"""
Evaluates classifier.py against a hand-labeled ground truth CSV.

Expected CSV format (evaluation/evaluation_70.csv):
    complaint,actual_category
    "The AC in C block is not cooling",Infrastructure

Run from the Backend/ folder (same place you run main.py and seed.py):
    python evaluation/evaluate_classifier.py

Outputs:
    - Overall accuracy
    - Per-category accuracy (so you can see which categories the
      classifier is weak on, not just one aggregate number)
    - A confusion table (actual category -> what it was predicted as)
    - The full list of misclassified rows, with the classifier's
      reasoning field, so you can see WHY it got each one wrong —
      this is what actually helps you improve classifier.py afterward
    - Writes evaluation/evaluation_results.csv with every row's
      prediction, for anything you want to inspect further yourself
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Make the parent folder's modules (classifier.py) importable when this
# script is run from inside evaluation/
sys.path.insert(0, str(Path(__file__).parent.parent))

from classifier import classify_complaint  # noqa: E402

CSV_PATH = Path(__file__).parent / "holdout_set.csv"
OUTPUT_PATH = Path(__file__).parent / "evaluation_results.csv"


def load_ground_truth(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("Put your evaluation CSV at that path, or edit CSV_PATH "
              "in this script.")
        sys.exit(1)

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "complaint" not in reader.fieldnames or \
           "actual_category" not in reader.fieldnames:
            print(f"ERROR: CSV must have columns 'complaint' and "
                  f"'actual_category'. Found: {reader.fieldnames}")
            sys.exit(1)
        for row in reader:
            rows.append(row)
    return rows


def main():
    rows = load_ground_truth(CSV_PATH)
    print(f"Loaded {len(rows)} labeled complaints from {CSV_PATH.name}\n")

    results = []
    correct = 0
    per_category_total = defaultdict(int)
    per_category_correct = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))  # actual -> predicted -> count

    for row in rows:
        text = row["complaint"]
        actual = row["actual_category"].strip()

        prediction = classify_complaint(text)
        # a restricted (safety) complaint has category "Safety" set
        # by the classifier regardless of what CATEGORY_KEYWORDS matched
        predicted = "Safety" if prediction.is_restricted else prediction.category

        is_correct = (predicted == actual)
        if is_correct:
            correct += 1
        per_category_total[actual] += 1
        if is_correct:
            per_category_correct[actual] += 1
        confusion[actual][predicted] += 1

        results.append({
            "complaint": text,
            "actual_category": actual,
            "predicted_category": predicted,
            "correct": is_correct,
            "priority": prediction.priority,
            "matched_keywords": prediction.reasoning,
        })

    # --- Overall accuracy ---
    total = len(rows)
    accuracy = correct / total if total else 0
    print(f"=== Overall accuracy: {correct}/{total} = {accuracy:.1%} ===\n")

    # --- Per-category accuracy ---
    print("--- Per-category accuracy ---")
    for cat in sorted(per_category_total):
        c = per_category_correct[cat]
        t = per_category_total[cat]
        print(f"  {cat:20s} {c:3d}/{t:<3d}  ({c/t:.1%})")

    # --- Confusion table ---
    print("\n--- Confusion (actual -> predicted counts, only errors) ---")
    any_confusion = False
    for actual in sorted(confusion):
        for predicted, count in sorted(confusion[actual].items()):
            if predicted != actual:
                any_confusion = True
                print(f"  {actual:20s} misclassified as {predicted:20s} "
                      f"x{count}")
    if not any_confusion:
        print("  (no misclassifications)")

    # --- Misclassified examples with reasoning ---
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n--- {len(wrong)} misclassified complaints (with reasoning) ---")
        for r in wrong:
            print(f"\n  Text: {r['complaint']}")
            print(f"  Actual: {r['actual_category']}  |  "
                  f"Predicted: {r['predicted_category']}")
            print(f"  Reasoning: {r['matched_keywords']}")

    # --- Write full results to CSV ---
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "complaint", "actual_category", "predicted_category",
            "correct", "priority", "matched_keywords"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nFull results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
