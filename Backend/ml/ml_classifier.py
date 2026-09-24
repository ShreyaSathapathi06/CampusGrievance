"""
Loads the trained SBERT + logistic regression model (produced by
train_classifier.py) and classifies new complaints.

Replaces classify_category() in classifier.py for anything that reaches
this stage — i.e. anything that did NOT trip the keyword-based safety
check, which still runs first and separately (see classifier.py's
docstring for why that stays keyword-based on purpose).

Unlike the keyword classifier's `matched_keywords` reasoning, this
returns a confidence score and the runner-up categories with their
probabilities — a different but equally honest way to show *why* the
model decided what it decided, and importantly, how SURE it was. A
60%-confidence prediction with a close second place is a genuinely
different situation from a 98%-confidence one, and the old keyword
system had no equivalent way to express that.
"""
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
import joblib

MODEL_DIR = Path(__file__).parent / "model"

_sbert_model = None
_classifier = None
_label_encoder = None


def _ensure_loaded():
    global _sbert_model, _classifier, _label_encoder
    if _classifier is not None:
        return

    clf_path = MODEL_DIR / "category_classifier.joblib"
    le_path = MODEL_DIR / "label_encoder.joblib"
    name_path = MODEL_DIR / "model_name.txt"

    if not clf_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {clf_path}. Run "
            "'python ml/train_classifier.py' from the Backend/ folder "
            "first — it needs evaluation/evaluation_70.csv and "
            "evaluation/holdout_set.csv to exist."
        )

    model_name = name_path.read_text().strip() if name_path.exists() \
        else "all-MiniLM-L6-v2"

    _sbert_model = SentenceTransformer(model_name)
    _classifier = joblib.load(clf_path)
    _label_encoder = joblib.load(le_path)


@dataclass
class MLClassificationResult:
    category: str
    confidence: float
    runner_up: list[tuple[str, float]]  # [(category, probability), ...]


def classify_category_ml(text: str, top_n_runner_up: int = 2) -> MLClassificationResult:
    """
    Classifies a single complaint's category using the trained model.
    Loads the model on first call and reuses it after (loading SBERT
    takes a moment; do it once, not per-request, in a real server —
    main.py should call _ensure_loaded() once at startup, not rely on
    lazy loading on the first live request).
    """
    _ensure_loaded()

    embedding = _sbert_model.encode([text], convert_to_numpy=True)
    probabilities = _classifier.predict_proba(embedding)[0]

    order = np.argsort(probabilities)[::-1]  # descending confidence
    top_idx = order[0]
    category = str(_label_encoder.inverse_transform([top_idx])[0])
    confidence = float(probabilities[top_idx])

    runner_up = [
        (str(_label_encoder.inverse_transform([idx])[0]), float(probabilities[idx]))
        for idx in order[1:1 + top_n_runner_up]
    ]

    return MLClassificationResult(category=category, confidence=confidence,
                                    runner_up=runner_up)
SAFETY_MIN_MARGIN = 0.05

def classify_category_guarded(text: str) -> MLClassificationResult:
    """
    Like classify_category_ml, but the model may not send a complaint to Safety on a
    close call. The keyword safety check runs before this, so it is the only thing that
    should decide "Safety"; a thin ML-only Safety is a guess. "Other" is a fallback
    rather than evidence, so a specific category that is nearly as likely wins over it.
    """
    result = classify_category_ml(text, top_n_runner_up=3)
    if result.category != "Safety" or not result.runner_up:
        return result
    if result.confidence - result.runner_up[0][1] >= SAFETY_MIN_MARGIN:
        return result                                    # clear win: leave it alone

    candidates = [(label, prob) for label, prob in result.runner_up if label != "Safety"]
    if not candidates:
        return result
    chosen = candidates[0]
    specific = [c for c in candidates if c[0] != "Other"]
    if chosen[0] == "Other" and specific and chosen[1] - specific[0][1] < SAFETY_MIN_MARGIN:
        chosen = specific[0]
    others = [c for c in result.runner_up if c[0] != chosen[0]]
    return MLClassificationResult(category=chosen[0], confidence=chosen[1], runner_up=others[:2])


if __name__ == "__main__":
    # Manual spot-check with sentences NOT in either training file —
    # a rough sanity check, not a real evaluation (see the "don't
    # re-evaluate against your training data" warning in
    # train_classifier.py — this applies here too, in reverse: these
    # specific sentences below are fine as a smoke test since they're
    # freshly written here, but don't treat this tiny set as a real
    # accuracy measurement either).
    samples = [
        "The vending machine near the library keeps eating coins without dispensing anything",
        "A classmate keeps mocking my accent whenever I speak in class discussions",
        "The paint on the hostel corridor walls is peeling badly",
        "I still haven't received my transcript despite requesting it a month ago",
    ]
    for s in samples:
        r = classify_category_ml(s)
        print(f"\n> {s}")
        print(f"  category={r.category} (confidence={r.confidence:.1%})")
        print(f"  runner-up: {r.runner_up}")
