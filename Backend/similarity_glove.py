"""
Similarity and clustering using pretrained GloVe word vectors, averaged
per complaint ("bag of word vectors"), compared against the TF-IDF
approach in similarity.py.

Why this can catch things TF-IDF can't: GloVe vectors are pretrained on
billions of words of real text, so words that mean similar things end up
close together in the vector space even if they share no letters — "AC"
and "air conditioning" and "cooling" all land near each other. TF-IDF has
no such knowledge; it only sees whether the same exact word appears.

What this does NOT fix: word order and context are still ignored (this
is "bag of vectors," not a sentence-level model) — "the professor didn't
show up" and "the professor did show up" would still look almost
identical here, because both are dominated by the same words. A real
sentence encoder (see the module docstring comparison in the project
notes) is needed to fix that; this is a middle ground between TF-IDF and
a full sentence-transformer.

Model file: glove50.kv, saved locally after one download so `seed.py`
and the server don't re-download it every run.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

_MODEL_PATH = Path(__file__).parent / "data" / "glove50.kv"
_model = None  # lazy-loaded singleton, ~66MB in memory once loaded


def _get_model() -> KeyedVectors:
    global _model
    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{_MODEL_PATH} not found. Run: python3 -c \"import "
                "gensim.downloader as api; v = api.load("
                "'glove-wiki-gigaword-50'); v.save('glove50.kv')\""
            )
        _model = KeyedVectors.load(str(_MODEL_PATH))
    return _model


# Recalibrated for GloVe cosine similarities, which run in a different
# range than TF-IDF's — see the __main__ block for the measured values
# this was tuned against.
DUPLICATE_THRESHOLD = 0.80


def embed(text: str) -> np.ndarray:
    """
    Averages the GloVe vector of every recognized word in the text.
    Words not in the vocabulary (typos, rare words) are simply skipped.
    Returns a zero vector if no words were recognized at all.
    """
    model = _get_model()
    words = [w.strip(".,!?").lower() for w in text.split()]
    vectors = [model[w] for w in words if w in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


@dataclass
class SimilarComplaint:
    complaint_id: int
    description: str
    similarity: float


def find_similar(new_text: str, existing: list[tuple[int, str]],
                  top_k: int = 5) -> list[SimilarComplaint]:
    if not existing:
        return []
    new_vec = embed(new_text).reshape(1, -1)
    existing_vecs = np.array([embed(desc) for _, desc in existing])
    sims = cosine_similarity(new_vec, existing_vecs)[0]

    results = []
    for (cid, desc), sim in zip(existing, sims):
        if sim >= DUPLICATE_THRESHOLD:
            results.append(SimilarComplaint(complaint_id=cid,
                                              description=desc,
                                              similarity=round(float(sim), 3)))
    results.sort(key=lambda r: r.similarity, reverse=True)
    return results[:top_k]


def cluster_complaints(complaints: list[tuple[int, str]],
                        min_cluster_size: int = 2
                        ) -> dict[int, list[int]]:
    if len(complaints) < 2:
        return {}
    ids = [c[0] for c in complaints]
    vectors = np.array([embed(desc) for _, desc in complaints])

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - DUPLICATE_THRESHOLD,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(vectors)

    groups: dict[int, list[int]] = {}
    for cid, label in zip(ids, labels):
        groups.setdefault(int(label), []).append(cid)
    return {label: members for label, members in groups.items()
            if len(members) >= min_cluster_size}


if __name__ == "__main__":
    # Same test set used for the TF-IDF version, for a direct comparison.
    sample = [
        (1, "AC in C block 3rd floor is not cooling"),
        (2, "The air conditioning on the third floor of C block is broken"),
        (3, "C block AC not working since morning"),
        (4, "Canteen food quality has gotten worse this week"),
        (5, "The food in the canteen has been really bad lately"),
        (6, "Professor keeps changing the assignment deadline"),
        (7, "Water leakage near the staircase"),
        (8, "There is a water leak near the stairs"),
        (9, "Projector in the classroom is not working"),
    ]

    print("--- Pairwise similarity matrix (for threshold calibration) ---")
    vecs = np.array([embed(t) for _, t in sample])
    sim_matrix = cosine_similarity(vecs)
    np.set_printoptions(precision=2, suppress=True)
    print(sim_matrix)

    print("\n--- Clustering at threshold", DUPLICATE_THRESHOLD, "---")
    clusters = cluster_complaints(sample)
    for label, members in clusters.items():
        texts = [t for cid, t in sample if cid in members]
        print(f"Cluster {label}: ids={members}")
        for t in texts:
            print(f"    - {t}")

    print("\n--- The case TF-IDF missed: AC vs 'air conditioning' ---")
    new_complaint = "AC on the 3rd floor of C block still not fixed"
    similar = find_similar(new_complaint, [(c[0], c[1]) for c in sample])
    print(f"New: {new_complaint}")
    for s in similar:
        print(f"  -> #{s.complaint_id} (sim={s.similarity}): {s.description}")
