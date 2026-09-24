"""
Similarity and clustering using SBERT sentence embeddings — the same
model family (all-MiniLM-L6-v2) used in the CopyShield plagiarism
pipeline for its retrieval stage.

UNTESTED IN THIS SANDBOX: this build environment cannot reach
huggingface.co, so this file has not been run or calibrated here. Run
the __main__ block below on your own machine first — it prints the same
pairwise similarity matrix and duplicate-check test used to calibrate
similarity.py (TF-IDF) and similarity_glove.py (GloVe averaging), so you
can compare all three side by side and pick real threshold values from
what you actually see, not from a guess.

Why this should outperform both of the other two approaches: unlike
TF-IDF (exact word overlap only) and GloVe averaging (word-level vectors
averaged together, order-blind), SBERT is trained specifically to
produce one vector for an entire sentence such that semantically similar
sentences end up close together — it was trained ON sentence pairs, not
just words. This is the same reason it was the right choice for
CopyShield's retrieval stage: catching paraphrases, not just synonyms.

Setup (run locally):
    pip install sentence-transformers
    python3 similarity_sbert.py
"""
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

_MODEL_NAME = "all-MiniLM-L6-v2"  # same model used in CopyShield's retrieval stage
_model = None  # lazy-loaded singleton


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# PLACEHOLDER — not calibrated in this sandbox. SBERT cosine similarities
# for genuinely different sentences typically sit lower than GloVe's did
# (SBERT is better at actually separating unrelated meaning, not just
# unrelated vocabulary), so 0.80 is a reasonable starting guess based on
# general SBERT behavior, but you MUST re-run the __main__ block below
# locally and adjust this against the real numbers you see — the same
# way similarity.py's 0.28 and similarity_glove.py's 0.80 were each
# tuned against their own actual output, not assumed.
DUPLICATE_THRESHOLD = 0.50


def embed(text: str) -> np.ndarray:
    model = _get_model()
    return model.encode(text, convert_to_numpy=True)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Batching is much faster than calling embed() in a loop for more
    than a few texts — use this when embedding a whole complaint list."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True)


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
    existing_vecs = embed_batch([desc for _, desc in existing])
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
    vectors = embed_batch([desc for _, desc in complaints])

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
    # Identical test set to similarity.py and similarity_glove.py, so you
    # can compare all three approaches on exactly the same sentences.
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

    print(f"Loading {_MODEL_NAME} (first run downloads the model)...")
    texts = [t for _, t in sample]
    vecs = embed_batch(texts)
    sim_matrix = cosine_similarity(vecs)
    np.set_printoptions(precision=2, suppress=True)
    print("\n--- Pairwise similarity matrix ---")
    print("Compare this against similarity.py's TF-IDF matrix and")
    print("similarity_glove.py's GloVe matrix for the same sentences.")
    print("Look specifically at:")
    print("  - items 0,1,2 (the three AC complaints) — should be HIGH")
    print("    even though item 1 shares almost no words with 0/2")
    print("  - item 0 vs item 8 (AC vs projector) — should be LOW,")
    print("    this is the false-positive case GloVe averaging failed")
    print(sim_matrix)

    print(f"\n--- Clustering at threshold {DUPLICATE_THRESHOLD} ---")
    print("If this threshold produces wrong clusters, adjust")
    print("DUPLICATE_THRESHOLD above based on the matrix printed above,")
    print("then re-run this file to check again.")
    clusters = cluster_complaints(sample)
    for label, members in clusters.items():
        texts_in_cluster = [t for cid, t in sample if cid in members]
        print(f"Cluster {label}: ids={members}")
        for t in texts_in_cluster:
            print(f"    - {t}")

    print("\n--- The case TF-IDF missed entirely, and GloVe over-matched ---")
    new_complaint = "AC on the 3rd floor of C block still not fixed"
    similar = find_similar(new_complaint, [(c[0], c[1]) for c in sample])
    print(f"New: {new_complaint}")
    print("Expect: items 1, 2, 3 (all real AC complaints) near the top,")
    print("        items 6, 9 (professor, projector) NOT appearing at all.")
    for s in similar:
        print(f"  -> #{s.complaint_id} (sim={s.similarity}): {s.description}")
