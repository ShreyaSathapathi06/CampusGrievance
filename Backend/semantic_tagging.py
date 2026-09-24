"""
Semantic issue-tagging, lost-and-found detection, and low-confidence
("couldn't understand this") detection using SBERT embeddings — replacing
brittle keyword substring matching.

Uses the same model as similarity_sbert.py and the category classifier
(all-MiniLM-L6-v2) so no new dependency is introduced.

Drop into: CampusGrievance/Backend/ml/semantic_tagging.py

IMPORTANT: ISSUE_TAG_PROTOTYPES below is a best-effort, comprehensive
construction covering plausible issue tags under your 8 known categories
(Infrastructure, Academic, Faculty, Student-related, Safety, Canteen,
Administration, Other) — built from general campus-grievance domain
knowledge, NOT validated against your actual live tag list (seed.py,
issue_clusters table, or wherever your tags are currently enumerated).
Tag names here may not exactly match what your DB/frontend expects.
Paste your real tag list and I'll reconcile this against it exactly —
until then, treat this as a strong draft, not ground truth.
"""

from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker

_model = SentenceTransformer("all-MiniLM-L6-v2")

# Campus/domain terms that a general English dictionary will flag as
# misspelled but are actually correct — add to this as you find more
# false-positive "corrections" in testing.
_DOMAIN_WHITELIST = {
    "wifi", "canteen", "hostel", "bonafide", "invigilator", "revaluation",
    "srm", "vadapalani", "ktr",
}

_spell = SpellChecker()
_spell.word_frequency.load_words(_DOMAIN_WHITELIST)


def normalize_text(text: str) -> str:
    """
    Lightweight spell-correction pass, run once before any embedding call.
    Corrects only words the dictionary doesn't recognize (leaving domain
    terms alone via the whitelist) — does not rewrite grammar or phrasing.
    NOT a silver bullet: won't fix words that happen to be *other* valid
    English words when misspelled, and won't fix multi-word garbling.
    """
    words = text.split()
    corrected = []
    for w in words:
        stripped = w.strip(".,!?;:\"'")
        if not stripped or stripped.lower() in _DOMAIN_WHITELIST:
            corrected.append(w)
            continue
        fix = _spell.correction(stripped)
        corrected.append(fix if fix else w)
    return " ".join(corrected)

# ---------------------------------------------------------------------------
# ISSUE-TAG PROTOTYPES, grouped by parent category for clarity
# (grouping is just organizational here — matching is still flat across all
# tags; category-scoping the search is a possible future refinement)
# ---------------------------------------------------------------------------
ISSUE_TAG_PROTOTYPES = {
    # --- Infrastructure ---
    "Water leakage": [
        "water is leaking from the pipe",
        "the tap won't stop dripping",
        "pipe burst and flooded the floor",
        "ceiling is leaking water during rain",
        "water is coming out from the wall",
        "there is a leak under the sink",
        "water is seeping through the wall near the window",
        "the washbasin pipe is leaking constantly",
        "water is pooling on the floor near the entrance",
        "there's a slow drip from the ceiling corner",
    ],
    "Cleanliness": [
        "the water looks dirty and discolored",
        "washroom is unclean and smells bad",
        "garbage has not been collected in days",
        "floor is dirty and hasn't been mopped",
        "dustbins are overflowing near the canteen",
        "there is dust and cobwebs everywhere in the room",
        "the classroom hasn't been cleaned in a while",
        "there's a bad smell coming from the corridor",
        "trash is piling up outside the building",
        "the corridor floor is stained and dirty",
    ],
    "Electrical": [
        "exposed wire near the switchboard",
        "there is no power in the classroom",
        "the light is flickering and not working",
        "fan is not working in the lab",
        "socket is sparking when plugged in",
        "the projector isn't turning on",
        "power keeps tripping in the room",
        "tubelight is not switching on",
        "the switchboard is loose and unsafe",
        "no electricity in half the classroom",
    ],
    "Furniture issues": [
        "the chair is broken and wobbly",
        "desk has a sharp broken edge",
        "bench is cracked and unsafe to sit on",
        "the classroom table is damaged",
        "there are not enough benches in the classroom",
        "no enough benches in my class",
        "we don't have enough chairs for everyone",
        "some students have nowhere to sit, not enough desks",
        "the classroom is short on benches",
        "furniture is missing from the room",
        "chairs are broken and there aren't enough to go around",
        "need more benches, several students are standing",
    ],
    "Internet/WiFi": [
        "wifi is not working in the hostel",
        "internet connection is extremely slow",
        "unable to connect to campus network",
        "wifi keeps disconnecting during class",
        "no wifi signal in the library",
        "internet has been down all day",
        "cannot access wifi from my room",
        "connection drops every few minutes",
    ],
    "Washroom issues": [
        "washroom door lock is broken",
        "no water supply in the restroom",
        "toilet is not flushing properly",
        "washroom tap is broken",
        "restroom has no soap or tissue",
        "washroom door doesn't close properly",
        "bathroom light isn't working",
    ],
    "Air conditioning/Ventilation": [
        "the air conditioner is not cooling",
        "classroom is too stuffy with no ventilation",
        "AC has been leaking water onto the floor",
        "the room gets extremely hot with no airflow",
        "AC is not working at all",
        "the room is very warm and there's no fan or AC",
        "poor ventilation makes the classroom uncomfortable",
        "air conditioning has been off for days",
    ],
    "Lift/Elevator": [
        "the lift has been out of service for a week",
        "elevator makes a loud grinding noise",
        "lift doors don't close properly",
        "elevator is stuck between floors",
        "lift button panel isn't responding",
    ],
    "Noise disturbance": [
        "construction noise is disrupting classes",
        "loud noise from outside during exams",
        "drilling sound is unbearable during lectures",
        "constant noise from the corridor during class",
        "loudspeaker outside is too disruptive",
    ],
    "General Infrastructure": [
        "the door is broken and won't close",
        "windows are cracked and unsafe",
        "building looks poorly maintained overall",
        "paint is peeling off the walls",
        "ceiling tiles are falling apart",
        "the wall has visible cracks",
    ],

    "Infrastructure Hazard": [
        "exposed live wire near the walkway",
        "staircase railing is broken and dangerous",
        "floor is extremely slippery, someone could fall",
        "fire extinguisher is missing from the corridor",
        "emergency exit is blocked by furniture",
        "ceiling looks like it could collapse",
        "loose railing on the stairs is a fall risk",
        "broken glass on the floor near the entrance",
    ],

    # --- Academic ---
    "Exam issues": [
        "exam schedule has a timing conflict",
        "two exams were scheduled at the same time",
        "seating arrangement for the exam was wrong",
        "exam hall was too crowded and noisy",
        "exam was rescheduled without proper notice",
        "not enough time was given to reach the exam hall",
    ],
    "Grading/Marks dispute": [
        "my marks were not updated correctly",
        "I think my answer sheet was evaluated wrongly",
        "grade posted doesn't match what I expected",
        "revaluation request has not been processed",
        "marks awarded don't match the answer I gave",
        "my score seems lower than it should be",
    ],
    "Timetable issues": [
        "class timetable has overlapping slots",
        "timetable was changed without informing students",
        "back to back classes with no break",
        "the schedule was updated but nobody was told",
    ],
    "Attendance issues": [
        "attendance was marked wrong for my class",
        "I was marked absent despite attending",
        "attendance shortage notice seems incorrect",
        "my attendance percentage looks wrong on the portal",
    ],
    "Course content issues": [
        "syllabus is not being covered on time",
        "course material provided is outdated",
        "lab equipment for the practical is insufficient",
        "the course is falling behind schedule",
    ],

    # --- Faculty ---
    "Faculty conduct": [
        "the professor was rude and dismissive in class",
        "faculty member used inappropriate language",
        "teacher shouted at students unnecessarily",
        "professor was disrespectful during the lecture",
    ],
    "Faculty absence": [
        "professor has not been showing up for class",
        "faculty is frequently unavailable during office hours",
        "substitute teacher was not arranged for the missed class",
        "the teacher has missed several classes in a row",
    ],
    "Biased grading": [
        "I feel my marks were unfairly low compared to others",
        "professor seems to favor certain students in grading",
        "grading feels inconsistent across students",
    ],

    # --- Student-related ---
    "Peer conflict": [
        "there is a dispute between students in my group",
        "conflict with my roommate over shared space",
        "disagreement with classmates during group project",
    ],
    "Group project disputes": [
        "one team member is not contributing to the project",
        "unfair division of work in the group assignment",
        "a group member isn't responding or doing their part",
    ],

    # --- Canteen ---
    "Food quality": [
        "food served in the canteen was stale",
        "found a foreign object in the food",
        "food was undercooked and tasted off",
        "the food quality has gotten worse recently",
    ],
    "Canteen hygiene": [
        "canteen kitchen looks unhygienic",
        "flies around the food counter",
        "utensils in the canteen are not properly cleaned",
    ],
    "Canteen pricing/billing": [
        "was overcharged for my order at the canteen",
        "canteen bill amount doesn't match what I ordered",
        "prices increased without notice",
    ],
    "Canteen wait time": [
        "canteen queue takes too long during lunch",
        "waited over 30 minutes to get served",
        "the line moves very slowly during peak hours",
    ],

    # --- Administration ---
    "Fee-related issues": [
        "fee payment was not reflected in my account",
        "extra charges appeared on my fee receipt",
        "fee deadline extension request was ignored",
    ],
    "Document/Certificate delays": [
        "my bonafide certificate has been delayed for weeks",
        "transcript request has not been processed",
        "ID card renewal is taking too long",
    ],
    "Administrative staff behavior": [
        "office staff was unhelpful and rude",
        "admin department is not responding to my emails",
    ],
    "Scholarship issues": [
        "scholarship amount has not been credited",
        "scholarship application status is unclear",
    ],

    # --- Other ---
    "General/Miscellaneous": [
        "this doesn't fit any specific category but needs attention",
        "general feedback about campus facilities",
    ],
    "Computer/network issues": [
        "lab computers are not turning on",
        "the computers in the lab are not working",
        "system is very slow in the computer lab",
        "keyboard and mouse not working in the lab",
        "lab systems keep freezing during class",
    ],
    "Portal/login issues": [
        "cannot log in to the student portal",
        "the portal is not opening",
        "unable to access the LMS or ERP website",
        "portal keeps showing an error on login",
    ],
}

# ---------------------------------------------------------------------------
# LOST & FOUND PROTOTYPES
# ---------------------------------------------------------------------------
LOST_FOUND_PROTOTYPES = [
    "I lost my water bottle in the classroom",
    "my phone is missing since this morning",
    "I misplaced my laptop charger somewhere on campus",
    "found a wallet near the library, does anyone know whose it is",
    "I can't find my bag anywhere",
    "left my umbrella in the lecture hall, can someone check",
    "my laptop is missing from the hostel room",
    "someone took my headphones from the study room",
    "lost my ID card somewhere near the canteen",
    "found a pair of earphones in the lab, whose are they",
    "my calculator went missing during the exam",
    "misplaced my notebook, has anyone seen it",
]

# Precomputed once at import time — cheap at this scale (a few hundred sentences)
_issue_tag_embeddings = {
    tag: _model.encode(examples, convert_to_tensor=True)
    for tag, examples in ISSUE_TAG_PROTOTYPES.items()
}
_lost_found_embeddings = _model.encode(LOST_FOUND_PROTOTYPES, convert_to_tensor=True)

# NOT yet tuned against her real data — starting points only.
# Run tune_thresholds() against evaluation_70.csv + holdout_set.csv and the
# known problem cases before trusting these numbers in production, the same
# way DUPLICATE_THRESHOLD=0.50 was empirically set for similarity_sbert.py.
ISSUE_TAG_THRESHOLD = 0.45
LOST_FOUND_THRESHOLD = 0.50

# Below this, nothing scored high enough anywhere to be trustworthy at all —
# treat as unintelligible input (gibberish, severe typos, empty/near-empty
# text) rather than force-fitting a low-confidence guess into any bucket.
NO_MATCH_FLOOR = 0.28

CLARIFICATION_MESSAGE = (
    "Sorry, I couldn't quite understand that. Could you describe the issue "
    "in a bit more detail or rephrase it?"
)


def get_issue_tag(text: str):
    """Returns (tag, confidence). tag is None if nothing clears threshold —
    caller should fall back to a "General <Category>" tag in that case."""
    text_emb = _model.encode(normalize_text(text), convert_to_tensor=True)
    best_tag, best_score = None, 0.0
    for tag, proto_embs in _issue_tag_embeddings.items():
        score = util.cos_sim(text_emb, proto_embs).max().item()
        if score > best_score:
            best_tag, best_score = tag, score
    if best_score < ISSUE_TAG_THRESHOLD:
        return None, best_score
    return best_tag, best_score


def check_lost_found_semantic(text: str):
    """Returns (is_lost_found: bool, confidence: float)."""
    text_emb = _model.encode(normalize_text(text), convert_to_tensor=True)
    score = util.cos_sim(text_emb, _lost_found_embeddings).max().item()
    return score >= LOST_FOUND_THRESHOLD, score


def check_understandable(text: str):
    """
    Returns (is_understandable: bool, best_overall_score: float).
    Call this FIRST, before category classification / issue-tagging / lost-found
    checks. If is_understandable is False, short-circuit and return
    CLARIFICATION_MESSAGE instead of attempting to classify at all — don't
    force gibberish or severe typos into any category.

    Note: this only checks against issue-tag and lost-found prototypes, since
    those are what this module owns. For full coverage, also check the
    category classifier's own confidence score (ml_classifier.py already
    returns confidence + runner-up probabilities per the existing pipeline)
    and treat a low max-probability there as an additional gibberish signal —
    combine both checks in main.py rather than relying on this alone.
    """
    text_emb = _model.encode(normalize_text(text), convert_to_tensor=True)
    issue_best = max(
        util.cos_sim(text_emb, proto_embs).max().item()
        for proto_embs in _issue_tag_embeddings.values()
    )
    lf_best = util.cos_sim(text_emb, _lost_found_embeddings).max().item()
    overall_best = max(issue_best, lf_best)
    return overall_best >= NO_MATCH_FLOOR, overall_best


def tune_thresholds(labeled_cases: list[tuple[str, str]]):
    """
    Helper to print similarity scores for known cases so you can pick real
    thresholds instead of guessing. Pass a list of (text, expected_tag_or_None)
    tuples. Prints best match, score, and understandability for each.
    """
    for text, expected in labeled_cases:
        tag, score = get_issue_tag(text)
        is_lf, lf_score = check_lost_found_semantic(text)
        understandable, overall_score = check_understandable(text)
        print(f"{text!r}")
        print(f"  issue_tag -> {tag} (score={score:.3f}, expected={expected})")
        print(f"  lost_found -> {is_lf} (score={lf_score:.3f})")
        print(f"  understandable -> {understandable} (overall_score={overall_score:.3f})")


if __name__ == "__main__":
    test_cases = [
        ("dirty water in the tank", "Cleanliness"),
        ("water is leaking from the pipe", "Water leakage"),
        ("the tap keeps dripping all day", "Water leakage"),
        ("I lost my water bottle in class", None),
        ("my phone is missing since morning", None),
        ("my laptop is missing from the hostel room", None),
        ("professor shouted at me in class today", "Faculty conduct"),
        ("canteen food had a hair in it", "Food quality"),
        ("asdkj qwoeiu xzxz", "GIBBERISH — should fail check_understandable"),
        ("wtaer leking pype", "MISSPELLED — should still roughly match Water leakage"),
        ("", "EMPTY STRING — should fail check_understandable"),
    ]
    tune_thresholds(test_cases)
