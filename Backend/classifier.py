"""
Classification + entity extraction pipeline for the Campus Grievance
Intelligence System.

Design (UPDATED from the original all-keyword version): category
classification now uses a trained SBERT + logistic regression model
(ml/ml_classifier.py, trained by ml/train_classifier.py on 114 hand-
labeled complaints) instead of keyword matching. The keyword classifier
scored 100% on data it had been repeatedly patched against, then 43.2%
on genuinely new phrasing — proof it was memorizing specific words, not
learning what each category means. See ml/train_classifier.py's
docstring for the full reasoning and the honest, cross-validated
accuracy number.

The keyword-based CATEGORY_KEYWORDS/classify_category() below are kept
in the file for reference/comparison (and because TANGLISH_KEYWORDS and
tag_issue() still use CATEGORY_KEYWORDS) but classify_category() itself
is no longer called by the main pipeline — classify_complaint() below
calls classify_category_ml() instead.

Pipeline order still matters:
  1. Safety check FIRST, before anything else, and this part is
     DELIBERATELY still keyword-based, not ML — a trained model can be
     confidently wrong, and for something as consequential as routing a
     harassment report to the confidential queue, a deterministic,
     over-sensitive check is the safer choice. If a complaint trips the
     safety trigger, no other classification runs — it goes straight to
     the restricted path. (False positives here cost nothing; false
     negatives cost a great deal.)
  2. Category classification (ML model)
  3. Entity extraction (location, department hint) — still regex/keyword,
     unaffected by this change
  4. Priority recommendation — still keyword-based, unaffected
  5. Issue-type tagging (feeds clustering) — still keyword-based

Every non-restricted result includes an `ml_confidence` and
`ml_runner_up` reasoning field — the model's own confidence and
runner-up categories, which is a more honest signal than a keyword list
was: a 60%-confidence prediction with a close second place is a
genuinely different situation from a 98%-confidence one.
"""
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ml/ml_classifier.py provides the trained SBERT + logistic regression
# category classifier, replacing the old keyword-based classify_category().
# See ml/train_classifier.py's docstring for why: the keyword approach
# scored 100% on data it had been repeatedly patched against, then 43.2%
# on genuinely new phrasing — proof it was memorizing words, not learning
# categories. The safety check below stays keyword-based on purpose (see
# its own comment) — only category classification moved to the trained
# model.
sys.path.insert(0, str(Path(__file__).parent / "ml"))
from ml.ml_classifier import classify_category_guarded, classify_category_ml  # noqa: E402
from ml.hazard_keywords import check_infrastructure_hazard
from semantic_tagging import get_issue_tag, check_lost_found_semantic
# ---------------------------------------------------------------------------
# Safety trigger — runs before everything else, deliberately over-sensitive.
# OR'd keyword match, not AND'd: one hit is enough to route to the
# restricted path. False positives are cheap; false negatives are not.
# ---------------------------------------------------------------------------

SAFETY_KEYWORDS = {
    "harassment": ["harass", "harassing", "harassed"],
    "bullying": ["bully", "bullying", "bullied", "intimidate", "intimidating",
                 "humiliate", "humiliating"],
    "threat": ["threat", "threatening", "threatened"],
    "stalking": ["stalk", "stalking", "following me", "follows me"],
    "physical": ["touch", "touched", "touching", "hit me", "hitting",
                 "assault", "fight", "fighting", "attacked", "attacking",
                 "molest", "molesting"],
    "abuse": ["abuse", "abusing", "abusive"],
    "unsafe": ["unsafe", "scared", "afraid", "threatened by",
               "do not feel safe", "don't feel safe", "not feel safe"],
    "inappropriate_conduct": ["inappropriate comment", "inappropriate remark",
                               "catcall", "catcalling"],
    "weapon": ["knife", "weapon", "gun", "pistol", "revolver", "blade",
           "sword", "armed", "stab", "stabbed", "stabbing",
           "brought a knife", "carrying a weapon"],
}


def _contains_keyword(lowered_text: str, keyword: str) -> bool:
    """
    Word-boundary match with tolerance for common suffixes (plurals,
    -ing, -ed), rather than exact-word-only matching. The exact-only
    version had a real, measured bug: "classmates" (plural) did not
    match the keyword "classmate" because \\b requires the match to
    end exactly at a word boundary. Complaint text is full of plurals
    and verb forms ("students", "damaged", "excluding"), so this
    matters in practice, not just in theory.

    Multi-word phrases are matched as an exact phrase (no suffix
    tolerance applied to phrases like "hit me") since adding suffix
    tolerance to every word in a phrase gets complicated fast for
    little benefit — phrases are already specific enough to not need it.

    Very short keywords (<=2 chars, e.g. "ac") skip suffix tolerance
    too — "ac" + "ed" would match "aced" (as in "I aced the exam"),
    a real false positive that's worse than missing a plural of a
    2-letter word.
    """
    if " " in keyword or len(keyword) <= 2:
        pattern = r"\b" + re.escape(keyword) + r"\b"
    else:
        pattern = r"\b" + re.escape(keyword) + r"(?:s|es|ing|ed)?\b"
    return re.search(pattern, lowered_text) is not None


def check_safety(text: str) -> Optional[str]:
    """
    Returns a restricted_reason string if the text trips the safety
    trigger, else None. Checked first, before any other classification.
    """
    lowered = text.lower()
    for reason, keywords in SAFETY_KEYWORDS.items():
        for kw in keywords:
            if _contains_keyword(lowered, kw):
                return reason
    return None


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "Infrastructure": [
        "ac", "air condition", "fan", "light", "electric", "socket",
        "wiring", "wire", "leak", "leaking", "water", "toilet", "washroom",
        "projector", "computer", "pc", "lab", "network", "wifi", "wi-fi",
        "internet", "connection", "signal", "furniture", "chair", "bench",
        "door", "window", "elevator", "lift", "generator",
        "power cut", "power outage", "outlet", "plug point", "whiteboard",
        "marker", "board", "dustbin", "garbage", "cleanliness", "ceiling",
        "wall", "paint", "drinking water",
        "water cooler", "water purifier", "slow", "not working", "broken",
        "damaged", "malfunction", "glitch", "glitching", "portal down",
        "system down",
    ],
    "Infrastructure Hazard": [
        "exposed wire", "exposed wiring", "exposed electrical wire",
        "exposed electrical wiring", "loose wire", "bare wire", "live wire",
        "spark", "sparking", "short circuit", "fire hazard", "smoke",
        "gas leak", "blocked exit", "emergency exit", "fire extinguisher",
        "no lighting", "poor lighting", "no proper lighting",
        "poorly lit", "not well lit", "insufficient lighting",
        "dark corridor", "dim light", "slippery", "slippery floor",
        "wet floor", "broken railing", "loose railing", "open manhole",
        "collapse", "collapsing", "ceiling falling", "structural damage",
        "smoking",
    ],
    "Academic": [
        "marks", "grade", "grading", "internal", "exam", "examination",
        "result", "syllabus", "timetable", "schedule", "attendance",
        "curriculum", "register", "registration", "course", "enroll",
        "enrollment", "study material", "notes", "portal", "access",
        "login", "credential", "password", "academic calendar", "gpa",
        "cgpa", "transcript", "backlog", "re-evaluation", "revaluation",
        "reevaluation", "semester", "credit", "submission",
        "lab session", "class cancelled",
    ],
    "Faculty": [
        "professor", "sir", "ma'am", "madam", "lecturer", "teacher",
        "faculty", "deadline", "biased", "bias", "favoritism", "favouritism",
        "unfair grading", "absent teacher", "substitute teacher",
        "class cancelled", "no show",
    ],
    "Canteen": [
        "food", "canteen", "mess", "meal", "hygiene", "quality of food",
        "stale", "overpriced", "expensive food", "menu", "vendor",
        "cold food", "spoiled", "spoilt", "unhygienic", "cockroach",
        "insect in food", "hair in food",
    ],
    "Administration": [
        "certificate", "bonafide", "admin office", "front office",
        "administration office", "fee",
        "payment", "refund", "document", "id card", "registration form",
        "form", "record", "records", "scholarship", "verify",
        "verification", "verified", "update details", "emergency contact",
        "noc", "transfer certificate", "admission", "counselling",
        "counseling", "transcript request", "application status",
        "pending approval",
    ],
    "Student-related": [
        "ragging", "peer", "classmate", "roommate", "hostel mate",
        "disturb", "disturbing", "noise", "noisy", "cooperate",
        "cooperating", "uncooperative", "exclude", "excluding", "excluded",
        "argument", "arguing", "dispute", "rumor", "rumour",
        "false information", "spreading rumors", "spreading rumours",
        "taking my seat", "group project", "group member", "teammate",
        "team member", "damaging property", "property damage",
        "vandalism", "vandalizing", "vandalising", "damaging",
    ],
}

# Tanglish / common campus-specific terms (review section 2.3: reduced to
# a keyword dictionary rather than a full multilingual model)
TANGLISH_KEYWORDS = {
    "Infrastructure": ["romba hot", "slow ah", "leak aaguthu", "kettu",
                        "velaila", "spoil aayiduchu"],
}


def classify_category(text: str) -> tuple[str, list[str]]:
    """
    Returns (category, matched_keywords). Falls back to 'Other'.

    Scoring weights each matched keyword by its word count, not just
    counting hits. This matters because generic single-word keywords
    overlap across categories in ways that used to cause wrong wins —
    e.g. "wire" (generic, in Infrastructure) matching alongside "exposed
    wire" (specific, in Infrastructure Hazard) used to make plain
    Infrastructure win on raw count even when the hazard-specific phrase
    was the better signal. Weighting by phrase length means a specific
    2-word match outweighs a generic 1-word match, which is the correct
    outcome in almost every case like this.
    """
    lowered = text.lower()
    scores: dict[str, list[str]] = {}

    all_keywords = {**CATEGORY_KEYWORDS}
    for cat, extra in TANGLISH_KEYWORDS.items():
        all_keywords.setdefault(cat, [])
        all_keywords[cat] = all_keywords[cat] + extra

    for category, keywords in all_keywords.items():
        matched = [kw for kw in keywords if _contains_keyword(lowered, kw)]
        if matched:
            scores[category] = matched

    if not scores:
        return "Other", []

    def weight(matched_list):
        # Role-identifying words (professor, teacher, faculty) are a
        # stronger signal than a generic subject-matter noun (notes,
        # syllabus) when both appear — the complaint is about a specific
        # person's conduct, not the academic system itself. Without this,
        # "Our faculty member has not uploaded the lecture notes" ties
        # between Faculty (['faculty']) and Academic (['notes']), and the
        # tie was silently going to whichever category happened to be
        # defined earlier in the dict — an arbitrary outcome, not a
        # real decision.
        ROLE_WORDS = {"professor", "teacher", "lecturer", "faculty",
                      "sir", "ma'am", "madam"}
        total = 0
        for kw in matched_list:
            words = kw.split()
            total += 2 if kw in ROLE_WORDS else len(words)
        return total

    best = max(scores.items(), key=lambda kv: (weight(kv[1]), len(kv[1])))
    return best[0], best[1]


# ---------------------------------------------------------------------------
# Issue-type tagging — coarser than category, feeds the clustering step
# ---------------------------------------------------------------------------

ISSUE_PATTERNS = [
    ("AC malfunction", ["ac", "air condition"]),
    ("Water leakage", ["leak", "leaking", "water"]),
    ("Computer/network issues", ["computer", "pc", "network", "wifi",
                                  "wi-fi", "internet", "connection",
                                  "signal", "slow"]),
    ("Projector not working", ["projector"]),
    ("Electrical hazard", ["spark", "wiring", "wire", "exposed wir",
                            "short circuit"]),
    ("Blocked emergency exit", ["blocked exit", "emergency exit"]),
    ("Poor lighting", ["no lighting", "poor lighting", "dark corridor",
                        "dim light"]),
    ("Slippery/unsafe flooring", ["slippery", "wet floor"]),
    ("Food quality", ["food", "quality", "stale", "hygiene", "spoiled",
                       "spoilt", "cockroach"]),
    ("Certificate processing delay", ["certificate", "bonafide"]),
    ("Internal marks not updated", ["marks", "internal"]),
    ("Assignment deadline changes", ["deadline", "assignment"]),
    ("Library facilities", ["library"]),
    ("Peer conflict", ["disturb", "noise", "exclude", "argument",
                        "dispute", "rumor", "rumour"]),
]


def tag_issue(text: str, category: str) -> str:
    lowered = text.lower()
    for issue_name, keywords in ISSUE_PATTERNS:
        if any(_contains_keyword(lowered, kw) for kw in keywords):
            return issue_name
    return f"General {category}" if category != "Other" else "Uncategorized"

WATER_LEAK_PHRASES = [
    "water leak", "leaking water", "leaking pipe", "pipe leak",
    "water dripping", "dripping tap", "tap leaking", "flooded",
    "flooding", "water seepage", "seepage", "burst pipe",
    "overflowing", "water logging", "waterlogged",
]

def check_water_leak(text: str) -> bool:
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in WATER_LEAK_PHRASES)

LOST_FOUND_PHRASES = [
    "lost my",
    "i lost",
    "missing my",
    "is missing",
    "are missing",
    "i misplaced",
    "misplaced my",
    "left behind",
    "left my",
    "can't find my",
    "cant find my",
    "cannot find my",
    "found a",
    "found an",
    "missed my",
]

LOST_FOUND_RESPONSE = (
    "This looks like a lost & found matter rather than a grievance. "
    "This system routes complaints to departments for resolution and isn't "
    "set up to track lost items. Please check with campus security or the "
    "nearest department office."
)

def check_lost_found(text: str) -> bool:
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in LOST_FOUND_PHRASES)
# ---------------------------------------------------------------------------
# Entity extraction — location and department hints
# ---------------------------------------------------------------------------

# Room code format: [Wing][W][Floor digit][Room 2-digits], e.g. CW204 = C Wing, 2nd Floor, room 04
ROOM_CODE_PATTERN = re.compile(r"\b([A-C])W(\d)(\d{2})\b", re.IGNORECASE)

# Plain wording fallback: "A wing", "C-wing"
WING_WORD_PATTERN = re.compile(r"\b([A-C])[\s-]?Wing\b", re.IGNORECASE)

FLOOR_PATTERN = re.compile(
    r"\b(\d)(?:st|nd|rd|th)\s?Floor\b|\bGround Floor\b", re.IGNORECASE
)
AREA_KEYWORDS = ["lab", "library", "canteen", "staircase", "hostel",
                  "parking", "auditorium", "washroom", "toilet"]
LAB_NUMBER_PATTERN = re.compile(r"\bLab\s?(\d)\b", re.IGNORECASE)


def _ordinal_floor(digit: str) -> str:
    n = int(digit)
    if n == 0:
        return "Ground Floor"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n, "th")
    return f"{n}{suffix} Floor"


@dataclass
class ExtractedLocation:
    building: Optional[str] = None   # holds wing, e.g. "C Wing" — kept as
                                      # "building" for schema compatibility
    floor: Optional[str] = None
    area: Optional[str] = None
    specific: Optional[str] = None
    confidence: str = "none"
    matched_text: list[str] = field(default_factory=list)

    @property
    def is_empty(self):
        return not any([self.building, self.floor, self.area, self.specific])


def extract_location(text: str) -> ExtractedLocation:
    loc = ExtractedLocation()

    # 1. Room code is the strongest signal — gives wing, floor, and room in one match
    room = ROOM_CODE_PATTERN.search(text)
    if room:
        wing_letter, floor_digit, room_num = room.groups()
        loc.building = f"{wing_letter.upper()} Wing"
        loc.floor = _ordinal_floor(floor_digit)
        loc.specific = f"Room {wing_letter.upper()}W{floor_digit}{room_num}"
        loc.matched_text.append(room.group(0))
    else:
        # 2. Fall back to plain wording — wing and floor can appear independently
        w = WING_WORD_PATTERN.search(text)
        if w:
            loc.building = f"{w.group(1).upper()} Wing"
            loc.matched_text.append(w.group(0))

        f = FLOOR_PATTERN.search(text)
        if f:
            matched = f.group(0)
            digit_match = re.search(r"\d", matched)
            if digit_match:
                loc.floor = _ordinal_floor(digit_match.group(0))
            else:
                loc.floor = "Ground Floor"  # matched "Ground Floor" with no digit
            loc.matched_text.append(matched)
    # 3. Area keywords checked independently — can co-occur with wing/floor either way
    lowered = text.lower()
    for kw in AREA_KEYWORDS:
        if _contains_keyword(lowered, kw):
            loc.area = kw.title()
            loc.matched_text.append(kw)
            break

    # 4. Lab number only if room code didn't already give a specific location
    if not loc.specific:
        lab = LAB_NUMBER_PATTERN.search(text)
        if lab:
            loc.specific = f"Lab {lab.group(1)}"
            loc.matched_text.append(lab.group(0))

    filled = sum(x is not None for x in [loc.building, loc.floor, loc.area,
                                          loc.specific])
    if filled == 0:
        loc.confidence = "none"
    elif filled == 1:
        loc.confidence = "partial"
    else:
        loc.confidence = "full"

    return loc

DEPARTMENT_MAP = {
    "Infrastructure": "Maintenance",
    "Infrastructure Hazard": "Maintenance",
    "Academic": "Academic Office",
    "Faculty": "Academic Office",
    "Administration": "Administration",
    "Canteen": "Canteen Services",
    "Safety": "Student Affairs",
    "Student-related": "Student Affairs",
    "Other": "Administration",
}


# ---------------------------------------------------------------------------
# Priority recommendation
# ---------------------------------------------------------------------------

CRITICAL_KEYWORDS = ["spark", "sparking", "fire", "smoke", "electrical hazard",
                      "exposed wire", "exposed wiring", "short circuit",
                      "collapse", "collapsing", "flooding", "gas leak",
                      "blocked exit", "emergency exit", "open manhole",
                      "structural damage"]
HIGH_KEYWORDS = ["urgent", "immediately", "exam", "deadline today",
                  "three days", "week", "again", "still not", "repeatedly",
                  "slippery", "poor lighting", "no lighting", "dark corridor"]
LOW_KEYWORDS = ["minor", "small", "whenever", "no rush"]


def recommend_priority(text: str, category: str,
                        safety_reason: Optional[str]) -> tuple[str, list[str]]:
    if safety_reason:
        lowered = text.lower()
        if any(_contains_keyword(lowered, kw) for kw in
           ["threat", "unsafe", "scared", "assault", "hit me",
            "knife", "weapon", "gun", "armed", "stab"]):
            return "Critical", [safety_reason, "explicit danger language"]
        return "High", [safety_reason]

    lowered = text.lower()
    matched_critical = [kw for kw in CRITICAL_KEYWORDS
                         if _contains_keyword(lowered, kw)]
    if matched_critical:
        return "Critical", matched_critical

    matched_high = [kw for kw in HIGH_KEYWORDS
                     if _contains_keyword(lowered, kw)]
    if matched_high:
        return "High", matched_high

    matched_low = [kw for kw in LOW_KEYWORDS
                    if _contains_keyword(lowered, kw)]
    if matched_low:
        return "Low", matched_low

    return "Medium", []  # default: nothing alarming, nothing trivial


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    is_restricted: bool
    restricted_reason: Optional[str]
    category: Optional[str]
    issue: Optional[str]
    priority: Optional[str]
    department_hint: Optional[str]
    location: ExtractedLocation
    reasoning: dict


def classify_complaint(text: str) -> ClassificationResult:
    """
    Runs the full pipeline in order. Safety check happens first and, if
    tripped, short-circuits category/issue/location classification —
    a restricted complaint is handed off, not auto-triaged in detail
    (review section 5.1: "does not ask follow-up questions, does not
    run similarity matching").
    """
    safety_reason = check_safety(text)

    if safety_reason:
        priority, priority_reasoning = recommend_priority(text, "Safety",
                                                            safety_reason)
        return ClassificationResult(
            is_restricted=True,
            restricted_reason=safety_reason,
            category="Safety",
            issue=None,          # deliberately not tagged — no clustering
            priority=priority,
            department_hint="Student Affairs",
            location=ExtractedLocation(),  # deliberately not extracted
            reasoning={
                "safety_trigger": safety_reason,
                "priority": priority_reasoning,
                "note": "Restricted path: category/issue/location "
                        "extraction skipped by design.",
            },
        )
        # Lost & Found redirect — do not classify as a grievance
    is_lost_found, lf_confidence = check_lost_found_semantic(text)
    if is_lost_found:
        return ClassificationResult(
            is_restricted=False,
            restricted_reason=None,
            category="Lost & Found",
            issue=None,
            priority=None,
            department_hint=None,
            location=ExtractedLocation(),
            reasoning={
                "redirect": True,
                "redirect_message": LOST_FOUND_RESPONSE,
                "lost_found_confidence": round(lf_confidence, 3),
            },
        )

    ml_result = classify_category_guarded(text)
    category = ml_result.category

     # Safety-oriented infrastructure hazard override:
    # only bump Infrastructure -> Infrastructure Hazard.
    if category == "Infrastructure" and check_infrastructure_hazard(text):
        category = "Infrastructure Hazard"
    # Water-leak override
    if check_water_leak(text):
        category = "Infrastructure"

    issue_tag, issue_confidence = get_issue_tag(text)
    issue = issue_tag if issue_tag else f"General {category}"

    location = extract_location(text)

    priority, priority_reasoning = recommend_priority(text, category, None)

    department = DEPARTMENT_MAP.get(category, "Administration")

    # Route IT-related issues to IT Support
    ISSUE_TAG_DEPARTMENT = {
    "Internet/WiFi": "IT Support",
    "Computer/network issues": "IT Support",
    "Portal/login issues": "IT Support",
    }

    department = ISSUE_TAG_DEPARTMENT.get(issue, department)

    return ClassificationResult(
        is_restricted=False,
        restricted_reason=None,
        category=category,
        issue=issue,
        priority=priority,
        department_hint=department,
        location=location,
        reasoning={
            "ml_confidence": round(ml_result.confidence, 3),
            "ml_runner_up": [(c, round(p, 3)) for c, p in ml_result.runner_up],
            "priority_keywords": priority_reasoning,
            "location_confidence": location.confidence,
            "location_matched": location.matched_text,
        },
    )

if __name__ == "__main__":
    # quick manual spot-check — requires the trained model to exist first
    # (run ml/train_classifier.py from the Backend/ folder before this)
    samples = [
        "C block 3rd floor AC romba hot ah irukku.",
        "A student keeps bullying me in class and I don't know what to do.",
        "There is an electrical spark coming from the socket in Lab 3.",
        "Our internal marks still haven't been updated.",
        "The professor keeps changing the assignment deadline.",
        "Someone spilled coffee on the table.",  # false-positive check: "fee"
        "The macaroni in the canteen was cold.",  # false-positive check: "ac"
        "I stacked my books on the rack.",  # false-positive check: "ac"
    ]
    try:
        for s in samples:
            r = classify_complaint(s)
            print(f"\n> {s}")
            print(f"  restricted={r.is_restricted} category={r.category} "
                  f"issue={r.issue} priority={r.priority}")
            print(f"  location: {r.location}")
            print(f"  reasoning: {r.reasoning}")
    except FileNotFoundError as e:
        print(f"\n{e}")
