HAZARD_KEYWORDS = [
    "exposed wire", "exposed wiring",
    "live wire", "bare wire", "loose wire", "loose wiring",
    "sparking", "spark",
    "blocked exit", "blocked fire exit", "fire exit",
    "gas leak", "gas smell", "smell of gas",
    "slippery floor", "wet floor",
    "no railing", "broken railing", "loose railing",
    "collapsed", "collapsing",
    "ceiling falling", "falling debris",
    "no lighting", "dark corridor", "poor lighting",
    "short circuit", "electric shock",
    "structural crack", "wall crack", "leaking roof",
]


def check_infrastructure_hazard(text: str) -> bool:
    """
    Returns True when a known infrastructure-hazard phrase
    is present in the complaint.
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in HAZARD_KEYWORDS)