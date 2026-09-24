"""
Seeds the database with realistic synthetic complaints, spread over the
past ~6 weeks, so that trend charts, hotspots, and recurring-issue tables
have something to show on demo day.

Run: python3 seed.py
This drops and recreates campus_grievance.db from scratch every time.
"""
import random
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, Department, Location, IssueCluster, Complaint

random.seed(42)  # reproducible demo data

DB_PATH = "sqlite:///campus_grievance.db"
NUM_COMPLAINTS = 180

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

DEPARTMENTS = ["Maintenance", "Academic Office", "Administration",
               "Student Affairs", "Canteen Services", "IT Support"]

LOCATIONS = [
    # (building, floor, area, specific, verified)
    ("C Block", "3rd Floor", None, None, True),
    ("C Block", "Ground Floor", None, None, True),
    ("A Block", "1st Floor", "Computer Lab", "Lab 3", True),
    ("A Block", "2nd Floor", "Computer Lab", "Lab 2", True),
    ("Main Building", None, "Library", None, True),
    ("Main Building", "Ground Floor", "Canteen", None, True),
    ("B Block", "1st Floor", None, None, False),  # discovered, unverified
    ("B Block", "4th Floor", None, None, True),
    (None, None, "Staircase near A Block", None, False),
    ("A Block", "1st Floor", "Computer Lab", "Lab 3 area", False),
]

# Each template: (category, issue, description_variants, priority_weights,
#                  location_index_or_None, is_restricted)
# priority_weights over [Low, Medium, High, Critical]
TEMPLATES = [
    ("Infrastructure", "AC malfunction", [
        "C block 3rd floor AC romba hot ah irukku.",
        "The AC in C Block third floor isn't cooling at all.",
        "AC in our classroom has been broken for three days.",
    ], [0.1, 0.3, 0.5, 0.1], 0, False),

    ("Infrastructure", "Computer/network issues", [
        "Lab 3 PC is very slow.",
        "Computer in Lab 3 takes forever to open applications.",
        "Lab 3 la PC romba slow ah iruku.",
        "The systems in Lab 3 are extremely slow and keep freezing.",
    ], [0.2, 0.5, 0.3, 0.0], 2, False),

    ("Infrastructure", "Computer/network issues", [
        "Lab 2 computers won't connect to the internet.",
        "Network in Lab 2 keeps disconnecting during class.",
    ], [0.2, 0.5, 0.3, 0.0], 3, False),

    ("Infrastructure", "Water leakage", [
        "Water tank is leaking.",
        "There's water leaking near the first floor area.",
        "Water leak aaguthu near the lab.",
    ], [0.0, 0.3, 0.5, 0.2], None, False),  # location intentionally unknown

    ("Infrastructure", "Projector not working", [
        "The projector in Lab 3 isn't working.",
        "Projector in our classroom has been down all week.",
    ], [0.3, 0.5, 0.2, 0.0], 2, False),

    ("Academic", "Internal marks not updated", [
        "Our internal marks still haven't been updated.",
        "Professor hasn't uploaded our internal marks yet.",
        "Marks for the last internal are still missing on the portal.",
    ], [0.1, 0.6, 0.3, 0.0], 4, False),

    ("Faculty", "Assignment deadline changes", [
        "The professor keeps changing the assignment deadline.",
        "Sir keeps changing our assignment deadline and it's hard to manage.",
    ], [0.2, 0.6, 0.2, 0.0], None, False),

    ("Canteen", "Food quality", [
        "The food has been really bad this week.",
        "Canteen food quality has dropped a lot recently.",
    ], [0.3, 0.5, 0.2, 0.0], 5, False),

    ("Administration", "Certificate processing delay", [
        "I've been trying to get my bonafide certificate but nobody is responding.",
        "Office staff hasn't processed my certificate request in two weeks.",
    ], [0.2, 0.6, 0.2, 0.0], None, False),

    ("Infrastructure", "Electrical hazard", [
        "There is an electrical spark coming from the socket in the lab.",
        "Exposed wiring near the staircase looks dangerous.",
    ], [0.0, 0.0, 0.3, 0.7], 7, False),

    ("Infrastructure", "Library facilities", [
        "AC in the library isn't working properly.",
        "Not enough seating available in the library during exam week.",
    ], [0.4, 0.4, 0.2, 0.0], 4, False),

    # --- Restricted / safety complaints ---
    ("Safety", "Bullying", [
        "A student keeps disturbing me during class.",
        "A student keeps bullying me in class and I don't know what to do.",
    ], [0.0, 0.1, 0.5, 0.4], None, True),

    ("Safety", "Harassment", [
        "Someone was harassing students near the staircase.",
        "I am being followed and threatened by another student.",
    ], [0.0, 0.0, 0.3, 0.7], 8, True),
]


def weighted_priority(weights):
    return random.choices(
        ["Low", "Medium", "High", "Critical"], weights=weights, k=1
    )[0]


def main():
    engine = create_engine(DB_PATH)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Departments
    dept_objs = {name: Department(department_name=name) for name in DEPARTMENTS}
    session.add_all(dept_objs.values())

    # Locations
    loc_objs = []
    for building, floor, area, specific, verified in LOCATIONS:
        loc = Location(building=building, floor=floor, area=area,
                        specific_location=specific, verified=verified,
                        discovered_at=datetime.utcnow() - timedelta(
                            days=random.randint(20, 45)))
        loc_objs.append(loc)
        session.add(loc)

    session.flush()  # get IDs

    category_to_dept = {
        "Infrastructure": "Maintenance",
        "Academic": "Academic Office",
        "Faculty": "Academic Office",
        "Administration": "Administration",
        "Canteen": "Canteen Services",
        "Safety": "Student Affairs",
        "Student-related": "Student Affairs",
    }

    now = datetime.utcnow()
    window_days = 42  # ~6 weeks of history

    complaints = []
    for i in range(NUM_COMPLAINTS):
        category, issue, variants, weights, loc_idx, restricted = random.choice(TEMPLATES)
        description = random.choice(variants)
        priority = weighted_priority(weights)

        # skew volume toward recent weeks so the trend chart shows growth
        day_offset = int(random.triangular(0, window_days, window_days * 0.15))
        created = now - timedelta(days=day_offset,
                                   hours=random.randint(0, 23))

        dept_name = category_to_dept.get(category, "Administration")
        department = dept_objs[dept_name]
        location = loc_objs[loc_idx] if loc_idx is not None else None

        # resolution: restricted complaints resolve slower and less often
        # (handled by Student Affairs, not auto-clustered); others resolve
        # per a rough per-department pace
        base_resolve_days = {
            "Maintenance": 1.8, "Academic Office": 2.4,
            "Administration": 3.1, "Student Affairs": 4.0,
            "Canteen Services": 1.5, "IT Support": 2.0,
        }.get(dept_name, 2.5)

        resolved_prob = 0.55 if not restricted else 0.35
        status = "Pending"
        resolved_at = None
        if day_offset > 3 and random.random() < resolved_prob:
            resolve_days = max(0.3, random.gauss(base_resolve_days, 0.8))
            resolved_at = created + timedelta(days=resolve_days)
            if resolved_at < now:
                status = "Resolved"
            else:
                resolved_at = None
                status = "Under Review"
        elif day_offset <= 5 and random.random() < 0.2:
            status = "Under Review"

        c = Complaint(
            student_id=None if restricted and random.random() < 0.4
                        else f"S{1000 + i}",
            description=description,
            category=category,
            issue=issue,
            priority=priority,
            original_priority=priority,
            priority_overridden=False,
            status=status,
            department=department,
            location=location,
            created_at=created,
            resolved_at=resolved_at,
            is_restricted=restricted,
            restricted_reason=issue.lower() if restricted else None,
        )
        complaints.append(c)
        session.add(c)

    session.flush()

    # A small number of admin priority overrides, logged rather than
    # silently overwritten (review section 3.3)
    non_restricted = [c for c in complaints if not c.is_restricted]
    for c in random.sample(non_restricted, k=12):
        levels = ["Low", "Medium", "High", "Critical"]
        cur = levels.index(c.priority)
        new_idx = max(0, min(3, cur + random.choice([-1, 1])))
        if new_idx != cur:
            c.priority = levels[new_idx]
            c.priority_overridden = True

    # Build issue clusters using SBERT embeddings on the
    # complaint description text (similarity_sbert.py), not just matching
    # category/issue/location labels. Restricted complaints are excluded
    # from clustering entirely, per the safety-handling design.
    from similarity_sbert import cluster_complaints

    pairs = [(c.complaint_id, c.description) for c in non_restricted]
    id_to_complaint = {c.complaint_id: c for c in non_restricted}
    clusters = cluster_complaints(pairs, min_cluster_size=2)

    for label, member_ids in clusters.items():
        members = [id_to_complaint[cid] for cid in member_ids]
        # name the cluster after the most common `issue` tag among its
        # members, since the classifier's tag is more readable than raw
        # complaint text for a dashboard label
        issue_names = [m.issue for m in members if m.issue]
        issue_name = (max(set(issue_names), key=issue_names.count)
                      if issue_names else "Related complaints")
        category = members[0].category

        cluster = IssueCluster(issue_name=issue_name, category=category,
                                complaint_count=len(members))
        session.add(cluster)
        session.flush()
        for m in members:
            m.cluster_id = cluster.cluster_id

    session.commit()

    total = session.query(Complaint).count()
    restricted_count = session.query(Complaint).filter_by(is_restricted=True).count()
    clusters = session.query(IssueCluster).count()
    print(f"Seeded {total} complaints ({restricted_count} restricted, "
          f"excluded from analytics), {clusters} issue clusters, "
          f"{len(loc_objs)} locations, {len(dept_objs)} departments.")
    print(f"Database written to: campus_grievance.db")

    session.close()


if __name__ == "__main__":
    main()
