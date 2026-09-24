from classifier import classify_complaint
from models import Complaint, Department
from main import SessionLocal

def get_or_create_department(db, name):
    dept = db.query(Department).filter_by(department_name=name).first()
    if not dept:
        dept = Department(department_name=name)
        db.add(dept)
        db.flush()  # get department_id without full commit
    return dept

def reclassify_all(db):
    complaints = db.query(Complaint).all()
    print(f"Found {len(complaints)} complaints to reclassify.\n")

    changed = 0
    skipped_restricted = 0

    for c in complaints:
        if c.is_restricted:
            skipped_restricted += 1
            continue

        result = classify_complaint(c.description)

        if result.is_restricted:
            print(f"[FLAG] complaint_id={c.complaint_id} now triggers safety check "
                  f"(was category={c.category}) — review manually, not auto-converting.")
            continue

        if result.category != c.category:
            print(f"complaint_id={c.complaint_id}: {c.category} -> {result.category} "
                  f"({c.issue!r} -> {result.issue!r})")
            changed += 1

        c.category = result.category
        c.issue = result.issue
        c.priority = result.priority
        c.original_priority = result.priority  # NOT priority_overridden — leave that alone
        c.department = get_or_create_department(db, result.department_hint)
        # location intentionally untouched — don't disturb verified discovery state

    db.commit()
    print(f"\nDone. {changed} changed category. {skipped_restricted} restricted skipped.")

if __name__ == "__main__":
    db = SessionLocal()
    try:
        reclassify_all(db)
    finally:
        db.close()