# check_library_ids.py
from models import Complaint
from main import SessionLocal

ids = [12, 74, 81, 93, 176]  # ADJUST: swap in whichever 5 IDs actually flipped Infra->Academic — 176 was one of the Hazard flips, double check the list from your output

db = SessionLocal()
for cid in ids:
    c = db.query(Complaint).filter_by(complaint_id=cid).first()
    if c:
        print(f"[{c.complaint_id}] category={c.category} | {c.description}")
db.close()