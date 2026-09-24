"""
FastAPI backend for the Campus Grievance Intelligence & Analytics System.

Endpoints are split cleanly:
  - /api/complaints        general complaint CRUD (excludes restricted ones
                            from any listing that isn't explicitly the
                            restricted queue)
  - /api/complaints/classify   run text through the NLP pipeline WITHOUT
                            saving — used by the chat interface to show a
                            confirmation screen before committing anything
  - /api/restricted        the confidential safety/misconduct queue
  - /api/analytics/*        dashboard aggregates (also excludes restricted)
  - /api/locations          location list + verification

Run: uvicorn main:app --reload --port 8000
"""
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
from models import Base, Department, Location, IssueCluster, Complaint
from classifier import classify_complaint
from similarity_sbert import find_similar
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from ml.ml_classifier import classify_category_ml, classify_category_guarded
import secrets
import os
DB_PATH = "sqlite:///data/campus_grievance.db"
import secrets
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

app = FastAPI(title="Campus Grievance Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)
security = HTTPBasic()
dashboard_basic = HTTPBasic(scheme_name="DashboardBasic")


def verify_dashboard(
    credentials: HTTPBasicCredentials = Depends(dashboard_basic)
):
    user = os.environ.get("DASHBOARD_USER")
    password = os.environ.get("DASHBOARD_PASS")

    if not user or not password:
        raise HTTPException(
            status_code=503,
            detail="Dashboard credentials are not configured on the server"
        )

    user_ok = secrets.compare_digest(
        credentials.username.encode(),
        user.encode()
    )

    pass_ok = secrets.compare_digest(
        credentials.password.encode(),
        password.encode()
    )

    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"}
        )

def verify_staff(
    credentials: HTTPBasicCredentials = Depends(security)
):
    correct_user = secrets.compare_digest(
        credentials.username,
        "studentaffairs"
    )

    correct_pass = secrets.compare_digest(
        credentials.password,
        "staff123"
    )

    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _find_or_stage_location(db: Session, location) -> Optional[int]:
    """
    Find an existing location matching the classifier output.

    If no usable location was extracted, return None.
    If a new location is discovered, create it as unverified so
    staff can verify it later.
    """
    if not location:
        return None

    building = getattr(location, "building", None)
    floor = getattr(location, "floor", None)
    area = getattr(location, "area", None)
    specific = getattr(location, "specific", None)

    if not any([building, floor, area, specific]):
        return None

    # Reuse an existing location when all extracted fields match.
    existing = (
        db.query(Location)
        .filter(
            Location.building == building,
            Location.floor == floor,
            Location.area == area,
            Location.specific_location == specific,
        )
        .first()
    )

    if existing:
        return existing.location_id

    # New location → stage it as unverified.
    new_location = Location(
        building=building,
        floor=floor,
        area=area,
        specific_location=specific,
        verified=False,
    )

    db.add(new_location)
    db.flush()

    return new_location.location_id
# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ComplaintOut(BaseModel):
    complaint_id: int
    description: str
    category: Optional[str]
    issue: Optional[str]
    priority: Optional[str]
    priority_overridden: bool
    status: str
    department: Optional[str]
    location: Optional[str]
    created_at: datetime
    resolved_at: Optional[datetime]
    resolution_days: Optional[float]

    class Config:
        from_attributes = True


class ComplaintCreate(BaseModel):
    description: str
    category: str
    issue: str
    priority: str
    department_name: Optional[str] = None
    location_id: Optional[int] = None
    is_restricted: bool = False
    restricted_reason: Optional[str] = None
    student_id: Optional[str] = None


class ComplaintSubmit(BaseModel):
    """What the chat interface actually sends: raw text, nothing pre-tagged."""
    description: str
    student_id: Optional[str] = None
    anonymous: bool = False

class StatusUpdate(BaseModel):
    status: str  # e.g. "Under Review", "Resolved"


def serialize(c: Complaint) -> dict:
    return {
        "complaint_id": c.complaint_id,
        "description": c.description,
        "category": c.category,
        "issue": c.issue,
        "priority": c.priority,
        "priority_overridden": c.priority_overridden,
        "status": c.status,
        "department": c.department.department_name if c.department else None,
        "location": c.location.display_name if c.location else "Unknown",
        "created_at": c.created_at,
        "resolved_at": c.resolved_at,
        "resolution_days": c.resolution_days,
    }


# ---------------------------------------------------------------------------
# Complaints (non-restricted only)
# ---------------------------------------------------------------------------

@app.get("/api/complaints", dependencies=[Depends(verify_dashboard)])
def list_complaints(
    department: str | None = None,
    priority: str | None = None,
    status: str | None = None,
    category: str | None = None,
    db: Session = Depends(get_db)
):
    q = db.query(Complaint).filter(
        Complaint.is_restricted == False
    )  # noqa: E712

    if department:
        q = q.join(Department).filter(
            Department.department_name == department
        )

    if priority:
        q = q.filter(Complaint.priority == priority)

    if status:
        q = q.filter(Complaint.status == status)

    if category:
        q = q.filter(Complaint.category == category)

    q = q.order_by(Complaint.created_at.desc())

    return [serialize(c) for c in q.all()]

@app.post("/api/complaints/classify")
def classify_only(payload: ComplaintSubmit):
    """
    Runs the NLP pipeline WITHOUT saving anything. This is what the chat
    interface calls to build the confirmation screen (doc: "lets the
    student correct the model" — a stated strength, kept in the pipeline
    deliberately). Nothing is written to the database until the student
    confirms via /api/complaints/submit.
    """
    result = classify_complaint(payload.description)
    return {
        "is_restricted": result.is_restricted,
        "restricted_reason": result.restricted_reason,
        "category": result.category,
        "issue": result.issue,
        "priority": result.priority,
        "department_hint": result.department_hint,
        "location": {
            "building": result.location.building,
            "floor": result.location.floor,
            "area": result.location.area,
            "specific": result.location.specific,
            "confidence": result.location.confidence,
        },
        "reasoning": result.reasoning,
      # Lost & Found redirect
        "redirect": result.reasoning.get("redirect", False),
        "redirect_message": result.reasoning.get("redirect_message"),
    }


@app.post("/api/complaints/submit")
def submit_complaint(payload: ComplaintSubmit, db: Session = Depends(get_db)):
    """
    Full pipeline + save. This is the real submission endpoint the chat
    interface calls after the student confirms the classification shown
    on the confirmation screen.
    """
    result = classify_complaint(payload.description)
    dept = db.query(Department).filter_by(
        department_name=result.department_hint).first()

    student_id = None if payload.anonymous else payload.student_id

    if result.is_restricted:
        c = Complaint(
            student_id=student_id,
            description=payload.description,
            category="Safety",
            issue=None,
            priority=result.priority,
            original_priority=result.priority,
            status="Pending",
            department=dept,
            location_id=None,
            is_restricted=True,
            restricted_reason=result.restricted_reason,
        )
        db.add(c)
        db.commit()
        db.refresh(c)
        return {
            "restricted": True,
            "reference_number": c.complaint_id,
            "message": ("Your report has been received confidentially. "
                        "Please reach out to the Student Affairs office "
                        "directly for support — quote reference #"
                        f"{c.complaint_id} if you contact them."),
        }

    location_id = _find_or_stage_location(db, result.location)

    # Duplicate/cluster check: compare against existing non-restricted
    # complaints in the same category (cheap prefilter — comparing against
    # the whole corpus on every submission doesn't scale, and cross-category
    # matches are usually noise anyway).
    existing_in_category = (
        db.query(Complaint)
        .filter(Complaint.is_restricted == False,  # noqa: E712
                Complaint.category == result.category)
        .all()
    )
    pairs = [(ec.complaint_id, ec.description) for ec in existing_in_category]
    similar = find_similar(payload.description, pairs, top_k=3)

    cluster_id = None
    if similar:
        # Join the cluster of the most similar existing complaint, if it
        # has one; otherwise start a new cluster containing both.
        best_match = db.query(Complaint).get(similar[0].complaint_id)
        if best_match.cluster_id:
            cluster_id = best_match.cluster_id
            cluster = db.query(IssueCluster).get(cluster_id)
            cluster.complaint_count += 1
        else:
            cluster = IssueCluster(
                issue_name=result.issue or "Related complaints",
                category=result.category,
                complaint_count=2,
            )
            db.add(cluster)
            db.flush()
            cluster_id = cluster.cluster_id
            best_match.cluster_id = cluster_id

    c = Complaint(
        student_id=student_id,
        description=payload.description,
        category=result.category,
        issue=result.issue,
        priority=result.priority,
        original_priority=result.priority,
        status="Pending",
        department=dept,
        location_id=location_id,
        cluster_id=cluster_id,
        is_restricted=False,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return {
        "restricted": False,
        "complaint": serialize(c),
        "reasoning": result.reasoning,
        "similar_complaints": [
            {"complaint_id": s.complaint_id, "similarity": s.similarity,
             "description": s.description}
            for s in similar
        ],
    }




VALID_STATUSES = {"Open", "Under Review", "Resolved"}

@app.patch("/api/complaints/{complaint_id}/status", dependencies=[Depends(verify_dashboard)])
def update_complaint_status(
    complaint_id: int,
    update: StatusUpdate,
    db: Session = Depends(get_db)
):
    if update.status not in VALID_STATUSES:
        raise HTTPException(
            400,
            f"status must be one of {VALID_STATUSES}"
        )

    c = db.query(Complaint).get(complaint_id)

    if not c:
        raise HTTPException(404, "Complaint not found")

    if c.is_restricted:
        raise HTTPException(
            403,
            "Restricted complaints must be updated through "
            "/api/restricted/{complaint_id}/status"
        )

    c.status = update.status

    if update.status == "Resolved":
        c.resolved_at = datetime.utcnow()
    elif c.resolved_at is not None:
        c.resolved_at = None

    db.commit()
    db.refresh(c)

    return serialize(c)
# ---------------------------------------------------------------------------
# Restricted queue (review section 5.1) — separate endpoint, not filterable
# into the general complaints list.
# ---------------------------------------------------------------------------

@app.get("/api/restricted")
def list_restricted(
    db: Session = Depends(get_db),
    staff_user: str = Depends(verify_staff)
):
    """
    Confidential queue. In a real deployment this endpoint would require
    a Student Affairs / designated-authority role — auth is out of scope
    for this build, but the separation of data and endpoint is the point.
    """
    q = db.query(Complaint).filter(Complaint.is_restricted == True)  # noqa: E712
    q = q.order_by(Complaint.created_at.desc())
    return [serialize(c) for c in q.all()]


@app.get("/api/restricted/count")
def restricted_count(
    db: Session = Depends(get_db),
    staff_user: str = Depends(verify_staff)
):
    """Aggregate-only view for the main dashboard: a count, no content."""
    count = db.query(Complaint).filter(
        Complaint.is_restricted == True
    ).count()  # noqa: E712

    return {"restricted_complaint_count": count}

@app.patch("/api/restricted/{complaint_id}/status")
def update_restricted_status(
    complaint_id: int,
    update: StatusUpdate,
    db: Session = Depends(get_db),
    staff_user: str = Depends(verify_staff)
):
    if update.status not in VALID_STATUSES:
        raise HTTPException(400, f"status must be one of {VALID_STATUSES}")

    c = db.query(Complaint).get(complaint_id)

    if not c:
        raise HTTPException(404, "Complaint not found")

    if not c.is_restricted:
        raise HTTPException(
            400,
            "This complaint is not restricted — use "
            "/api/complaints/{id}/status instead"
        )

    c.status = update.status

    if update.status == "Resolved":
        c.resolved_at = datetime.utcnow()
    elif c.resolved_at is not None:
        c.resolved_at = None

    db.commit()
    db.refresh(c)

    return serialize(c)
# ---------------------------------------------------------------------------
# Analytics (all exclude restricted complaints)
# ---------------------------------------------------------------------------

@app.get("/api/analytics/kpis", dependencies=[Depends(verify_dashboard)])
def kpis(db: Session = Depends(get_db)):
    base = db.query(Complaint).filter(Complaint.is_restricted == False)  # noqa: E712
    total = base.count()
    pending = base.filter(Complaint.status == "Pending").count()
    high_priority = base.filter(
        Complaint.priority.in_(["High", "Critical"])).count()
    resolved = base.filter(Complaint.status == "Resolved").all()
    avg_resolution = (
        round(sum(c.resolution_days for c in resolved) / len(resolved), 2)
        if resolved else 0
    )
    return {
        "total_complaints": total,
        "pending": pending,
        "high_priority": high_priority,
        "avg_resolution_days": avg_resolution,
    }


@app.get("/api/analytics/categories", dependencies=[Depends(verify_dashboard)])
def category_distribution(db: Session = Depends(get_db)):
    rows = (
        db.query(Complaint.category, func.count(Complaint.complaint_id))
        .filter(Complaint.is_restricted == False)  # noqa: E712
        .group_by(Complaint.category)
        .order_by(func.count(Complaint.complaint_id).desc())
        .all()
    )
    return [{"category": cat, "count": cnt} for cat, cnt in rows]


@app.get("/api/analytics/hotspots", dependencies=[Depends(verify_dashboard)])
def location_hotspots(db: Session = Depends(get_db)):
    rows = (
        db.query(Location, func.count(Complaint.complaint_id).label("cnt"))
        .join(Complaint, Complaint.location_id == Location.location_id)
        .filter(Complaint.is_restricted == False)  # noqa: E712
        .group_by(Location.location_id)
        .order_by(func.count(Complaint.complaint_id).desc())
        .limit(10)
        .all()
    )
    return [{"location": loc.display_name, "verified": loc.verified,
             "count": cnt} for loc, cnt in rows]


@app.get("/api/analytics/recurring-issues", dependencies=[Depends(verify_dashboard)])
def recurring_issues(db: Session = Depends(get_db)):
    rows = (
        db.query(
            Complaint.issue.label("issue_name"),
            Complaint.category.label("category"),
            func.count(Complaint.complaint_id).label("complaint_count"),
        )
        .filter(Complaint.is_restricted == False)  # keep Safety complaints out of public analytics
        .group_by(Complaint.issue, Complaint.category)
        .order_by(func.count(Complaint.complaint_id).desc())
        .limit(10)
        .all()
    )
    return [
        {"issue_name": r.issue_name, "category": r.category, "complaint_count": r.complaint_count}
        for r in rows
    ]

@app.get("/api/analytics/trends", dependencies=[Depends(verify_dashboard)])
def trends(weeks: int = 6, db: Session = Depends(get_db)):
    now = datetime.utcnow()
    result = []
    for i in range(weeks, 0, -1):
        start = now - timedelta(weeks=i)
        end = now - timedelta(weeks=i - 1)
        count = (
            db.query(Complaint)
            .filter(Complaint.is_restricted == False)  # noqa: E712
            .filter(Complaint.created_at >= start, Complaint.created_at < end)
            .count()
        )
        result.append({"week_label": f"Week {weeks - i + 1}", "count": count})
    return result


@app.get("/api/analytics/resolution-by-department", dependencies=[Depends(verify_dashboard)])
def resolution_by_department(db: Session = Depends(get_db)):
    depts = db.query(Department).all()
    result = []
    for d in depts:
        resolved = [c for c in d.complaints
                    if c.resolved_at and not c.is_restricted]
        avg = (round(sum(c.resolution_days for c in resolved) / len(resolved), 2)
               if resolved else None)
        backlog = sum(1 for c in d.complaints
                      if c.status != "Resolved" and not c.is_restricted)
        result.append({"department": d.department_name,
                        "avg_resolution_days": avg,
                        "unresolved_backlog": backlog})
    return result


# ---------------------------------------------------------------------------
# Locations (progressive discovery + verification, doc section 22)
# ---------------------------------------------------------------------------

@app.get("/api/locations", dependencies=[Depends(verify_dashboard)])
def list_locations(db: Session = Depends(get_db)):
    locs = db.query(Location).all()
    return [{"location_id": l.location_id, "display_name": l.display_name,
             "verified": l.verified} for l in locs]


@app.post("/api/locations/{location_id}/verify", dependencies=[Depends(verify_dashboard)])
def verify_location(location_id: int, db: Session = Depends(get_db)):
    loc = db.query(Location).get(location_id)
    if not loc:
        raise HTTPException(404, "Location not found")
    loc.verified = True
    db.commit()
    return {"location_id": loc.location_id, "verified": True}


@app.get("/")
def root():
    return {"status": "ok", "service": "campus-grievance-api"}
