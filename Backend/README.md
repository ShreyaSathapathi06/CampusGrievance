# Campus Grievance Intelligence & Analytics System

A conversational, NLP-powered grievance reporting and analytics platform for campus (scoped to SRM Vadapalani). Students describe issues in plain natural language through a chat interface; the backend classifies, prioritizes, and routes each complaint automatically, while an admin dashboard gives staff a queue to act on and campus-level analytics to track patterns over time.

This is a full-fledged system, not a hackathon prototype — the classification pipeline is trained on real hand-labeled data rather than hardcoded keyword rules, and sensitive reports (harassment, threats, weapons) are routed to a confidential queue instead of the public dashboard.

---

## Features

- **Conversational complaint intake** — students describe an issue in their own words; no rigid form-filling
- **Two-step confirm flow** — the system classifies and shows a draft (category, issue, priority, department, location) before anything is saved, with the option to edit or add location detail
- **Trained ML classification** — SBERT sentence embeddings (`all-MiniLM-L6-v2`) + logistic regression, trained on 114 hand-labeled complaints, replacing an earlier keyword-based classifier that turned out to be overfitting (100% on its own dev set, 43.2% on a genuinely held-out set)
- **Semantic issue tagging** — issue sub-categories (e.g. "Water leakage," "Furniture issues") are matched via prototype-sentence embeddings rather than brittle keyword matching
- **Confidential Safety queue** — harassment, threats, weapons, and similar reports are keyword-detected up front, excluded from public analytics/clustering, and routed to a restricted queue behind basic auth
- **Infrastructure Hazard separation** — physical hazards (exposed wire, blocked exit, slippery floor) stay public and go to Maintenance, since hiding safety-relevant infrastructure issues from the dashboard would itself be unsafe
- **Duplicate/cluster detection** — new complaints are compared against existing ones via SBERT similarity so repeated issues surface as a cluster instead of duplicate tickets
- **Progressive location discovery** — locations aren't hardcoded; they're extracted from complaint text and admin-verified over time
- **Admin dashboard** — filterable queue (department, priority, status, category), status updates, and a separate restricted-queue view for confidential reports
- **Analytics** — KPIs, category breakdown, hotspots, recurring issues, trends, and resolution time by department

---

## Tech Stack

- **Backend:** Python, FastAPI, SQLAlchemy
- **Database:** SQLite (local)
- **ML/NLP:** Sentence-Transformers (`all-MiniLM-L6-v2`), scikit-learn (logistic regression), `pyspellchecker`
- **Frontend:** Plain HTML/CSS/JS (no framework) — chat interface and admin dashboard as self-contained pages
- **Auth:** FastAPI HTTP Basic — two separate schemes: `DashboardBasic` gates the general dashboard/analytics/complaint-management routes, `HTTPBasic` separately gates the confidential restricted-queue routes

---

## Project Structure

```
CampusGrievance/
├── Backend/
│   ├── main.py                  # FastAPI app, routes
│   ├── models.py                # SQLAlchemy schema
│   ├── classifier.py            # Safety check → ML category → location/priority/issue pipeline
│   ├── semantic_tagging.py      # SBERT prototype-based issue tagging + Lost & Found detection
│   ├── seed.py                  # Generates synthetic seeded complaints for testing
│   ├── ml/
│   │   ├── train_classifier.py  # Trains SBERT + LogisticRegression classifier
│   │   ├── ml_classifier.py     # Loads trained model, runs inference
│   │   └── hazard_keywords.py   # Keyword bump: Infrastructure → Infrastructure Hazard
│   ├── evaluation/
│   │   ├── evaluation_70.csv    # Hand-labeled complaints
│   │   └── holdout_set.csv      # Additional hand-labeled complaints
│   │                            # (both combined into one training set, deduped, 5-fold CV)
│   └── data/                     # local SQLite database (not committed)
└── Frontend/
    ├── index.html                # Student-facing chat interface
    └── admin_dashboard.html      # Staff-facing queue + analytics dashboard
```

---

## Setup

### Backend

```
cd CampusGrievance/Backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000`.

### Frontend

```
cd CampusGrievance/Frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500/index.html` for the student chat interface, or `admin_dashboard.html` for the staff dashboard.

> Both frontend pages call the backend directly at `127.0.0.1:8000` — CORS must be enabled in `main.py` for the two to talk to each other.

---

## Key API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/complaints/classify` | — | Preview classification without saving |
| POST | `/api/complaints/submit` | — | Classify and save a complaint |
| GET | `/api/complaints` | DashboardBasic | List complaints, filterable by department/priority/status/category |
| PATCH | `/api/complaints/{id}/status` | DashboardBasic | Update a complaint's status |
| GET | `/api/restricted` | HTTPBasic | List confidential Safety complaints |
| GET | `/api/restricted/count` | HTTPBasic | Count of restricted complaints (no content) |
| PATCH | `/api/restricted/{id}/status` | HTTPBasic | Update a restricted complaint's status |
| GET | `/api/locations` | DashboardBasic | List discovered locations |
| POST | `/api/locations/{id}/verify` | DashboardBasic | Mark a location as verified |
| GET | `/api/analytics/kpis` | DashboardBasic | Summary KPIs |
| GET | `/api/analytics/categories` | DashboardBasic | Complaint breakdown by category |
| GET | `/api/analytics/hotspots` | DashboardBasic | Location-based hotspots |
| GET | `/api/analytics/recurring-issues` | DashboardBasic | Recurring issue clusters |
| GET | `/api/analytics/trends` | DashboardBasic | Complaint volume over time (`weeks` query param, default 6) |
| GET | `/api/analytics/resolution-by-department` | DashboardBasic | Average resolution time by department |

---

## Classification Pipeline

1. **Safety keyword check** (deliberately non-ML, zero-tolerance) — harassment, threats, weapons, etc. short-circuit straight to the confidential queue
2. **Lost & Found redirect** — SBERT semantic match against lost-item prototype sentences
3. **Category classification** — trained SBERT + logistic regression model
4. **Infrastructure Hazard bump** — keyword check promotes borderline Infrastructure complaints to Infrastructure Hazard
5. **Issue tagging** — SBERT prototype-sentence matching for the specific issue sub-type
6. **Location extraction** — parses wing/floor/room codes (e.g. `CW204` = C Wing, 2nd Floor) or plain-language location mentions
7. **Priority recommendation** and **department routing**

---

## Known Limitations

- No full role-based auth — the restricted queue uses a single shared HTTP Basic credential, not per-user accounts
- Older seeded location data uses a legacy building-naming scheme that predates the current wing/floor/room-code extraction logic and hasn't been backfilled
- When the same underlying issue is resolved on one report, earlier duplicate reports of it aren't automatically marked resolved
- Spell-correction is generic English, not domain-aware, and can occasionally "fix" a word into something semantically wrong on heavily misspelled input

---

## Author

Shreya — BTech CSE, SRM Institute of Science and Technology
