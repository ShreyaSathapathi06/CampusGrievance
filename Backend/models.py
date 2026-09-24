"""
Database models for the Campus Grievance Intelligence & Analytics System.

Schema follows SRPROJECT.docx section 20, with additions from the review:
- `is_restricted` / `restricted_reason` on Complaint, for safety/misconduct
  grievances that must be excluded from clustering and public analytics.
- `verified` on Location, to support the progressive location-discovery flow.
- `overridden_priority` fields, so admin corrections become a logged signal
  rather than silently replacing the model's recommendation.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Department(Base):
    __tablename__ = "departments"

    department_id = Column(Integer, primary_key=True)
    department_name = Column(String(100), unique=True, nullable=False)

    complaints = relationship("Complaint", back_populates="department")


class Location(Base):
    __tablename__ = "locations"

    location_id = Column(Integer, primary_key=True)
    building = Column(String(100))
    floor = Column(String(50))
    area = Column(String(150))
    specific_location = Column(String(200))
    # Progressive location discovery (doc section 6 / 22): a location
    # extracted from a complaint starts unverified until an admin confirms it.
    verified = Column(Boolean, default=False)
    discovered_at = Column(DateTime, default=datetime.utcnow)

    complaints = relationship("Complaint", back_populates="location")

    @property
    def display_name(self):
        parts = [p for p in [self.building, self.floor, self.area,
                              self.specific_location] if p]
        return " - ".join(parts) if parts else "Unknown"


class IssueCluster(Base):
    __tablename__ = "issue_clusters"

    cluster_id = Column(Integer, primary_key=True)
    issue_name = Column(String(200), nullable=False)
    category = Column(String(50))
    complaint_count = Column(Integer, default=0)

    complaints = relationship("Complaint", back_populates="cluster")


class Complaint(Base):
    __tablename__ = "complaints"

    complaint_id = Column(Integer, primary_key=True)
    student_id = Column(String(50))  # nullable in practice -> anonymity
    description = Column(Text, nullable=False)

    category = Column(String(50))       # Infrastructure, Academic, Faculty,
                                         # Student-related, Safety, Canteen,
                                         # Administration, Other
    issue = Column(String(200))
    priority = Column(String(20))              # Low / Medium / High / Critical
    priority_overridden = Column(Boolean, default=False)
    original_priority = Column(String(20))     # what the model recommended,
                                                # kept even after admin edits
    status = Column(String(30), default="Pending")  # Pending / Under Review /
                                                      # Resolved

    department_id = Column(Integer, ForeignKey("departments.department_id"))
    location_id = Column(Integer, ForeignKey("locations.location_id"))
    cluster_id = Column(Integer, ForeignKey("issue_clusters.cluster_id"),
                         nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    # --- Safety / misconduct handling (review section 5.1) ---
    # Restricted complaints are excluded from clustering and from all
    # aggregate dashboard charts. They're visible only via the restricted
    # queue, and reported on the main dashboard as a count with no content.
    is_restricted = Column(Boolean, default=False)
    restricted_reason = Column(String(100), nullable=True)  # e.g. "harassment"

    department = relationship("Department", back_populates="complaints")
    location = relationship("Location", back_populates="complaints")
    cluster = relationship("IssueCluster", back_populates="complaints")

    @property
    def resolution_days(self):
        if self.resolved_at:
            return round((self.resolved_at - self.created_at).total_seconds()
                         / 86400, 2)
        return None
