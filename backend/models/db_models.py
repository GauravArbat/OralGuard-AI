"""
OralGuard AI — SQLAlchemy ORM Models
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, Text, Boolean,
    DateTime, ForeignKey, JSON, Enum as SAEnum
)
from sqlalchemy.orm import relationship
from database import Base

import enum


# ── Enums ──

class UserRole(str, enum.Enum):
    PATIENT = "patient"
    CHW = "chw"  # Community Health Worker
    DENTIST = "dentist"
    ADMIN = "admin"


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ScreeningStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    ANALYZING = "analyzing"
    QUESTIONNAIRE = "questionnaire"
    COMPLETED = "completed"
    REFERRED = "referred"
    FAILED = "failed"


class LesionType(str, enum.Enum):
    APHTHOUS_ULCER = "aphthous_ulcer"
    OSCC = "oscc"
    OTHER = "other"


class AphthousSubtype(str, enum.Enum):
    MINOR = "minor"
    MAJOR = "major"
    HERPETIFORM = "herpetiform"


# ── Helper ──

def generate_uuid():
    return str(uuid.uuid4())


# ── Models ──

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=True)
    phone = Column(String, unique=True, nullable=True)
    hashed_password = Column(String, nullable=True)
    full_name = Column(String, nullable=True)
    role = Column(SAEnum(UserRole), default=UserRole.PATIENT)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    state = Column(String, nullable=True)
    district = Column(String, nullable=True)
    abha_id = Column(String, nullable=True)  # Ayushman Bharat Health Account
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    screenings = relationship("Screening", back_populates="user")


class Screening(Base):
    __tablename__ = "screenings"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    status = Column(SAEnum(ScreeningStatus), default=ScreeningStatus.UPLOADED)

    # AI Results
    primary_diagnosis = Column(SAEnum(LesionType), nullable=True)
    aphthous_subtype = Column(SAEnum(AphthousSubtype), nullable=True)
    confidence_score = Column(Float, nullable=True)
    risk_score = Column(Integer, nullable=True)  # 0-100
    risk_level = Column(SAEnum(RiskLevel), nullable=True)

    # AI Classification Probabilities
    classification_probs = Column(JSON, nullable=True)
    # Example: {"aphthous_ulcer": 0.89, "oscc": 0.02, "other": 0.09}

    # Detected Features
    detected_features = Column(JSON, nullable=True)
    # Example: {"ulceration": true, "border_type": "regular", ...}

    # Differential Diagnosis
    differential_diagnoses = Column(JSON, nullable=True)
    # Example: [{"condition": "RAS", "probability": 0.89}, ...]

    # Recommendations
    recommendations = Column(JSON, nullable=True)
    referral_urgency = Column(String, nullable=True)

    # Metadata
    anatomical_location = Column(String, nullable=True)
    image_quality_score = Column(Integer, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="screenings")
    images = relationship("ScreeningImage", back_populates="screening")
    questionnaire = relationship("QuestionnaireResponse", back_populates="screening", uselist=False)
    report = relationship("DiagnosisReport", back_populates="screening", uselist=False)


class ScreeningImage(Base):
    __tablename__ = "screening_images"

    id = Column(String, primary_key=True, default=generate_uuid)
    screening_id = Column(String, ForeignKey("screenings.id"), nullable=False)

    # File info
    original_filename = Column(String, nullable=True)
    stored_path = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    mime_type = Column(String, nullable=True)

    # Processed versions
    preprocessed_path = Column(String, nullable=True)
    gradcam_path = Column(String, nullable=True)
    segmentation_mask_path = Column(String, nullable=True)
    detection_bbox_path = Column(String, nullable=True)

    # Detection results
    bbox_x = Column(Float, nullable=True)
    bbox_y = Column(Float, nullable=True)
    bbox_w = Column(Float, nullable=True)
    bbox_h = Column(Float, nullable=True)
    detection_confidence = Column(Float, nullable=True)

    # Image quality
    resolution_score = Column(Integer, nullable=True)
    focus_score = Column(Integer, nullable=True)
    lighting_score = Column(Integer, nullable=True)
    fov_score = Column(Integer, nullable=True)
    quality_total = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    screening = relationship("Screening", back_populates="images")


class QuestionnaireResponse(Base):
    __tablename__ = "questionnaire_responses"

    id = Column(String, primary_key=True, default=generate_uuid)
    screening_id = Column(String, ForeignKey("screenings.id"), nullable=False, unique=True)

    # Structured questionnaire answers stored as JSON
    responses = Column(JSON, nullable=False, default=dict)
    # Example structure:
    # {
    #   "onset_duration_days": 5,
    #   "recurrent": true,
    #   "recurrence_frequency": "monthly",
    #   "pain_level": 6,
    #   "vesicle_preceded": false,
    #   "ulcer_count": 1,
    #   "ulcer_size_mm": 6,
    #   "location": "buccal_mucosa",
    #   "tobacco_use": "none",
    #   "alcohol_use": "none",
    #   "medical_conditions": [],
    #   "medications": [],
    #   "weight_loss": false,
    #   "lymphadenopathy": false,
    #   ...
    # }

    # Risk factor scores computed from responses
    tobacco_risk_score = Column(Float, default=0.0)
    alcohol_risk_score = Column(Float, default=0.0)
    history_risk_score = Column(Float, default=0.0)
    clinical_risk_score = Column(Float, default=0.0)
    total_questionnaire_risk = Column(Float, default=0.0)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    screening = relationship("Screening", back_populates="questionnaire")


class DiagnosisReport(Base):
    __tablename__ = "diagnosis_reports"

    id = Column(String, primary_key=True, default=generate_uuid)
    screening_id = Column(String, ForeignKey("screenings.id"), nullable=False, unique=True)

    # Report content
    report_data = Column(JSON, nullable=False)
    pdf_path = Column(String, nullable=True)
    icd10_code = Column(String, nullable=True)

    # Sharing
    share_token = Column(String, unique=True, nullable=True)
    shared_with_doctor = Column(Boolean, default=False)
    doctor_email = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    screening = relationship("Screening", back_populates="report")
