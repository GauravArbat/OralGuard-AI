"""
OralGuard AI — Pydantic Schemas (Request/Response Models)
"""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


# ──────────────────────────────────────────────
# Auth Schemas
# ──────────────────────────────────────────────

class UserCreate(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    password: str
    full_name: Optional[str] = None
    role: str = "patient"
    age: Optional[int] = None
    gender: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    full_name: Optional[str] = None
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class LoginRequest(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    password: str


# ──────────────────────────────────────────────
# Image Upload Schemas
# ──────────────────────────────────────────────

class ImageQualityScore(BaseModel):
    resolution: int = Field(ge=0, le=25)
    focus: int = Field(ge=0, le=25)
    lighting: int = Field(ge=0, le=25)
    field_of_view: int = Field(ge=0, le=25)
    total: int = Field(ge=0, le=100)
    acceptable: bool


class ImageUploadResponse(BaseModel):
    screening_id: str
    image_id: str
    quality_score: ImageQualityScore
    status: str
    message: str


# ──────────────────────────────────────────────
# Detection Schemas
# ──────────────────────────────────────────────

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float


class DetectionResult(BaseModel):
    lesion_detected: bool
    bounding_boxes: list[BoundingBox] = []
    num_lesions: int = 0


# ──────────────────────────────────────────────
# Segmentation Schemas
# ──────────────────────────────────────────────

class SegmentationResult(BaseModel):
    mask_path: str
    dice_score: Optional[float] = None
    lesion_area_pixels: int = 0
    lesion_area_percentage: float = 0.0


# ──────────────────────────────────────────────
# Classification Schemas
# ──────────────────────────────────────────────

class ClassificationProbability(BaseModel):
    label: str
    probability: float
    display_name: str


class ClassificationResult(BaseModel):
    primary_class: str
    primary_class_display: str
    confidence: float
    subtype: Optional[str] = None
    subtype_display: Optional[str] = None
    probabilities: list[ClassificationProbability]


# ──────────────────────────────────────────────
# Clinical Feature Schemas
# ──────────────────────────────────────────────

class ClinicalFeatures(BaseModel):
    ulceration: bool = False
    border_type: str = "unknown"  # regular, irregular, rolled, diffuse
    red_component: str = "none"  # none, mild, moderate, severe
    white_component: str = "none"  # none, mild, moderate, severe
    mixed_red_white: bool = False
    exophytic_growth: bool = False
    necrotic_surface: bool = False
    anatomical_location: str = "unknown"


# ──────────────────────────────────────────────
# Questionnaire Schemas
# ──────────────────────────────────────────────

class QuestionOption(BaseModel):
    value: str
    label: str


class QuestionItem(BaseModel):
    id: str
    question: str
    question_type: str  # text, number, select, multiselect, boolean, scale
    options: Optional[list[QuestionOption]] = None
    required: bool = True
    category: str  # onset, recurrence, symptoms, risk_factors, systemic
    clinical_rationale: str  # Why we ask this


class QuestionnaireSchema(BaseModel):
    screening_id: str
    questions: list[QuestionItem]
    total_questions: int


class QuestionnaireSubmission(BaseModel):
    screening_id: Optional[str] = None
    responses: Optional[dict] = None  # question_id -> answer
    answers: Optional[dict] = None  # alias for responses

    def get_responses(self) -> dict:
        return self.responses if self.responses is not None else (self.answers or {})


# ──────────────────────────────────────────────
# Differential Diagnosis Schemas
# ──────────────────────────────────────────────

class DifferentialItem(BaseModel):
    rank: int
    condition: str
    display_name: str
    probability: float
    supporting_features: list[str] = []
    opposing_features: list[str] = []
    icd10_code: Optional[str] = None


# ──────────────────────────────────────────────
# Risk Assessment Schemas
# ──────────────────────────────────────────────

class RiskAssessment(BaseModel):
    risk_score: int = Field(ge=0, le=100)
    risk_level: str  # low, medium, high, urgent
    image_risk: float
    questionnaire_risk: float
    clinical_features_risk: float
    contributing_factors: list[str] = []


# ──────────────────────────────────────────────
# Full Screening Result Schema
# ──────────────────────────────────────────────

class ScreeningResult(BaseModel):
    screening_id: str
    status: str
    created_at: datetime

    # Images
    original_image_url: str
    gradcam_image_url: Optional[str] = None

    # Detection
    detection: DetectionResult

    # Classification
    classification: ClassificationResult

    # Clinical features
    features: ClinicalFeatures

    # Risk
    risk: RiskAssessment

    # Differentials
    differentials: list[DifferentialItem]

    # Recommendations
    recommendations: list[str]
    referral_urgency: str

    # Disclaimer
    disclaimer: str = (
        "This is an AI-assisted screening tool and does not replace "
        "professional clinical examination or histopathological diagnosis. "
        "Consult a qualified dental professional for definitive diagnosis."
    )


# ──────────────────────────────────────────────
# Report Schemas
# ──────────────────────────────────────────────

class ReportRequest(BaseModel):
    screening_id: str
    include_gradcam: bool = True
    language: str = "en"


class ReportResponse(BaseModel):
    report_id: str
    screening_id: str
    pdf_url: Optional[str] = None
    share_token: str
    created_at: datetime


# ──────────────────────────────────────────────
# Dashboard Schemas
# ──────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_screenings: int
    screenings_today: int
    positive_rate: float
    referral_rate: float
    risk_distribution: dict  # {"low": 120, "medium": 45, ...}
    top_conditions: list[dict]
    screenings_by_district: list[dict]


class ScreeningListItem(BaseModel):
    id: str
    user_name: Optional[str]
    primary_diagnosis: Optional[str]
    risk_level: Optional[str]
    risk_score: Optional[int]
    status: str
    created_at: datetime
