"""
OralGuard AI — Application Configuration
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Application ──
    APP_NAME: str = "OralGuard AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    # ── Server ──
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Database ──
    DATABASE_URL: str = f"sqlite+aiosqlite:///{Path(__file__).parent.resolve() / 'oralguard.db'}"
    # For production PostgreSQL:
    # DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/oralguard"

    # ── Security ──
    SECRET_KEY: str = "oralguard-dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # ── File Storage ──
    UPLOAD_DIR: str = os.environ.get("UPLOAD_DIR", str(Path(__file__).parent / "uploads"))
    MAX_IMAGE_SIZE_MB: int = 20
    ALLOWED_IMAGE_TYPES: list[str] = ["image/jpeg", "image/png", "image/heic", "image/webp"]

    # ── AI Model Paths ──
    MODEL_DIR: str = str(Path(__file__).parent / "model_weights")
    DETECTION_MODEL_PATH: str = str(
        (Path(__file__).parent / "model_weights" / "yolov8m.pt")
        if (Path(__file__).parent / "model_weights" / "yolov8m.pt").exists()
        else (Path(__file__).parent / "yolov8m.pt")
    )
    CLASSIFICATION_MODEL_PATH: str = str(Path(__file__).parent / "model_weights" / "classification_model.pt")
    SEGMENTATION_MODEL_PATH: str = str(Path(__file__).parent / "model_weights" / "segmentation_model.pt")
    FEATURE_MODEL_PATH: str = str(Path(__file__).parent / "model_weights" / "feature_model.pt")
    FUSION_MODEL_PATH: str = str(Path(__file__).parent / "model_weights" / "fusion_model.json")

    # ── AI Model Settings ──
    DETECTION_INPUT_SIZE: int = 640
    CLASSIFICATION_INPUT_SIZE: int = 380
    SEGMENTATION_INPUT_SIZE: int = 512
    CONFIDENCE_THRESHOLD: float = 0.5
    DEVICE: str = "cpu"  # "cuda" for GPU

    # ── Classification Labels ──
    PRIMARY_CLASSES: list[str] = [
        "aphthous_ulcer",
        "oscc",
        "other"
    ]
    APHTHOUS_SUBTYPES: list[str] = [
        "minor",
        "major",
        "herpetiform"
    ]
    OSCC_SUBTYPES: list[str] = [
        "endophytic",
        "exophytic",
        "verrucous"
    ]

    # ── Risk Thresholds ──
    RISK_LOW_MAX: int = 25
    RISK_MEDIUM_MAX: int = 50
    RISK_HIGH_MAX: int = 75
    # Above 75 = URGENT

    # ── Redis / Celery ──
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── External APIs ──
    SMS_API_KEY: str = ""
    SMS_API_URL: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)
