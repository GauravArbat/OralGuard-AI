"""
OralGuard AI — Screening API Routes

Core endpoints for the screening workflow:
  POST /screening/upload    — Upload image for analysis
  GET  /screening/{id}      — Get screening status/results
  POST /screening/{id}/questionnaire — Submit questionnaire
  GET  /screening/{id}/results — Get full results
"""

import os
import uuid
import time
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from loguru import logger
from PIL import Image
import io

from database import get_db
from config import settings
from models.db_models import (
    Screening, ScreeningImage, QuestionnaireResponse, DiagnosisReport,
    ScreeningStatus, LesionType, RiskLevel,
)
from models.schemas import (
    ImageUploadResponse, ImageQualityScore,
    QuestionnaireSubmission, ScreeningResult,
)
from ai.image_preprocessor import image_preprocessor
from clinical.decision_engine import clinical_engine
from clinical.questionnaire import generate_questionnaire, filter_dependent_questions


router = APIRouter(prefix="/screening", tags=["Screening"])


# ── In-memory cache for pipeline results (use Redis in production) ──
_screening_cache: dict = {}


@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload an intraoral image for AI screening.

    Flow:
    1. Validate image format and size
    2. Compute image quality score
    3. Create screening record
    4. Run AI pipeline (detection + classification + features)
    5. Return screening ID and quality score
    """
    # Read file
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"

    # Validate
    validation = image_preprocessor.validate_image(image_bytes, mime_type)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation["message"])

    img = validation["image"]

    # Quality score
    quality = image_preprocessor.compute_quality_score(img)

    if quality["total"] < 50:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Image quality too low for reliable analysis. Please retake.",
                "quality_score": quality,
                "tips": [
                    "Ensure good lighting",
                    "Hold camera steady and close to the lesion",
                    "Make sure the lesion is fully visible",
                ],
            },
        )

    # Create screening record
    screening_id = str(uuid.uuid4())
    screening = Screening(
        id=screening_id,
        user_id="anonymous",  # Replace with auth user
        status=ScreeningStatus.ANALYZING,
        image_quality_score=quality["total"],
    )
    db.add(screening)

    # Save image
    save_dir = os.path.join(settings.UPLOAD_DIR, screening_id)
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"original{Path(file.filename or 'img.jpg').suffix}")

    img.save(image_path, quality=95)

    # Save image record
    image_record = ScreeningImage(
        screening_id=screening_id,
        original_filename=file.filename,
        stored_path=image_path,
        file_size_bytes=len(image_bytes),
        mime_type=mime_type,
        resolution_score=quality["resolution"],
        focus_score=quality["focus"],
        lighting_score=quality["lighting"],
        fov_score=quality["field_of_view"],
        quality_total=quality["total"],
    )
    db.add(image_record)

    # Run AI pipeline
    try:
        pipeline_results = clinical_engine.process_image(
            image=img,
            save_dir=save_dir,
        )

        # Cache results for questionnaire phase
        _screening_cache[screening_id] = pipeline_results

        # Update screening with initial results
        summary = pipeline_results["summary"]
        screening.status = ScreeningStatus.QUESTIONNAIRE
        screening.primary_diagnosis = LesionType(summary["primary_diagnosis"])
        screening.confidence_score = summary["confidence"]
        screening.detected_features = summary["features"]
        screening.classification_probs = {
            p["label"]: p["probability"]
            for p in pipeline_results["stages"]["classification"]["probabilities"]
        }
        screening.processing_time_ms = pipeline_results.get("processing_time_ms")

        # Update image record with detection
        detection = pipeline_results["stages"]["detection"]
        if detection["bounding_boxes"]:
            bbox = detection["bounding_boxes"][0]
            image_record.bbox_x = bbox["x"]
            image_record.bbox_y = bbox["y"]
            image_record.bbox_w = bbox["width"]
            image_record.bbox_h = bbox["height"]
            image_record.detection_confidence = bbox["confidence"]

        # Set Grad-CAM path
        gradcam_path = summary.get("gradcam_path")
        if gradcam_path:
            image_record.gradcam_path = gradcam_path

        await db.flush()

    except Exception as e:
        logger.error(f"Pipeline failed for screening {screening_id}: {e}")
        screening.status = ScreeningStatus.FAILED
        await db.flush()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    return ImageUploadResponse(
        screening_id=screening_id,
        image_id=image_record.id,
        quality_score=ImageQualityScore(**quality),
        status="questionnaire",
        message="Image analyzed. Please complete the questionnaire.",
    )


@router.get("/{screening_id}/questionnaire")
async def get_questionnaire(
    screening_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get the dynamic clinical questionnaire for a screening.
    Questions are generated based on AI initial assessment.
    """
    cached = _screening_cache.get(screening_id)
    if not cached:
        result = await db.execute(
            select(Screening).where(Screening.id == screening_id)
        )
        screening = result.scalar_one_or_none()
        if not screening:
            raise HTTPException(status_code=404, detail="Screening not found")

        diag = screening.primary_diagnosis.value if screening.primary_diagnosis else "aphthous_ulcer"
        oscc_p = 0.0
        if screening.classification_probs:
            oscc_p = float(screening.classification_probs.get("oscc", 0.0))

        q_list = generate_questionnaire(
            initial_classification=diag,
            oscc_probability=oscc_p,
            clinical_features=screening.detected_features or {},
        )
        return {
            "screening_id": screening_id,
            "questions": q_list,
            "total_questions": len(q_list),
            "initial_assessment": {
                "primary_diagnosis": diag,
                "confidence": screening.confidence_score or 0.85,
            },
        }

    questionnaire = cached.get("questionnaire", [])

    return {
        "screening_id": screening_id,
        "questions": questionnaire,
        "total_questions": len(questionnaire),
        "initial_assessment": {
            "primary_diagnosis": cached["summary"]["primary_diagnosis"],
            "confidence": cached["summary"]["confidence"],
        },
    }


@router.post("/{screening_id}/questionnaire")
async def submit_questionnaire(
    screening_id: str,
    submission: QuestionnaireSubmission,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit questionnaire responses and get final diagnosis.
    """
    # Get screening from DB
    result = await db.execute(
        select(Screening).where(Screening.id == screening_id)
    )
    screening = result.scalar_one_or_none()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    cached = _screening_cache.get(screening_id)
    if not cached:
        probs = []
        if screening.classification_probs:
            for k, v in screening.classification_probs.items():
                probs.append({"label": k, "probability": float(v), "display_name": k.replace("_", " ").title()})
        else:
            probs = [{"label": "aphthous_ulcer", "probability": 0.8, "display_name": "Aphthous Ulcer"}]

        diag = screening.primary_diagnosis.value if screening.primary_diagnosis else "aphthous_ulcer"
        cached = {
            "summary": {
                "primary_diagnosis": diag,
                "primary_diagnosis_display": diag.replace("_", " ").title(),
                "confidence": screening.confidence_score or 0.85,
                "subtype": screening.aphthous_subtype.value if screening.aphthous_subtype else None,
                "subtype_display": screening.aphthous_subtype.value.title() if screening.aphthous_subtype else None,
                "features": screening.detected_features or {},
            },
            "stages": {
                "classification": {
                    "primary_class": diag,
                    "primary_class_display": diag.replace("_", " ").title(),
                    "confidence": screening.confidence_score or 0.85,
                    "subtype": screening.aphthous_subtype.value if screening.aphthous_subtype else None,
                    "subtype_display": screening.aphthous_subtype.value.title() if screening.aphthous_subtype else None,
                    "probabilities": probs,
                },
                "features": screening.detected_features or {},
            }
        }

    responses = submission.get_responses()

    # Process questionnaire through clinical engine
    final_results = clinical_engine.process_questionnaire(
        initial_results=cached,
        questionnaire_responses=responses,
    )

    # Save questionnaire
    questionnaire_record = QuestionnaireResponse(
        screening_id=screening_id,
        responses=responses,
        total_questionnaire_risk=final_results["stages"]["fusion"]["questionnaire_risk"],
    )
    db.add(questionnaire_record)

    # Update screening
    summary = final_results["summary"]
    screening.status = ScreeningStatus.COMPLETED
    screening.risk_score = summary["risk_score"]
    screening.risk_level = RiskLevel(summary["risk_level"])
    screening.differential_diagnoses = final_results["differentials"]
    screening.recommendations = final_results["recommendations"]
    screening.referral_urgency = final_results["referral_urgency"]["risk_level"]

    await db.flush()

    # Update cache
    _screening_cache[screening_id] = final_results

    return {
        "screening_id": screening_id,
        "status": "completed",
        "results": {
            "primary_diagnosis": summary["primary_diagnosis"],
            "primary_diagnosis_display": summary.get("primary_diagnosis_display"),
            "subtype": summary.get("subtype"),
            "subtype_display": summary.get("subtype_display"),
            "confidence": summary["confidence"],
            "risk_score": summary["risk_score"],
            "risk_level": summary["risk_level"],
            "contributing_factors": summary.get("contributing_factors", []),
        },
        "differentials": final_results["differentials"],
        "recommendations": final_results["recommendations"],
        "referral": final_results["referral_urgency"],
    }


def _to_web_url(filepath: str) -> str:
    """Convert local file path inside upload dir to a web URL."""
    if not filepath:
        return None
    p = Path(filepath).as_posix()
    if "/uploads/" in p:
        return "/uploads/" + p.split("/uploads/")[1]
    return filepath


@router.get("/{screening_id}/results")
async def get_results(
    screening_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get complete screening results with browser-accessible image URLs."""
    result = await db.execute(
        select(Screening).where(Screening.id == screening_id)
    )
    screening = result.scalar_one_or_none()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    # Get associated images
    img_result = await db.execute(
        select(ScreeningImage).where(ScreeningImage.screening_id == screening_id)
    )
    images = img_result.scalars().all()
    image_data = images[0] if images else None

    # Check for gradcam or segmentation on disk if not in record
    save_dir = Path(settings.UPLOAD_DIR) / screening_id
    gradcam_file = save_dir / "gradcam_overlay.jpg"
    seg_file = save_dir / "segmentation_mask.png"

    gradcam_path = str(gradcam_file) if gradcam_file.exists() else (image_data.gradcam_path if image_data else None)
    seg_path = str(seg_file) if seg_file.exists() else (image_data.segmentation_mask_path if image_data else None)

    return {
        "screening_id": screening_id,
        "status": screening.status.value,
        "created_at": screening.created_at.isoformat(),
        "primary_diagnosis": screening.primary_diagnosis.value if screening.primary_diagnosis else None,
        "confidence": screening.confidence_score,
        "risk_score": screening.risk_score,
        "risk_level": screening.risk_level.value if screening.risk_level else None,
        "detected_features": screening.detected_features,
        "classification_probabilities": screening.classification_probs,
        "differential_diagnoses": screening.differential_diagnoses,
        "recommendations": screening.recommendations,
        "referral_urgency": screening.referral_urgency,
        "image_quality_score": screening.image_quality_score,
        "processing_time_ms": screening.processing_time_ms,
        "images": {
            "original": _to_web_url(image_data.stored_path if image_data else None),
            "gradcam": _to_web_url(gradcam_path),
            "segmentation": _to_web_url(seg_path),
        },
    }


@router.get("/{screening_id}/report")
async def download_report(
    screening_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate and download the official clinical PDF screening report.
    """
    from fastapi.responses import Response
    from services.report_generator import generate_pdf_report

    result = await db.execute(
        select(Screening).where(Screening.id == screening_id)
    )
    screening = result.scalar_one_or_none()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    img_result = await db.execute(
        select(ScreeningImage).where(ScreeningImage.screening_id == screening_id)
    )
    images = img_result.scalars().all()
    image_data = images[0] if images else None

    save_dir = Path(settings.UPLOAD_DIR) / screening_id
    gradcam_file = save_dir / "gradcam_overlay.jpg"
    orig_file = image_data.stored_path if image_data else str(save_dir / "original.jpg")

    report_payload = {
        "screening_id": screening.id,
        "created_at": screening.created_at,
        "primary_diagnosis": screening.primary_diagnosis.value if screening.primary_diagnosis else "aphthous_ulcer",
        "primary_diagnosis_display": (screening.primary_diagnosis.value if screening.primary_diagnosis else "aphthous_ulcer").replace("_", " ").title(),
        "subtype_display": screening.aphthous_subtype.value.title() if screening.aphthous_subtype else "Minor Type",
        "confidence": screening.confidence_score or 0.85,
        "risk_score": screening.risk_score or 15,
        "risk_level": screening.risk_level.value if screening.risk_level else "low",
        "detected_features": screening.detected_features or {},
        "differential_diagnoses": screening.differential_diagnoses or [],
        "recommendations": screening.recommendations or [],
        "referral_urgency": {"risk_level": screening.referral_urgency or "low"},
        "images": {
            "original": orig_file if os.path.exists(orig_file) else None,
            "gradcam": str(gradcam_file) if gradcam_file.exists() else None,
        }
    }

    pdf_bytes = generate_pdf_report(report_payload)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=OralGuard_Report_{screening_id[:8]}.pdf"
        }
    )


@router.get("/{screening_id}/status")
async def get_status(
    screening_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get screening status (for polling during analysis)."""
    result = await db.execute(
        select(Screening).where(Screening.id == screening_id)
    )
    screening = result.scalar_one_or_none()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    return {
        "screening_id": screening_id,
        "status": screening.status.value,
        "primary_diagnosis": screening.primary_diagnosis.value if screening.primary_diagnosis else None,
        "risk_level": screening.risk_level.value if screening.risk_level else None,
    }


@router.get("/history/list")
async def list_screenings(
    limit: int = 30,
    db: AsyncSession = Depends(get_db),
):
    """
    List past screenings with image thumbnails and summary data.
    """
    result = await db.execute(
        select(Screening).order_by(Screening.created_at.desc()).limit(limit)
    )
    screenings = result.scalars().all()

    items = []
    for s in screenings:
        # Check thumbnail
        img_result = await db.execute(
            select(ScreeningImage).where(ScreeningImage.screening_id == s.id)
        )
        img = img_result.scalars().first()
        thumb_url = _to_web_url(img.stored_path if img else None)

        diag_name = s.primary_diagnosis.value if s.primary_diagnosis else "aphthous_ulcer"
        items.append({
            "id": s.id,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "status": s.status.value,
            "primary_diagnosis": diag_name,
            "primary_diagnosis_display": diag_name.replace("_", " ").title(),
            "subtype_display": s.aphthous_subtype.value.title() if s.aphthous_subtype else None,
            "confidence": s.confidence_score,
            "risk_score": s.risk_score,
            "risk_level": s.risk_level.value if s.risk_level else "low",
            "image_url": thumb_url,
        })

    return {"screenings": items, "total": len(items)}


@router.delete("/history/clear/all")
async def clear_all_screenings(
    db: AsyncSession = Depends(get_db),
):
    """
    Clear all past screening history records and associated files.
    """
    try:
        await db.execute(delete(DiagnosisReport))
        await db.execute(delete(QuestionnaireResponse))
        await db.execute(delete(ScreeningImage))
        await db.execute(delete(Screening))
        await db.commit()
        _screening_cache.clear()

        # Clean upload dir folders
        upload_dir = Path(settings.UPLOAD_DIR)
        if upload_dir.exists():
            for item in upload_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)

        logger.info("All screening history cleared.")
        return {"status": "success", "message": "All screening history cleared"}
    except Exception as e:
        logger.error(f"Failed to clear screening history: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


@router.delete("/history/{screening_id}")
@router.delete("/{screening_id}")
async def delete_screening(
    screening_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a single screening record and its associated data (images, questionnaire, report).
    """
    result = await db.execute(
        select(Screening).where(Screening.id == screening_id)
    )
    screening = result.scalar_one_or_none()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    try:
        # Delete associated reports
        await db.execute(
            delete(DiagnosisReport).where(DiagnosisReport.screening_id == screening_id)
        )

        # Delete associated questionnaire responses
        await db.execute(
            delete(QuestionnaireResponse).where(QuestionnaireResponse.screening_id == screening_id)
        )

        # Delete associated image records
        await db.execute(
            delete(ScreeningImage).where(ScreeningImage.screening_id == screening_id)
        )

        # Delete the screening record
        await db.delete(screening)
        await db.commit()

        _screening_cache.pop(screening_id, None)

        # Remove screening upload directory
        try:
            save_dir = Path(settings.UPLOAD_DIR) / screening_id
            if save_dir.exists():
                shutil.rmtree(save_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Could not remove uploads for {screening_id}: {e}")

        logger.info(f"Screening {screening_id} deleted successfully.")
        return {"status": "success", "message": "Screening deleted successfully", "id": screening_id}
    except Exception as e:
        logger.error(f"Error deleting screening {screening_id}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Could not delete screening: {str(e)}")



