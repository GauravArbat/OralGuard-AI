"""
OralGuard AI — Dashboard API Routes

Government health official dashboard endpoints:
  GET /dashboard/stats      — Aggregate screening statistics
  GET /dashboard/screenings — List all screenings with filters
  GET /dashboard/export     — Export data as CSV
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from database import get_db
from models.db_models import Screening, ScreeningStatus, RiskLevel, LesionType


router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/stats")
async def get_dashboard_stats(
    days: int = Query(default=30, ge=1, le=365),
    district: str = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """
    Get aggregate screening statistics for the dashboard.

    Returns total screenings, positive rate, risk distribution,
    and geographic breakdown.
    """
    since = datetime.utcnow() - timedelta(days=days)
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # Total screenings in period
    total_q = select(func.count(Screening.id)).where(
        Screening.created_at >= since,
        Screening.status == ScreeningStatus.COMPLETED,
    )
    total_result = await db.execute(total_q)
    total_screenings = total_result.scalar() or 0

    # Today's screenings
    today_q = select(func.count(Screening.id)).where(
        Screening.created_at >= today_start,
    )
    today_result = await db.execute(today_q)
    screenings_today = today_result.scalar() or 0

    # Positive (OSCC) rate
    oscc_q = select(func.count(Screening.id)).where(
        Screening.created_at >= since,
        Screening.primary_diagnosis == LesionType.OSCC,
    )
    oscc_result = await db.execute(oscc_q)
    oscc_count = oscc_result.scalar() or 0
    positive_rate = (oscc_count / total_screenings * 100) if total_screenings > 0 else 0

    # Referral rate (high + urgent)
    referral_q = select(func.count(Screening.id)).where(
        Screening.created_at >= since,
        Screening.risk_level.in_([RiskLevel.HIGH, RiskLevel.URGENT]),
    )
    referral_result = await db.execute(referral_q)
    referral_count = referral_result.scalar() or 0
    referral_rate = (referral_count / total_screenings * 100) if total_screenings > 0 else 0

    # Risk distribution
    risk_dist = {}
    for level in RiskLevel:
        level_q = select(func.count(Screening.id)).where(
            Screening.created_at >= since,
            Screening.risk_level == level,
        )
        level_result = await db.execute(level_q)
        risk_dist[level.value] = level_result.scalar() or 0

    # Top conditions
    top_conditions = []
    for lesion_type in LesionType:
        type_q = select(func.count(Screening.id)).where(
            Screening.created_at >= since,
            Screening.primary_diagnosis == lesion_type,
        )
        type_result = await db.execute(type_q)
        count = type_result.scalar() or 0
        if count > 0:
            top_conditions.append({
                "condition": lesion_type.value,
                "count": count,
                "percentage": round(count / total_screenings * 100, 1) if total_screenings else 0,
            })

    top_conditions.sort(key=lambda x: x["count"], reverse=True)

    return {
        "period_days": days,
        "total_screenings": total_screenings,
        "screenings_today": screenings_today,
        "positive_rate": round(positive_rate, 1),
        "referral_rate": round(referral_rate, 1),
        "risk_distribution": risk_dist,
        "top_conditions": top_conditions,
        "oscc_detected": oscc_count,
        "referrals_generated": referral_count,
    }


@router.get("/screenings")
async def list_screenings(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    risk_level: str = Query(default=None),
    diagnosis: str = Query(default=None),
    status: str = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """
    List screenings with pagination and filters.
    For government dashboard oversight.
    """
    query = select(Screening).order_by(Screening.created_at.desc())

    if risk_level:
        query = query.where(Screening.risk_level == RiskLevel(risk_level))
    if diagnosis:
        query = query.where(Screening.primary_diagnosis == LesionType(diagnosis))
    if status:
        query = query.where(Screening.status == ScreeningStatus(status))

    # Pagination
    offset = (page - 1) * per_page
    query = query.offset(offset).limit(per_page)

    result = await db.execute(query)
    screenings = result.scalars().all()

    # Count total
    count_q = select(func.count(Screening.id))
    if risk_level:
        count_q = count_q.where(Screening.risk_level == RiskLevel(risk_level))
    count_result = await db.execute(count_q)
    total = count_result.scalar() or 0

    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": (total + per_page - 1) // per_page,
        "screenings": [
            {
                "id": s.id,
                "user_id": s.user_id,
                "primary_diagnosis": s.primary_diagnosis.value if s.primary_diagnosis else None,
                "risk_level": s.risk_level.value if s.risk_level else None,
                "risk_score": s.risk_score,
                "confidence": s.confidence_score,
                "status": s.status.value,
                "created_at": s.created_at.isoformat(),
            }
            for s in screenings
        ],
    }
