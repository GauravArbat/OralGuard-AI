"""
OralGuard AI — Comprehensive Production API Integration Test Suite

Tests all stages of the clinical screening workflow:
1. System Health Check
2. Intraoral Image Upload & Real-time AI Pipeline Processing (Detection, Segmentation, Multi-task Feature Extraction, Classification, Grad-CAM)
3. Dynamic Questionnaire Fetching (Adaptive clinical question generation)
4. Questionnaire Submission & Multimodal Clinical Decision Fusion
5. Comprehensive Results Retrieval (Differential diagnosis matrix, risk stratification)
6. Diagnostic Clinical PDF Report Generation (ReportLab dual-image layout)
7. Screening History Listing
8. Population Health Surveillance Dashboard Statistics
"""

import sys
import io
import requests
from PIL import Image, ImageDraw

BASE_URL = "http://localhost:8000"


def create_synthetic_oral_image() -> io.BytesIO:
    """Creates a high-contrast synthetic oral lesion image for testing."""
    img = Image.new("RGB", (640, 640), color=(155, 60, 60))
    draw = ImageDraw.Draw(img)
    # Erythematous inflammatory border
    draw.ellipse([210, 210, 430, 430], fill=(215, 35, 35))
    # Fibrinous necrotic central base
    draw.ellipse([250, 250, 390, 390], fill=(245, 235, 195))
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf


def run_tests():
    print("=" * 65)
    print("   ORALGUARD AI — PRODUCTION SYSTEM END-TO-END TEST")
    print("=" * 65)

    # 1. Health check
    print("\n[1/8] Verifying API Health...")
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    health_data = r.json()
    print(f"      Status: {health_data.get('status')} | Version: {health_data.get('version')}")

    # 2. Upload image and run AI pipeline
    print("\n[2/8] Uploading intraoral image to /api/v1/screening/upload...")
    img_buf = create_synthetic_oral_image()
    files = {"file": ("test_oral_lesion.jpg", img_buf, "image/jpeg")}
    r = requests.post(f"{BASE_URL}/api/v1/screening/upload", files=files, timeout=60)
    assert r.status_code == 200, f"Upload failed: {r.text}"
    upload_res = r.json()
    screening_id = upload_res["screening_id"]
    quality = upload_res.get("quality_score", {})
    print(f"      Screening ID: {screening_id}")
    print(f"      Image Quality Total Score: {quality.get('total')}/100 (Sharpness: {quality.get('sharpness')}, Lighting: {quality.get('lighting')})")
    print(f"      Next Workflow Status: {upload_res.get('status')}")

    # 3. Dynamic Questionnaire
    print(f"\n[3/8] Fetching dynamic clinical questionnaire for {screening_id}...")
    r = requests.get(f"{BASE_URL}/api/v1/screening/{screening_id}/questionnaire", timeout=10)
    assert r.status_code == 200, f"Questionnaire failed: {r.text}"
    q_data = r.json()
    questions = q_data.get("questions", [])
    init_assess = q_data.get("initial_assessment", {})
    print(f"      Total questions generated: {len(questions)}")
    print(f"      Initial Assessment: {init_assess.get('primary_diagnosis')} (Confidence: {init_assess.get('confidence', 0)*100:.1f}%)")
    for q in questions[:3]:
        q_txt = q.get("question") or q.get("text", "")
        print(f"      - [{q['id']}] {q_txt} ({q.get('category')})")

    # 4. Submit Questionnaire
    print(f"\n[4/8] Submitting clinical questionnaire responses...")
    answers = {
        "onset_duration_days": 5,
        "recurrent": True,
        "recurrence_frequency": "few_per_year",
        "pain_present": True,
        "pain_level": "6",
        "ulcer_count": "single",
        "size_estimate": "under_10mm",
        "bleeding_spontaneous": False,
        "tobacco_user": False,
        "alcohol_regular": False
    }
    r = requests.post(
        f"{BASE_URL}/api/v1/screening/{screening_id}/questionnaire",
        json={"answers": answers},
        timeout=15
    )
    assert r.status_code == 200, f"Questionnaire submit failed: {r.text}"
    sub_res = r.json()
    res_data = sub_res.get("results", {})
    print(f"      Primary Diagnosis: {res_data.get('primary_diagnosis_display') or res_data.get('primary_diagnosis')}")
    print(f"      Confidence: {res_data.get('confidence', 0)*100:.1f}%")
    print(f"      Risk Score: {res_data.get('risk_score')} / 100 ({res_data.get('risk_level')})")
    referral = sub_res.get("referral", {})
    print(f"      Referral Urgency: {referral.get('risk_level')} ({referral.get('description', '')[:50]}...)")

    # 5. Fetch Full Clinical Results
    print(f"\n[5/8] Fetching full multimodal diagnostic results...")
    r = requests.get(f"{BASE_URL}/api/v1/screening/{screening_id}/results", timeout=10)
    assert r.status_code == 200, f"Results fetch failed: {r.text}"
    full_res = r.json()
    diffs = full_res.get("differential_diagnoses", [])
    print(f"      Differential Diagnoses evaluated: {len(diffs)}")
    for d in diffs[:3]:
        d_name = d.get("display_name") or d.get("condition")
        d_icd = d.get("icd10_code", "N/A")
        print(f"      • {d_name}: {d.get('probability', 0)*100:.1f}% (ICD-10: {d_icd})")
    recs = full_res.get("recommendations", [])
    rec_text = recs[0] if isinstance(recs, list) and recs else (recs.get("patient_action", "N/A") if isinstance(recs, dict) else str(recs))
    print(f"      Patient Guidance: {rec_text[:60]}...")
    images_dict = full_res.get("images", {})
    print(f"      Original Image URL: {images_dict.get('original')}")
    print(f"      Grad-CAM Heatmap URL: {images_dict.get('gradcam')}")
    print(f"      Segmentation Mask URL: {images_dict.get('segmentation')}")

    # 6. Generate Clinical PDF Report
    print(f"\n[6/8] Generating Clinical PDF Diagnostic Report...")
    r = requests.get(f"{BASE_URL}/api/v1/screening/{screening_id}/report", timeout=15)
    assert r.status_code == 200, f"PDF report failed: {r.status_code}"
    assert r.headers.get("content-type") == "application/pdf", "Expected application/pdf content type"
    pdf_bytes = r.content
    assert pdf_bytes.startswith(b"%PDF"), "Response is not a valid PDF file"
    print(f"      PDF Report generated successfully ({len(pdf_bytes):,} bytes)")

    # 7. Screening History Listing
    print(f"\n[7/8] Querying Screening History...")
    r = requests.get(f"{BASE_URL}/api/v1/screening/history/list", timeout=10)
    assert r.status_code == 200, f"History list failed: {r.text}"
    history_data = r.json()
    items = history_data.get("screenings", history_data) if isinstance(history_data, dict) else history_data
    print(f"      Total recorded screening sessions: {len(items)}")
    if items:
        latest = items[0]
        print(f"      Latest Record: {latest.get('id') or latest.get('screening_id')} -> {latest.get('primary_diagnosis')} ({latest.get('risk_level')})")

    # 8. Surveillance Dashboard Metrics
    print(f"\n[8/8] Querying Surveillance Dashboard Analytics...")
    r = requests.get(f"{BASE_URL}/api/v1/dashboard/stats", timeout=10)
    assert r.status_code == 200, f"Dashboard stats failed: {r.text}"
    stats = r.json()
    print(f"      Total Screenings: {stats.get('total_screenings')}")
    print(f"      Risk Stratification Breakdown: {stats.get('risk_distribution')}")
    print(f"      Prevalent Conditions: {stats.get('prevalent_conditions')}")

    print("\n" + "=" * 65)
    print("   ALL 8 END-TO-END PRODUCTION CHECKS PASSED PERFECTLY!")
    print("=" * 65)


if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print(f"\nTEST RUNNER ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
