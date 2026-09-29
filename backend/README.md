# OralGuard AI — Backend Architecture

## Complete Project Structure

```
backend/
|-- main.py                          # FastAPI entry point + lifespan + routers
|-- config.py                        # Pydantic settings (env vars, model paths, thresholds)
|-- database.py                      # Async SQLAlchemy engine + session factory
|-- requirements.txt                 # Python dependencies
|
|-- models/
|   |-- db_models.py                 # SQLAlchemy ORM (User, Screening, Image, Questionnaire, Report)
|   |-- schemas.py                   # Pydantic request/response schemas (30+ schemas)
|
|-- ai/
|   |-- image_preprocessor.py        # Validation, quality scoring, normalization
|   |-- lesion_detector.py           # YOLOv8-m detection model
|   |-- lesion_segmenter.py          # HF-UNet segmentation model
|   |-- lesion_classifier.py         # EfficientNet-B4 multi-task classifier
|   |-- feature_extractor.py         # ResNet-50 multi-task feature extraction (8 features)
|   |-- clinical_fusion.py           # XGBoost/rule-based multimodal fusion + risk scoring
|   |-- gradcam.py                   # Grad-CAM heatmap generation
|   |-- pipeline.py                  # End-to-end 4-stage inference orchestrator
|
|-- clinical/
|   |-- questionnaire.py             # 20 aphthous + 6 OSCC dynamic questions
|   |-- differential.py              # 8-condition differential diagnosis engine
|   |-- decision_engine.py           # Master clinical logic + recommendations + referrals
|
|-- api/
|   |-- auth.py                      # JWT register/login/profile
|   |-- screening.py                 # Image upload, questionnaire, results
|   |-- dashboard.py                 # Government analytics, screening list
|
|-- services/                        # (Extensible: image storage, PDF reports, SMS)
|-- utils/                           # (Extensible: helpers, constants)
```

## 4-Stage AI Pipeline

```
Image Upload
    |
    v
[Stage 1] YOLOv8-m Detection -----> Bounding box (640x640)
    |
    v
[Stage 2] HF-UNet Segmentation ---> Pixel mask (512x512)
    |
    v
[Stage 3a] EfficientNet-B4 -------> Classification (Aphthous/OSCC/Other + subtypes)
[Stage 3b] ResNet-50 Multi-task ---> 8 clinical features
[Stage 3c] Grad-CAM ---------------> Heatmap overlay
    |
    v
[Stage 4] Clinical Fusion ---------> Risk score (0-100) + adjusted probabilities
    |
    v
Differential Diagnosis Engine -----> Ranked differentials (8 conditions)
    |
    v
Recommendation Engine -------------> Referral urgency + treatment suggestions
```

## API Endpoints

| Method | Endpoint                               | Description                    |
|--------|----------------------------------------|--------------------------------|
| POST   | /api/v1/auth/register                  | Register user                  |
| POST   | /api/v1/auth/login                     | Login (JWT)                    |
| GET    | /api/v1/auth/me                        | Current user profile           |
| POST   | /api/v1/screening/upload               | Upload image for analysis      |
| GET    | /api/v1/screening/{id}/questionnaire   | Get dynamic questionnaire      |
| POST   | /api/v1/screening/{id}/questionnaire   | Submit answers, get diagnosis  |
| GET    | /api/v1/screening/{id}/results         | Full screening results         |
| GET    | /api/v1/screening/{id}/status          | Polling status                 |
| GET    | /api/v1/dashboard/stats                | Aggregate statistics           |
| GET    | /api/v1/dashboard/screenings           | List screenings with filters   |
| GET    | /health                                | Health check + model status    |

## Running the Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
# Server starts at http://localhost:8000
# API docs at http://localhost:8000/api/docs
```

## Risk Scoring Algorithm

```
Risk Score = 0.40 * Image Risk + 0.25 * Features Risk + 0.35 * Questionnaire Risk

Risk Levels:
  0-25  = LOW    (green)  -> Self-care + reassess in 2 weeks
  26-50 = MEDIUM (yellow) -> Routine specialist review in 2-4 weeks
  51-75 = HIGH   (orange) -> Priority referral, biopsy recommended
  76-100= URGENT (red)    -> Immediate specialist referral in 24-48 hours
```
