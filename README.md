---
title: OralGuard AI
emoji: 🦷
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# OralGuard AI — Clinical Decision Support System

AI-powered differential diagnosis of Aphthous Ulcers and Oral Squamous Cell Carcinoma (OSCC).

## Tech Stack
- **Deep Learning**: PyTorch, YOLOv8-m, HF-UNet Segmentation, ResNet-50 & EfficientNet-B4 Classification
- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (Responsive PWA UI)

## Running Locally
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```
Open `http://localhost:8000` in your browser.
