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

## One-Click Setup & Run (Windows / Friend's Laptop)

Simply double-click or run:
```bat
run.bat
```
This automatically:
1. Detects Python or Docker.
2. Fixes any conflicting packages (`jose`/`jwt`).
3. Sets up a dedicated virtual environment (`.venv`).
4. Installs all PyTorch & AI dependencies safely.
5. Launches the app in your browser at `http://localhost:8000`.

### PowerShell / Linux / macOS
- **PowerShell:** `.\run.ps1`
- **Linux/Mac/WSL:** `chmod +x run.sh && ./run.sh`
- **Docker:** `docker build -t oralguard-ai . && docker run -p 8000:7860 oralguard-ai`
