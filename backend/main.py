"""
OralGuard AI — FastAPI Application Entry Point

The main server that bootstraps the API, loads AI models,
initializes the database, and configures CORS/middleware.
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from database import init_db


# ── Lifespan (startup/shutdown) ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown logic."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Initialize database
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database ready")

    # Load AI models (lazy load on first request if DEBUG)
    if not settings.DEBUG:
        logger.info("Pre-loading AI models...")
        from ai.pipeline import inference_pipeline
        inference_pipeline.load_all_models()
        logger.info("AI models loaded")
    else:
        logger.info("DEBUG mode: AI models will lazy-load on first request")

    yield

    # Shutdown
    logger.info("Shutting down OralGuard AI")


# ── FastAPI App ──

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "AI-powered Clinical Decision Support System for oral lesion "
        "screening. Combines deep learning image analysis with structured "
        "clinical questionnaires for differential diagnosis of Aphthous "
        "Ulcers and Oral Squamous Cell Carcinoma."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


# ── CORS ──

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static Files ──

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")


# ── Register API Routers ──

from api.auth import router as auth_router
from api.screening import router as screening_router
from api.dashboard import router as dashboard_router

app.include_router(auth_router, prefix=settings.API_PREFIX)
app.include_router(screening_router, prefix=settings.API_PREFIX)
app.include_router(dashboard_router, prefix=settings.API_PREFIX)


# ── Frontend Web Application Routes ──

from fastapi.responses import FileResponse

frontend_dir = Path(__file__).parent.parent

@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "OralGuard AI Backend running. index.html not found."}


@app.get("/styles.css", include_in_schema=False)
async def serve_css():
    css_path = frontend_dir / "styles.css"
    if css_path.exists():
        return FileResponse(css_path, media_type="text/css")
    return FileResponse(frontend_dir / "index.html")


@app.get("/app.js", include_in_schema=False)
async def serve_js():
    js_path = frontend_dir / "app.js"
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    return FileResponse(frontend_dir / "index.html")


# ── Root API Endpoints ──

@app.get("/api")
async def root():
    """API root — health & endpoint directory."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/api/docs",
        "endpoints": {
            "auth": f"{settings.API_PREFIX}/auth",
            "screening": f"{settings.API_PREFIX}/screening",
            "dashboard": f"{settings.API_PREFIX}/dashboard",
        },
    }



@app.get("/health")
async def health_check():
    """Detailed health check."""
    health = {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "database": "connected",
        "models": {},
    }

    # Check model status
    try:
        from ai.lesion_detector import lesion_detector
        health["models"]["detection"] = "loaded" if lesion_detector._loaded else "not_loaded"
    except Exception:
        health["models"]["detection"] = "error"

    try:
        from ai.lesion_classifier import lesion_classifier
        health["models"]["classification"] = "loaded" if lesion_classifier._loaded else "not_loaded"
    except Exception:
        health["models"]["classification"] = "error"

    try:
        from ai.lesion_segmenter import lesion_segmenter
        health["models"]["segmentation"] = "loaded" if lesion_segmenter._loaded else "not_loaded"
    except Exception:
        health["models"]["segmentation"] = "error"

    try:
        from ai.feature_extractor import feature_extractor
        health["models"]["feature_extraction"] = "loaded" if feature_extractor._loaded else "not_loaded"
    except Exception:
        health["models"]["feature_extraction"] = "error"

    return health


# ── Run ──

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", settings.PORT))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.DEBUG,
        log_level="info",
    )

