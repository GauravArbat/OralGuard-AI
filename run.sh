#!/usr/bin/env bash
# =====================================================================
# OralGuard AI — All-in-One Setup & Launcher (Linux / macOS / WSL)
# =====================================================================

set -e

echo "====================================================================="
echo "             OralGuard AI -- Automatic Setup & Launcher              "
echo "====================================================================="

# Check for Docker
if command -v docker &> /dev/null && docker info &> /dev/null; then
    echo "[Docker Detected] Running with Docker..."
    docker build -t oralguard-ai .
    xdg-open "http://localhost:8000" 2>/dev/null || open "http://localhost:8000" 2>/dev/null || true
    docker run -it --rm -p 8000:7860 -v "$(pwd)/backend/uploads:/app/backend/uploads" oralguard-ai
    exit 0
fi

# Python Setup
PY_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PY_CMD="python"
fi

echo "Cleaning conflicting packages..."
$PY_CMD -m pip uninstall -y jose jwt python-jose 2>/dev/null || true

echo "Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    $PY_CMD -m venv .venv
fi
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
pip install -r backend/requirements.txt --quiet
pip uninstall -y jose jwt 2>/dev/null || true
pip install pyjwt --quiet

echo "Opening browser..."
xdg-open "http://localhost:8000" 2>/dev/null || open "http://localhost:8000" 2>/dev/null || true

echo "Starting OralGuard AI server..."
cd backend
python main.py
