# =====================================================================
# OralGuard AI — 1-Click Setup & Launcher (PowerShell)
# =====================================================================

$ErrorActionPreference = "Stop"
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "             OralGuard AI -- Automatic Setup & Launcher              " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

Set-Location $PSScriptRoot

# 1. Detect Python
Write-Host "`n[1/5] Checking Python installation..." -ForegroundColor Yellow
$pyCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pyCmd = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pyCmd = "py"
} else {
    Write-Host "`n[ERROR] Python is not found in PATH." -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/ (check 'Add to PATH')." -ForegroundColor Red
    exit 1
}

$pyVersion = & $pyCmd --version 2>&1
Write-Host "Found: $pyVersion" -ForegroundColor Green

# 2. Clean conflicting old jose/jwt packages
Write-Host "`n[2/5] Cleaning conflicting packages (jose/jwt)..." -ForegroundColor Yellow
& $pyCmd -m pip uninstall -y jose jwt python-jose 2>$null

# 3. Virtual Environment
Write-Host "`n[3/5] Setting up virtual environment (.venv)..." -ForegroundColor Yellow
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "Creating .venv..."
    & $pyCmd -m venv .venv
}
& .\.venv\Scripts\Activate.ps1

# 4. Install Dependencies
Write-Host "`n[4/5] Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "- Installing PyTorch (CPU-optimized)..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
Write-Host "- Installing application dependencies..."
python -m pip install -r backend/requirements.txt --quiet
python -m pip uninstall -y jose jwt python-jose 2>$null
python -m pip install pyjwt --quiet

# 5. Open Browser & Run Server
Write-Host "`n[5/5] Launching OralGuard AI..." -ForegroundColor Yellow
Start-Process "http://localhost:8000"

Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host "             OralGuard AI is Running!                                " -ForegroundColor Green
Write-Host "  App URL:   http://localhost:8000                                   " -ForegroundColor Green
Write-Host "  API Docs:  http://localhost:8000/api/docs                          " -ForegroundColor Green
Write-Host "  Press CTRL+C to stop the server.                                   " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

Set-Location backend
python main.py
