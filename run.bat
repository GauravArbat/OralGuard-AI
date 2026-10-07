@echo off
setlocal enabledelayedexpansion
title OralGuard AI - 1-Click Launcher
color 0B

echo =====================================================================
echo                OralGuard AI -- Automatic Setup ^& Launcher
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Detect Python
echo [1/5] Detecting Python installation...

set PYTHON_CMD=
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set PYTHON_CMD=python
) else (
    where py >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set PYTHON_CMD=py
    )
)

if "%PYTHON_CMD%"=="" (
    echo.
    echo [ERROR] Python is not installed or not in your system PATH!
    echo Please install Python (3.11 recommended) from: https://www.python.org/downloads/
    echo Make sure to check "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('%PYTHON_CMD% --version 2^>^&1') do set PY_VER=%%v
echo Found: !PY_VER!

:: 2. Fix known global package conflicts (uninstall broken Python 2 jose/jwt if present)
echo.
echo [2/5] Cleaning up conflicting packages (jose/jwt)...
%PYTHON_CMD% -m pip uninstall -y jose jwt python-jose >nul 2>&1

:: 3. Create & Activate Virtual Environment
echo.
echo [3/5] Setting up virtual environment (.venv)...
if not exist ".venv\Scripts\activate.bat" (
    echo Creating clean virtual environment for OralGuard AI...
    %PYTHON_CMD% -m venv .venv
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Existing virtual environment found.
)

call .venv\Scripts\activate.bat

:: 4. Upgrade pip & Install Dependencies
echo.
echo [4/5] Installing dependencies (first run may take 1-2 minutes)...
python -m pip install --upgrade pip --quiet

:: Install CPU PyTorch directly (pre-compiled binary, works with Python 3.11 - 3.13)
echo - Installing PyTorch (CPU-optimized)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

:: Install core requirements
echo - Installing application dependencies...
python -m pip install -r backend\requirements.txt --quiet

:: Ensure modern PyJWT is installed and purge conflicting jose in venv
python -m pip uninstall -y jose jwt python-jose >nul 2>&1
python -m pip install pyjwt --quiet

:: 5. Open browser & Start Server
echo.
echo [5/5] Launching OralGuard AI...
start http://localhost:8000

echo.
echo =====================================================================
echo                OralGuard AI is Running!
echo   App URL:         http://localhost:8000
echo   API Docs:        http://localhost:8000/api/docs
echo   Press CTRL+C in this window to stop the server.
echo =====================================================================
echo.

cd backend
python main.py

if !ERRORLEVEL! neq 0 (
    echo.
    echo [WARNING] Direct run exited. Retrying with uvicorn...
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
)

pause
