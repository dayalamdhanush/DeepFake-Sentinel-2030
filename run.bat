@echo off
title DeepFake Sentinel 2030 Launcher
echo ========================================================
echo       DEEPFAKE SENTINEL 2030 - FORENSIC SYSTEM
echo ========================================================
echo.
echo Starting FastAPI backend server and Web Dashboard...
echo.

:: Check port 8000 availability
netstat -ano | findstr :8000 | findstr LISTENING >nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Port 8000 is currently in use by another application.
    echo [INFO] Launching DeepFake Sentinel 2030 on Port 8001...
    set PORT=8001
) else (
    set PORT=8000
)

echo.
echo Access URL: http://127.0.0.1:%PORT%
echo Opening browser in 3 seconds...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:%PORT%

python -m uvicorn backend.main:app --host 127.0.0.1 --port %PORT% --reload

pause
