@echo off
echo ========================================
echo ASL Sign Language Translator
echo ========================================
echo.

cd /d "%~dp0"

echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo ========================================
echo Starting Web App on http://localhost:5001
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
