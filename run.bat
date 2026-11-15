@echo off
title SENTINEL AI - Multimodal Sentiment Intelligence Platform
color 0b

echo ===============================================================================
echo                SENTINEL AI: MULTIMODAL SENTIMENT ANALYSIS
echo               Windows 11 CPU-Optimized Command Center Launcher
echo ===============================================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python runtime environment...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not found in system PATH.
    pause
    exit /b %errorlevel%
)

echo [2/3] Verifying core model assets...
if not exist "assets\models\text_sentiment_pipeline.joblib" (
    echo [*] Training text sentiment pipeline...
    python -m src.text_analysis.train_text_model
)
if not exist "assets\models\vision_expression_pipeline.joblib" (
    echo [*] Training FACS vision expression pipeline...
    python -m src.vision_analysis.train_vision_model
)

echo [3/3] Launching Sentinel AI Command Center...
echo.
echo Application will open in your default browser at http://localhost:8501
echo Press Ctrl+C in this terminal window to stop the server.
echo.

streamlit run app.py --server.port=8501 --server.headless=false --theme.base="dark"

pause
