@echo off
REM ASD Gaze Tracker - Windows Launch Script
REM Run with: run.bat

echo 🧠 ASD Gaze Tracking Research Tool
echo ==================================
echo ⚠️  IMPORTANT: This is for RESEARCH ONLY - Not medical diagnosis!
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update requirements
echo 📥 Installing requirements...
pip install -r requirements.txt

REM Check for optional dependencies
echo 🔍 Checking optional dependencies...
python -c "
try:
    import cv2
    print('✅ OpenCV available')
except ImportError:
    print('⚠️  OpenCV not available - webcam features limited')

try:
    import mediapipe as mp
    print('✅ MediaPipe available')
except ImportError:
    print('⚠️  MediaPipe not available - face detection limited')
"

echo.
echo 🚀 Starting ASD Gaze Tracker...
echo 📱 Web interface will be available at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

REM Run the application
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0

pause