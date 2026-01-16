#!/bin/bash
# ASD Gaze Tracker - Easy Launch Script
# Run with: ./run.sh or bash run.sh

echo "🧠 ASD Gaze Tracking Research Tool"
echo "=================================="
echo "⚠️  IMPORTANT: This is for RESEARCH ONLY - Not medical diagnosis!"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Check for optional dependencies
echo "🔍 Checking optional dependencies..."
python3 -c "
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

echo ""
echo "🚀 Starting ASD Gaze Tracker..."
echo "📱 Web interface will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Run the application
python3 -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0