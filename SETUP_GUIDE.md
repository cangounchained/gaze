# 🚀 ASD Gaze Tracker - Complete Setup Guide

## Quick Start (30 seconds)

### Windows
```batch
run.bat
```

### Linux/macOS
```bash
chmod +x run.sh
./run.sh
```

## Manual Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Webcam (for live tracking)

### Step-by-Step Setup

#### 1. Create Virtual Environment
```bash
python -m venv gaze_env

# Activate it
# Windows:
gaze_env\Scripts\activate
# Linux/macOS:
source gaze_env/bin/activate
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## What You'll Get

### 🎯 Live Session Tab
- **Webcam Preview**: See your face in real-time
- **5-Point Calibration**: Calibrate your gaze for accuracy
- **Gaze Tracking**: Real-time pupil tracking with red dot visualization
- **Session Duration**: Configurable tracking duration (5-60 seconds)

### 📹 Upload & Analyze Tab
- Upload images or videos for analysis
- Extract gaze features automatically
- Get predictions using trained models

### 🤖 Train Model Tab
- Upload your own CSV dataset
- Select label column (automatic detection)
- Train custom RandomForest models
- Model is saved and reused for predictions

### 📊 Results Tab
- View all session data
- Analyze gaze patterns
- Download detailed metrics
- Export results for further analysis

## Features in Detail

### Real-Time Gaze Tracking
- Uses MediaPipe Face Mesh for face detection
- Tracks pupil position in real-time
- Draws red visualization markers
- Logs gaze coordinates to CSV

### Machine Learning
- RandomForest classifier for ASD risk assessment
- Train on custom datasets
- Flexible label detection (string or numeric)
- Model persistence (saves/loads trained models)

### Data Logging
- Automatic CSV creation
- Logs timestamp, target, x, y coordinates
- Configurable participant IDs
- Optional session notes

## Troubleshooting

### Issue: "OpenCV not available"
**Fix**: Install it with:
```bash
pip install opencv-python
```

### Issue: "MediaPipe not available"
**Fix**: Install it with:
```bash
pip install mediapipe
```

### Issue: Webcam not working
1. Check webcam permissions
2. Ensure no other app is using the webcam
3. Try selecting Camera 0 (default)

### Issue: Model training fails
1. Ensure CSV has numeric features
2. Check label column is selected correctly
3. Ensure no rows are completely empty
4. Use "Train Model" tab to see detailed errors

### Issue: Gaze tracking very slow
- Reduce session duration
- Ensure webcam is not running other processes
- Close other applications

## System Requirements

### Minimum
- CPU: Intel Core i5 or equivalent
- RAM: 4GB
- Disk: 2GB free space

### Recommended
- CPU: Intel Core i7 or better
- RAM: 8GB+
- Disk: 5GB free space
- GPU: NVIDIA GPU (for faster processing)

## File Structure

```
gaze/
├── app.py                 # Main Streamlit app
├── helpers.py            # Utility functions
├── calibration.py        # Calibration module
├── gaze_simple.py        # Simplified gaze tracking
├── gaze.py              # Advanced gaze tracking
├── main.py              # CLI interface
├── requirements.txt     # Python dependencies
├── environment.yml      # Conda environment
├── run.sh              # Linux/macOS launcher
├── run.bat             # Windows launcher
└── README.md           # Main documentation
```

## Environment Variables

Set these for advanced configuration:

```bash
# Logging level (DEBUG, INFO, WARNING, ERROR)
export LOG_LEVEL=INFO

# Webcam device index (0=default, 1=external, etc.)
export WEBCAM_DEVICE=0

# Enable GPU acceleration (if available)
export USE_GPU=0
```

## Advanced Usage

### Training Custom Models

1. Prepare CSV with features and label column
2. Go to "Train Model" tab
3. Upload CSV
4. Select label column from dropdown
5. Click "Train Model"
6. Model is saved as `asd_model.pkl`

### CSV Format Example

```csv
pupil_left_x,pupil_left_y,pupil_right_x,pupil_right_y,mouth_x,mouth_y,nose_x,nose_y,dist_eye_left,dist_eye_right,label
150.5,120.3,165.2,118.9,155.0,145.0,160.0,100.0,20.5,21.3,ASD
145.2,119.8,160.1,119.2,150.0,140.0,158.0,99.5,19.8,20.9,Typical
```

### API Usage (if extending)

```python
from app import ASDClassifier
import pandas as pd

# Load trained model
classifier = ASDClassifier()
classifier.load_model("asd_model.pkl")

# Make prediction
features = pd.DataFrame({
    'pupil_left_x': [150.5],
    'pupil_left_y': [120.3],
    # ... other features
})

prediction, confidence = classifier.predict(features)
print(f"Prediction: {prediction}, Confidence: {confidence:.2%}")
```

## Data Privacy

✅ **All processing is local** - No data sent to internet
✅ **No cloud storage** - Files stay on your machine
✅ **No tracking** - No analytics or telemetry
✅ **Secure** - Works completely offline

## Performance Tips

1. **Better accuracy**: Run longer sessions (20-30 seconds)
2. **Faster processing**: Use simpler models or reduce resolution
3. **Less lag**: Close other applications
4. **Better lighting**: Ensure adequate lighting for face detection

## Support & Troubleshooting

For issues:
1. Check the error message in red box
2. Try closing and reopening the app
3. Clear browser cache if UI looks broken
4. Reinstall dependencies: `pip install --upgrade -r requirements.txt`

## Additional Resources

- [MediaPipe Face Mesh Docs](https://google.github.io/mediapipe/solutions/face_mesh)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-Learn RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## License

Research use only. See LICENSE file for details.

## Citation

If you use this tool in research, please cite:
```
ASD Gaze Tracker (2025)
https://github.com/cangounchained/gaze
```

---

**Last Updated**: January 2025  
**Version**: 2.0  
**Status**: ✅ Fully Functional
