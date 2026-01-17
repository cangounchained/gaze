# 🚀 Quick Start Guide

## 1. Installation (2 minutes)

```bash
# Clone and setup
git clone <repo-url>
cd research_tool

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Launch Webcam Interface (30 seconds)

```bash
streamlit run src/webcam_ui.py
```

Opens at: http://localhost:8501

## 3. Use the Interface

### Step 1: Calibration 🎯
- Click "Start 5-Point Calibration"
- Follow the red dots with your eyes
- Takes ~10 seconds

### Step 2: Gaze Tracking 📹
- Click "Start Gaze Tracking"
- Keep your face visible
- 15 seconds recording (adjustable)
- Red dot shows pupil position

### Step 3: Analysis 📊
- View gaze trajectory
- Check detection rate
- Load trained model for predictions

## 4. Train Your Own Model (Optional)

### 4a. Prepare Dataset
```
data/my_dataset/
├── asd/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── typical/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

### 4b. Run Training
```bash
python train_example.py
```

Creates model at: `models/asd_detector.pkl`

### 4c. Load in Webcam UI
- Go to sidebar "Model"
- Enter: `models/asd_detector.pkl`
- Click "Load Model"
- Use predictions in "Live Analysis" tab

## 5. File Structure

```
research_tool/
├── src/
│   ├── preprocessing.py      ← Face & eye detection
│   ├── feature_extraction.py ← Gaze metrics
│   ├── model.py              ← Classifiers
│   ├── train.py              ← Training pipeline
│   └── webcam_ui.py          ← Streamlit interface
├── data/                      ← Datasets
├── models/                    ← Trained models
├── train_example.py          ← Example training script
├── requirements.txt
└── README.md
```

## 6. Common Commands

```bash
# Install all dependencies
pip install -r requirements.txt

# Launch webcam interface
streamlit run src/webcam_ui.py

# Train model (requires dataset)
python train_example.py

# Check OpenCV version
python -c "import cv2; print(cv2.__version__)"

# Verify MediaPipe
python -c "import mediapipe; print('✅ MediaPipe OK')"
```

## 7. Troubleshooting

| Problem | Solution |
|---------|----------|
| Webcam not opening | Check permissions, try `python -c "import cv2; cv2.VideoCapture(0).isOpened()"` |
| Face not detected | Ensure good lighting, adjust distance from camera |
| Module not found | Make sure you're in `research_tool/` directory |
| Memory error | Set `USE_CNN = False` in `train_example.py` |
| Slow processing | Reduce `max_frames` or use CPU instead of GPU |

## 8. Dataset Options

### Option 1: Use Existing Datasets
- MIT GazeCapture: https://gazecapture.csail.mit.edu/
- Kaggle ASD Dataset: https://www.kaggle.com/
- Your own collected data

### Option 2: Create Synthetic Dataset
```bash
python train_example.py  # Creates demo dataset automatically
```

### Option 3: Collect Your Own
1. Launch webcam interface
2. Run sessions with different users
3. Export data from "Results" tab
4. Organize into `data/` folder

## 9. Key Features

✅ **5-Point Red Dot Calibration** - Auto-calibrates gaze tracking
✅ **Live Gaze Visualization** - Red dot at pupil position in real-time
✅ **Feature Extraction** - 10+ gaze metrics automatically computed
✅ **ML Classification** - RandomForest or Neural Network
✅ **Model Training** - Complete training pipeline
✅ **Data Export** - JSON export of sessions
✅ **Ethical Disclaimer** - Research-only, not for diagnosis
✅ **Local Processing** - All data stays on your machine

## 10. Ethical Use ⚠️

**This tool is for research only:**
- ❌ Cannot be used for clinical diagnosis
- ❌ Not validated for medical decisions
- ❌ Cannot replace healthcare professionals
- ✅ Use only for academic research
- ✅ Always include ethical disclaimers
- ✅ Obtain proper informed consent

---

**Ready to start?**
```bash
streamlit run src/webcam_ui.py
```

**Questions?** See `README.md` for full documentation.
