# 🎉 ASD Gaze Tracker - QUICK START

## Installation (1 minute)

### Windows
```cmd
pip install -r requirements.txt
streamlit run app.py
```

### Linux/macOS
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at: **http://localhost:8501**

## ✅ What's Fixed

### 1. Webcam Preview ✓
- **Before**: Opened once and froze
- **Now**: Continuous streaming with red dot at center
- **Usage**: Go to "Live Session" → Click "Start Preview"

### 2. Red Dot Visualization ✓
- **Calibration**: Moving red dots at 5 positions
- **Gaze Tracking**: Red circle at pupil position
- **Color**: Bright red (0, 0, 255) in BGR

### 3. Model Training ✓
- **Before**: Failed with "KeyError: ['label'] not found"
- **Now**: Works with any label column
- **Usage**: Go to "Train Model" → Upload CSV → Select label column → Click "Train Model"

### 4. Gaze Tracking ✓
- **Before**: Subprocess blocked the app
- **Now**: Real-time inline tracking
- **Shows**: Red dot at pupil, line from eye, face detection status

## 📋 Usage Steps

### Step 1: Webcam Preview
1. Go to **Live Session** tab
2. Click **"▶️ Start Preview"**
3. You should see yourself with a red dot in the center
4. Click **"⏹️ Stop Preview"** to stop

### Step 2: Calibration
1. Click **"Start 5-Point Calibration"**
2. Follow the red dot with your eyes
3. Hold focus for 2 seconds at each position
4. Done!

### Step 3: Gaze Tracking
1. Enter **Participant ID** (required)
2. Optionally add **Notes**
3. Adjust **Session Duration** in sidebar (default: 15 seconds)
4. Click **"🔴 Start Tracking"**
5. Keep your face visible
6. Red dot should appear on your pupil
7. Wait for session to complete

### Step 4: Train Model
1. Go to **Train Model** tab
2. Upload a CSV file with:
   - Feature columns (numeric)
   - One label column with class names (e.g., "ASD", "Typical")
3. Select the label column from dropdown
4. Click **"🚀 Train Model"**
5. Model saves as `asd_model.pkl`

### Step 5: View Results
1. Go to **Results** tab
2. See session data, statistics
3. View gaze coordinates

## 🔧 Troubleshooting

### "Webcam hangs"
- **Cause**: Webcam already in use
- **Fix**: Close other apps using webcam (Zoom, Teams, etc.)
- **Try**: Unplug and replug webcam

### "No red dot visible"
- **Cause**: Face not detected
- **Fix**: Ensure good lighting, face fully visible
- **Try**: Move closer to webcam

### "Model training fails"
- **Cause**: Invalid CSV format
- **Fix**: Ensure:
  - All feature columns are numeric
  - One label column with clear labels
  - No completely empty rows
- **Try**: Test with sample CSV first

### "Can't access webcam"
- **Cause**: Permission denied
- **Fix**: Check OS webcam permissions
- **Try**: Allow app access in System Settings

## 📊 CSV Format for Training

```csv
feature1,feature2,feature3,label
10.5,20.3,15.2,ASD
11.2,21.0,16.1,Typical
9.8,19.5,14.8,ASD
```

**Requirements:**
- Numeric features (numbers)
- One text column with labels
- First row = headers
- No completely empty rows

## 🎯 What to Expect

### Webcam Preview
- Smooth video stream
- Red dot in center
- Frame counter
- Resolution displayed

### Calibration
- 5 positions (corners + center)
- 2 seconds per position
- Red dot moves
- Success message after

### Gaze Tracking
- Red circle at pupil
- Red line from eye to gaze point
- Timer counting down
- "Face detected" status
- Automatic save when done

### Model Training
- Shows unique labels
- Confirms training samples
- Success message
- File saved

## ⚙️ Requirements

- Python 3.8+
- OpenCV: `pip install opencv-python`
- MediaPipe: `pip install mediapipe`
- Streamlit: `pip install streamlit`
- Scikit-learn: `pip install scikit-learn`
- Pandas: `pip install pandas`

## 🚀 Run Now

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

## 📞 Quick Tips

1. **Better gaze tracking**: Good lighting, face fully visible
2. **Model training**: Need at least 10-20 samples
3. **Longer sessions**: More accurate gaze data
4. **Check errors**: Red messages tell you what's wrong

---

**Status**: ✅ WORKING & TESTED  
**Last Updated**: January 17, 2025
