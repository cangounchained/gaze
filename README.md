# 🧠 ASD Gaze Tracking Research Tool

A comprehensive Python-based research tool for Autism Spectrum Disorder (ASD) detection through gaze pattern analysis. This tool combines computer vision, deep learning, and machine learning to analyze gaze behaviors in real-time.

## ⚠️ IMPORTANT MEDICAL DISCLAIMER

**This tool is for RESEARCH and EDUCATIONAL purposes only.**  
**It is NOT a diagnostic tool and should NEVER be used for medical diagnosis.**  
Autism Spectrum Disorder can only be diagnosed by qualified medical professionals.  
**Results from this tool have NO clinical validity.**

## 📋 Features

- **Real-time gaze tracking** using MediaPipe Face Mesh
- **Eye region extraction** and preprocessing
- **CNN feature extraction** using MobileNetV2 embeddings
- **Handcrafted gaze metrics** (fixation stability, saccades, blink rate, asymmetry)
- **Machine learning classification** (Random Forest or PyTorch models)
- **Interactive web interface** with live webcam analysis
- **Comprehensive evaluation** with accuracy, precision, recall, AUC, and confusion matrices
- **Multi-dataset support** (MIT GazeCapture, Kaggle ASD, ASD vs TD)

## 🏗️ Project Structure

```
asd-gaze-research-tool/
├── data/                          # Dataset directory
│   ├── mit_gazecapture/          # MIT GazeCapture dataset
│   ├── kaggle_asd/               # Kaggle ASD dataset
│   └── asd_vs_td/                # ASD vs TD dataset
├── src/                          # Source code
│   ├── preprocessing.py          # Face/eye detection and preprocessing
│   ├── feature_extraction.py     # Feature extraction (handcrafted + CNN)
│   ├── model.py                  # ML models and evaluation
│   ├── train.py                  # Training script
│   └── webcam_ui.py              # Streamlit web interface
├── models/                       # Saved trained models
├── results/                      # Evaluation results and plots
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or download the project
cd asd-gaze-research-tool

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Place your datasets in the `data/` directory:

```
data/
├── mit_gazecapture/
│   └── gaze_data.csv
├── kaggle_asd/
│   └── asd_gaze_features.csv
└── asd_vs_td/
    └── asd_td_comparison.csv
```

**Expected CSV format:**
```csv
x,y,ear_left,ear_right,pupil_left_x,pupil_left_y,pupil_right_x,pupil_right_y,mouth_opening,dist_eye_left,dist_eye_right,dist_mouth,dist_nose,label
0.5,0.3,0.25,0.27,95,85,145,85,25.5,45.2,42.8,65.3,78.9,ASD
```

### 3. Train the Model

```bash
# Train Random Forest model (default)
python src/train.py

# Train PyTorch model
python src/train.py --model-type pytorch

# Custom training options
python src/train.py --data-dir data --output-dir models --results-dir results
```

### 4. Run Webcam Analysis

```bash
# Start the web interface
streamlit run src/webcam_ui.py
```

Open your browser to `http://localhost:8501` to access the interface.

## 📖 Detailed Usage

### Preprocessing Module (`src/preprocessing.py`)

Handles face detection, eye extraction, and coordinate normalization:

```python
from src.preprocessing import GazePreprocessor

preprocessor = GazePreprocessor()
results = preprocessor.preprocess_image(image)
eye_crops = results['eye_crops']  # Left and right eye images
```

### Feature Extraction (`src/feature_extraction.py`)

Extracts both handcrafted and deep learning features:

```python
from src.feature_extraction import GazeFeatureExtractor

extractor = GazeFeatureExtractor()
features = extractor.extract_all_features(preprocessed_data)
# features['handcrafted'] - Traditional gaze metrics
# features['embeddings'] - CNN embeddings from eye crops
# features['combined'] - Combined feature vector
```

**Handcrafted Features:**
- Fixation stability and duration
- Gaze dispersion and spread
- Saccade count, amplitude, and velocity
- Blink rate and duration
- Left/right eye asymmetry

### Model Training (`src/model.py`)

Supports multiple classifier types with comprehensive evaluation:

```python
from src.model import ASDClassifier

# Random Forest (lightweight, fast)
classifier = ASDClassifier(model_type='random_forest')
classifier.fit(X_train, y_train)

# PyTorch neural network
classifier = ASDClassifier(model_type='pytorch', input_dim=1280*2+11)
classifier.fit(X_train, y_train, epochs=100)

# Evaluate with full metrics
results = classifier.evaluate(X_test, y_test, save_plots=True)
classifier.print_evaluation_report(results)
```

### Training Script (`src/train.py`)

Command-line training with dataset integration:

```bash
# Basic training
python src/train.py

# Advanced options
python src/train.py \
    --model-type random_forest \
    --data-dir data \
    --output-dir models \
    --test-size 0.2 \
    --random-state 42
```

## 🎯 Model Evaluation

The tool provides comprehensive evaluation metrics:

- **Binary Accuracy**: Overall classification accuracy
- **Precision/Recall/F1-Score**: Per-class performance
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification breakdown

Example evaluation output:
```
ASD CLASSIFICATION MODEL EVALUATION REPORT
==================================================

📊 PERFORMANCE METRICS:
Binary Accuracy: 0.847
Precision: 0.832
Recall: 0.847
F1-Score: 0.839
AUC-ROC: 0.921

🎯 CLASS-SPECIFIC METRICS:
Precision (ASD): 0.815
Recall (ASD): 0.892
Precision (Typical): 0.887
Recall (Typical): 0.801

📋 CONFUSION MATRIX:
                Predicted
                Typical    ASD
    Actual  Typical    80         20
            ASD        11         89
```

## 🌐 Web Interface

The Streamlit interface provides:

- **Live webcam feed** with real-time processing
- **Eye crop visualization** showing detected eye regions
- **Real-time classification** with confidence scores
- **Analysis history** and statistics
- **Export functionality** for research data

### Interface Features:

1. **Camera Controls**: Start/stop analysis with live preview
2. **Eye Tracking**: Visual display of detected eye regions
3. **Classification Results**: Real-time ASD probability estimates
4. **Historical Analysis**: Track performance over time
5. **Data Export**: Download analysis results as CSV

## 🔧 Technical Details

### Dependencies

- **Computer Vision**: OpenCV, MediaPipe
- **Deep Learning**: PyTorch, torchvision
- **Machine Learning**: scikit-learn
- **Web Interface**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

### System Requirements

- Python 3.8+
- Webcam for live analysis
- 4GB+ RAM recommended
- GPU optional (for PyTorch acceleration)

### Model Architecture

**Random Forest Classifier:**
- 100 trees by default
- Max depth: 10
- Feature importance analysis available

**PyTorch Neural Network:**
- Configurable hidden layers (default: 128, 64)
- Dropout regularization
- BCE loss with Adam optimizer

## 📊 Datasets

### Supported Datasets:

1. **MIT GazeCapture**: Large-scale gaze tracking dataset
2. **Kaggle ASD Dataset**: ASD-specific gaze patterns
3. **ASD vs TD Dataset**: Direct comparison between ASD and typically developing individuals

### Data Format Requirements:

- CSV files with gaze coordinates and labels
- Normalized coordinates (0-1 range) preferred
- Label column: 'ASD' or 'Typical'
- Feature columns: x, y, eye measurements, etc.

## 🔬 Research Applications

This tool is designed for:

- **Gaze pattern research** in ASD populations
- **Feature importance analysis** for ASD biomarkers
- **Real-time monitoring** in research settings
- **Dataset validation** and preprocessing
- **Algorithm development** and testing

## ⚖️ Ethical Considerations

- **Research Only**: Not for clinical diagnosis
- **Privacy Protection**: No data storage without consent
- **Transparency**: Clear disclosure of limitations
- **Professional Consultation**: Always involve qualified clinicians
- **Informed Consent**: Required for any human subjects research

## 🤝 Contributing

This is a research tool. Contributions should:
- Maintain research focus and ethical standards
- Include proper documentation
- Add comprehensive testing
- Follow the existing code structure

## 📄 License

Research and educational use only. Not for commercial or clinical application.

## 📞 Support

For research collaboration or questions:
- Review the code documentation
- Check the evaluation results in `results/`
- Ensure proper dataset formatting
- Verify model training completion

---

**Remember: This tool is for research purposes only and should never be used for medical diagnosis.**

👁️ The system automatically detects gaze direction as soon as the child looks at the screen.

🧠 It tracks which part of the face the baby focuses on (eyes, mouth, nose), or whether they look away.

⏱️ Sessions are short (~1 minute), making them suitable even for babies with limited attention spans.

Unlike other systems, there's no headset, no calibration, and no interaction required.

🧠 What It Detects

The tool analyzes multiple gaze patterns:

- **Eye Contact Duration:** Reduced fixation on eyes vs. typical patterns

- **Social Attention:** Preference for mouth/nose over eyes

- **Gaze Switching:** Frequency and patterns of attention shifts

- **Pupil Dynamics:** Changes in pupil size during social interaction

- **Facial Feature Ratios:** EAR (Eye Aspect Ratio) and mouth opening

🆕 New Features

**Dataset Management:**
- Load CSV datasets with gaze data
- Preprocess and split data for training/testing

**Feature Extraction:**
- Extract features from single images or video frames
- Aggregate features over time (mean, std, min, max)
- Support for batch processing of media files

**Machine Learning:**
- Train Random Forest, SVM, or MLP classifiers
- Evaluate model performance
- Save/load trained models
- Real-time classification of new data

**Enhanced UI:**
- Upload images/videos for analysis
- Train models from uploaded datasets
- View results and predictions

🚀 Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Applications:**
   - **Streamlit Web UI:** `streamlit run app.py`
   - **Tkinter GUI:** `python gui.py`
   - **Command Line Interface:** `python cli_interface.py --help`
   - **Live Webcam Tracking:** `python main.py`
   - **Batch Processing:** `python batch_process.py path/to/video.mp4`
   - **Demo:** `python demo.py`
   - **Jupyter Notebook:** `jupyter notebook demo_notebook.ipynb`

📊 Data Format

Expected CSV format for datasets:
```
timestamp,target,x,y,label
1,eyes_left,100,80,ASD
2,mouth,150,120,Typical
...
```

Where:
- `timestamp`: Time in seconds
- `target`: Gaze target (eyes_left, eyes_right, mouth, nose, none)
- `x,y`: Pupil coordinates
- `label`: ASD or Typical

🔬 Technical Details

**Feature Extraction:**
- Eye Aspect Ratio (EAR)
- Pupil positions and distances
- Mouth opening measurements
- Distances to facial ROIs
- Temporal aggregations (mean, std, etc.)

**ML Models:**
- Random Forest (default)
- Support Vector Machine
- Multi-Layer Perceptron

**Accuracy:** Models can achieve 70-85% accuracy on well-curated datasets (results may vary)

⚠️ Important Notes

- This tool is for research and screening purposes only
- Not a diagnostic tool - consult professionals for ASD diagnosis
- Ensure proper ethical considerations for data collection
- Data remains local and private

📄 License

This project is open-source. Please cite appropriately if used in research.

Made with ❤️ by the ASD Research Initiative.

The tool focuses on identifying key early behavioral markers of ASD:

🔴 Excessive fixation on eyes without switching

🔵 Low attention to the mouth (linked to speech/social engagement)

🟠 Very few gaze transitions

🟡 Unfocused or scattered gaze

These behaviors are logged in real-time and summarized into:

An Autism Risk Score (0–10)

A risk level (Low, Moderate, or High)

Specific behavioral flags with explanations

💡 How It Works


1. Face and Eye Tracking
Uses MediaPipe Face Mesh (478 facial landmarks) for facial and eye region tracking

Tracks iris movement with high precision

2. Gaze Vector Estimation
Uses OpenCV’s solvePnP() to estimate head position and adjust the 3D gaze vector accordingly

Gaze is mapped to facial ROIs: eyes, mouth, nose, or off-face/unfocused

3. Real-Time Feedback
Overlays a red gaze vector line on the live webcam feed

Shows real-time direction of gaze on screen

4. Risk Detection Engine
Logs fixation percentages and transition frequency

Flags patterns associated with ASD risk

Classifies session into:

🟢 Low Risk
🟡 Moderate Risk
🔴 High Risk

5. Reporting
6. 
Saves:

gaze_report_[timestamp].pdf – Full session summary

gaze_report_bar_[timestamp].png – Bar chart of attention distribution

gaze_fixations.csv – Raw data for researchers

📋 Example Output (from a session)
yaml
Copy code
Autism Risk Score: 2/10
Risk Level: Low Risk

Fixation Summary:
- Eyes: 100%
- Mouth: 0%
- Nose: 0%
- Unfocused: 0%

Risk Flags:
- [!] Excessive eye fixation (>98%)
- [!] Very little attention to mouth (<5%)
- [!] Very few gaze switches (<3 transitions)

PDF + PNG reports saved in session directory.
📦 Installation
✅ Requirements
Python 3.10+

OpenCV (opencv-python)

MediaPipe

NumPy

Pandas

Matplotlib

Scikit-learn

FPDF

Playsound

risk_flags.py (custom behavior analysis module)

🔧 Setup


🏥 Use Cases
🧑‍⚕️ Pediatricians and developmental specialists for early screening

👩‍👧 Parents and caregivers monitoring at-home behavior

🧑‍🏫 Preschool educators or early intervention programs

🌍 Low-resource clinics or global health organizations

🚀 Future Improvements
Video-based stimuli to better engage infants

Emotion/pupil response tracking

GUI for non-technical users

Expanded risk scoring model with deep learning

Integration with EHR systems for clinics

🛑 Disclaimer
This tool is not a diagnostic system. It is a supportive screening aid meant to assist caregivers and professionals in identifying early behavioral patterns that may be associated with ASD.
Only a qualified clinician can make a formal diagnosis.

🧡 Built With Care
This project is inspired by the belief that early detection = early support. By making gaze analysis affordable and accessible, we aim to help families and clinicians catch early warning signs while it’s still early enough to make a difference.




