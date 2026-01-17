# 🧠 ASD Gaze Detection Research Tool

A comprehensive Python-based gaze-tracking research tool for studying autism spectrum disorder (ASD) detection through eye movement analysis.

## ⚠️ ETHICAL DISCLAIMER

**THIS TOOL IS FOR RESEARCH PURPOSES ONLY**

- **NOT a diagnostic tool** - Cannot be used for clinical diagnosis of ASD
- **NOT a medical device** - Not validated for medical decision-making
- **NOT a replacement** for professional healthcare evaluation
- **FOR RESEARCH ONLY** - Design, development, and testing of gaze-based ASD detection methods

Gaze-based ASD detection is an active area of research with ongoing validation. Results from this tool should **NEVER** be used for:
- Clinical diagnosis
- Medical decision-making
- Screening or assessment of individuals
- Any clinical application without proper validation

**Always consult qualified healthcare professionals for proper diagnosis and assessment.**

## Features

### 📷 Preprocessing Module (`preprocessing.py`)
- **Face & Eye Detection**: Uses MediaPipe Face Mesh (468 landmarks)
- **Eye Crop Extraction**: Extracts left and right eye regions automatically
- **Coordinate Normalization**: Normalizes gaze coordinates to [-1, 1] range
- **Image Preprocessing**: Resizes images to standard size for CNN input (224×224)
- **Landmark Visualization**: Draw landmarks and eye regions for debugging

### 🔍 Feature Extraction Module (`feature_extraction.py`)
- **Handcrafted Gaze Metrics**:
  - Fixation Stability (standard deviation of gaze positions)
  - Gaze Dispersion (range of gaze movement)
  - Saccade Analysis (velocity and frequency)
  - Left/Right Eye Asymmetry
  - Blink Rate Detection

- **CNN Embeddings**: Optional MobileNetV2 embeddings from eye crops
  - Pre-trained feature extraction
  - 1280-dimensional embeddings per eye
  - Automatic GPU support if available

### 🤖 Model Module (`model.py`)
- **RandomForest Classifier**: Fast, interpretable, efficient
  - 100 estimators with balanced class weights
  - Feature importance analysis
  - Automatic hyperparameter optimization

- **Lightweight Neural Network**: Optional PyTorch-based model
  - 3-layer fully connected network
  - Dropout regularization
  - Early stopping with validation

- **Comprehensive Evaluation**:
  - Accuracy, Precision, Recall, AUC metrics
  - Confusion matrix analysis
  - Classification reports

### 📊 Training Pipeline (`train.py`)
- **Dataset Processing**:
  - Support for image directories
  - CSV-based dataset loading
  - Video file processing with frame extraction

- **Integration**:
  - End-to-end preprocessing → feature extraction → training
  - Automatic handling of missing faces
  - Data augmentation support

- **Evaluation**:
  - Train/test split with stratification
  - Cross-validation support
  - Detailed training metrics and history

### 🎥 Webcam Interface (`webcam_ui.py`)
- **5-Point Red Dot Calibration**: 
  - Top-left, Top-right, Bottom-left, Bottom-right, Center
  - Instant visual feedback
  - Non-blocking UI for real-time interaction

- **Live Gaze Tracking**:
  - Real-time face detection from webcam
  - Red dot visualization at pupil center
  - Adjustable session duration
  - Face detection indicator

- **Real-time Analysis**:
  - Live gaze trajectory visualization
  - Gaze stability and dispersion metrics
  - Temporal feature computation
  - Classifier prediction with confidence scores

- **Data Export**:
  - JSON export of session data
  - Gaze trajectory logging
  - Calibration point records

## Project Structure

```
research_tool/
├── data/
│   ├── mit_gazecapture/          # MIT GazeCapture dataset
│   ├── kaggle_asd/                # Kaggle ASD detection dataset
│   └── asd_vs_td/                 # ASD vs Typical Development dataset
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Face/eye detection & preprocessing
│   ├── feature_extraction.py       # Gaze metrics & CNN embeddings
│   ├── model.py                   # Classifiers (RF & NN)
│   ├── train.py                   # Training pipeline
│   └── webcam_ui.py               # Streamlit interface
├── models/
│   └── asd_detector.pkl           # Trained model storage
├── features/
│   └── extracted_features.npz     # Cached features
├── requirements.txt
├── README.md                      # This file
└── LICENSE                        # Research license
```

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd research_tool
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import cv2, mediapipe, sklearn; print('✅ All dependencies installed')"
```

## Quick Start

### 1. Launch Webcam Interface
```bash
streamlit run src/webcam_ui.py
```

Access at `http://localhost:8501`

### 2. Training Pipeline (example)
```python
from src.train import GazeTrainingPipeline

# Initialize
pipeline = GazeTrainingPipeline(model_type='randomforest', use_cnn=False)

# Process dataset
X, y = pipeline.process_dataset('data/asd_vs_td')

# Train model
metrics = pipeline.train(X, y, test_size=0.2)

# Save model
pipeline.save_model('models/asd_detector.pkl')

# Print summary
pipeline.print_summary()
```

### 3. Real-time Prediction
```python
from src.preprocessing import GazePreprocessor
from src.feature_extraction import GazeFeatureExtractor
from src.model import ASDClassifier
import cv2

# Initialize
preprocessor = GazePreprocessor()
extractor = GazeFeatureExtractor(use_cnn=False)
classifier = ASDClassifier()
classifier.load('models/asd_detector.pkl')

# Capture and process frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    # Preprocess
    data = preprocessor.process_frame(frame)
    data = preprocessor.extract_eye_crops(data)
    data = preprocessor.normalize_gaze_coordinates(data)
    
    # Extract features
    features = extractor.extract_all_features(data)
    feature_vector = extractor.create_feature_vector(features)
    
    # Predict
    predictions, probs = classifier.predict(feature_vector.reshape(1, -1))
    print(f"Prediction: {'ASD' if predictions[0]==1 else 'Typical'}")
    print(f"Confidence: {max(probs[0])*100:.1f}%")

cap.release()
```

## Dataset Integration

### MIT GazeCapture Dataset
1. Download from: https://gazecapture.csail.mit.edu/
2. Place in `data/mit_gazecapture/`
3. Organize as: `images/person_*/frame_*.jpg`

### Kaggle ASD Detection Dataset
1. Download from: https://www.kaggle.com/datasets/...
2. Place in `data/kaggle_asd/`
3. Organize as: `asd/*.jpg` and `typical/*.jpg`

### Custom Dataset
Create directory structure:
```
data/my_dataset/
├── asd/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── typical/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Then:
```python
X, y = pipeline.process_dataset('data/my_dataset')
```

## API Reference

### GazePreprocessor
```python
from src.preprocessing import GazePreprocessor

# Initialize
prep = GazePreprocessor(image_size=224)

# Process frame
data = prep.process_frame(frame)  # Detect face
data = prep.extract_eye_crops(data)  # Extract eyes
data = prep.normalize_gaze_coordinates(data)  # Normalize coordinates
```

### GazeFeatureExtractor
```python
from src.feature_extraction import GazeFeatureExtractor

# Initialize (with optional CNN)
extractor = GazeFeatureExtractor(use_cnn=True, device='cuda')

# Extract all features
features = extractor.extract_all_features(data)
feature_vector = extractor.create_feature_vector(features)
```

### ASDClassifier
```python
from src.model import ASDClassifier

# Create classifier
clf = ASDClassifier(model_type='randomforest', input_size=10)

# Train
clf.fit(X_train, y_train)

# Predict
predictions, probabilities = clf.predict(X_test)

# Evaluate
metrics = clf.evaluate(X_test, y_test)

# Save/Load
clf.save('model.pkl')
clf.load('model.pkl')
```

### GazeTrainingPipeline
```python
from src.train import GazeTrainingPipeline

# Create pipeline
pipeline = GazeTrainingPipeline(model_type='randomforest')

# Process dataset (image directory or video)
X, y = pipeline.process_dataset('data/dataset')

# Train model
metrics = pipeline.train(X, y, test_size=0.2)

# Make predictions
predictions, probs = pipeline.predict(X_new)
```

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended for CNN embeddings)
- **GPU**: Optional (CUDA for faster inference)
- **Webcam**: Required for live tracking interface
- **Operating System**: Windows, macOS, or Linux

## Performance Notes

### Preprocessing
- Face detection: ~30-50ms per frame
- Eye crop extraction: ~5ms
- Coordinate normalization: <1ms

### Feature Extraction
- Handcrafted metrics: <2ms
- CNN embeddings: ~50-100ms (GPU), ~500ms (CPU)

### Classification
- RandomForest prediction: <1ms
- Neural network prediction: ~5ms

## Troubleshooting

### Webcam Issues
```bash
# Check available cameras
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### MediaPipe Issues
```bash
# Verify installation
python -c "import mediapipe; print(mediapipe.__version__)"
```

### GPU Issues (PyTorch)
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Research Applications

This tool can be used for:
- **Feature validation**: Test new gaze-based features
- **Model comparison**: Compare different classifiers
- **Dataset exploration**: Analyze gaze patterns in ASD populations
- **Method validation**: Validate novel ASD detection approaches
- **Educational demonstrations**: Teach gaze tracking concepts

## Contributing

We welcome research contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add comprehensive docstrings
4. Include unit tests
5. Submit a pull request

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{asd_gaze_tool_2024,
  title={ASD Gaze Detection Research Tool},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## References

1. **MediaPipe Face Mesh**
   - Bazarevsky, V., et al. (2020). "Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs"
   - https://google.github.io/mediapipe/

2. **Gaze-based ASD Detection**
   - Chawarska, K., et al. (2013). "Atypical development of social orienting in autism spectrum disorders"
   - Swanson, M. R., et al. (2017). "Gaze fixation during dynamic social scenes in autism spectrum disorder"

3. **CNN Feature Extraction**
   - Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
   - https://arxiv.org/abs/1801.04381

## License

Research Use Only License - See LICENSE file for details

This tool is provided for research purposes. Commercial use or use for clinical diagnosis is strictly prohibited without proper validation and regulatory approval.

## Authors

Created for ASD gaze detection research.

## Support

For issues, questions, or suggestions:
1. Check existing GitHub issues
2. Create a new issue with detailed information
3. Include system information and error messages
4. Provide minimal reproducible example

## Acknowledgments

- MediaPipe team (Google) for face mesh detection
- scikit-learn for ML implementations
- Streamlit for the interactive interface
- Research community for gaze-based ASD detection insights

---

**Last Updated**: January 2024

**Status**: ✅ Active Development

**Warning**: ⚠️ Research Prototype - Not for Clinical Use
