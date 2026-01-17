# 📋 Setup Guide - ASD Gaze Detection Research Tool

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8 | 3.10+ |
| RAM | 4GB | 8GB+ |
| Storage | 2GB | 10GB+ |
| GPU | Optional | NVIDIA (CUDA 11.8+) |
| OS | Windows/Mac/Linux | Linux |

## Step-by-Step Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/gaze-asd-detector.git
cd research_tool
```

### 2. Create Virtual Environment

#### Linux/macOS:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

#### Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 4. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "
import cv2
import mediapipe as mp
import sklearn
import streamlit as st
import torch
print('✅ All dependencies installed successfully!')
print(f'   OpenCV: {cv2.__version__}')
print(f'   MediaPipe: {mp.__version__}')
print(f'   scikit-learn: {sklearn.__version__}')
print(f'   PyTorch: {torch.__version__}')
"
```

## Configuration

### 1. Model Configuration

Edit `src/train.py` to adjust:

```python
# Model parameters
MODEL_TYPE = 'randomforest'  # or 'neural_network'
USE_CNN = False  # Set to True if you have GPU memory
N_ESTIMATORS = 100  # RandomForest trees
TEST_SIZE = 0.2  # Train/test split
```

### 2. Preprocessing Configuration

Edit `src/preprocessing.py`:

```python
# Image size for CNN input
IMAGE_SIZE = 224

# Face detection confidence
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
```

### 3. Feature Extraction Configuration

Edit `src/feature_extraction.py`:

```python
# Temporal window for feature computation
WINDOW_SIZE = 30  # frames

# CNN embeddings
USE_CNN = True  # Requires CUDA for speed
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### 4. Webcam UI Configuration

Edit `src/webcam_ui.py`:

```python
# Session parameters
DEFAULT_DURATION = 15  # seconds
PARTICIPANT_ID = "TEST001"
CALIBRATION_POINTS = 5
```

## Dataset Setup

### Directory Structure

```
research_tool/
└── data/
    ├── mit_gazecapture/
    │   ├── train/
    │   │   ├── asd/
    │   │   │   ├── img001.jpg
    │   │   │   └── ...
    │   │   └── typical/
    │   │       ├── img001.jpg
    │   │       └── ...
    │   └── test/
    │
    ├── kaggle_asd/
    │   ├── asd/
    │   │   ├── img001.jpg
    │   │   └── ...
    │   └── typical/
    │       ├── img001.jpg
    │       └── ...
    │
    └── asd_vs_td/  # Your custom dataset
        ├── asd/
        │   ├── img001.jpg
        │   └── ...
        └── typical/
            ├── img001.jpg
            └── ...
```

### Downloading Public Datasets

#### MIT GazeCapture
```bash
cd data/mit_gazecapture
# Download from: https://gazecapture.csail.mit.edu/
# Extract to this directory
```

#### Kaggle ASD Dataset
```bash
pip install kaggle
kaggle datasets download -d <dataset-id>
unzip -d data/kaggle_asd/
```

### Organizing Custom Data

```bash
# Create structure
mkdir -p data/my_study/{asd,typical}

# Add images
cp /path/to/asd/images/* data/my_study/asd/
cp /path/to/typical/images/* data/my_study/typical/

# Verify
ls -la data/my_study/asd/ | wc -l   # Count ASD images
ls -la data/my_study/typical/ | wc -l  # Count typical images
```

## Running the System

### 1. Launch Webcam Interface

```bash
streamlit run src/webcam_ui.py --logger.level=info
```

Visit: http://localhost:8501

### 2. Train Model (First Time)

```bash
python train_example.py
```

This will:
- Load data from `data/asd_vs_td/`
- Extract features from all images
- Train RandomForest classifier
- Save model to `models/asd_detector.pkl`
- Print metrics and summary

### 3. Use Trained Model in UI

1. Launch webcam interface: `streamlit run src/webcam_ui.py`
2. Go to sidebar → "Model"
3. Enter path: `models/asd_detector.pkl`
4. Click "Load Model"
5. Use predictions in "Live Analysis" tab

## Advanced Configuration

### GPU Acceleration (CUDA)

#### Check CUDA availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

#### Enable GPU in code:
```python
# In src/feature_extraction.py
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# In src/model.py (if using Neural Network)
# PyTorch automatically uses GPU if available
```

#### Install CUDA (if needed):
```bash
# For PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Multi-GPU Setup

```python
# In src/feature_extraction.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use GPUs 0 and 1

# In src/model.py (PyTorch)
# Use DataParallel for multi-GPU
model = nn.DataParallel(model, device_ids=[0, 1])
```

### Batch Processing

```python
from src.train import GazeTrainingPipeline

pipeline = GazeTrainingPipeline()

# Process videos in batch
for video_file in glob('data/videos/*.mp4'):
    X, indices = pipeline.process_video_dataset(video_file)
    # Use X for feature extraction
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'mediapipe'"

```bash
# Solution: Reinstall MediaPipe
pip uninstall mediapipe -y
pip install --upgrade mediapipe
```

### Issue: Webcam not opening

```bash
# Check webcam availability
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# If False, check system permissions and restart
# On Linux: sudo usermod -a -G video $USER
```

### Issue: Out of Memory error

```python
# In train_example.py, set:
USE_CNN = False  # Disable CNN embeddings

# Or reduce batch size in training:
# In src/model.py, reduce n_estimators
```

### Issue: Slow preprocessing

```python
# Enable GPU processing in feature_extraction.py
device = 'cuda'

# Use fewer frames for demo
max_frames = 100  # Instead of 300
```

### Issue: "CUDA out of memory"

```python
# Reduce model size
USE_CNN = False
BATCH_SIZE = 8  # Smaller batches

# Clear GPU cache
import torch
torch.cuda.empty_cache()
```

## Performance Optimization

### 1. Enable Caching

```python
# In train.py, add:
import functools

@functools.lru_cache(maxsize=1000)
def get_features(image_hash):
    # Compute features once
    pass
```

### 2. Parallel Processing

```bash
# Install parallel processing
pip install ray

# Use in preprocessing
from ray import tune
```

### 3. Model Quantization

```python
# In model.py, quantize PyTorch model
from torch.quantization import quantize_dynamic
quantized_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

## Monitoring & Logging

### Enable detailed logging

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Monitor training progress

```bash
# Terminal 1: Run training
python train_example.py

# Terminal 2: Monitor resources
watch nvidia-smi  # GPU usage
htop  # CPU usage
```

## Security & Privacy

### 1. Data Encryption

```python
# Encrypt sensitive data
from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher = Fernet(key)
encrypted_data = cipher.encrypt(data)
```

### 2. Anonymization

```python
import hashlib

def anonymize_participant_id(pid):
    return hashlib.sha256(pid.encode()).hexdigest()[:8]
```

### 3. Secure Storage

```bash
# Encrypt model files
gpg --symmetric models/asd_detector.pkl

# Secure deletion
pip install secure-delete
srm models/sensitive_data.pkl
```

## Version Control

```bash
# Initialize git (if not done)
git init

# Add files
git add src/ requirements.txt README.md

# Commit
git commit -m "Initial ASD gaze detection tool"

# Do NOT commit:
# - data/
# - models/*.pkl
# - venv/
# - __pycache__/
```

## Docker Deployment (Optional)

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8501

CMD ["streamlit", "run", "src/webcam_ui.py"]
```

### Build and run

```bash
docker build -t asd-gaze-detector .
docker run -p 8501:8501 --device /dev/video0 asd-gaze-detector
```

## Testing & Validation

### Run unit tests (create tests/)

```bash
pytest tests/ -v
```

### Validate on test data

```python
from src.train import GazeTrainingPipeline

pipeline = GazeTrainingPipeline()
pipeline.load_model('models/asd_detector.pkl')

# Test on known dataset
test_accuracy = pipeline.classifier.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy['accuracy']:.4f}")
```

## Support & Documentation

- **Full README**: See [README.md](README.md)
- **Quick Start**: See [QUICK_START.md](QUICK_START.md)
- **License**: See [LICENSE](LICENSE)
- **API Docs**: Code docstrings in each module

## Frequently Asked Questions

**Q: Can I use this for clinical diagnosis?**
A: No. This is research-only. Always consult healthcare professionals.

**Q: How much data do I need to train?**
A: Minimum 100 samples per class (ASD/Typical). 500+ recommended.

**Q: Do I need a GPU?**
A: No, but it's faster. CPU works fine for most use cases.

**Q: Can I publish results with this tool?**
A: Yes, but you must cite it and disclose its research-only nature.

**Q: How do I collect my own data?**
A: Use the webcam interface to record sessions, then export data.

---

**Last Updated**: January 2024
**Status**: ✅ Ready for Research Use
**License**: Research Use Only - See LICENSE file
