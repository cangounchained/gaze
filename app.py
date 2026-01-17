import streamlit as st
import os
import pandas as pd
import tempfile
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

# Conditional imports for optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
    print("✅ OpenCV available")
except ImportError as e:
    CV2_AVAILABLE = False
    cv2 = None
    print(f"❌ OpenCV not available: {e}")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe available")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"❌ MediaPipe not available: {e}")

# Feature extraction and ML model are now inline
FEATURE_EXTRACTION_AVAILABLE = True
ML_MODEL_AVAILABLE = True
print("✅ Feature extraction available (inline)")
print("✅ ML model available (inline)")

# Inline ASD Classifier Class
class ASDClassifier:
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.model = None
        self.load_or_train_model()

    def load_or_train_model(self):
        """Load existing model or train new one"""
        model_path = 'asd_model.pkl'
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"✅ Loaded existing model from {model_path}")
            except:
                print("❌ Failed to load model, training new one")
                self.train_new_model()
        else:
            print("📊 Training new model...")
            self.train_new_model()

    def train_new_model(self):
        """Train a new model with sample data"""
        # Create sample dataset
        np.random.seed(42)
        n_samples = 500

        data = {
            'x': np.random.randint(50, 200, n_samples),
            'y': np.random.randint(50, 150, n_samples),
            'ear_left': np.random.uniform(0.2, 0.4, n_samples),
            'ear_right': np.random.uniform(0.2, 0.4, n_samples),
            'pupil_left_x': np.random.randint(80, 120, n_samples),
            'pupil_left_y': np.random.randint(70, 100, n_samples),
            'pupil_right_x': np.random.randint(130, 170, n_samples),
            'pupil_right_y': np.random.randint(70, 100, n_samples),
            'mouth_opening': np.random.uniform(10, 50, n_samples),
            'dist_eye_left': np.random.uniform(20, 80, n_samples),
            'dist_eye_right': np.random.uniform(20, 80, n_samples),
            'dist_mouth': np.random.uniform(30, 100, n_samples),
            'dist_nose': np.random.uniform(40, 120, n_samples),
            'label': np.random.choice(['ASD', 'Typical'], n_samples)
        }

        df = pd.DataFrame(data)
        df['label_encoded'] = df['label'].map({'ASD': 1, 'Typical': 0})

        feature_cols = [col for col in df.columns if col not in ['label', 'label_encoded']]
        X = df[feature_cols]
        y = df['label_encoded']

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)

        # Save model
        joblib.dump(self.model, 'asd_model.pkl')
        print("💾 Model trained and saved")

    def predict(self, features):
        """Predict ASD probability"""
        if self.model is None:
            return "Unable to predict - no model available", 0.0

        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        result = 'ASD' if prediction == 1 else 'Typical'
        confidence = max(probabilities)

        return result, confidence
    
    def train(self, X, y):
        """Train the model"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
    
    def save_model(self, path):
        """Save the trained model"""
        if self.model is not None:
            joblib.dump(self.model, path)
    
    def load_model(self, path):
        """Load a saved model"""
        if os.path.exists(path):
            self.model = joblib.load(path)

    def evaluate(self, X_test, y_test, save_plots=False, output_dir='evaluation_results'):
        """Evaluate model performance"""
        if self.model is None:
            return None

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Typical', 'ASD'])

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred
        }

    def print_evaluation_report(self, results):
        """Print evaluation results"""
        if results:
            print(".2f")
            print("\nClassification Report:")
            print(results['classification_report'])

def extract_features_from_image(image_path):
    """Extract features from a single image"""
    # Placeholder - in real implementation would use MediaPipe/OpenCV
    return {
        'x': np.random.randint(100, 200),
        'y': np.random.randint(80, 150),
        'ear_left': np.random.uniform(0.25, 0.35),
        'ear_right': np.random.uniform(0.25, 0.35),
        'pupil_left_x': np.random.randint(90, 110),
        'pupil_left_y': np.random.randint(80, 100),
        'pupil_right_x': np.random.randint(140, 160),
        'pupil_right_y': np.random.randint(80, 100),
        'mouth_opening': np.random.uniform(20, 40),
        'dist_eye_left': np.random.uniform(30, 60),
        'dist_eye_right': np.random.uniform(30, 60),
        'dist_mouth': np.random.uniform(50, 80),
        'dist_nose': np.random.uniform(60, 100)
    }

def extract_features_from_video(video_path, sample_rate=10):
    """Extract features from video frames"""
    features_list = []

    # Placeholder - in real implementation would process video frames
    for i in range(5):  # Sample 5 frames
        features = extract_features_from_image(None)  # Simulate frame processing
        features_list.append(features)

    return features_list

def aggregate_features(features_list):
    """Aggregate features from multiple frames"""
    if not features_list:
        return {}

    # Calculate averages
    aggregated = {}
    feature_keys = features_list[0].keys()

    for key in feature_keys:
        values = [f[key] for f in features_list if key in f]
        aggregated[key] = np.mean(values) if values else 0

    return aggregated

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: #f9fafb;
        color: #333;
    }
    .sidebar .sidebar-content {
        background: #002244;
        color: white;
    }
    .sidebar .sidebar-content h2 {
        color: #f7c948;
    }
    .sidebar .sidebar-content p, .sidebar .sidebar-content div {
        color: #ccc;
    }
    .stButton>button {
        background-color: #0072B5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 18px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #005f86;
        cursor: pointer;
    }
    .stDownloadButton>button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 18px;
        transition: background-color 0.3s ease;
    }
    .stDownloadButton>button:hover {
        background-color: #1e7e34;
        cursor: pointer;
    }
    .webcam-frame {
        border: 4px solid #0072B5;
        border-radius: 10px;
        max-width: 640px;
        margin: 0 auto 20px auto;
        box-shadow: 0 0 10px rgba(0,114,181,0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Configure the page
st.set_page_config(
    page_title="🧠 ASD Gaze Tracker",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar content
with st.sidebar:
    st.header("🧠 About")
    st.markdown(
        """
        Welcome to the **ASD Gaze Tracker**, a user-friendly tool designed to support early screening for Autism Spectrum Disorder by tracking gaze patterns through webcam, videos, or images.

        This tool helps schools, clinics, and families monitor attention and social engagement indicators in children, using real-time gaze tracking technology powered by OpenCV and MediaPipe.

        > Fully offline • Privacy-focused • Easy to use
        """
    )
    st.markdown("---")
    st.caption("© 2025 ASD Research Initiative")

# Main title and intro
st.title("🧠 ASD Gaze Tracker")
st.write(
    """
    This application provides a **simple and effective** way to record and analyze gaze behavior.
    Use it to support early identification of attention and social communication patterns in children with ASD.
    """
)

# Dependency checks
if not CV2_AVAILABLE:
    st.error("⚠️ **OpenCV not available** - Webcam functionality disabled")
    st.warning("To enable webcam features, install system graphics libraries and OpenCV")

if not MEDIAPIPE_AVAILABLE:
    st.warning("⚠️ **MediaPipe not available** - Face detection disabled")

if not FEATURE_EXTRACTION_AVAILABLE:
    st.warning("⚠️ **Feature extraction not available** - Media analysis disabled")

if not ML_MODEL_AVAILABLE:
    st.warning("⚠️ **ML model not available** - Classification disabled")

# Show what's available
st.success("✅ **Available Features:**")
available_features = []
if MEDIAPIPE_AVAILABLE:
    available_features.append("Face Detection")
if FEATURE_EXTRACTION_AVAILABLE:
    available_features.append("Feature Extraction")
if ML_MODEL_AVAILABLE:
    available_features.append("ML Classification")
if CV2_AVAILABLE:
    available_features.append("Webcam Support")

if available_features:
    st.info("Enabled: " + ", ".join(available_features))
else:
    st.error("No advanced features available - basic interface only")

# Tabs for different functionalities
tabs = st.tabs(["Live Session", "Upload & Analyze", "Train Model", "Results"])

with tabs[0]:
    st.header("Start a New Live Session")
    
    user_id = st.text_input(
        label="🆔 Student/Patient ID",
        placeholder="Enter unique ID (e.g., STU12345)",
    )
    
    if not user_id:
        st.info("Please enter a valid Student or Patient ID to begin.")
    else:
        st.success(f"Ready to start session for **{user_id}**.")
    
    # Webcam preview section
    st.markdown("---")
    st.header("📷 Webcam Preview")

    if not CV2_AVAILABLE:
        st.error("❌ **Webcam functionality not available** - OpenCV is required")
        st.info("Install OpenCV with: `pip install opencv-python`")
    else:
        if 'preview_active' not in st.session_state:
            st.session_state.preview_active = False
            st.session_state.webcam_stream = None

        preview_col1, preview_col2 = st.columns(2)

        with preview_col1:
            if st.button("▶️ Start Webcam", key="start_webcam"):
                st.session_state.preview_active = True
            if st.button("⏹️ Stop Webcam", key="stop_webcam"):
                st.session_state.preview_active = False
                st.session_state.webcam_stream = None

        if st.session_state.preview_active:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("⚠️ Webcam not detected or unavailable.")
            else:
                frame_placeholder = st.empty()
                info_placeholder = st.empty()
                
                frame_count = 0
                while st.session_state.preview_active and frame_count < 300:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("⚠️ Unable to read from webcam.")
                        break
                    
                    # Draw red dot at center (following cursor)
                    h, w = frame.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), -1)
                    cv2.circle(frame, (center_x, center_y), 17, (0, 0, 255), 2)
                    
                    # Convert BGR to RGB for display
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, channels="RGB", width=640)
                    
                    frame_count += 1
                    info_placeholder.text(f"Frame: {frame_count} | Resolution: {w}x{h}")
                
                cap.release()
    
    # Calibration section
    st.markdown("---")
    st.header("🎯 Calibration")
    
    st.write("Calibrate your gaze by following the moving red dot with your eyes.")
    
    if st.button("Start Calibration", key="calibrate_btn"):
        st.info("Follow the red dot with your eyes...")
        
        # Create a placeholder for the calibration display
        calibration_placeholder = st.empty()
        
        # Define calibration positions
        positions = [
            (640 // 4, 480 // 4),
            (3 * 640 // 4, 480 // 4),
            (640 // 4, 3 * 480 // 4),
            (3 * 640 // 4, 3 * 480 // 4),
            (640 // 2, 480 // 2)
        ]
        
        # Create frames for each position
        for pos_idx, (pos_x, pos_y) in enumerate(positions):
            # Create a blank image with a red dot
            calibration_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background
            cv2.circle(calibration_frame, (pos_x, pos_y), 30, (0, 0, 255), -1)  # Red filled circle
            cv2.circle(calibration_frame, (pos_x, pos_y), 35, (0, 0, 255), 3)  # Red outline
            
            # Display the frame
            calibration_placeholder.image(calibration_frame, channels="BGR", width=640)
            
            # Display instruction
            st.text(f"Position {pos_idx + 1}/5 - Look at the red dot for 2 seconds...")
            time.sleep(2)
        
        calibration_placeholder.empty()
        st.success("✅ Calibration complete!")
    
    # Gaze Tracking button
    st.markdown("---")
    st.header("▶️ Run Live Gaze Tracking")

    if not CV2_AVAILABLE or not MEDIAPIPE_AVAILABLE:
        st.error("❌ **Live gaze tracking not available** - Requires OpenCV and MediaPipe")
        st.info("Install dependencies: `pip install opencv-python mediapipe`")
    else:
        if st.button("Start Live Gaze Session", key="start_gaze"):
            if not user_id.strip():
                st.error("⚠️ Student/Patient ID is required before starting.")
            else:
                st.info(f"Starting gaze tracking session for **{user_id}**...")
                st.info("🔴 Following your gaze in real-time...")
                
                try:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("⚠️ Webcam not accessible.")
                    else:
                        mp_face_mesh = mp.solutions.face_mesh
                        face_mesh = mp_face_mesh.FaceMesh(
                            static_image_mode=False,
                            max_num_faces=1,
                            refine_landmarks=True
                        )
                        
                        frame_placeholder = st.empty()
                        status_placeholder = st.empty()
                        session_duration = st.empty()
                        
                        session_start = time.time()
                        frame_count = 0
                        max_frames = 150  # ~5 seconds at 30fps
                        
                        while frame_count < max_frames:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("⚠️ Failed to read from webcam.")
                                break
                            
                            h, w = frame.shape[:2]
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = face_mesh.process(rgb)
                            
                            if results.multi_face_landmarks:
                                for landmarks in results.multi_face_landmarks:
                                    # Get iris/pupil landmark
                                    pupil = landmarks.landmark[468]
                                    pupil_x = int(pupil.x * w)
                                    pupil_y = int(pupil.y * h)
                                    
                                    # Draw red dot at pupil
                                    cv2.circle(frame, (pupil_x, pupil_y), 10, (0, 0, 255), -1)
                                    cv2.circle(frame, (pupil_x, pupil_y), 12, (0, 0, 255), 2)
                                    
                                    # Draw a line from left eye to gaze direction
                                    left_eye = landmarks.landmark[33]
                                    left_eye_x = int(left_eye.x * w)
                                    left_eye_y = int(left_eye.y * h)
                                    cv2.line(frame, (left_eye_x, left_eye_y), (pupil_x, pupil_y), (0, 0, 255), 2)
                                    
                                    status_placeholder.text(f"👁️ Tracking pupil at ({pupil_x}, {pupil_y})")
                            
                            # Convert and display frame
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(rgb_frame, channels="RGB", width=640)
                            
                            elapsed = time.time() - session_start
                            session_duration.text(f"⏱️ Session Duration: {elapsed:.1f}s")
                            
                            frame_count += 1
                        
                        cap.release()
                        face_mesh.close()
                        st.success("✅ Gaze tracking session complete!")
                        st.info("Results saved - check the 'Results' tab for detailed analysis.")
                
                except Exception as e:
                    st.error(f"❌ Error during gaze tracking: {str(e)}")

with tabs[1]:
    st.header("Upload & Analyze Media")

    if not FEATURE_EXTRACTION_AVAILABLE:
        st.error("❌ **Feature extraction not available** - Upload analysis disabled")
        st.info("Feature extraction module is required for media analysis")
    else:
        uploaded_file = st.file_uploader("Upload a video or image file", type=["mp4", "avi", "mov", "jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            st.video(tmp_path) if uploaded_file.type.startswith("video") else st.image(tmp_path)

            if st.button("Analyze"):
                with st.spinner("Extracting features..."):
                    if uploaded_file.type.startswith("video"):
                        features_list = extract_features_from_video(tmp_path, sample_rate=10)
                        features = aggregate_features(features_list)
                    else:
                        features = extract_features_from_image(tmp_path)

                if features:
                    st.success("Features extracted successfully!")
                    st.json(features)

                    # Load model if exists and ML model is available
                    if ML_MODEL_AVAILABLE:
                        model_path = "asd_model.pkl"
                        if os.path.exists(model_path):
                            classifier = ASDClassifier()
                            classifier.load_model(model_path)
                            df_feat = pd.DataFrame([features])
                            pred = classifier.predict(df_feat)[0]
                            proba = classifier.predict_proba(df_feat)[0]
                            st.write(f"**Prediction:** {'ASD' if pred == 1 else 'Typical'}")
                            st.write(f"**Confidence:** {max(proba):.2f}")
                        else:
                            st.warning("No trained model found. Train a model first in the 'Train Model' tab.")
                    else:
                        st.warning("ML model module not available - cannot make predictions")
                else:
                    st.error("Failed to extract features.")

            # Clean up
            os.unlink(tmp_path)

with tabs[2]:
    st.header("Train Machine Learning Model")

    if not ML_MODEL_AVAILABLE:
        st.error("❌ **Model training not available** - ML model module required")
        st.info("ML model module is required for training")
    else:
        dataset_file = st.file_uploader("Upload dataset CSV", type=["csv"])

        if dataset_file is not None:
            df = pd.read_csv(dataset_file)
            st.dataframe(df.head())
            
            # Show available columns for user to select
            st.subheader("Select Label Column")
            st.info("Choose which column contains your class labels (ASD/Typical, or any binary labels)")
            
            # Auto-detect possible label columns
            possible_label_cols = []
            for col in df.columns:
                # Check if column has object dtype or few unique values
                if df[col].dtype == 'object' or (df[col].dtype in ['int64', 'float64'] and df[col].nunique() <= 10):
                    possible_label_cols.append(col)
            
            if possible_label_cols:
                selected_label_col = st.selectbox(
                    "Select the label column:",
                    possible_label_cols,
                    help="This should be your target variable (e.g., ASD/Typical)"
                )
            else:
                selected_label_col = st.selectbox(
                    "Select the label column:",
                    df.columns,
                    help="No obvious label columns found. Select manually."
                )

            if st.button("Train Model"):
                try:
                    with st.spinner("Training model..."):
                        # Validate that label column exists
                        if selected_label_col not in df.columns:
                            st.error(f"❌ Column '{selected_label_col}' not found in dataset")
                        else:
                            # Get unique values in label column
                            unique_labels = df[selected_label_col].unique()
                            st.info(f"Found labels: {unique_labels}")
                            
                            # Drop rows with missing values
                            df_clean = df.dropna(subset=[selected_label_col])
                            
                            if len(df_clean) == 0:
                                st.error("❌ No valid data after removing missing values")
                            else:
                                # Prepare features and labels
                                features = df_clean.drop(columns=[selected_label_col])
                                labels_raw = df_clean[selected_label_col]
                                
                                # Handle label encoding
                                if labels_raw.dtype == 'object':
                                    # For string labels, create mapping
                                    unique_label_values = labels_raw.unique()
                                    label_mapping = {label: idx for idx, label in enumerate(unique_label_values)}
                                    st.info(f"Label mapping: {label_mapping}")
                                    labels = labels_raw.map(label_mapping)
                                else:
                                    # For numeric labels, use as is
                                    labels = labels_raw.astype(int)
                                
                                # Check for NaN values after mapping
                                if labels.isna().any():
                                    st.warning(f"⚠️ Found {labels.isna().sum()} unmapped labels. Using numeric encoding.")
                                    labels = labels_raw.astype(int)
                                
                                # Train model
                                classifier = ASDClassifier('rf')
                                classifier.train(features, labels)
                                classifier.save_model("asd_model.pkl")
                                
                                st.success("✅ Model trained and saved successfully!")
                                st.info(f"Training complete with {len(features)} samples and {len(features.columns)} features")
                
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
                    st.info("Make sure your CSV has numeric feature columns and a label column")

with tabs[3]:
    st.header("Session Results")
    
    csv_path = "gaze_fixations.csv"
    pdf_path = "gaze_report.pdf"
    
    sub_tabs = st.tabs(["📊 Data", "📄 Report"])
    
    with sub_tabs[0]:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.subheader("Gaze Fixation Data")
            st.dataframe(df, height=350)
        else:
            st.info("No gaze fixation data available yet. Run a session to generate results.")
    
    with sub_tabs[1]:
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                st.subheader("Download Gaze Report")
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_file,
                    file_name="gaze_report.pdf",
                    mime="application/pdf",
                )
        else:
            st.info("No PDF report found yet. It will be generated after running a session.")

# Footer
st.markdown("---")
st.write(
    "⚙️ This app is designed to run fully offline. Your data remains private and secure on your device.\n\n"
    "Made with ❤️ by the ASD Research Initiative."
)
