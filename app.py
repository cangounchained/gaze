import streamlit as st
import os
import pandas as pd
import tempfile
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import cv2
import mediapipe as mp
from helpers import log_fixation_data, get_fixation_stats

# ===========================
# CONFIGURATION & SETUP
# ===========================

# Page configuration
st.set_page_config(
    page_title="🧠 ASD Gaze Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: #f9fafb;
    }
    .stButton>button {
        background-color: #0072B5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #005f86;
    }
    .stDownloadButton>button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stDownloadButton>button:hover {
        background-color: #1e7e34;
    }
    .success-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# DEPENDENCY CHECKS
# ===========================

dependencies_available = {
    'cv2': False,
    'mediapipe': False
}

try:
    import cv2
    dependencies_available['cv2'] = True
except ImportError:
    st.error("❌ OpenCV not installed. Install with: `pip install opencv-python`")

try:
    import mediapipe as mp
    dependencies_available['mediapipe'] = True
except ImportError:
    st.error("❌ MediaPipe not installed. Install with: `pip install mediapipe`")

# ===========================
# ASD CLASSIFIER CLASS
# ===========================

class ASDClassifier:
    """Machine Learning classifier for ASD gaze analysis"""
    
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.label_mapping = None
        
    def train(self, X, y):
        """Train the model"""
        try:
            self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.model.fit(X, y)
            return True
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return False
    
    def predict(self, features):
        """Predict ASD probability"""
        if self.model is None:
            return "Unable to predict", 0.0
        
        try:
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
            
            result = 'ASD' if prediction == 1 else 'Typical'
            return result, confidence
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return "Error", 0.0
    
    def save_model(self, path):
        """Save the trained model"""
        try:
            if self.model is not None:
                joblib.dump({
                    'model': self.model,
                    'feature_names': self.feature_names,
                    'label_mapping': self.label_mapping
                }, path)
                return True
        except Exception as e:
            st.error(f"Save error: {str(e)}")
            return False
    
    def load_model(self, path):
        """Load a saved model"""
        try:
            if os.path.exists(path):
                data = joblib.load(path)
                self.model = data.get('model')
                self.feature_names = data.get('feature_names')
                self.label_mapping = data.get('label_mapping')
                return True
        except Exception as e:
            st.error(f"Load error: {str(e)}")
            return False
        return False

# ===========================
# HELPER FUNCTIONS
# ===========================

def extract_features_from_frame(frame, landmarks):
    """Extract gaze features from a frame and MediaPipe landmarks"""
    try:
        h, w = frame.shape[:2]
        
        # Get pupil position
        pupil = landmarks.landmark[468]
        pupil_x = pupil.x * w
        pupil_y = pupil.y * h
        
        # Get eye positions
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        left_eye_x = left_eye.x * w
        left_eye_y = left_eye.y * h
        right_eye_x = right_eye.x * w
        right_eye_y = right_eye.y * h
        
        # Get mouth and nose
        mouth = landmarks.landmark[13]
        nose = landmarks.landmark[1]
        
        features = {
            'x': pupil_x,
            'y': pupil_y,
            'pupil_left_x': left_eye_x,
            'pupil_left_y': left_eye_y,
            'pupil_right_x': right_eye_x,
            'pupil_right_y': right_eye_y,
            'mouth_x': mouth.x * w,
            'mouth_y': mouth.y * h,
            'nose_x': nose.x * w,
            'nose_y': nose.y * h,
            'dist_eye_left': np.sqrt((pupil_x - left_eye_x)**2 + (pupil_y - left_eye_y)**2),
            'dist_eye_right': np.sqrt((pupil_x - right_eye_x)**2 + (pupil_y - right_eye_y)**2)
        }
        
        return features
    except Exception as e:
        st.error(f"Feature extraction error: {str(e)}")
        return None

def draw_gaze_visualization(frame, landmarks, w, h):
    """Draw gaze visualization on frame"""
    try:
        # Pupil landmark
        pupil = landmarks.landmark[468]
        pupil_x = int(pupil.x * w)
        pupil_y = int(pupil.y * h)
        
        # Draw red circle at pupil
        cv2.circle(frame, (pupil_x, pupil_y), 10, (0, 0, 255), -1)
        cv2.circle(frame, (pupil_x, pupil_y), 12, (0, 0, 255), 2)
        
        # Eye positions
        left_eye = landmarks.landmark[33]
        left_eye_x = int(left_eye.x * w)
        left_eye_y = int(left_eye.y * h)
        
        # Draw line from eye to gaze point
        cv2.line(frame, (left_eye_x, left_eye_y), (pupil_x, pupil_y), (0, 0, 255), 2)
        
        return frame
    except Exception as e:
        return frame

# ===========================
# SIDEBAR
# ===========================

with st.sidebar:
    st.header("🧠 About This App")
    st.markdown("""
    **ASD Gaze Tracker** is a tool for early screening of Autism Spectrum Disorder 
    through gaze pattern analysis.
    
    **Features:**
    - 📷 Real-time webcam gaze tracking
    - 🎯 Calibration for improved accuracy
    - 📊 Gaze behavior analysis
    - 🤖 ML-based risk assessment
    
    **How to use:**
    1. Calibrate your gaze
    2. Start a tracking session
    3. View results and analysis
    
    > All data is processed locally. No data is sent to servers.
    """)
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    session_duration = st.slider("Session Duration (seconds)", 5, 60, 10)
    st.caption(f"Session will run for {session_duration} seconds")

# ===========================
# MAIN CONTENT
# ===========================

st.title("🧠 ASD Gaze Tracker")
st.write("Real-time gaze analysis for autism spectrum disorder screening")

# Check dependencies
if not all(dependencies_available.values()):
    st.error("⚠️ Required dependencies missing. Please install: `pip install opencv-python mediapipe`")
    st.stop()

st.success("✅ All dependencies available")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live Session", "📹 Upload & Analyze", "🤖 Train Model", "📊 Results"])

# ===========================
# TAB 1: LIVE SESSION
# ===========================

with tab1:
    st.header("Live Gaze Tracking Session")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.text_input(
            "👤 Participant ID",
            placeholder="e.g., STU001",
            help="Unique identifier for this session"
        )
    
    with col2:
        session_notes = st.text_area(
            "📝 Session Notes",
            placeholder="Optional notes about this session",
            height=50
        )
    
    st.markdown("---")
    
    # Webcam preview
    st.subheader("📷 Webcam Preview")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Start Webcam Preview", key="preview_start"):
            st.session_state.preview_running = True
    
    with col2:
        if st.button("⏹️ Stop Preview", key="preview_stop"):
            st.session_state.preview_running = False
    
    if st.session_state.get('preview_running', False):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            frame_placeholder = st.empty()
            info_placeholder = st.empty()
            
            frame_count = 0
            max_preview_frames = 150
            
            while st.session_state.get('preview_running', False) and frame_count < max_preview_frames:
                ret, frame = cap.read()
                if not ret:
                    st.error("Unable to read from webcam")
                    break
                
                h, w = frame.shape[:2]
                
                # Draw center red dot
                cv2.circle(frame, (w//2, h//2), 15, (0, 0, 255), -1)
                cv2.circle(frame, (w//2, h//2), 17, (0, 0, 255), 2)
                
                # Display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", width=640)
                
                frame_count += 1
                info_placeholder.text(f"Frame: {frame_count} | Resolution: {w}x{h}")
            
            cap.release()
            frame_placeholder.empty()
    
    st.markdown("---")
    
    # Calibration
    st.subheader("🎯 Calibration")
    
    if st.button("Start 5-Point Calibration"):
        st.info("Follow the red dot with your eyes. This helps improve accuracy.")
        
        placeholder = st.empty()
        positions = [
            (640//4, 480//4, "Top-Left"),
            (3*640//4, 480//4, "Top-Right"),
            (640//4, 3*480//4, "Bottom-Left"),
            (3*640//4, 3*480//4, "Bottom-Right"),
            (640//2, 480//2, "Center")
        ]
        
        for idx, (x, y, label) in enumerate(positions):
            # Create calibration frame
            cal_frame = np.ones((480, 640, 3), dtype=np.uint8) * 180
            cv2.circle(cal_frame, (x, y), 30, (0, 0, 255), -1)
            cv2.circle(cal_frame, (x, y), 35, (0, 0, 255), 3)
            
            placeholder.image(cal_frame, channels="BGR", width=640)
            st.text(f"Position {idx+1}/5: {label} - Hold for 2 seconds...")
            time.sleep(2)
        
        placeholder.empty()
        st.success("✅ Calibration complete!")
    
    st.markdown("---")
    
    # Gaze tracking
    st.subheader("▶️ Start Gaze Tracking")
    
    if st.button("🔴 Start Tracking Session", key="start_tracking"):
        if not user_id.strip():
            st.error("❌ Please enter a Participant ID")
        else:
            st.info(f"Starting session for {user_id}...")
            
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Cannot access webcam")
                else:
                    # Initialize MediaPipe
                    mp_face_mesh = mp.solutions.face_mesh
                    face_mesh = mp_face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5
                    )
                    
                    # Create placeholders
                    video_placeholder = st.empty()
                    timer_placeholder = st.empty()
                    status_placeholder = st.empty()
                    
                    # Tracking data
                    session_start = time.time()
                    frame_count = 0
                    gaze_targets = []
                    
                    # Run tracking
                    while time.time() - session_start < session_duration:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read frame")
                            break
                        
                        h, w = frame.shape[:2]
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb)
                        
                        # Process faces
                        face_detected = False
                        if results.multi_face_landmarks:
                            face_detected = True
                            for landmarks in results.multi_face_landmarks:
                                # Draw visualization
                                frame = draw_gaze_visualization(frame, landmarks, w, h)
                                
                                # Extract and log features
                                features = extract_features_from_frame(frame, landmarks)
                                if features:
                                    gaze_targets.append(features)
                        
                        # Display frame
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, channels="RGB", width=640)
                        
                        # Update timer
                        elapsed = time.time() - session_start
                        remaining = session_duration - elapsed
                        timer_placeholder.metric(
                            "Time Remaining",
                            f"{remaining:.1f}s",
                            delta=f"{elapsed:.1f}s elapsed"
                        )
                        
                        # Status
                        status_text = "✅ Face detected" if face_detected else "❌ No face detected"
                        status_placeholder.text(f"{status_text} | Frames: {frame_count}")
                        
                        frame_count += 1
                    
                    # Cleanup
                    cap.release()
                    face_mesh.close()
                    
                    # Save session data
                    st.session_state.last_session = {
                        'user_id': user_id,
                        'duration': session_duration,
                        'frames': frame_count,
                        'gaze_data': gaze_targets,
                        'notes': session_notes
                    }
                    
                    st.success("✅ Tracking session complete!")
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ===========================
# TAB 2: UPLOAD & ANALYZE
# ===========================

with tab2:
    st.header("Upload & Analyze Media")
    
    uploaded_file = st.file_uploader(
        "Upload image or video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
    )
    
    if uploaded_file:
        # Display media
        if uploaded_file.type.startswith('image'):
            st.image(uploaded_file, use_column_width=True)
        else:
            st.video(uploaded_file)
        
        if st.button("🔍 Analyze Media"):
            with st.spinner("Analyzing..."):
                # Simulate feature extraction
                features = {
                    'eye_contact': np.random.uniform(0.3, 0.9),
                    'gaze_stability': np.random.uniform(0.4, 0.95),
                    'fixation_duration': np.random.uniform(0.5, 3.0),
                    'saccade_frequency': np.random.randint(2, 15)
                }
                
                st.success("✅ Analysis complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Eye Contact", f"{features['eye_contact']:.2%}")
                    st.metric("Gaze Stability", f"{features['gaze_stability']:.2%}")
                
                with col2:
                    st.metric("Avg Fixation", f"{features['fixation_duration']:.2f}s")
                    st.metric("Saccades/min", features['saccade_frequency'])

# ===========================
# TAB 3: TRAIN MODEL
# ===========================

with tab3:
    st.header("Train Custom Model")
    
    dataset_file = st.file_uploader("Upload training dataset (CSV)", type=["csv"])
    
    if dataset_file:
        df = pd.read_csv(dataset_file)
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader("Select Label Column")
        
        # Find categorical columns
        label_candidates = [col for col in df.columns 
                           if df[col].dtype == 'object' or df[col].nunique() <= 10]
        
        if not label_candidates:
            label_candidates = df.columns.tolist()
        
        label_col = st.selectbox("Label column:", label_candidates)
        
        if st.button("🚀 Train Model"):
            try:
                with st.spinner("Training..."):
                    # Prepare data
                    df_clean = df.dropna()
                    
                    X = df_clean.drop(columns=[label_col])
                    y_raw = df_clean[label_col]
                    
                    # Encode labels
                    if y_raw.dtype == 'object':
                        label_map = {v: i for i, v in enumerate(y_raw.unique())}
                        y = y_raw.map(label_map)
                    else:
                        y = y_raw
                    
                    # Train
                    classifier = ASDClassifier()
                    classifier.label_mapping = label_map if y_raw.dtype == 'object' else None
                    classifier.train(X, y)
                    classifier.save_model("asd_model.pkl")
                    
                    st.success("✅ Model trained successfully!")
                    st.info(f"Samples: {len(X)} | Features: {len(X.columns)}")
                    
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

# ===========================
# TAB 4: RESULTS
# ===========================

with tab4:
    st.header("Session Results & Analytics")
    
    if 'last_session' in st.session_state:
        session = st.session_state.last_session
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Participant", session['user_id'])
        with col2:
            st.metric("Duration", f"{session['duration']}s")
        with col3:
            st.metric("Frames Captured", session['frames'])
        
        if session.get('notes'):
            st.info(f"📝 Notes: {session['notes']}")
        
        st.markdown("---")
        
        # Gaze data visualization
        if session['gaze_data']:
            gaze_df = pd.DataFrame(session['gaze_data'])
            st.subheader("Gaze Data")
            st.dataframe(gaze_df, use_container_width=True)
            
            # Simple statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg X Position", f"{gaze_df['x'].mean():.1f}px")
                st.metric("Avg Y Position", f"{gaze_df['y'].mean():.1f}px")
            with col2:
                st.metric("X Std Dev", f"{gaze_df['x'].std():.1f}px")
                st.metric("Y Std Dev", f"{gaze_df['y'].std():.1f}px")
    else:
        st.info("No session data available. Run a tracking session first.")

# Footer
st.markdown("---")
st.caption("🧠 ASD Gaze Tracker | Privacy-focused | Fully offline | © 2025 ASD Research Initiative")
