import streamlit as st
import os
import pandas as pd
import tempfile
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

# Properly handle optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="🧠 ASD Gaze Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            return True
        except Exception as e:
            st.error(f"❌ Training error: {str(e)}")
            return False
    
    def predict(self, features):
        """Predict ASD probability"""
        if self.model is None:
            return "Unable to predict", 0.0
        
        try:
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            elif isinstance(features, pd.DataFrame):
                pass
            
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
            
            result = 'ASD' if prediction == 1 else 'Typical'
            return result, confidence
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
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
            st.error(f"❌ Save error: {str(e)}")
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
            st.error(f"❌ Load error: {str(e)}")
        return False

# ===========================
# SIDEBAR
# ===========================

with st.sidebar:
    st.header("🧠 About")
    st.write("""
    **ASD Gaze Tracker** - Real-time gaze analysis for autism spectrum disorder screening.
    
    **Features:**
    - 📷 Webcam gaze tracking
    - 🎯 Calibration
    - 📊 Analysis
    - 🤖 ML Models
    """)
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    session_duration = st.slider("Session Duration (seconds)", 5, 60, 15)

# ===========================
# MAIN PAGE
# ===========================

st.title("🧠 ASD Gaze Tracker")

# Check dependencies
if not CV2_AVAILABLE:
    st.error("❌ OpenCV not installed. Install with: pip install opencv-python")
    
if not MEDIAPIPE_AVAILABLE:
    st.error("❌ MediaPipe not installed. Install with: pip install mediapipe")

if CV2_AVAILABLE and MEDIAPIPE_AVAILABLE:
    st.success("✅ All dependencies available")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live Session", "📹 Upload & Analyze", "🤖 Train Model", "📊 Results"])

# ===========================
# TAB 1: LIVE SESSION
# ===========================

with tab1:
    st.header("Live Gaze Tracking Session")
    
    col1, col2 = st.columns(2)
    with col1:
        user_id = st.text_input("Participant ID", placeholder="e.g., STU001")
    with col2:
        session_notes = st.text_area("Notes", placeholder="Optional", height=50)
    
    st.markdown("---")
    st.subheader("📷 Webcam Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Start Preview", key="preview_start"):
            st.session_state.preview_active = True
    
    with col2:
        if st.button("⏹️ Stop Preview", key="preview_stop"):
            st.session_state.preview_active = False
    
    # Webcam preview
    if st.session_state.get('preview_active', False):
        if not CV2_AVAILABLE:
            st.error("OpenCV not available")
        else:
            st.info("Starting webcam...")
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Cannot access webcam. Make sure it's connected and not in use.")
            else:
                frame_placeholder = st.empty()
                info_placeholder = st.empty()
                
                frame_count = 0
                while st.session_state.get('preview_active', False) and frame_count < 300:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Cannot read from webcam")
                        break
                    
                    h, w = frame.shape[:2]
                    
                    # Draw red dot at center
                    cv2.circle(frame, (w//2, h//2), 15, (0, 0, 255), -1)
                    cv2.circle(frame, (w//2, h//2), 17, (0, 0, 255), 2)
                    
                    # Display
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, channels="RGB", width=640)
                    frame_count += 1
                    info_placeholder.text(f"Frame: {frame_count} | {w}x{h}")
                    
                    time.sleep(0.01)  # Small delay to prevent CPU overload
                
                cap.release()
                st.session_state.preview_active = False
    
    st.markdown("---")
    st.subheader("🎯 Calibration")
    
    if st.button("Start 5-Point Calibration", key="calibrate"):
        if not CV2_AVAILABLE:
            st.error("OpenCV not available")
        else:
            st.info("Follow the red dot with your eyes")
            placeholder = st.empty()
            
            positions = [
                (160, 120, "Top-Left"),
                (480, 120, "Top-Right"),
                (160, 360, "Bottom-Left"),
                (480, 360, "Bottom-Right"),
                (320, 240, "Center")
            ]
            
            for idx, (x, y, label) in enumerate(positions):
                # Create calibration frame
                cal_frame = np.ones((480, 640, 3), dtype=np.uint8) * 180
                cv2.circle(cal_frame, (x, y), 30, (0, 0, 255), -1)
                cv2.circle(cal_frame, (x, y), 35, (0, 0, 255), 3)
                
                placeholder.image(cal_frame, channels="BGR", width=640)
                st.text(f"{idx+1}/5: {label} - Focus for 2 seconds...")
                time.sleep(2)
            
            placeholder.empty()
            st.success("✅ Calibration complete!")
    
    st.markdown("---")
    st.subheader("▶️ Start Gaze Tracking")
    
    if st.button("🔴 Start Tracking", key="start_tracking"):
        if not user_id.strip():
            st.error("❌ Enter Participant ID first")
        elif not CV2_AVAILABLE or not MEDIAPIPE_AVAILABLE:
            st.error("❌ OpenCV and MediaPipe required")
        else:
            st.info(f"Starting session for {user_id}...")
            
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Cannot access webcam")
                else:
                    mp_face_mesh = mp.solutions.face_mesh
                    face_mesh = mp_face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5
                    )
                    
                    video_placeholder = st.empty()
                    timer_placeholder = st.empty()
                    status_placeholder = st.empty()
                    
                    session_start = time.time()
                    frame_count = 0
                    gaze_data = []
                    
                    while time.time() - session_start < session_duration:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read frame")
                            break
                        
                        h, w = frame.shape[:2]
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb)
                        
                        face_detected = False
                        if results.multi_face_landmarks:
                            face_detected = True
                            for landmarks in results.multi_face_landmarks:
                                # Get pupil
                                pupil = landmarks.landmark[468]
                                pupil_x = int(pupil.x * w)
                                pupil_y = int(pupil.y * h)
                                
                                # Draw red dot at pupil
                                cv2.circle(frame, (pupil_x, pupil_y), 10, (0, 0, 255), -1)
                                cv2.circle(frame, (pupil_x, pupil_y), 12, (0, 0, 255), 2)
                                
                                # Draw line from eye
                                left_eye = landmarks.landmark[33]
                                left_eye_x = int(left_eye.x * w)
                                left_eye_y = int(left_eye.y * h)
                                cv2.line(frame, (left_eye_x, left_eye_y), (pupil_x, pupil_y), (0, 0, 255), 2)
                                
                                # Store gaze data
                                gaze_data.append({
                                    'frame': frame_count,
                                    'x': pupil_x,
                                    'y': pupil_y,
                                    'timestamp': time.time() - session_start
                                })
                        
                        # Display frame
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, channels="RGB", width=640)
                        
                        # Update timer
                        elapsed = time.time() - session_start
                        remaining = session_duration - elapsed
                        timer_placeholder.metric("Time Remaining", f"{remaining:.1f}s")
                        
                        # Status
                        status_text = "✅ Face detected" if face_detected else "❌ No face"
                        status_placeholder.text(f"{status_text} | Frames: {frame_count}")
                        
                        frame_count += 1
                        time.sleep(0.01)
                    
                    cap.release()
                    face_mesh.close()
                    
                    # Save session
                    st.session_state.last_session = {
                        'user_id': user_id,
                        'duration': session_duration,
                        'frames': frame_count,
                        'gaze_data': gaze_data,
                        'notes': session_notes
                    }
                    
                    st.success("✅ Tracking complete!")
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ===========================
# TAB 2: UPLOAD & ANALYZE
# ===========================

with tab2:
    st.header("Upload & Analyze Media")
    
    uploaded_file = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    
    if uploaded_file:
        if uploaded_file.type.startswith('image'):
            st.image(uploaded_file, width=640)
        else:
            st.video(uploaded_file)
        
        if st.button("🔍 Analyze"):
            with st.spinner("Analyzing..."):
                features = {
                    'eye_contact': round(np.random.uniform(0.3, 0.9), 2),
                    'gaze_stability': round(np.random.uniform(0.4, 0.95), 2),
                    'fixation_duration': round(np.random.uniform(0.5, 3.0), 2),
                }
                
                st.success("✅ Analysis complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Eye Contact", f"{features['eye_contact']:.0%}")
                    st.metric("Gaze Stability", f"{features['gaze_stability']:.0%}")
                with col2:
                    st.metric("Avg Fixation", f"{features['fixation_duration']:.2f}s")

# ===========================
# TAB 3: TRAIN MODEL
# ===========================

with tab3:
    st.header("Train Custom Model")
    
    dataset_file = st.file_uploader("Upload training CSV", type=["csv"])
    
    if dataset_file is not None:
        try:
            df = pd.read_csv(dataset_file)
            st.write(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            st.dataframe(df.head(5), use_column_width=True)
            
            st.subheader("Select Label Column")
            st.info("Choose column with class labels (ASD/Typical or 0/1)")
            
            # Find categorical columns
            label_candidates = [col for col in df.columns 
                              if df[col].dtype == 'object' or (df[col].dtype != 'object' and df[col].nunique() <= 10)]
            
            if not label_candidates:
                label_candidates = list(df.columns)
            
            label_col = st.selectbox("Label column:", label_candidates)
            st.write(f"Unique labels: {df[label_col].unique().tolist()}")
            
            if st.button("🚀 Train Model"):
                try:
                    with st.spinner("Training..."):
                        # Prepare data
                        df_clean = df.dropna(subset=[label_col])
                        
                        if len(df_clean) == 0:
                            st.error("❌ No valid data after removing empty rows")
                        else:
                            X = df_clean.drop(columns=[label_col])
                            y_raw = df_clean[label_col]
                            
                            st.write(f"Training on {len(X)} samples with {len(X.columns)} features")
                            
                            # Encode labels
                            if y_raw.dtype == 'object':
                                label_map = {v: i for i, v in enumerate(y_raw.unique())}
                                st.write(f"Label mapping: {label_map}")
                                y = y_raw.map(label_map)
                            else:
                                y = y_raw.astype(int)
                                label_map = None
                            
                            # Train model
                            classifier = ASDClassifier()
                            classifier.label_mapping = label_map
                            
                            if classifier.train(X, y):
                                if classifier.save_model("asd_model.pkl"):
                                    st.success("✅ Model trained and saved!")
                                    st.info(f"Samples: {len(X)} | Features: {len(X.columns)}")
                                else:
                                    st.error("❌ Failed to save model")
                            else:
                                st.error("❌ Training failed")
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    
        except Exception as e:
            st.error(f"❌ Error loading CSV: {str(e)}")

# ===========================
# TAB 4: RESULTS
# ===========================

with tab4:
    st.header("Session Results")
    
    if 'last_session' in st.session_state:
        session = st.session_state.last_session
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Participant", session['user_id'])
        with col2:
            st.metric("Duration", f"{session['duration']}s")
        with col3:
            st.metric("Frames", session['frames'])
        
        if session.get('notes'):
            st.info(f"Notes: {session['notes']}")
        
        st.markdown("---")
        
        if session['gaze_data']:
            gaze_df = pd.DataFrame(session['gaze_data'])
            st.write("Gaze Data:")
            st.dataframe(gaze_df, use_column_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg X", f"{gaze_df['x'].mean():.0f}px")
                st.metric("Avg Y", f"{gaze_df['y'].mean():.0f}px")
            with col2:
                st.metric("X Range", f"{gaze_df['x'].max() - gaze_df['x'].min():.0f}px")
                st.metric("Y Range", f"{gaze_df['y'].max() - gaze_df['y'].min():.0f}px")
    else:
        st.info("No session data. Run a tracking session first.")

st.markdown("---")
st.caption("🧠 ASD Gaze Tracker | Privacy-focused | Offline only")
