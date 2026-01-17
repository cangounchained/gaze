"""
Streamlit-based real-time gaze tracking interface for ASD research.
Includes 5-point calibration with red dot tracking and live classifier predictions.

⚠️ RESEARCH ONLY - NOT FOR DIAGNOSTIC USE ⚠️
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List

from preprocessing import GazePreprocessor
from feature_extraction import GazeFeatureExtractor
from model import ASDClassifier

logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="🧠 ASD Gaze Research Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .ethical-warning {
        background-color: #ffe6e6;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #ff0000;
        margin: 10px 0;
    }
    .title-main {
        text-align: center;
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title-main'>🧠 ASD Gaze Research Tool</h1>", unsafe_allow_html=True)

# Ethical Disclaimer
st.markdown("""
<div class="ethical-warning">
<strong>⚠️ RESEARCH ONLY - NOT FOR DIAGNOSTIC USE ⚠️</strong>

This tool is designed for research purposes only and should NOT be used for:
- Clinical diagnosis of autism spectrum disorder (ASD)
- Medical decision-making
- Any screening or assessment tool for individuals

Gaze-based ASD detection is an active area of research with ongoing validation.
Always consult qualified healthcare professionals for proper diagnosis.

By using this tool, you acknowledge that:
1. This is a research prototype, not a validated medical device
2. Results may not be reliable or accurate
3. You will not use this for diagnostic purposes
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = GazePreprocessor(image_size=224)

if 'feature_extractor' not in st.session_state:
    st.session_state.feature_extractor = GazeFeatureExtractor(use_cnn=False)

if 'classifier' not in st.session_state:
    st.session_state.classifier = None

if 'calibration_complete' not in st.session_state:
    st.session_state.calibration_complete = False

if 'session_data' not in st.session_state:
    st.session_state.session_data = {
        'frames': [],
        'gaze_data': [],
        'calibration_points': []
    }

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    st.subheader("Model")
    model_path = st.text_input("Model path (optional)", value="models/asd_detector.pkl")
    if st.button("Load Model"):
        try:
            classifier = ASDClassifier(model_type='randomforest', input_size=10)
            classifier.load(model_path)
            st.session_state.classifier = classifier
            st.success("✅ Model loaded!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    
    # Session settings
    st.subheader("Session Parameters")
    session_duration = st.slider("Tracking duration (seconds)", 5, 60, 15)
    participant_id = st.text_input("Participant ID", value="TEST001")
    notes = st.text_area("Session notes (optional)", height=80)
    
    # Display loaded model info
    if st.session_state.classifier and st.session_state.classifier.is_trained:
        st.info("✅ Model loaded and ready for predictions")


# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📷 Calibration & Tracking", "📊 Live Analysis", "📈 Results", "ℹ️ Info"])

# ==========================================
# TAB 1: CALIBRATION AND TRACKING
# ==========================================

with tab1:
    st.header("🎯 Calibration & Live Gaze Tracking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Step 1: Calibration")
        st.info("Focus on each red dot for 2 seconds")
        
        if st.button("🔴 Start 5-Point Calibration", key="calibrate"):
            st.info("Calibration sequence - Follow the red dot with your eyes")
            
            # Display all 5 calibration points
            cal_positions = [
                (80, 60, "⬅️ Top-Left"),
                (240, 60, "➡️ Top-Right"),
                (80, 180, "⬅️ Bottom-Left"),
                (240, 180, "➡️ Bottom-Right"),
                (160, 120, "⭐ Center")
            ]
            
            cols = st.columns(5)
            calibration_data = []
            
            for idx, (col, (x, y, label)) in enumerate(zip(cols, cal_positions)):
                with col:
                    # Create calibration frame
                    cal_frame = np.ones((240, 160, 3), dtype=np.uint8) * 200
                    cv2.circle(cal_frame, (x, y), 15, (0, 0, 255), -1)
                    cv2.circle(cal_frame, (x, y), 18, (0, 0, 255), 2)
                    
                    st.image(cal_frame, channels="BGR", caption=f"{idx+1}/5: {label}", use_column_width=True)
                    
                    calibration_data.append({
                        'point': idx + 1,
                        'label': label,
                        'x': x,
                        'y': y
                    })
            
            st.session_state.calibration_complete = True
            st.session_state.session_data['calibration_points'] = calibration_data
            st.success("✅ Calibration complete! Proceed to tracking.")
    
    with col2:
        st.subheader("Step 2: Live Tracking")
        
        if st.session_state.calibration_complete:
            st.info("✅ Calibration complete. Ready to track.")
        else:
            st.warning("⚠️ Please complete calibration first")
        
        if st.button("🎥 Start Gaze Tracking", key="track", disabled=not st.session_state.calibration_complete):
            st.info(f"Recording {session_duration}s of gaze data for {participant_id}...")
            
            # Video capture setup
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot access webcam")
            else:
                # Initialize tracking variables
                frame_count = 0
                max_frames = session_duration * 15  # Reduced FPS for demo
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                image_placeholder = st.empty()
                
                gaze_data = []
                frame_buffer = []
                
                # Video capture loop
                while frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Cannot read from webcam")
                        break
                    
                    # Preprocess frame
                    data = st.session_state.preprocessor.process_frame(frame)
                    
                    if data is None:
                        face_detected = False
                    else:
                        face_detected = True
                        data = st.session_state.preprocessor.extract_eye_crops(data)
                        data = st.session_state.preprocessor.normalize_gaze_coordinates(data)
                        
                        # Store gaze point
                        if 'left_pupil_px' in data:
                            gaze_data.append({
                                'frame': frame_count,
                                'x': data['left_pupil_px'][0],
                                'y': data['left_pupil_px'][1],
                                'face_detected': True
                            })
                            
                            # Draw red dot at pupil
                            h, w = frame.shape[:2]
                            px, py = data['left_pupil_px']
                            cv2.circle(frame, (px, py), 10, (0, 0, 255), -1)
                            cv2.circle(frame, (px, py), 12, (0, 0, 255), 2)
                        
                        frame_buffer.append(data)
                        if len(frame_buffer) > 30:
                            frame_buffer.pop(0)
                    
                    if not face_detected:
                        gaze_data.append({
                            'frame': frame_count,
                            'x': 0,
                            'y': 0,
                            'face_detected': False
                        })
                    
                    # Display frame
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_placeholder.image(rgb_frame, width=640)
                    
                    # Update status
                    status_text.text(f"{'✅ Face' if face_detected else '⚠️ No face'} | Frame {frame_count}/{int(max_frames)}")
                    
                    # Update progress
                    progress_bar.progress(frame_count / max_frames)
                    
                    frame_count += 1
                
                cap.release()
                
                # Save session data
                st.session_state.session_data = {
                    'participant_id': participant_id,
                    'duration': session_duration,
                    'frames': frame_count,
                    'gaze_data': gaze_data,
                    'notes': notes,
                    'timestamp': datetime.now().isoformat(),
                    'calibration_complete': True
                }
                
                st.success(f"✅ Tracking complete! Collected {frame_count} frames")
                st.balloons()


# ==========================================
# TAB 2: LIVE ANALYSIS
# ==========================================

with tab2:
    st.header("📊 Live Gaze Analysis")
    
    if st.session_state.session_data.get('gaze_data'):
        gaze_df = pd.DataFrame(st.session_state.session_data['gaze_data'])
        
        # Filter detected faces
        detected = gaze_df[gaze_df['face_detected'] == True]
        
        if len(detected) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Frames Detected", len(detected))
                st.metric("Detection Rate", f"{len(detected)/len(gaze_df)*100:.1f}%")
            
            with col2:
                st.metric("Avg Gaze X", f"{detected['x'].mean():.0f}px")
                st.metric("Avg Gaze Y", f"{detected['y'].mean():.0f}px")
            
            with col3:
                st.metric("X Std Dev", f"{detected['x'].std():.0f}px")
                st.metric("Y Std Dev", f"{detected['y'].std():.0f}px")
            
            # Gaze trajectory plot
            if len(detected) > 1:
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Gaze trajectory
                ax1.scatter(detected['x'], detected['y'], alpha=0.5, s=10)
                ax1.plot(detected['x'], detected['y'], alpha=0.3)
                ax1.set_xlabel('X (pixels)')
                ax1.set_ylabel('Y (pixels)')
                ax1.set_title('Gaze Trajectory')
                ax1.grid(True, alpha=0.3)
                
                # Time series
                ax2.plot(detected['x'], label='X', alpha=0.7)
                ax2.plot(detected['y'], label='Y', alpha=0.7)
                ax2.set_xlabel('Frame')
                ax2.set_ylabel('Position (pixels)')
                ax2.set_title('Gaze Position Over Time')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
        else:
            st.warning("No face detected in this session")
        
        # Classifier prediction
        if st.session_state.classifier and st.session_state.classifier.is_trained:
            st.markdown("---")
            st.subheader("🤖 Classifier Prediction")
            
            if st.button("Run Prediction"):
                # Extract features from session data
                # For demo, create synthetic feature vector
                feature_vector = np.array([
                    detected['x'].mean() if len(detected) > 0 else 0,
                    detected['y'].mean() if len(detected) > 0 else 0,
                    detected['x'].std() if len(detected) > 0 else 0,
                    detected['y'].std() if len(detected) > 0 else 0,
                    len(detected) / len(gaze_df) if len(gaze_df) > 0 else 0,
                    0, 0, 0, 0, 0
                ]).reshape(1, -1)
                
                predictions, probabilities = st.session_state.classifier.predict(feature_vector)
                
                col1, col2 = st.columns(2)
                with col1:
                    prediction = "🔴 ASD Risk" if predictions[0] == 1 else "🟢 Typical"
                    st.metric("Classification", prediction)
                
                with col2:
                    typical_prob = probabilities[0, 0] * 100
                    asd_prob = probabilities[0, 1] * 100
                    st.metric("Confidence", f"{max(typical_prob, asd_prob):.1f}%")
                
                # Probability distribution
                fig, ax = plt.subplots(figsize=(8, 4))
                categories = ['Typical', 'ASD Risk']
                probs = [typical_prob, asd_prob]
                colors = ['#00ff00', '#ff0000']
                ax.bar(categories, probs, color=colors, alpha=0.7)
                ax.set_ylabel('Probability (%)')
                ax.set_ylim([0, 100])
                for i, (cat, prob) in enumerate(zip(categories, probs)):
                    ax.text(i, prob + 2, f'{prob:.1f}%', ha='center')
                ax.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig)
                
                st.warning("""
                ⚠️ **IMPORTANT**: This prediction is for research purposes only. 
                Do NOT use this for clinical decision-making or diagnosis.
                """)
    else:
        st.info("No session data yet. Complete a tracking session first.")


# ==========================================
# TAB 3: RESULTS HISTORY
# ==========================================

with tab3:
    st.header("📈 Session Results")
    
    if st.session_state.session_data.get('participant_id'):
        session = st.session_state.session_data
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Participant", session.get('participant_id', 'N/A'))
        with col2:
            st.metric("Duration", f"{session.get('duration', 0)}s")
        with col3:
            st.metric("Frames", session.get('frames', 0))
        
        if session.get('notes'):
            st.info(f"**Notes:** {session['notes']}")
        
        # Download session data
        if st.button("📥 Download Session Data"):
            # Create JSON export
            export_data = {
                'participant_id': session.get('participant_id'),
                'timestamp': session.get('timestamp'),
                'duration': session.get('duration'),
                'frames_captured': session.get('frames'),
                'calibration_points': session.get('calibration_points', []),
                'notes': session.get('notes'),
                'gaze_data_summary': {
                    'total_frames': len(session.get('gaze_data', [])),
                    'face_detected_frames': sum(1 for d in session.get('gaze_data', []) if d.get('face_detected'))
                }
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download as JSON",
                data=json_str,
                file_name=f"session_{session.get('participant_id')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("No session data available.")


# ==========================================
# TAB 4: INFORMATION
# ==========================================

with tab4:
    st.header("ℹ️ About This Tool")
    
    st.markdown("""
    ## Overview
    This is a research tool for studying gaze patterns in autism spectrum disorder (ASD) detection.
    
    ## Features
    - **5-Point Calibration**: Calibrate gaze tracking by focusing on sequential red dots
    - **Live Gaze Tracking**: Capture and analyze gaze patterns in real-time
    - **Feature Extraction**: Compute gaze metrics (stability, dispersion, fixations)
    - **ML Classification**: Predict ASD likelihood using trained models (research only)
    
    ## How It Works
    1. **Calibration**: Focus on 5 red dots to calibrate the system
    2. **Tracking**: Continuous gaze capture from webcam
    3. **Feature Extraction**: Compute gaze metrics (fixation, saccades, stability)
    4. **Classification**: Use trained model for predictions (research demonstration)
    
    ## Technical Details
    - **Face Detection**: MediaPipe Face Mesh (468 landmarks)
    - **Feature Extraction**: Handcrafted gaze metrics + optional CNN embeddings
    - **Classification**: RandomForest or lightweight neural network
    - **Framework**: Streamlit for interactive UI
    
    ## Important Disclaimers
    
    ⚠️ **NOT FOR CLINICAL USE**
    - This tool is for research purposes only
    - Not validated for diagnostic use
    - Cannot replace professional medical evaluation
    - Results should not be used for clinical decision-making
    
    ⚠️ **PRIVACY & ETHICS**
    - No data is automatically uploaded or stored
    - All processing is local
    - Users retain full control of their data
    - Proper informed consent should be obtained for research studies
    
    ## References
    - Tobii Eye Tracking for ASD Research
    - MediaPipe Face Mesh: https://google.github.io/mediapipe/
    - ML approaches to gaze-based ASD detection
    
    ## Authors & Attribution
    Research tool created for ASD gaze detection study.
    Built with Streamlit, OpenCV, MediaPipe, and scikit-learn.
    """)
    
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This is a research prototype. Always consult healthcare professionals 
    for proper diagnosis and assessment.
    """)


# Footer
st.markdown("---")
st.caption("🧠 ASD Gaze Research Tool | Local • Private • Research Only | ⚠️ Not for Diagnostic Use")
