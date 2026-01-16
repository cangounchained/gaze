import streamlit as st
import subprocess
import os
import pandas as pd
import tempfile

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

try:
    from feature_extraction import extract_features_from_image, extract_features_from_video, aggregate_features
    FEATURE_EXTRACTION_AVAILABLE = True
    print("✅ Feature extraction available")
except ImportError as e:
    FEATURE_EXTRACTION_AVAILABLE = False
    print(f"❌ Feature extraction not available: {e}")

try:
    from ml_model import ASDClassifier
    ML_MODEL_AVAILABLE = True
    print("✅ ML model available")
except ImportError as e:
    ML_MODEL_AVAILABLE = False
    print(f"❌ ML model not available: {e}")

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
    st.info("Current environment: Container without graphics support")

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
    st.header("📷 Preview Webcam (Optional)")

    if not CV2_AVAILABLE:
        st.error("❌ **Webcam functionality not available** - OpenCV is required")
        st.info("Install OpenCV with: `pip install opencv-python`")
    else:
        if 'preview_active' not in st.session_state:
            st.session_state.preview_active = False

        def start_preview():
            st.session_state.preview_active = True

        def stop_preview():
            st.session_state.preview_active = False

        preview_col1, preview_col2 = st.columns([1, 3])

        with preview_col1:
            if not st.session_state.preview_active:
                if st.button("▶️ Start Webcam Preview"):
                    start_preview()
            else:
                if st.button("⏹️ Stop Webcam Preview"):
                    stop_preview()

        with preview_col2:
            if st.session_state.preview_active:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        st.image(frame, channels="BGR", caption="Live Webcam Feed", output_format="JPEG", clamp=True, use_column_width=True, caption_visibility="visible")
                    else:
                        st.error("⚠️ Unable to read from webcam.")
                    cap.release()
                else:
                    st.error("⚠️ Webcam not detected or unavailable.")
    
    # Gaze Tracking button
    st.markdown("---")
    st.header("▶️ Run Gaze Tracking")

    if not CV2_AVAILABLE:
        st.error("❌ **Live gaze tracking not available** - Requires OpenCV")
        st.info("Install dependencies: `pip install opencv-python mediapipe`")
    else:
        if st.button("Start Live Session"):
            if not user_id.strip():
                st.error("⚠️ Student/Patient ID is required before starting.")
            else:
                with st.spinner(f"Starting gaze tracking session for **{user_id}**... Please ensure your webcam is on."):
                    subprocess.run(["python", "main.py"])
                st.success("Session complete! Check results in the Results tab.")

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

            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    # Assuming dataset has features and 'label' column
                    features = df.drop('label', axis=1)
                    labels = df['label'].map({'ASD': 1, 'Typical': 0})

                    classifier = ASDClassifier('rf')
                    classifier.train(features, labels)
                    classifier.save_model("asd_model.pkl")

                    st.success("Model trained and saved!")

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
