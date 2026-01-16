import streamlit as st
import subprocess
import os
import pandas as pd
import cv2

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
        Welcome to the **ASD Gaze Tracker**, a user-friendly tool designed to support early screening for Autism Spectrum Disorder by tracking gaze patterns through a simple webcam.

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

# Start session input
st.markdown("---")
st.header("Start a New Session")

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

if st.button("Start Session"):
    if not user_id.strip():
        st.error("⚠️ Student/Patient ID is required before starting.")
    else:
        with st.spinner(f"Starting gaze tracking session for **{user_id}**... Please ensure your webcam is on."):
            # Run gaze tracking script here with user_id argument
            # Example: subprocess.run(["python", "main.py", user_id])
            subprocess.run(["python", "main.py"])
        st.success("Session complete! Check results below.")

# Session results in tabs
st.markdown("---")
st.header("Session Results")

csv_path = "gaze_fixations.csv"
pdf_path = "gaze_report.pdf"

tabs = st.tabs(["📊 Data", "📄 Report"])

with tabs[0]:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.subheader("Gaze Fixation Data")
        st.dataframe(df, height=350)
    else:
        st.info("No gaze fixation data available yet. Run a session to generate results.")

with tabs[1]:
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            st.subheader("Download Gaze Report")
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_file,
                file_name=f"gaze_report_{user_id or 'latest'}.pdf",
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
