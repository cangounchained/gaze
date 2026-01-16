👀 Child Gaze Tracker – Real-Time Autism Risk Screening (6 Months – 4 Years)


A privacy-preserving, webcam-based tool that analyzes gaze behavior in infants and young children (ages 6 months to 4 years) to identify early signs of Autism Spectrum Disorder (ASD).

Using MediaPipe and OpenCV, this Python app provides non-invasive, real-time tracking of visual attention toward key social cues like the eyes, mouth, and nose, and flags behavioral patterns backed by clinical research.

🎯 Key Features


✅ Webcam-based gaze tracking (no special equipment needed)

✅ Real-time visual feedback (live gaze vector overlay)

✅ Detects behavioral markers related to ASD

✅ Fully offline, low-cost, and private

✅ Generates PDF + PNG reports

✅ Designed for children as young as 6 months

👶 Intended Age Range

6 Months to 4 Years

The tool is optimized for early developmental stages when social attention and gaze behaviors begin to emerge. It’s ideal for:

Infants (6–18 months) showing early signs of gaze avoidance or delayed responsiveness

Toddlers (1.5–4 years) during routine developmental screening

Low-resource settings where full clinical assessments aren't always accessible

👶 How Infants (6 Months+) Can Use It

This tool is uniquely designed to work even with preverbal infants:

👩‍🍼 The baby can sit on a caregiver’s lap or in a baby seat, 30–50 cm from the laptop.

👁️ The system automatically detects gaze direction as soon as the child looks at the screen.

🧠 It tracks which part of the face the baby focuses on (eyes, mouth, nose), or whether they look away.

⏱️ Sessions are short (~1 minute), making them suitable even for babies with limited attention spans.

Unlike other systems, there’s no headset, no calibration, and no interaction required.

🧠 What It Detects

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




