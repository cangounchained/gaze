import cv2
import mediapipe as mp
from gaze_simple import gaze
import time
import pandas as pd
import matplotlib.pyplot as plt
from helpers import get_fixation_stats
from playsound import playsound
from fpdf import FPDF
import os
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from risk_flags import get_gaze_flags


# Face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access the webcam.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('autism_gaze_session.avi', fourcc, 20.0, (width, height))

print("Starting real-time autism screening via webcam...\nPress 'q' to quit and generate report.")

start_time = time.time()
fixation_counter = {k: 0 for k in ['eyes_left', 'eyes_right', 'mouth', 'nose', 'none']}
prev_target = "none"
fixation_start_time = time.time()
fixation_durations = []
pupil_diameters = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam error.")
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            target = gaze(frame, landmarks)
            fixation_counter[target] += 1

            # Estimate pupil diameter from iris landmarks
            left_iris = [landmarks.landmark[474], landmarks.landmark[476]]
            dx = (left_iris[0].x - left_iris[1].x) * width
            dy = (left_iris[0].y - left_iris[1].y) * height
            pupil_diameter = round(np.sqrt(dx ** 2 + dy ** 2), 2)
            pupil_diameters.append(pupil_diameter)

            if target != prev_target:
                try:
                    playsound(f"{target}.wav", block=False)
                except Exception as e:
                    print(f"Sound error: {e}")
                fixation_end_time = time.time()
                duration = fixation_end_time - fixation_start_time
                if prev_target != "none":
                    fixation_durations.append({"target": prev_target, "duration": round(duration, 2)})
                fixation_start_time = time.time()
                prev_target = target

            label = f"Fixation: {target}"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Fixation bar
    x, y = 20, height - 40
    bar_length = 300
    total = sum(fixation_counter.values()) or 1
    colors = {
        'eyes_left': (0, 200, 0),
        'eyes_right': (0, 255, 0),
        'mouth': (0, 255, 255),
        'nose': (255, 255, 0),
        'none': (100, 100, 100)
    }
    start = x
    for label in fixation_counter:
        pct = fixation_counter[label] / total
        seg_len = int(pct * bar_length)
        cv2.rectangle(frame, (start, y), (start + seg_len, y + 20), colors[label], -1)
        start += seg_len
    cv2.rectangle(frame, (x, y), (x + bar_length, y + 20), (255, 255, 255), 2)
    cv2.putText(frame, "Live Fixation Distribution", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Timer overlay
    elapsed = int(time.time() - start_time)
    mins, secs = divmod(elapsed, 60)
    cv2.putText(frame, f"Time: {mins:02}:{secs:02}", (width - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

    out.write(frame.copy())
    cv2.imshow("Autism Screening (Live)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Session ended.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save fixation durations
with open("gaze_fixation_durations.csv", "w", newline="") as csvfile:
    fieldnames = ["target", "duration"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in fixation_durations:
        writer.writerow(row)

# Machine Learning-based risk classification (mocked with logistic regression if needed)
total_duration = sum(row['duration'] for row in fixation_durations)
pupil_mean = round(np.mean(pupil_diameters), 2) if pupil_diameters else 0
features = [[fixation_counter['eyes_left'], fixation_counter['mouth'], fixation_counter['none'], total_duration, pupil_mean]]
labels = ['Low Risk', 'Moderate Risk', 'High Risk']
le = LabelEncoder()
le.fit(labels)
classifier = LogisticRegression()
X_train = [[10, 8, 2, 30, 3.5], [4, 3, 6, 15, 2.9], [2, 1, 12, 8, 2.0]]
y_train = le.transform(['Low Risk', 'Moderate Risk', 'High Risk'])
classifier.fit(X_train, y_train)
pred_label = le.inverse_transform(classifier.predict(features))[0]

# Summary & stats
session_time = time.time() - start_time
risk_score, report = get_fixation_stats(fixation_counter, session_time)
report += f"\nML-Predicted Risk Level: {pred_label}"

# Replace emojis with text equivalents for PDF compatibility
report = report.replace("📊", "[Report]")
report = report.replace("📈", "[Score]")
report = report.replace("🧠 Risk Level:", "[Risk Level]")
report = report.replace("🔴", "High Risk")
report = report.replace("🟡", "Moderate Risk")
report = report.replace("🟢", "Low Risk")
report = report.replace("--------", "------------------------------")

lines = report.splitlines()
cleaned_lines = []
seen_risk = False
for line in lines:
    if line.startswith("[Risk Level]"):
        if seen_risk:
            continue
        seen_risk = True
    cleaned_lines.append(line)

# Add session-length warning
if session_time < 15:
    cleaned_lines.append("⚠️ Session duration too short for reliable analysis. Please record at least 20–30 seconds.")

report = "\n".join(cleaned_lines)
# Append gaze behavior flags
flags = get_gaze_flags(fixation_counter, fixation_durations, session_time)
if flags:
    report += "\n\n[Risk Flags]\n" + "\n".join(flags)




print(report)

# PNG report
df = pd.read_csv("gaze_fixations.csv")
counts = df['target'].value_counts()
labels = counts.index.tolist()
values = counts.values

timestamp = int(time.time())
pie_file = f"gaze_report_{timestamp}.png"
bar_file = f"gaze_report_bar_{timestamp}.png"
pdf_file = f"gaze_report_{timestamp}.pdf"

plt.figure(figsize=(6, 6))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Gaze Fixation Distribution")
plt.savefig(pie_file)
plt.close()

plt.figure(figsize=(8, 4))
plt.barh(labels, values, color='skyblue')
plt.xlabel("Fixation Count")
plt.title("Gaze Fixation Summary")
plt.tight_layout()
plt.savefig(bar_file)
plt.close()

# PDF Report
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Autism Gaze Tracking Report", ln=True, align='C')
pdf.set_font("Arial", size=12)
pdf.ln(10)
pdf.multi_cell(0, 8, report)
pdf.image(pie_file, x=10, y=80, w=90)
pdf.image(bar_file, x=110, y=80, w=90)
pdf.output(pdf_file)

print(f"PNG and PDF reports saved as: 
