import cv2
import mediapipe as mp
from gaze_simple import gaze
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Load the ASD model
model_path = 'asd_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    print("No model found, using default RandomForest")
    model = RandomForestClassifier(n_estimators=100, random_state=42)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access the webcam.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Starting calibration... Follow the red dot with your eyes.")
print("Press 'q' to quit calibration.")

# Calibration positions for the dot
positions = [
    (width//4, height//4),
    (3*width//4, height//4),
    (width//4, 3*height//4),
    (3*width//4, 3*height//4),
    (width//2, height//2)
]

features_list = []
frame_count = 0
pos_index = 0
pos_start_time = time.time()
pos_duration = 3  # seconds per position

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam error.")
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # Get current dot position
    current_time = time.time()
    if current_time - pos_start_time > pos_duration:
        pos_index = (pos_index + 1) % len(positions)
        pos_start_time = current_time

    dot_x, dot_y = positions[pos_index]

    # Draw the red dot
    cv2.circle(frame, (dot_x, dot_y), 20, (0, 0, 255), -1)

    # Track gaze
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            target = gaze(frame, landmarks)

            # Extract features (simplified version)
            pupil = landmarks.landmark[468]
            left_eye = mid(landmarks.landmark[33], landmarks.landmark[133])
            right_eye = mid(landmarks.landmark[362], landmarks.landmark[263])
            mouth = mid(landmarks.landmark[13], landmarks.landmark[14])
            nose = landmarks.landmark[1]

            features = {
                'x': pupil.x * width,
                'y': pupil.y * height,
                'ear_left': 0.3,  # placeholder
                'ear_right': 0.3,
                'pupil_left_x': left_eye.x * width,
                'pupil_left_y': left_eye.y * height,
                'pupil_right_x': right_eye.x * width,
                'pupil_right_y': right_eye.y * height,
                'mouth_opening': 20.0,
                'dist_eye_left': np.linalg.norm(np.array([pupil.x*width, pupil.y*height]) - np.array([left_eye.x*width, left_eye.y*height])),
                'dist_eye_right': np.linalg.norm(np.array([pupil.x*width, pupil.y*height]) - np.array([right_eye.x*width, right_eye.y*height])),
                'dist_mouth': np.linalg.norm(np.array([pupil.x*width, pupil.y*height]) - np.array([mouth.x*width, mouth.y*height])),
                'dist_nose': np.linalg.norm(np.array([pupil.x*width, pupil.y*height]) - np.array([nose.x*width, nose.y*height]))
            }
            features_list.append(features)

    cv2.imshow('Calibration - Follow the red dot', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Aggregate features
if features_list:
    df = pd.DataFrame(features_list)
    avg_features = df.mean().to_dict()

    # Predict ASD
    pred = model.predict([list(avg_features.values())])[0]
    proba = model.predict_proba([list(avg_features.values())])[0]

    result = "ASD" if pred == 1 else "Typical"
    confidence = max(proba)

    print(f"Calibration complete!")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}")

    # Save result
    with open('calibration_result.txt', 'w') as f:
        f.write(f"Prediction: {result}\nConfidence: {confidence:.2f}\n")
else:
    print("No features collected during calibration.")