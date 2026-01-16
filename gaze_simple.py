import cv2
import numpy as np
import time
from helpers import relative, log_fixation_data

# Landmarks used for ROIs
ROIS = {
    'eyes_left': [33, 133],
    'eyes_right': [362, 263],
    'mouth': [13, 14],  # upper and lower lip approx.
    'nose': [1]
}

def mid(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def gaze(frame, landmarks):
    print("👁️ [Simple Gaze] Entered gaze()")

    try:
        # Iris landmark
        pupil = relative(landmarks.landmark[468], frame.shape)
        cv2.circle(frame, pupil, 5, (255, 0, 0), -1)  # blue dot

        # Eye regions
        left_eye_center = mid(
            relative(landmarks.landmark[33], frame.shape),
            relative(landmarks.landmark[133], frame.shape)
        )
        right_eye_center = mid(
            relative(landmarks.landmark[362], frame.shape),
            relative(landmarks.landmark[263], frame.shape)
        )

        cv2.circle(frame, left_eye_center, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_eye_center, 5, (0, 255, 0), -1)

        # Mouth region center
        mouth_center = mid(
            relative(landmarks.landmark[13], frame.shape),
            relative(landmarks.landmark[14], frame.shape)
        )
        nose_tip = relative(landmarks.landmark[1], frame.shape)

        cv2.circle(frame, mouth_center, 5, (0, 255, 255), -1)
        cv2.circle(frame, nose_tip, 5, (0, 100, 255), -1)

        # Distance from pupil to eye centers
        dist_left = np.linalg.norm(np.array(pupil) - np.array(left_eye_center))
        dist_right = np.linalg.norm(np.array(pupil) - np.array(right_eye_center))
        eye_thresh = 80  # px

        target = "none"

        if dist_left < eye_thresh:
            target = "eyes_left"
        elif dist_right < eye_thresh:
            target = "eyes_right"
        elif pupil[1] > nose_tip[1] + 30:  # if pupil is significantly below nose
            target = "mouth"

        print(f"🎯 Simple Gaze Detected: {target}")
        ts = time.time()
        log_fixation_data(ts, target, pupil)
        return target

    except Exception as e:
        print(f"⚠️ Exception in simple gaze(): {e}")
        return "none"
