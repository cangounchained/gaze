import cv2
import numpy as np
import time
from helpers import relative, relativeT, log_fixation_data

# Regions of interest (ROIs) on the face
ROIS = {
    'eyes_left': [33, 133],
    'eyes_right': [362, 263],
    'mouth': [78, 308],
    'nose': [1]
}

def mid(p1, p2):
    """Return midpoint between two points."""
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def within(p, center, r=30):
    """Check if point `p` is within radius `r` of `center`."""
    return np.linalg.norm(np.array(p) - np.array(center)) <= r

def gaze(frame, landmarks):
    """Compute gaze direction and draw red line to indicate it."""
    
    # Convert facial landmarks to image coordinates
    img_pts = np.array([
        relative(landmarks.landmark[4], frame.shape),     # Nose
        relative(landmarks.landmark[152], frame.shape),   # Chin
        relative(landmarks.landmark[263], frame.shape),   # Right eye corner
        relative(landmarks.landmark[33], frame.shape),    # Left eye corner
        relative(landmarks.landmark[287], frame.shape),   # Right cheek
        relative(landmarks.landmark[57], frame.shape)     # Left cheek
    ], dtype="double")

    img_pts3d = np.array([
        relativeT(landmarks.landmark[4], frame.shape),
        relativeT(landmarks.landmark[152], frame.shape),
        relativeT(landmarks.landmark[263], frame.shape),
        relativeT(landmarks.landmark[33], frame.shape),
        relativeT(landmarks.landmark[287], frame.shape),
        relativeT(landmarks.landmark[57], frame.shape)
    ], dtype="double")

    model_pts = np.array([
        (0, 0, 0), 
        (0, -63, -12),
        (-43, 32, -26),
        (43, 32, -26),
        (-29, -28, -24),
        (29, -28, -24)
    ])

    eye_center = np.array([[29.05], [32.7], [-39.5]])

    h, w = frame.shape[:2]
    f_len = w
    cam_mtx = np.array([
        [f_len, 0, w / 2],
        [0, f_len, h / 2],
        [0, 0, 1]
    ], dtype="double")

    dist = np.zeros((4, 1))
    success, rvec, tvec = cv2.solvePnP(model_pts, img_pts, cam_mtx, dist)

    left_pupil = relative(landmarks.landmark[468], frame.shape)

    _, trans, _ = cv2.estimateAffine3D(img_pts3d, model_pts)

    if trans is not None:
        pupil_3d = trans @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
        gaze_vec = eye_center + (pupil_3d - eye_center) * 10

        gaze_2d, _ = cv2.projectPoints(gaze_vec.T[0], rvec, tvec, cam_mtx, dist)
        head_fix, _ = cv2.projectPoints((int(pupil_3d[0]), int(pupil_3d[1]), 40), rvec, tvec, cam_mtx, dist)

        final_gaze = left_pupil + (gaze_2d[0][0] - left_pupil) - (head_fix[0][0] - left_pupil)

        # 🔴 Draw red dot at pupil center and red line to gaze direction
        cv2.circle(frame, tuple(map(int, left_pupil)), 10, (0, 0, 255), -1)  # Red filled dot at pupil
        cv2.circle(frame, tuple(map(int, left_pupil)), 12, (0, 0, 255), 2)   # Red outline for emphasis
        cv2.line(frame, tuple(map(int, left_pupil)), tuple(map(int, final_gaze)), (0, 0, 255), 2)

        # Determine which facial region the child is gazing at
        target = "none"
        for label, ids in ROIS.items():
            pt1 = relative(landmarks.landmark[ids[0]], frame.shape)
            pt2 = relative(landmarks.landmark[ids[1]], frame.shape) if len(ids) > 1 else pt1
            center = mid(pt1, pt2)

            inside = within(final_gaze, center)
            color = (0, 255, 0) if inside else (180, 180, 180)
            cv2.circle(frame, center, 5, color, -1)

            if inside:
                target = label

        # Log data
        ts = time.time()
        log_fixation_data(ts, target, final_gaze)

        return target

    return "none"

