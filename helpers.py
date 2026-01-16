import csv
import os
from collections import Counter

FILE = "gaze_fixations.csv"

# Create CSV file with headers if it doesn't exist
if not os.path.exists(FILE):
    with open(FILE, 'w', newline='') as f:
        csv.writer(f).writerow(['timestamp', 'target', 'x', 'y'])

def relative(point, shape):
    """
    Convert normalized landmark coordinates to pixel coordinates.
    """
    h, w = shape[:2]
    return (int(point.x * w), int(point.y * h))

def relativeT(point, shape):
    """
    Convert normalized landmark coordinates to float (3D coords).
    """
    h, w = shape[:2]
    return (point.x * w, point.y * h, point.z * w)

def log_fixation_data(ts, target, pt):
    """
    Log timestamp, gaze target region, and fixation coordinates to CSV.
    """
    with open(FILE, 'a', newline='') as f:
        csv.writer(f).writerow([ts, target, int(pt[0]), int(pt[1])])

def get_fixation_stats(counter, duration_sec):
    """
    Given fixation counts and total duration,
    calculate percentage times spent on regions and autism risk score.
    """
    # Prevent division by zero
    total = sum(counter.values()) or 1

    # Calculate % time on each region
    perc = {k: round(v / total * 100, 2) for k, v in counter.items()}

    # Simple risk scoring logic based on typical ASD indicators:
    score = 0
    if perc.get('eyes_left', 0) + perc.get('eyes_right', 0) < 30:
        score += 3  # low eye contact
    if perc.get('none', 0) > 40:
        score += 3  # unfocused or random gaze
    if perc.get('mouth', 0) < 10:
        score += 2  # low interest in social cues
    if duration_sec < 60:
        score += 2  # short session (less reliable)

    # Risk level interpretation
    if score <= 3:
        level = "🟢 Low Risk"
    elif score <= 6:
        level = "🟡 Moderate Risk"
    else:
        level = "🔴 High Risk - Consider professional evaluation"

    # Create a report string to display to the user
    report = f"""
-------- Autism Gaze Screening Report --------
Eyes Fixation: {perc.get('eyes_left', 0) + perc.get('eyes_right', 0)}%
Mouth Fixation: {perc.get('mouth', 0)}%
Nose Fixation: {perc.get('nose', 0)}%
Unfocused Gaze: {perc.get('none', 0)}%
Total Time: {round(duration_sec, 1)} sec

📈 Autism Risk Score: {score}/10
🧠 Risk Level: {level}
----------------------------------------------
"""
    return score, report


