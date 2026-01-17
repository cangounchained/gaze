"""
Preprocessing module for gaze tracking data.
Handles face/eye detection, eye crop extraction, and coordinate normalization.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class GazePreprocessor:
    """
    Detects faces and eyes using MediaPipe, extracts eye crops,
    normalizes gaze coordinates, and prepares images for CNN input.
    """
    
    def __init__(self, image_size: int = 224):
        """
        Initialize preprocessor.
        
        Args:
            image_size: Target size for CNN input (e.g., 224 for MobileNetV2)
        """
        self.image_size = image_size
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices for MediaPipe Face Mesh
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.PUPIL_LEFT = 468  # Iris center left
        self.PUPIL_RIGHT = 469  # Iris center right
        
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process a single frame and extract facial landmarks.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Dictionary with face detection results or None if no face detected
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        return {
            'landmarks': landmarks,
            'height': h,
            'width': w,
            'frame': frame.copy()
        }
    
    def extract_eye_crops(self, data: Dict, margin: int = 20) -> Dict:
        """
        Extract left and right eye crop regions.
        
        Args:
            data: Dictionary from process_frame()
            margin: Pixel margin around eye region
            
        Returns:
            Dictionary with eye crops and bounding boxes
        """
        landmarks = data['landmarks']
        h, w = data['height'], data['width']
        frame = data['frame']
        
        # Extract eye regions
        def get_eye_crop(eye_indices):
            pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_indices], dtype=np.int32)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            
            # Add margin
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            crop = frame[y_min:y_max, x_min:x_max].copy()
            
            # Ensure minimum size
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                return None, None
            
            # Resize to standard size
            crop_resized = cv2.resize(crop, (self.image_size, self.image_size))
            
            return crop_resized, (x_min, y_min, x_max, y_max)
        
        left_crop, left_bbox = get_eye_crop(self.LEFT_EYE_INDICES)
        right_crop, right_bbox = get_eye_crop(self.RIGHT_EYE_INDICES)
        
        return {
            'left_eye': left_crop,
            'right_eye': right_crop,
            'left_bbox': left_bbox,
            'right_bbox': right_bbox,
            'landmarks': landmarks,
            'height': h,
            'width': w
        }
    
    def normalize_gaze_coordinates(self, data: Dict) -> Dict:
        """
        Normalize gaze coordinates to [-1, 1] range.
        
        Args:
            data: Dictionary from extract_eye_crops()
            
        Returns:
            Dictionary with normalized gaze coordinates
        """
        landmarks = data['landmarks']
        h, w = data['height'], data['width']
        
        # Get pupil positions
        left_pupil = landmarks[self.PUPIL_LEFT]
        right_pupil = landmarks[self.PUPIL_RIGHT]
        
        # Normalize to [-1, 1]
        left_gaze_x = (left_pupil.x * w - w/2) / (w/2)
        left_gaze_y = (left_pupil.y * h - h/2) / (h/2)
        right_gaze_x = (right_pupil.x * w - w/2) / (w/2)
        right_gaze_y = (right_pupil.y * h - h/2) / (h/2)
        
        # Average gaze
        avg_gaze_x = (left_gaze_x + right_gaze_x) / 2
        avg_gaze_y = (left_gaze_y + right_gaze_y) / 2
        
        data.update({
            'left_gaze': (left_gaze_x, left_gaze_y),
            'right_gaze': (right_gaze_x, right_gaze_y),
            'avg_gaze': (avg_gaze_x, avg_gaze_y),
            'left_pupil_px': (int(left_pupil.x * w), int(left_pupil.y * h)),
            'right_pupil_px': (int(right_pupil.x * w), int(right_pupil.y * h))
        })
        
        return data
    
    def get_face_landmarks_normalized(self, data: Dict) -> np.ndarray:
        """
        Get all face landmarks normalized to [-1, 1].
        
        Args:
            data: Dictionary from process_frame()
            
        Returns:
            Array of normalized landmarks (468 * 3)
        """
        landmarks = data['landmarks']
        h, w = data['height'], data['width']
        
        normalized = []
        for lm in landmarks:
            norm_x = (lm.x * w - w/2) / (w/2)
            norm_y = (lm.y * h - h/2) / (h/2)
            norm_z = lm.z
            normalized.append([norm_x, norm_y, norm_z])
        
        return np.array(normalized)
    
    def preprocess_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Process multiple frames efficiently.
        
        Args:
            frames: List of BGR images
            
        Returns:
            List of processed data dictionaries
        """
        results = []
        for frame in frames:
            data = self.process_frame(frame)
            if data is None:
                continue
            
            data = self.extract_eye_crops(data)
            data = self.normalize_gaze_coordinates(data)
            results.append(data)
        
        return results
    
    def draw_landmarks(self, frame: np.ndarray, data: Dict, draw_eye_regions: bool = True) -> np.ndarray:
        """
        Draw face landmarks and eye regions on frame for visualization.
        
        Args:
            frame: BGR image
            data: Dictionary from process_frame()
            draw_eye_regions: Whether to draw eye bounding boxes
            
        Returns:
            Frame with landmarks drawn
        """
        frame_copy = frame.copy()
        landmarks = data['landmarks']
        h, w = data['height'], data['width']
        
        # Draw face mesh points
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame_copy, (x, y), 2, (0, 255, 0), -1)
        
        # Draw eye regions if available
        if draw_eye_regions and 'left_bbox' in data and data['left_bbox']:
            x1, y1, x2, y2 = data['left_bbox']
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        if draw_eye_regions and 'right_bbox' in data and data['right_bbox']:
            x1, y1, x2, y2 = data['right_bbox']
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw pupils if detected
        if 'left_pupil_px' in data:
            px, py = data['left_pupil_px']
            cv2.circle(frame_copy, (px, py), 5, (0, 0, 255), -1)
        
        if 'right_pupil_px' in data:
            px, py = data['right_pupil_px']
            cv2.circle(frame_copy, (px, py), 5, (0, 0, 255), -1)
        
        return frame_copy
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


def load_image(image_path: str, target_size: int = 224) -> Optional[np.ndarray]:
    """
    Load and preprocess a single image file.
    
    Args:
        image_path: Path to image file
        target_size: Target size for preprocessing
        
    Returns:
        Preprocessed image array or None if loading failed
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None
        
        # Resize to target size
        img = cv2.resize(img, (target_size, target_size))
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None
