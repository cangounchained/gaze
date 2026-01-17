"""
Feature extraction module for gaze analysis.
Computes handcrafted gaze metrics and CNN embeddings for ASD detection.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional
from collections import deque
import logging

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v2
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class GazeFeatureExtractor:
    """
    Extracts handcrafted gaze metrics and CNN embeddings from eye crops.
    """
    
    def __init__(self, use_cnn: bool = True, device: str = 'cpu'):
        """
        Initialize feature extractor.
        
        Args:
            use_cnn: Whether to use CNN embeddings (requires PyTorch)
            device: 'cpu' or 'cuda'
        """
        self.use_cnn = use_cnn and PYTORCH_AVAILABLE
        self.device = device
        
        if self.use_cnn:
            self._init_cnn_model()
        
        # Window for computing temporal features
        self.gaze_history = deque(maxlen=30)  # ~1 second at 30 FPS
        self.blink_history = deque(maxlen=30)
        
        # Transforms for CNN input
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _init_cnn_model(self):
        """Initialize MobileNetV2 for feature extraction."""
        try:
            self.cnn_model = mobilenet_v2(pretrained=True)
            # Remove classification head to get embeddings
            self.cnn_model = torch.nn.Sequential(*list(self.cnn_model.children())[:-1])
            self.cnn_model.to(self.device)
            self.cnn_model.eval()
            logger.info("MobileNetV2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CNN model: {e}")
            self.use_cnn = False
    
    def extract_handcrafted_features(self, data: Dict, frame_buffer: Optional[List[Dict]] = None) -> Dict:
        """
        Extract handcrafted gaze metrics.
        
        Args:
            data: Dictionary from preprocessing module
            frame_buffer: Optional buffer of previous frames for temporal features
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Static gaze features
        if 'avg_gaze' in data:
            gaze_x, gaze_y = data['avg_gaze']
            features['gaze_x'] = float(gaze_x)
            features['gaze_y'] = float(gaze_y)
            features['gaze_magnitude'] = float(np.sqrt(gaze_x**2 + gaze_y**2))
        
        # Left/right eye asymmetry
        if 'left_gaze' in data and 'right_gaze' in data:
            lx, ly = data['left_gaze']
            rx, ry = data['right_gaze']
            features['left_right_asymmetry_x'] = float(abs(lx - rx))
            features['left_right_asymmetry_y'] = float(abs(ly - ry))
        
        # Temporal features (require frame buffer)
        if frame_buffer and len(frame_buffer) >= 2:
            features.update(self._compute_temporal_features(frame_buffer))
        else:
            # Default values
            features['fixation_stability'] = 0.0
            features['gaze_dispersion'] = 0.0
            features['saccade_velocity'] = 0.0
            features['saccade_frequency'] = 0.0
            features['blink_rate'] = 0.0
        
        return features
    
    def _compute_temporal_features(self, frame_buffer: List[Dict]) -> Dict:
        """
        Compute temporal features from a buffer of frames.
        
        Args:
            frame_buffer: List of frame data dictionaries
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        gaze_coords = []
        for frame in frame_buffer:
            if 'avg_gaze' in frame:
                x, y = frame['avg_gaze']
                gaze_coords.append([x, y])
        
        if len(gaze_coords) < 2:
            return {
                'fixation_stability': 0.0,
                'gaze_dispersion': 0.0,
                'saccade_velocity': 0.0,
                'saccade_frequency': 0.0,
                'blink_rate': 0.0
            }
        
        gaze_coords = np.array(gaze_coords)
        
        # Fixation Stability (lower = more stable)
        # Standard deviation of gaze positions
        fixation_stability = float(np.std(gaze_coords, axis=0).mean())
        features['fixation_stability'] = fixation_stability
        
        # Gaze Dispersion (range of gaze movement)
        gaze_range = gaze_coords.max(axis=0) - gaze_coords.min(axis=0)
        features['gaze_dispersion'] = float(np.linalg.norm(gaze_range))
        
        # Saccade Analysis
        gaze_velocity = np.diff(gaze_coords, axis=0)
        velocity_magnitude = np.linalg.norm(gaze_velocity, axis=1)
        
        features['saccade_velocity'] = float(velocity_magnitude.mean())
        
        # Saccade frequency (number of rapid movements)
        saccade_threshold = velocity_magnitude.mean() + 1.5 * velocity_magnitude.std()
        num_saccades = np.sum(velocity_magnitude > saccade_threshold)
        features['saccade_frequency'] = float(num_saccades / len(gaze_coords))
        
        # Blink Rate (simplified - assumes blink detection elsewhere)
        features['blink_rate'] = 0.0  # To be computed with eye closure detection
        
        return features
    
    def extract_cnn_features(self, eye_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract CNN embeddings from eye crop.
        
        Args:
            eye_crop: BGR image of eye region (should be resized to 224x224)
            
        Returns:
            Embedding vector or None if CNN not available
        """
        if not self.use_cnn or eye_crop is None:
            return None
        
        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            img_tensor = self.transforms(rgb).unsqueeze(0).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                embeddings = self.cnn_model(img_tensor)
                embeddings = torch.nn.functional.adaptive_avg_pool2d(embeddings, (1, 1))
                embeddings = embeddings.squeeze()
            
            return embeddings.cpu().numpy()
        
        except Exception as e:
            logger.error(f"Error extracting CNN features: {e}")
            return None
    
    def extract_all_features(self, data: Dict, frame_buffer: Optional[List[Dict]] = None) -> Dict:
        """
        Extract all features (handcrafted + CNN).
        
        Args:
            data: Dictionary from preprocessing module
            frame_buffer: Optional buffer for temporal features
            
        Returns:
            Dictionary containing all features
        """
        all_features = {}
        
        # Handcrafted features
        handcrafted = self.extract_handcrafted_features(data, frame_buffer)
        all_features.update(handcrafted)
        
        # CNN embeddings
        if self.use_cnn and data.get('left_eye') is not None:
            left_emb = self.extract_cnn_features(data['left_eye'])
            if left_emb is not None:
                all_features['left_eye_embedding'] = left_emb
        
        if self.use_cnn and data.get('right_eye') is not None:
            right_emb = self.extract_cnn_features(data['right_eye'])
            if right_emb is not None:
                all_features['right_eye_embedding'] = right_emb
        
        return all_features
    
    def create_feature_vector(self, features: Dict) -> np.ndarray:
        """
        Create a flat feature vector from extracted features.
        Concatenates all numeric features and embeddings.
        
        Args:
            features: Dictionary from extract_all_features()
            
        Returns:
            Flat numpy array suitable for classifier input
        """
        feature_list = []
        
        # Add handcrafted features in order
        handcrafted_names = [
            'gaze_x', 'gaze_y', 'gaze_magnitude',
            'left_right_asymmetry_x', 'left_right_asymmetry_y',
            'fixation_stability', 'gaze_dispersion',
            'saccade_velocity', 'saccade_frequency', 'blink_rate'
        ]
        
        for name in handcrafted_names:
            if name in features:
                feature_list.append(features[name])
            else:
                feature_list.append(0.0)
        
        # Add CNN embeddings if available
        if 'left_eye_embedding' in features:
            feature_list.extend(features['left_eye_embedding'].flatten())
        if 'right_eye_embedding' in features:
            feature_list.extend(features['right_eye_embedding'].flatten())
        
        return np.array(feature_list, dtype=np.float32)
    
    def get_feature_names(self, include_embeddings: bool = False) -> List[str]:
        """
        Get list of feature names for documentation.
        
        Args:
            include_embeddings: Whether to include embedding feature names
            
        Returns:
            List of feature name strings
        """
        names = [
            'gaze_x', 'gaze_y', 'gaze_magnitude',
            'left_right_asymmetry_x', 'left_right_asymmetry_y',
            'fixation_stability', 'gaze_dispersion',
            'saccade_velocity', 'saccade_frequency', 'blink_rate'
        ]
        
        if include_embeddings:
            names.extend([f'left_eye_emb_{i}' for i in range(1280)])
            names.extend([f'right_eye_emb_{i}' for i in range(1280)])
        
        return names
