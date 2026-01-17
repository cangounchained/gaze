"""
ASD Gaze Detection Research Tool
Modules for preprocessing, feature extraction, model training, and webcam interface.
"""

from .preprocessing import GazePreprocessor
from .feature_extraction import GazeFeatureExtractor
from .model import ASDClassifier
from .train import GazeTrainingPipeline

__version__ = "0.1.0"
__all__ = [
    'GazePreprocessor',
    'GazeFeatureExtractor',
    'ASDClassifier',
    'GazeTrainingPipeline'
]
