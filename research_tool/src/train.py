"""
Training pipeline for ASD detection model.
Integrates preprocessing, feature extraction, and model training.
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
import json
from datetime import datetime

from src.preprocessing import GazePreprocessor
from src.feature_extraction import GazeFeatureExtractor
from src.model import ASDClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GazeTrainingPipeline:
    """
    Complete training pipeline: preprocessing → feature extraction → model training.
    """
    
    def __init__(self, model_type: str = 'randomforest', use_cnn: bool = True):
        """
        Initialize pipeline.
        
        Args:
            model_type: 'randomforest' or 'neural_network'
            use_cnn: Whether to use CNN embeddings
        """
        self.preprocessor = GazePreprocessor(image_size=224)
        self.feature_extractor = GazeFeatureExtractor(use_cnn=use_cnn)
        self.classifier = None
        self.model_type = model_type
        self.training_metrics = {}
    
    def process_dataset(self, dataset_path: str, label_column: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a dataset and extract features.
        
        Args:
            dataset_path: Path to dataset (directory with images or CSV with image paths)
            label_column: Name of label column in CSV
            
        Returns:
            Tuple of (feature_matrix, labels)
        """
        dataset_path = Path(dataset_path)
        
        features_list = []
        labels_list = []
        
        # Check if CSV file
        if dataset_path.suffix == '.csv':
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded CSV with {len(df)} samples")
            
            for idx, row in df.iterrows():
                image_path = row.get('image_path')
                label = row.get(label_column)
                
                if image_path is None or label is None:
                    continue
                
                features = self._process_single_image(image_path)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)
                
                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx + 1} samples")
        
        else:  # Directory of images
            image_files = list(dataset_path.glob('**/*.jpg')) + list(dataset_path.glob('**/*.png'))
            logger.info(f"Found {len(image_files)} images")
            
            for idx, image_path in enumerate(image_files):
                # Infer label from directory name
                label = image_path.parent.name  # e.g., 'asd' or 'typical'
                
                features = self._process_single_image(str(image_path))
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)
                
                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx + 1} samples")
        
        if not features_list:
            raise ValueError("No valid samples processed from dataset")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        logger.info(f"Dataset processed: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _process_single_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Process single image and extract features.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Feature vector or None if processing failed
        """
        try:
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                return None
            
            # Preprocess
            data = self.preprocessor.process_frame(frame)
            if data is None:
                return None
            
            data = self.preprocessor.extract_eye_crops(data)
            data = self.preprocessor.normalize_gaze_coordinates(data)
            
            # Extract features
            features = self.feature_extractor.extract_all_features(data)
            feature_vector = self.feature_extractor.create_feature_vector(features)
            
            return feature_vector
        
        except Exception as e:
            logger.debug(f"Error processing {image_path}: {e}")
            return None
    
    def process_video_dataset(self, video_path: str, fps: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process video file and extract features from frames.
        
        Args:
            video_path: Path to video file
            fps: Frames per second to extract
            
        Returns:
            Tuple of (feature_matrix, frame_indices)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(video_fps / fps))
        
        features_list = []
        frame_indices = []
        frame_count = 0
        
        logger.info(f"Processing video: {total_frames} frames, {video_fps:.1f} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                features = self._process_single_frame_with_buffer(frame)
                if features is not None:
                    features_list.append(features)
                    frame_indices.append(frame_count)
            
            frame_count += 1
            
            if frame_count % 500 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        cap.release()
        
        X = np.array(features_list)
        indices = np.array(frame_indices)
        
        logger.info(f"Video processed: {X.shape[0]} features extracted from {frame_count} frames")
        return X, indices
    
    def _process_single_frame_with_buffer(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process frame with feature buffer for temporal features.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Feature vector or None
        """
        try:
            data = self.preprocessor.process_frame(frame)
            if data is None:
                return None
            
            data = self.preprocessor.extract_eye_crops(data)
            data = self.preprocessor.normalize_gaze_coordinates(data)
            
            features = self.feature_extractor.extract_all_features(data)
            feature_vector = self.feature_extractor.create_feature_vector(features)
            
            return feature_vector
        except Exception as e:
            logger.debug(f"Error processing frame: {e}")
            return None
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
              encode_labels: bool = True) -> Dict:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix
            y: Labels (strings like 'asd', 'typical' or 0, 1)
            test_size: Fraction of data for test set
            encode_labels: Whether to encode string labels to integers
            
        Returns:
            Dictionary with training results and metrics
        """
        # Encode labels if needed
        if encode_labels and y.dtype == 'object':
            unique_labels = np.unique(y)
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            y_encoded = np.array([label_map[label] for label in y])
            logger.info(f"Label mapping: {label_map}")
        else:
            y_encoded = y.astype(int)
            label_map = None
        
        # Split data
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        split_idx = int(n_samples * (1 - test_size))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train classifier
        self.classifier = ASDClassifier(
            model_type=self.model_type,
            input_size=X.shape[1]
        )
        
        train_metrics = self.classifier.fit(X_train, y_train)
        test_metrics = self.classifier.evaluate(X_test, y_test)
        
        # Combine metrics
        self.training_metrics = {
            'timestamp': datetime.now().isoformat(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'label_map': label_map,
            'input_features': X.shape[1],
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Training complete!")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
        
        return self.training_metrics
    
    def save_model(self, model_path: str):
        """
        Save trained model and training metadata.
        
        Args:
            model_path: Path to save model
        """
        if self.classifier is None:
            raise ValueError("No trained model to save")
        
        # Save model
        self.classifier.save(model_path)
        
        # Save training metadata
        metadata_path = Path(model_path).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2, default=str)
        
        logger.info(f"Training metadata saved to {metadata_path}")
    
    def load_model(self, model_path: str):
        """
        Load trained model.
        
        Args:
            model_path: Path to load model from
        """
        input_size = self._get_input_size_from_dataset()
        self.classifier = ASDClassifier(
            model_type=self.model_type,
            input_size=input_size
        )
        self.classifier.load(model_path)
        
        # Load metadata
        metadata_path = Path(model_path).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.training_metrics = json.load(f)
    
    def _get_input_size_from_dataset(self) -> int:
        """Get expected input size from training metadata."""
        if 'input_features' in self.training_metrics:
            return self.training_metrics['input_features']
        return 10  # Default
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.classifier is None:
            raise ValueError("No trained model. Call train() or load_model() first.")
        
        return self.classifier.predict(X)
    
    def print_summary(self):
        """Print training summary."""
        if not self.training_metrics:
            print("No training metrics available")
            return
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Timestamp: {self.training_metrics.get('timestamp')}")
        print(f"Train samples: {self.training_metrics.get('train_samples')}")
        print(f"Test samples: {self.training_metrics.get('test_samples')}")
        print(f"Input features: {self.training_metrics.get('input_features')}")
        print(f"\nTest Metrics:")
        for key, val in self.training_metrics.get('test_metrics', {}).items():
            if key != 'classification_report':
                print(f"  {key}: {val}")
        print("="*60)


if __name__ == '__main__':
    # Example usage
    pipeline = GazeTrainingPipeline(model_type='randomforest', use_cnn=False)
    
    # Process dataset
    X, y = pipeline.process_dataset('data/asd_vs_td')
    
    # Train
    metrics = pipeline.train(X, y)
    
    # Save
    pipeline.save_model('models/asd_detector.pkl')
    
    # Print summary
    pipeline.print_summary()
