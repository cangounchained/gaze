"""
Model definitions for ASD detection classifier.
Supports both RandomForest and PyTorch neural networks.
"""

import numpy as np
import joblib
from typing import Tuple, Optional, Dict
import logging
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ASDClassifier:
    """
    Machine learning classifier for ASD detection using gaze features.
    Supports both RandomForest and lightweight neural networks.
    """
    
    def __init__(self, model_type: str = 'randomforest', input_size: int = 10):
        """
        Initialize classifier.
        
        Args:
            model_type: 'randomforest' or 'neural_network'
            input_size: Number of input features
        """
        self.model_type = model_type
        self.input_size = input_size
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
        if model_type == 'randomforest':
            self._init_random_forest()
        elif model_type == 'neural_network':
            if PYTORCH_AVAILABLE:
                self._init_neural_network()
            else:
                logger.warning("PyTorch not available, using RandomForest instead")
                self._init_random_forest()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _init_random_forest(self):
        """Initialize RandomForest classifier."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        logger.info("RandomForest classifier initialized")
    
    def _init_neural_network(self):
        """Initialize lightweight neural network."""
        self.model = SimpleGazeNN(self.input_size)
        logger.info("Neural network classifier initialized")
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0 = Typical, 1 = ASD)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training metrics
        """
        # Check data
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        if self.model_type == 'randomforest':
            # Train RandomForest
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Get training metrics
            y_pred = self.model.predict(X_scaled)
            train_acc = accuracy_score(y, y_pred)
            
            metrics = {
                'train_accuracy': train_acc,
                'train_samples': len(X),
                'feature_importance': dict(zip(range(self.input_size), self.model.feature_importances_))
            }
        
        else:  # neural network
            metrics = self._train_neural_network(X_scaled, y, validation_split)
        
        logger.info(f"Model trained. Metrics: {metrics}")
        return metrics
    
    def _train_neural_network(self, X: np.ndarray, y: np.ndarray, validation_split: float) -> Dict:
        """
        Train neural network with validation.
        
        Args:
            X: Scaled feature matrix
            y: Labels
            validation_split: Validation fraction
            
        Returns:
            Training metrics dictionary
        """
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            # Training step
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Final metrics
        self.model.eval()
        with torch.no_grad():
            train_preds = self.model(X_train_t).argmax(dim=1).numpy()
            val_preds = self.model(X_val_t).argmax(dim=1).numpy()
        
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'final_val_loss': float(best_val_loss),
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels and probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'randomforest':
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
        else:
            X_t = torch.FloatTensor(X_scaled)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_t)
                probs = torch.softmax(outputs, dim=1).numpy()
            predictions = probs.argmax(axis=1)
            probabilities = probs
        
        return predictions, probabilities
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate classifier on test set.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions, probabilities = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'auc': roc_auc_score(y, probabilities[:, 1]) if len(probabilities.shape) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(y, predictions).tolist(),
            'classification_report': classification_report(y, predictions)
        }
        
        logger.info(f"Evaluation metrics:\n{metrics['classification_report']}")
        return metrics
    
    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: File path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'input_size': self.input_size,
            'is_trained': self.is_trained
        }
        
        joblib.dump(data, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: File path to load model from
        """
        data = joblib.load(path)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data['model_type']
        self.input_size = data['input_size']
        self.is_trained = data['is_trained']
        
        logger.info(f"Model loaded from {path}")
    
    def get_feature_importance(self) -> Optional[Dict]:
        """
        Get feature importance (RandomForest only).
        
        Returns:
            Dictionary mapping feature index to importance score
        """
        if self.model_type == 'randomforest' and hasattr(self.model, 'feature_importances_'):
            return dict(zip(range(self.input_size), self.model.feature_importances_))
        return None


class SimpleGazeNN(nn.Module):
    """
    Lightweight neural network for gaze-based ASD detection.
    """
    
    def __init__(self, input_size: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 2)  # Binary classification
        )
    
    def forward(self, x):
        return self.net(x)
