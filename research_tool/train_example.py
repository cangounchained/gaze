"""
Example training script for ASD Gaze Detection model.
Demonstrates the complete pipeline from data loading to model evaluation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.train import GazeTrainingPipeline
from src.preprocessing import GazePreprocessor
from src.feature_extraction import GazeFeatureExtractor


def create_synthetic_dataset():
    """
    Create synthetic dataset for demonstration.
    In practice, use real ASD and control datasets.
    """
    print("Creating synthetic dataset for demonstration...")
    
    data_dir = Path('data/synthetic')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directories
    (data_dir / 'typical').mkdir(exist_ok=True)
    (data_dir / 'asd').mkdir(exist_ok=True)
    
    import cv2
    
    # Create synthetic images
    np.random.seed(42)
    
    for label, count in [('typical', 50), ('asd', 50)]:
        for i in range(count):
            # Create random image (480x640x3)
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some face-like features
            cv2.circle(img, (320, 240), 50, (100, 100, 100), -1)  # Head
            cv2.circle(img, (300, 230), 8, (50, 50, 50), -1)  # Left eye
            cv2.circle(img, (340, 230), 8, (50, 50, 50), -1)  # Right eye
            cv2.circle(img, (320, 260), 5, (100, 100, 100), -1)  # Nose
            
            # Save image
            img_path = data_dir / label / f'sample_{i:03d}.jpg'
            cv2.imwrite(str(img_path), img)
    
    print(f"✅ Synthetic dataset created at {data_dir}")
    return str(data_dir)


def main():
    """Main training pipeline."""
    
    print("=" * 60)
    print("ASD GAZE DETECTION - TRAINING PIPELINE")
    print("=" * 60)
    print()
    
    # Configuration
    MODEL_TYPE = 'randomforest'  # or 'neural_network'
    USE_CNN = False  # Set to True if you have enough memory
    TEST_SIZE = 0.2
    
    # Paths
    DATA_DIR = 'data/asd_vs_td'  # Change to your dataset
    MODEL_PATH = 'models/asd_detector.pkl'
    
    # Check if data exists
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        print(f"⚠️ Data directory not found: {DATA_DIR}")
        print("Creating synthetic dataset for demonstration...")
        DATA_DIR = create_synthetic_dataset()
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = GazeTrainingPipeline(
        model_type=MODEL_TYPE,
        use_cnn=USE_CNN
    )
    print(f"   ✅ Model type: {MODEL_TYPE}")
    print(f"   ✅ CNN embeddings: {USE_CNN}")
    
    # Process dataset
    print("\n2. Processing dataset...")
    try:
        X, y = pipeline.process_dataset(DATA_DIR)
        print(f"   ✅ Loaded {X.shape[0]} samples")
        print(f"   ✅ Features: {X.shape[1]}")
        print(f"   ✅ Labels: {np.unique(y, return_counts=True)}")
    except Exception as e:
        print(f"   ❌ Error processing dataset: {e}")
        print("\n   Tip: Ensure dataset directory contains 'asd' and 'typical' subdirectories")
        print("        with .jpg or .png images inside.")
        return
    
    # Train model
    print("\n3. Training model...")
    try:
        metrics = pipeline.train(X, y, test_size=TEST_SIZE)
        print(f"   ✅ Training complete!")
        print(f"   ✅ Train samples: {metrics['train_samples']}")
        print(f"   ✅ Test samples: {metrics['test_samples']}")
        print(f"   ✅ Test Accuracy: {metrics['test_metrics']['accuracy']:.4f}")
        print(f"   ✅ Test Precision: {metrics['test_metrics']['precision']:.4f}")
        print(f"   ✅ Test Recall: {metrics['test_metrics']['recall']:.4f}")
        print(f"   ✅ Test AUC: {metrics['test_metrics']['auc']:.4f}")
    except Exception as e:
        print(f"   ❌ Training error: {e}")
        return
    
    # Save model
    print("\n4. Saving model...")
    try:
        Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        pipeline.save_model(MODEL_PATH)
        print(f"   ✅ Model saved to: {MODEL_PATH}")
    except Exception as e:
        print(f"   ❌ Save error: {e}")
        return
    
    # Print summary
    print("\n5. Training Summary:")
    pipeline.print_summary()
    
    # Demonstrate prediction
    print("\n6. Testing predictions...")
    try:
        # Get test sample
        test_idx = np.random.randint(0, len(X))
        test_sample = X[test_idx:test_idx+1]
        
        predictions, probs = pipeline.predict(test_sample)
        
        pred_label = 'ASD' if predictions[0] == 1 else 'Typical'
        confidence = max(probs[0]) * 100
        
        print(f"   Sample prediction: {pred_label} (confidence: {confidence:.1f}%)")
        print(f"   Probabilities: Typical={probs[0,0]*100:.1f}%, ASD={probs[0,1]*100:.1f}%")
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Launch webcam interface: streamlit run src/webcam_ui.py")
    print(f"2. Load model path: {MODEL_PATH}")
    print(f"3. Run live gaze tracking and predictions")
    print()


if __name__ == '__main__':
    main()
