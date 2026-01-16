#!/usr/bin/env python3
"""
ASD Gaze Tracker - Working Command Line Version

This is a fully functional command-line version that works without
requiring OpenCV or complex dependencies.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys

def main():
    print("🧠 ASD Gaze Tracker - Command Line Version")
    print("=" * 50)

    # Check if sample dataset exists, if not create it
    if not os.path.exists('sample_gaze_dataset.csv'):
        print("📊 Creating sample dataset...")
        create_sample_dataset()

    # Load dataset
    print("📂 Loading dataset...")
    features, labels = load_dataset('sample_gaze_dataset.csv')

    if features is None:
        print("❌ Failed to load dataset")
        return

    # Split data
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")

    # Train model
    print("🤖 Training model...")
    model = train_model(X_train, y_train)

    # Evaluate
    print("📊 Evaluating model...")
    from ml_model import ASDClassifier
    classifier = ASDClassifier()
    classifier.model = model  # Use the trained model
    results = classifier.evaluate(X_test, y_test, save_plots=True, output_dir='evaluation_results')
    classifier.print_evaluation_report(results)

    # Save model
    print("💾 Saving model...")
    save_model(model, 'asd_model.pkl')

    # Test classification
    print("🔍 Testing classification...")
    sample_features = get_sample_features()
    result, confidence = classify_sample(model, sample_features)

    print("\n" + "=" * 50)
    print("✅ SUCCESS! ASD Gaze Tracker is working!")
    print(f"🎯 Sample Prediction: {result} ({confidence:.2f} confidence)")

    print("\n🚀 How to use with your data:")
    print("1. Create a CSV file with columns: x,y,ear_left,ear_right,...label")
    print("2. Run: python working_cli.py your_data.csv")
    print("3. The trained model will be saved as 'asd_model.pkl'")
    print("4. Check 'evaluation_results/' for detailed performance metrics")

def create_sample_dataset():
    """Create sample dataset"""
    np.random.seed(42)
    n_samples = 500

    data = {
        'x': np.random.randint(50, 200, n_samples),
        'y': np.random.randint(50, 150, n_samples),
        'ear_left': np.random.uniform(0.2, 0.4, n_samples),
        'ear_right': np.random.uniform(0.2, 0.4, n_samples),
        'pupil_left_x': np.random.randint(80, 120, n_samples),
        'pupil_left_y': np.random.randint(70, 100, n_samples),
        'pupil_right_x': np.random.randint(130, 170, n_samples),
        'pupil_right_y': np.random.randint(70, 100, n_samples),
        'mouth_opening': np.random.uniform(10, 50, n_samples),
        'dist_eye_left': np.random.uniform(20, 80, n_samples),
        'dist_eye_right': np.random.uniform(20, 80, n_samples),
        'dist_mouth': np.random.uniform(30, 100, n_samples),
        'dist_nose': np.random.uniform(40, 120, n_samples),
        'label': np.random.choice(['ASD', 'Typical'], n_samples)
    }

    df = pd.DataFrame(data)
    df.to_csv('sample_gaze_dataset.csv', index=False)
    print(f"   Created {len(df)} samples")

def load_dataset(csv_path):
    """Load and preprocess dataset"""
    try:
        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} samples")

        # Encode labels
        df['label_encoded'] = df['label'].map({'ASD': 1, 'Typical': 0})

        # Select features
        feature_cols = [col for col in df.columns if col not in ['label', 'label_encoded']]
        features = df[feature_cols]
        labels = df['label_encoded']

        print(f"   Features: {len(feature_cols)} ({', '.join(feature_cols)})")
        print(f"   Labels: {df['label_encoded'].value_counts().to_dict()}")

        return features, labels

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None, None

def train_model(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """Save trained model"""
    joblib.dump(model, filename)
    print(f"   Model saved as {filename}")

def get_sample_features():
    """Get sample features for testing"""
    return {
        'x': 100,
        'y': 80,
        'ear_left': 0.28,
        'ear_right': 0.30,
        'pupil_left_x': 95,
        'pupil_left_y': 85,
        'pupil_right_x': 145,
        'pupil_right_y': 85,
        'mouth_opening': 25.5,
        'dist_eye_left': 45.2,
        'dist_eye_right': 42.8,
        'dist_mouth': 65.3,
        'dist_nose': 78.9
    }

def classify_sample(model, features):
    """Classify a sample"""
    df_features = pd.DataFrame([features])
    prediction = model.predict(df_features)[0]
    probabilities = model.predict_proba(df_features)[0]

    result = 'ASD' if prediction == 1 else 'Typical'
    confidence = max(probabilities)

    print(f"   Input features: {len(features)} values")
    print(f"   Prediction: {result}")
    print(f"   Confidence: {confidence:.2f}")
    return result, confidence

if __name__ == "__main__":
    # Allow custom dataset file as argument
    if len(sys.argv) > 1:
        custom_file = sys.argv[1]
        if os.path.exists(custom_file):
            print(f"Using custom dataset: {custom_file}")
            # Rename to expected filename
            import shutil
            shutil.copy(custom_file, 'sample_gaze_dataset.csv')
        else:
            print(f"❌ Custom file {custom_file} not found")
            sys.exit(1)

    main()