#!/usr/bin/env python3
"""
Check library availability for ASD Gaze Tracker
"""

print("🔍 ASD GAZE TRACKER - LIBRARY AVAILABILITY CHECK")
print("=" * 60)

# Test OpenCV
print("\n📹 Testing OpenCV...")
try:
    import cv2
    print(f"✅ OpenCV available - version: {cv2.__version__}")
    print("   Webcam features: ENABLED")
    opencv_ok = True
except ImportError as e:
    print(f"❌ OpenCV not available: {e}")
    print("   Webcam features: DISABLED")
    opencv_ok = False

# Test MediaPipe
print("\n👁️ Testing MediaPipe...")
try:
    import mediapipe as mp
    print(f"✅ MediaPipe available - version: {mp.__version__}")
    print("   Face detection: ENABLED")
    mediapipe_ok = True
except ImportError as e:
    print(f"❌ MediaPipe not available: {e}")
    print("   Face detection: DISABLED")
    mediapipe_ok = False

# Test feature extraction
print("\n🔬 Testing Feature Extraction...")
try:
    from feature_extraction import extract_features_from_image
    print("✅ Feature extraction available")
    print("   Media file analysis: ENABLED")
    feature_ok = True
except ImportError as e:
    print(f"❌ Feature extraction not available: {e}")
    print("   Media file analysis: DISABLED")
    feature_ok = False

# Test ML model
print("\n🤖 Testing ML Model...")
try:
    from ml_model import ASDClassifier
    print("✅ ML model available")
    print("   ASD classification: ENABLED")
    ml_ok = True
except ImportError as e:
    print(f"❌ ML model not available: {e}")
    print("   ASD classification: DISABLED")
    ml_ok = False

print("\n" + "=" * 60)
print("📊 SUMMARY:")
print(f"OpenCV: {'✅' if opencv_ok else '❌'}")
print(f"MediaPipe: {'✅' if mediapipe_ok else '❌'}")
print(f"Feature Extraction: {'✅' if feature_ok else '❌'}")
print(f"ML Model: {'✅' if ml_ok else '❌'}")

if opencv_ok and mediapipe_ok and feature_ok and ml_ok:
    print("\n🎉 ALL FEATURES AVAILABLE!")
    print("   Full webcam and media analysis enabled!")
else:
    print("\n⚠️ SOME FEATURES LIMITED")
    print("   Missing libraries disable certain features")

print("\n🚀 Web interface will show available features")
print("   Run: streamlit run app.py")