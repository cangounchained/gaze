#!/usr/bin/env python3
"""
ASD Gaze Tracker - Setup Verification Script
Run this to check if your installation is working correctly.
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_import(module_name, display_name=None):
    """Check if a module can be imported"""
    if display_name is None:
        display_name = module_name

    try:
        __import__(module_name)
        print(f"✅ {display_name} - Available")
        return True
    except ImportError as e:
        print(f"❌ {display_name} - Not available: {e}")
        return False

def check_file_exists(filename, description):
    """Check if a file exists"""
    if os.path.exists(filename):
        print(f"✅ {description} - Found")
        return True
    else:
        print(f"❌ {description} - Missing")
        return False

def main():
    print("🧠 ASD Gaze Tracker - Setup Verification")
    print("=" * 50)

    all_good = True

    # Check Python version
    all_good &= check_python_version()

    print("\n📦 Checking Core Dependencies:")
    # Required dependencies
    all_good &= check_import("numpy", "NumPy")
    all_good &= check_import("pandas", "Pandas")
    all_good &= check_import("sklearn", "Scikit-learn")
    all_good &= check_import("joblib", "Joblib")
    all_good &= check_import("streamlit", "Streamlit")

    print("\n📷 Checking Optional Dependencies:")
    # Optional dependencies
    opencv_ok = check_import("cv2", "OpenCV")
    mediapipe_ok = check_import("mediapipe", "MediaPipe")

    print("\n📁 Checking Required Files:")
    # Required files
    all_good &= check_file_exists("app.py", "Main application")
    all_good &= check_file_exists("requirements.txt", "Requirements file")
    all_good &= check_file_exists("README.md", "Documentation")

    print("\n🤖 Checking ML Model:")
    # Check if model exists or can be created
    if check_file_exists("asd_model.pkl", "Trained model"):
        print("   Model file exists - ready to use")
    else:
        print("   Model will be created on first run")

    print("\n" + "=" * 50)

    if all_good:
        print("🎉 Setup verification PASSED!")
        print("🚀 You can now run: python3 -m streamlit run app.py")
        print("   Or use: ./run.sh (Linux/macOS) or run.bat (Windows)")
    else:
        print("⚠️  Setup verification FAILED!")
        print("📖 Check the errors above and follow the installation instructions in README.md")

    # Additional recommendations
    print("\n💡 Recommendations:")
    if not opencv_ok:
        print("   - Install OpenCV for full webcam functionality")
        print("   - Linux: sudo apt-get install libgl1-mesa-glx")
    if not mediapipe_ok:
        print("   - Install MediaPipe for advanced face detection")

    print("\n🔗 Web interface will be available at: http://localhost:8501")

if __name__ == "__main__":
    main()