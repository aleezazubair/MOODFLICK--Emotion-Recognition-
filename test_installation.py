"""
Installation Test Script
Run this to verify all dependencies are correctly installed
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""

    print("=" * 60)
    print("Testing Emotion Recognition Project Installation")
    print("=" * 60)

    tests = []

    # Test TensorFlow
    print("\n[1/8] Testing TensorFlow...")
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow {tf.__version__} installed successfully")
        print(f"  GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        tests.append(("TensorFlow", True))
    except ImportError as e:
        print(f"[FAIL] TensorFlow import failed: {e}")
        tests.append(("TensorFlow", False))

    # Test Keras
    print("\n[2/8] Testing Keras...")
    try:
        from tensorflow import keras
        print(f"[OK] Keras (via TensorFlow) available")
        tests.append(("Keras", True))
    except ImportError as e:
        print(f"[FAIL] Keras import failed: {e}")
        tests.append(("Keras", False))

    # Test OpenCV
    print("\n[3/8] Testing OpenCV...")
    try:
        import cv2
        print(f"[OK] OpenCV {cv2.__version__} installed successfully")
        tests.append(("OpenCV", True))
    except ImportError as e:
        print(f"[FAIL] OpenCV import failed: {e}")
        tests.append(("OpenCV", False))

    # Test NumPy
    print("\n[4/8] Testing NumPy...")
    try:
        import numpy as np
        print(f"[OK] NumPy {np.__version__} installed successfully")
        tests.append(("NumPy", True))
    except ImportError as e:
        print(f"[FAIL] NumPy import failed: {e}")
        tests.append(("NumPy", False))

    # Test Pandas
    print("\n[5/8] Testing Pandas...")
    try:
        import pandas as pd
        print(f"[OK] Pandas {pd.__version__} installed successfully")
        tests.append(("Pandas", True))
    except ImportError as e:
        print(f"[FAIL] Pandas import failed: {e}")
        tests.append(("Pandas", False))

    # Test Scikit-learn
    print("\n[6/8] Testing Scikit-learn...")
    try:
        import sklearn
        print(f"[OK] Scikit-learn {sklearn.__version__} installed successfully")
        tests.append(("Scikit-learn", True))
    except ImportError as e:
        print(f"[FAIL] Scikit-learn import failed: {e}")
        tests.append(("Scikit-learn", False))

    # Test Matplotlib
    print("\n[7/8] Testing Matplotlib...")
    try:
        import matplotlib
        print(f"[OK] Matplotlib {matplotlib.__version__} installed successfully")
        tests.append(("Matplotlib", True))
    except ImportError as e:
        print(f"[FAIL] Matplotlib import failed: {e}")
        tests.append(("Matplotlib", False))

    # Test Seaborn
    print("\n[8/8] Testing Seaborn...")
    try:
        import seaborn as sns
        print(f"[OK] Seaborn {sns.__version__} installed successfully")
        tests.append(("Seaborn", True))
    except ImportError as e:
        print(f"[FAIL] Seaborn import failed: {e}")
        tests.append(("Seaborn", False))

    # Summary
    print("\n" + "=" * 60)
    print("INSTALLATION TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, status in tests if status)
    total = len(tests)

    for package, status in tests:
        symbol = "[OK]" if status else "[FAIL]"
        print(f"{symbol} {package}")

    print(f"\nResult: {passed}/{total} packages installed correctly")

    if passed == total:
        print("\nSUCCESS! All dependencies are installed correctly.")
        print("You can now run the emotion recognition system!")
        return True
    else:
        print("\nWARNING! Some dependencies are missing.")
        print("\nTo fix, run:")
        print("  pip install -r requirements.txt")
        return False


def test_dataset():
    """Check if dataset directories exist"""

    print("\n" + "=" * 60)
    print("DATASET CHECK")
    print("=" * 60)

    project_root = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(project_root, "dataset", "train")
    test_dir = os.path.join(project_root, "dataset", "test")

    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    print(f"\nChecking training dataset at: {train_dir}")
    if os.path.exists(train_dir):
        print("[OK] Training directory exists")
        for emotion in emotions:
            emotion_dir = os.path.join(train_dir, emotion)
            if os.path.exists(emotion_dir):
                count = len([f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                print(f"  [OK] {emotion}: {count} images")
            else:
                print(f"  [FAIL] {emotion}: directory not found")
    else:
        print("[FAIL] Training directory not found")
        print("  Please ensure dataset/train/ exists with emotion subdirectories")

    print(f"\nChecking test dataset at: {test_dir}")
    if os.path.exists(test_dir):
        print("[OK] Test directory exists")
        for emotion in emotions:
            emotion_dir = os.path.join(test_dir, emotion)
            if os.path.exists(emotion_dir):
                count = len([f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                print(f"  [OK] {emotion}: {count} images")
            else:
                print(f"  [FAIL] {emotion}: directory not found")
    else:
        print("[FAIL] Test directory not found")
        print("  Please ensure dataset/test/ exists with emotion subdirectories")


def test_camera():
    """Test if camera is accessible"""
    print("\n" + "=" * 60)
    print("CAMERA TEST")
    print("=" * 60)

    try:
        import cv2
        print("\nAttempting to access camera...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("[OK] Camera is accessible and working")
                print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("[FAIL] Camera opened but cannot read frames")
            cap.release()
        else:
            print("[FAIL] Cannot open camera")
            print("  Try checking camera permissions or use index 1:")
            print("  cv2.VideoCapture(1)")
    except Exception as e:
        print(f"[FAIL] Camera test failed: {e}")


if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("  EMOTION RECOGNITION PROJECT - INSTALLATION TEST")
    print("=" * 60)

    # Test imports
    imports_ok = test_imports()

    # Test dataset
    test_dataset()

    # Test camera
    test_camera()

    # Final message
    print("\n" + "=" * 60)
    if imports_ok:
        print("System is ready! Next steps:")
        print("  1. cd training")
        print("  2. python preprocessing.py")
        print("  3. python train.py")
        print("  4. python run_emotion_detection.py")
    else:
        print("Please install missing dependencies first:")
        print("  pip install -r requirements.txt")
    print("=" * 60)
    print("\nFor detailed instructions, see QUICKSTART.md")
    print()
