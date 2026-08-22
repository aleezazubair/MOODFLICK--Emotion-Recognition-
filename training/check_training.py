"""
Quick script to check if training is complete and model exists
"""
import os
import time

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TRAINING_DIR, "emotion_model.h5")

if os.path.exists(MODEL_PATH):
    # Get file modification time
    mod_time = os.path.getmtime(MODEL_PATH)
    mod_time_readable = time.ctime(mod_time)
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB

    print("=" * 60)
    print("MODEL STATUS")
    print("=" * 60)
    print(f"[OK] Model file exists: {MODEL_PATH}")
    print(f"Last modified: {mod_time_readable}")
    print(f"File size: {file_size:.2f} MB")
    print("\nModel is ready for evaluation and testing!")
    print("=" * 60)
else:
    print("Model file not found. Training may still be in progress...")
