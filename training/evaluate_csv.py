import os
import cv2
import numpy as np
import csv
from tensorflow.keras.models import load_model

# Get directories
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
MODEL_PATH = os.path.join(TRAINING_DIR, "emotion_model.h5")
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please train the model first by running train.py")
    exit()

# Load trained model
model = load_model(MODEL_PATH)

# Corrected emotion order to match training
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE = 48

# CSV file path
CSV_FILE = os.path.join(TRAINING_DIR, "test_predictions.csv")

# 4️⃣ Open CSV file to write
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "True Label", "Predicted Label"])  # header

    # 5️⃣ Loop through test folder and predict
    for emotion in EMOTIONS:
        folder = os.path.join(TEST_DIR, emotion)
        if not os.path.exists(folder):
            continue

        images = os.listdir(folder)
        print(f"Processing {len(images)} images for emotion: {emotion}")

        for img_name in images:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            # 6️⃣ Prediction
            pred = model.predict(img)
            predicted_label = EMOTIONS[np.argmax(pred)]

            # Write to CSV
            writer.writerow([img_name, emotion, predicted_label])

print(f"\nDone! All predictions saved in {CSV_FILE}")
