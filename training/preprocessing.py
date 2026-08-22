import os
import cv2
import numpy as np

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")

# Emotions (folder names) - corrected order to match standard FER-2013
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

IMG_SIZE = 48

X = []
y = []

# 🔹 Check if dataset folder exists
if not os.path.exists(DATASET_DIR):
    print("Dataset folder not found!")
    print("Check your DATASET_DIR path.")
    exit()

print("Starting Preprocessing...")

for idx, emotion in enumerate(EMOTIONS):
    folder = os.path.join(DATASET_DIR, emotion)

    # Check if emotion folder exists
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    images = os.listdir(folder)
    if len(images) == 0:
        print(f"No images in folder: {folder}")

    print(f"Processing {len(images)} images for emotion: {emotion}")

    for img_name in images:
        img_path = os.path.join(folder, img_name)

        # read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        # resize to 48x48
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(img)
        y.append(idx)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize (0–255 → 0–1)
X = X / 255.0

# Add channel dimension (48,48 → 48,48,1)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Save files in the training directory
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
np.save(os.path.join(TRAINING_DIR, "X_data.npy"), X)
np.save(os.path.join(TRAINING_DIR, "y_labels.npy"), y)

print("Preprocessing Completed!")
print("Total Images Processed:", len(X))
print(f"Saved: {os.path.join(TRAINING_DIR, 'X_data.npy')} & {os.path.join(TRAINING_DIR, 'y_labels.npy')}")
