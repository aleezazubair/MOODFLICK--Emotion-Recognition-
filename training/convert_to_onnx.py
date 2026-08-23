"""
Convert emotion_model.h5 to ONNX, then verify the two agree.

The served model is ONNX because onnxruntime is ~20MB where TensorFlow is
~400MB, and the deployed bundle has to fit inside a 250MB serverless function.
The conversion is lossless up to float32 rounding, which this script checks
against real test images rather than taking on faith.

Only needed if the model is retrained. Requires the training environment plus
the converter, which is deliberately not in requirements-dev.txt because it
drags protobuf forward and can break a working TensorFlow install:

    pip install tf2onnx onnx onnxruntime

Usage:
    python training/convert_to_onnx.py
"""

import glob
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERAS_PATH = os.path.join(PROJECT_ROOT, "training", "emotion_model.h5")
ONNX_PATH = os.path.join(PROJECT_ROOT, "model", "emotion_model.onnx")
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")

SAMPLE_SIZE = 300
TOLERANCE = 1e-5


def convert():
    model = tf.keras.models.load_model(KERAS_PATH)
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

    spec = (tf.TensorSpec((None, 48, 48, 1), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=13,
                               output_path=ONNX_PATH)

    size_mb = os.path.getsize(ONNX_PATH) / 1e6
    print(f"Wrote {ONNX_PATH} ({size_mb:.2f} MB)")
    return model


def verify(keras_model):
    """Compare both runtimes on real test images, not random noise."""
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*", "*.jpg")))
    if not files:
        print("No test images found - skipping verification.")
        print("Unpack the dataset (see README) to check the conversion.")
        return

    random.seed(0)
    files = random.sample(files, min(SAMPLE_SIZE, len(files)))

    batch = np.stack([
        cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), (48, 48)) / 255.0
        for f in files
    ]).reshape(-1, 48, 48, 1).astype(np.float32)

    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    expected = keras_model.predict(batch, verbose=0)
    actual = session.run(None, {input_name: batch})[0]

    max_diff = float(np.abs(expected - actual).max())
    agree = int((expected.argmax(1) == actual.argmax(1)).sum())

    print(f"Compared {len(files)} images")
    print(f"  max abs difference : {max_diff:.3e}")
    print(f"  argmax agreement   : {agree}/{len(files)}")

    if max_diff > TOLERANCE or agree != len(files):
        raise SystemExit("ONNX output does not match Keras - not safe to deploy.")

    print("ONNX model matches Keras.")


if __name__ == "__main__":
    verify(convert())
