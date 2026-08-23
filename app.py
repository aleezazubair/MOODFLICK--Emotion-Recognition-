"""
MoodFlick / Emotion Recognition - Flask backend

The camera lives in the browser, not on the server: the page captures frames
with getUserMedia and POSTs them to /predict. That keeps every endpoint
stateless, so this runs unchanged on a laptop or in a container.

Serves:
  - MoodFlick UI at /
  - raw detector UI at /detector
  - POST /predict  → emotion for one uploaded frame (web UI and mobile app)
"""

import os

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# template_folder is explicit because the serverless bundle does not keep the
# repo layout the Flask default assumes.
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
CORS(app)  # Enable CORS for React Native to connect

# 🔹 Load trained model
# ONNX rather than the original .h5: onnxruntime is ~20MB against TensorFlow's
# ~400MB, which is the difference between fitting a serverless bundle and not.
# training/convert_to_onnx.py regenerates this file and checks it still matches
# Keras to within float32 rounding.
MODEL_PATH = os.path.join(BASE_DIR, "model", "emotion_model.onnx")
try:
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    model = session  # kept under the old name so /api/health reads the same
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    session = None
    model = None

# 🔹 Emotion labels (same order as the dataset folders in preprocessing.py)
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def predict_face(gray_face):
    """Predict emotion from an already-cropped grayscale face."""
    face = cv2.resize(gray_face, (48, 48)) / 255.0
    face = face.reshape(1, 48, 48, 1).astype(np.float32)

    # InferenceSession.run is thread-safe, so no lock is needed here.
    prediction = session.run(None, {input_name: face})[0][0]

    index = int(np.argmax(prediction))

    return (
        emotions[index],
        float(prediction[index]),
        {emotions[i]: float(prediction[i]) for i in range(len(emotions))},
    )


def largest_face(faces):
    """Haar can return several boxes; the biggest one is the subject."""
    return max(faces, key=lambda f: f[2] * f[3])


@app.route("/", methods=["GET"])
def moodflick():
    """MoodFlick — scan your face, get movie recommendations"""
    return render_template("moodflick.html")


@app.route("/detector", methods=["GET"])
def index():
    """Raw emotion recognition UI (live feed + confidence breakdown)"""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "message": "Emotion Recognition API is active",
        "model_loaded": model is not None,
        "emotions": emotions
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Predict emotion from one uploaded frame (web UI and mobile app)"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded. Please check server logs."
            }), 500

        # Check if image file is provided
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided"
            }), 400

        file = request.files["image"]

        # Check if file is empty
        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "Empty file provided"
            }), 400

        # Read and decode image
        npimg = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "success": False,
                "error": "Invalid image format"
            }), 400

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # A live preview frame with nobody in it is normal, not an error, so
        # the default is to report "no face" rather than guess. Callers sending
        # one deliberate photo -- or an already-cropped face, which Haar tends
        # to miss -- can pass fallback=1 to predict on the whole frame instead.
        fallback = request.form.get("fallback", "").lower() in ("1", "true", "yes")

        if len(faces):
            x, y, w, h = largest_face(faces)
            gray_face = gray[y:y + h, x:x + w]
            box = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        elif fallback:
            gray_face, box = gray, None
        else:
            return jsonify({
                "success": True,
                "face_detected": False,
                "faces": 0,
                "emotion": None,
                "confidence": 0.0,
                "all_predictions": {}
            })

        emotion, confidence, all_predictions = predict_face(gray_face)

        return jsonify({
            "success": True,
            "face_detected": bool(len(faces)),
            "faces": int(len(faces)),
            "emotion": emotion,
            "confidence": confidence,
            "box": box,
            "all_predictions": all_predictions
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Starting Emotion Recognition Server...")
    print(f"MoodFlick:      http://localhost:{port}")
    print(f"Raw detector:   http://localhost:{port}/detector")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True, use_reloader=False)
