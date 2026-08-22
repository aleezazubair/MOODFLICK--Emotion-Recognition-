"""
Emotion Recognition - Flask backend

Serves:
  - the live web UI at /
  - an MJPEG webcam stream with face boxes + emotion labels at /video_feed
  - JSON endpoints for the web UI (/status) and the mobile app (/predict)
"""

import os
import threading

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native to connect

# 🔹 Load trained model
MODEL_PATH = os.path.join("training", "emotion_model.h5")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 🔹 Emotion labels (same order as the dataset folders in preprocessing.py)
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Running the model on every frame stalls the stream, so we predict every
# PREDICT_EVERY frames and keep drawing the most recent result in between.
PREDICT_EVERY = 5

_camera = None
_camera_lock = threading.Lock()

_latest = {"emotion": None, "confidence": 0.0, "all_predictions": {}, "faces": 0}
_latest_lock = threading.Lock()


def predict_face(gray_face):
    """Predict emotion from an already-cropped grayscale face."""
    face = cv2.resize(gray_face, (48, 48)) / 255.0
    face = face.reshape(1, 48, 48, 1)

    prediction = model.predict(face, verbose=0)[0]
    index = int(np.argmax(prediction))

    return (
        emotions[index],
        float(prediction[index]),
        {emotions[i]: float(prediction[i]) for i in range(len(emotions))},
    )


def largest_face(faces):
    """Haar can return several boxes; the biggest one is the subject."""
    return max(faces, key=lambda f: f[2] * f[3])


def draw_label(frame, box, text):
    """Draw the face box with a filled caption bar above it."""
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 230, 118), 2)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(y - th - 12, 0)
    cv2.rectangle(frame, (x, top), (x + tw + 12, top + th + 10), (0, 230, 118), -1)
    cv2.putText(
        frame, text, (x + 6, top + th + 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 12, 16), 2,
    )


def generate_frames():
    """Yield webcam frames as an MJPEG stream."""
    frame_count = 0

    while True:
        with _camera_lock:
            if _camera is None:
                break
            ok, frame = _camera.read()

        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror, so it behaves like a selfie camera
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) and model is not None:
            x, y, w, h = largest_face(faces)

            if frame_count % PREDICT_EVERY == 0:
                emotion, confidence, all_predictions = predict_face(gray[y:y + h, x:x + w])
                with _latest_lock:
                    _latest["emotion"] = emotion
                    _latest["confidence"] = confidence
                    _latest["all_predictions"] = all_predictions

            with _latest_lock:
                _latest["faces"] = len(faces)
                label = _latest["emotion"]
                score = _latest["confidence"]

            if label:
                draw_label(frame, (x, y, w, h), f"{label} ({score * 100:.1f}%)")
        else:
            with _latest_lock:
                _latest["faces"] = 0
                _latest["emotion"] = None
                _latest["confidence"] = 0.0
                _latest["all_predictions"] = {}

        frame_count += 1

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


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
        "camera_on": _camera is not None,
        "emotions": emotions
    })


@app.route("/camera/start", methods=["POST"])
def camera_start():
    """Open the webcam"""
    global _camera

    with _camera_lock:
        if _camera is None:
            # CAP_DSHOW avoids the slow startup OpenCV otherwise has on Windows
            _camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not _camera.isOpened():
                _camera.release()
                _camera = None
                return jsonify({
                    "success": False,
                    "error": "Could not open webcam. Is another app using it?"
                }), 500

    return jsonify({"success": True, "camera_on": True})


@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    """Release the webcam"""
    global _camera

    with _camera_lock:
        if _camera is not None:
            _camera.release()
            _camera = None

    with _latest_lock:
        _latest["emotion"] = None
        _latest["confidence"] = 0.0
        _latest["all_predictions"] = {}
        _latest["faces"] = 0

    return jsonify({"success": True, "camera_on": False})


@app.route("/video_feed", methods=["GET"])
def video_feed():
    """MJPEG stream of the annotated webcam feed"""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status", methods=["GET"])
def status():
    """Most recent prediction from the live stream, polled by the web UI"""
    with _latest_lock:
        return jsonify({
            "camera_on": _camera is not None,
            "faces": _latest["faces"],
            "emotion": _latest["emotion"],
            "confidence": _latest["confidence"],
            "all_predictions": _latest["all_predictions"],
        })


@app.route("/predict", methods=["POST"])
def predict():
    """Predict emotion from an uploaded image (used by the mobile app)"""
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

        # The model was trained on cropped faces, so crop before predicting.
        # Without a face we fall back to the whole frame rather than failing.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        if len(faces):
            x, y, w, h = largest_face(faces)
            gray = gray[y:y + h, x:x + w]

        emotion, confidence, all_predictions = predict_face(gray)

        return jsonify({
            "success": True,
            "emotion": emotion,
            "confidence": confidence,
            "face_detected": bool(len(faces)),
            "all_predictions": all_predictions
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print("Starting Emotion Recognition Server...")
    print("MoodFlick:      http://localhost:5000")
    print("Raw detector:   http://localhost:5000/detector")
    # use_reloader=False keeps Flask from starting a second process and
    # fighting over the webcam.
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
