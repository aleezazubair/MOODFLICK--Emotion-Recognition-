# Emotion Recognition System Using Deep Learning

https://moodflick-emotion-recognition.onrender.com/
An AI-based system that detects emotions from facial expressions using Convolutional Neural Networks (CNN). This project provides real-time emotion detection capabilities using the FER-2013 dataset.

## Project Overview

This project implements a complete emotion recognition pipeline including:
- CNN-based emotion recognition model
- Training using FER-2013 dataset
- Real-time emotion detection using webcam
- Comprehensive evaluation with confusion matrices and accuracy metrics
- Mobile-ready architecture

## Detected Emotions

The system can detect 7 different emotions:
1. **Angry**
2. **Disgust**
3. **Fear**
4. **Happy**
5. **Neutral**
6. **Sad**
7. **Surprise**

## Project Structure

```
Emotion_Recognition_Project/
│
├── dataset/
│   ├── train/              # Training images organized by emotion folders
│   └── test/               # Test images organized by emotion folders
│
├── training/
│   ├── preprocessing.py    # Data preprocessing and augmentation
│   ├── model.py           # CNN model architecture
│   ├── train.py           # Model training script
│   ├── evaluate_csv.py    # Generate predictions CSV
│   ├── evaluate_analysis.py # Analyze predictions and create confusion matrix
│   ├── run_emotion_detection.py # Real-time webcam detection
│   └── emotion_model.h5   # Trained model
│
├── api/index.py           # Vercel entry point (imports app.py)
├── app.py                 # Flask server: pages + /predict endpoint
├── model/                 # emotion_model.onnx, the served model
├── vercel.json            # serverless routing and bundled files
├── mobile-app/            # React Native (Expo) app
│
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time detection)
- GPU recommended (for faster training)

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd Emotion_Recognition_Project
   ```

2. **Install required packages**

   For training and evaluation:
   ```bash
   pip install -r requirements-dev.txt
   ```

   To only run the web app, the slimmer runtime set is enough:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**
   - Ensure your FER-2013 dataset is organized in the following structure:
   ```
   dataset/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── neutral/
   │   ├── sad/
   │   └── surprise/
   └── test/
       ├── angry/
       ├── disgust/
       ├── fear/
       ├── happy/
       ├── neutral/
       ├── sad/
       └── surprise/
   ```

## Usage

### Step 1: Preprocess the Data

Run the preprocessing script to prepare images for training:

```bash
cd training
python preprocessing.py
```

This will:
- Load all training images from the dataset
- Resize images to 48x48 pixels
- Normalize pixel values (0-1)
- Save preprocessed data as `X_data.npy` and `y_labels.npy`

### Step 2: Train the Model

Train the CNN model on the preprocessed data:

```bash
python train.py
```

Features:
- 80-20 train-validation split
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Saves best model as `emotion_model.h5`
- Generates training history plot (`training_history.png`)

Training typically takes 30-50 epochs depending on early stopping.

### Step 3: Evaluate the Model

#### Option A: Generate predictions on test set

```bash
python evaluate_csv.py
```

This creates `test_predictions.csv` with predictions for all test images.

#### Option B: Analyze predictions and create confusion matrix

```bash
python evaluate_analysis.py
```

This will:
- Calculate overall accuracy
- Display classification report (precision, recall, F1-score)
- Generate confusion matrix visualization
- Save confusion matrix as `confusion_matrix.png`

### Step 4: Real-Time Emotion Detection

Run the webcam detection script:

```bash
python run_emotion_detection.py
```

Features:
- Real-time face detection using Haar Cascade
- Emotion prediction for detected faces
- Bounding boxes around faces with emotion labels
- Press 'q' to quit

### Step 5: Web App (Live Camera)

Start the Flask server from the project root:

```bash
python app.py
```

Then open **http://localhost:5000** in your browser and press **Start Camera**.

Features:
- Live camera preview with a confidence breakdown across all 7 emotions
- Start/Stop camera controls
- MoodFlick movie shelves driven by the detected emotion

The camera is opened by the **browser** (`getUserMedia`), not by the server. The
page grabs the current video frame onto an offscreen canvas every 400ms and POSTs
that JPEG to `/predict`, which detects the face and returns the CNN's output. That
keeps the server stateless, so the same code runs locally and when deployed --
a server-side `cv2.VideoCapture(0)` would only ever see a camera attached to the
machine running Flask.

> Browsers only grant camera access on `https://` or on `localhost`. Opening the
> page over plain `http://` on a LAN IP will silently fail to start the camera.

---

## Movie Catalogue (TMDB)

The page ships with a hand-written catalogue, so it works with no key and no
network. Set a TMDB key and the same shelves are filled from TMDB instead,
which adds real poster art, official trailers, and per-country streaming
availability.

To be clear about what this is: TMDB serves **metadata**. No free API streams
films, and this one does not either. What it can do is say which services carry
a title in a given country -- through JustWatch -- so the app links straight to
it rather than guessing with a search URL.

1. Sign up at <https://www.themoviedb.org/signup> and request an API key at
   Settings -> API. It is free.
2. Set it in the environment:

   ```bash
   # Windows PowerShell
   $env:TMDB_API_KEY = "your-key"
   # macOS / Linux
   export TMDB_API_KEY=your-key
   ```

   On Render, add it under Environment; `render.yaml` already declares it with
   `sync: false`, so the value stays out of the repository.

3. Optionally set `TMDB_REGION` (default `PK`) to change which country's
   streaming availability is reported.

`GET /api/health` reports which source is in use: `movie_api` reads either
`tmdb` or `built-in catalogue`.

| Endpoint | Purpose |
|---|---|
| `GET /api/catalogue` | Every shelf, deduplicated. `configured: false` means the page keeps its own list. |
| `GET /api/movie/<id>` | One title: full cast, official trailer, and where it can be watched. |

Responses are cached in-process for six hours, which keeps the app well inside
the free rate limit and off a small instance's CPU. Any TMDB failure leaves the
built-in catalogue in place rather than emptying the shelves.

---

## Deployment (Vercel)

The app is deployed as a single Python serverless function. `vercel.json` routes
every path to `api/index.py`, which imports the same `app` object that
`python app.py` runs locally, so there is one codebase and no separate build.

```bash
npm i -g vercel
vercel            # first run links the project and gives a preview URL
vercel --prod
```

Or import the GitHub repo at <https://vercel.com/new> and let it deploy on push.

### Why the model is ONNX

Serving with TensorFlow is what makes this app undeployable. A serverless
function is capped at 250MB, and on Linux `pip install tensorflow` pulls the
`nvidia-*` CUDA wheels -- around 1.1GB of GPU libraries a CPU host cannot use.
A first attempt at deploying weighed **2527MB**.

`model/emotion_model.onnx` is the same network exported to ONNX, served by
`onnxruntime` at roughly 40MB instead of TensorFlow's 1.1GB:

| | TensorFlow | ONNX |
|---|---:|---:|
| Inference runtime | ~1101 MB | ~41 MB |
| Model file | 4.2 MB (`.h5`) | 1.4 MB (`.onnx`) |
| Bundle vs 250MB limit | over | under |

The conversion is lossless up to float32 rounding. Regenerate and re-verify it
with `python training/convert_to_onnx.py`, which compares both runtimes over 300
real test images and fails if the predictions diverge -- measured max difference
is 6.6e-07, with identical argmax on all 300.

Training still uses Keras and `training/emotion_model.h5`; only serving uses ONNX.

### Other hosts

`Dockerfile` builds the same app for any container host (Render, Fly.io, Cloud
Run). Note that **Hugging Face Spaces now requires a paid plan** for Docker and
Gradio Spaces -- only Static Spaces are free, and those cannot run Python.
**GitHub Pages will not work** either: it serves static files only, so the UI
would load but every `/predict` call would 404.

## Model Architecture

The CNN model consists of:

```
Input (48x48x1 grayscale image)
    ↓
Conv2D (32 filters, 3x3) + ReLU + MaxPooling
    ↓
Conv2D (64 filters, 3x3) + ReLU + MaxPooling
    ↓
Conv2D (128 filters, 3x3) + ReLU + MaxPooling
    ↓
Flatten
    ↓
Dense (128 units) + ReLU + Dropout (0.5)
    ↓
Dense (7 units) + Softmax
    ↓
Output (7 emotion classes)
```

**Total Parameters:** 355,847

## Training Configuration

- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Batch Size:** 64
- **Max Epochs:** 50
- **Validation Split:** 20%
- **Early Stopping Patience:** 10 epochs
- **Learning Rate Reduction:** Factor 0.5, Patience 5

## Evaluation Metrics

The system provides comprehensive evaluation:

1. **Accuracy Score:** Overall prediction accuracy
2. **Confusion Matrix:** Visual representation of predictions vs. true labels
3. **Classification Report:**
   - Precision per emotion
   - Recall per emotion
   - F1-score per emotion
   - Support (number of samples)
4. **Training History Graphs:**
   - Training vs. Validation Accuracy
   - Training vs. Validation Loss

## Measured Performance

Evaluated on the full 7,178-image test set (see [REPORT.md](REPORT.md) for the full analysis):

- **Test Accuracy:** 57.09%
- **Training Accuracy:** ~72% (final epoch)
- **Validation Accuracy:** ~56% (plateau)
- **Macro F1:** 0.514

Best classes: happy (F1 0.78), surprise (F1 0.71).
Worst: disgust (recall 0.17 — only 1.5% of the training data).

Note: Emotion recognition is inherently challenging due to subjective labeling and image quality variations. Human agreement on FER-2013 labels is only around 65%.

## Troubleshooting

### Issue: TensorFlow import errors
**Solution:** Install TensorFlow
```bash
pip install tensorflow>=2.10.0
```

### Issue: OpenCV error - cannot open webcam
**Solution:** Check webcam permissions and ensure no other application is using it

### Issue: Model file not found
**Solution:** Train the model first using `python train.py`

### Issue: CUDA/GPU errors
**Solution:** Install GPU-compatible TensorFlow or use CPU version
```bash
pip install tensorflow-cpu
```

## Mobile Application

A React Native (Expo) app lives in `mobile-app/`. It captures a photo and posts it to
the Flask API for prediction. See [MOBILE_APP_SETUP.md](MOBILE_APP_SETUP.md) for setup.

Note: `node_modules/` is not committed — run `npm install` inside `mobile-app/` first.

Because the model runs on the server, each frame has to travel over WiFi, so the mobile
app does single-photo capture rather than a live stream. For live detection use the
webcam script or the web frontend.

## Future Enhancements

- [ ] Data augmentation for improved accuracy
- [ ] Transfer learning with pre-trained models (VGG, ResNet)
- [ ] Multi-face detection and emotion tracking
- [ ] Emotion intensity prediction
- [ ] Mobile app with TensorFlow Lite
- [x] Web interface using Flask
- [ ] Export to ONNX format for broader deployment

## Dataset Information

**FER-2013 Dataset:**
- 35,887 grayscale images (48x48 pixels)
- 7 emotion categories
- Training set: 28,709 images
- Test set: 7,178 images (public + private test splits combined)

## Technologies Used

- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** Scikit-learn

## License

This project is for educational purposes. Please ensure you have proper rights to use the FER-2013 dataset.

## Contributors

- Development: 5th Semester Mobile Application Project

## Acknowledgments

- FER-2013 dataset creators
- TensorFlow and Keras teams
- OpenCV community

## Contact & Support

For issues, questions, or contributions, please create an issue in the project repository.

---

**Last Updated:** December 2025
**Version:** 1.0.0
