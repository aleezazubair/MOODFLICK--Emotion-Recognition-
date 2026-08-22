# Emotion Recognition System Using Deep Learning

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
├── app.py                 # Flask API server (/predict endpoint)
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
- Live MJPEG webcam stream with face boxes and emotion labels drawn server-side
- Live confidence breakdown across all 7 emotions
- Start/Stop camera controls

The model runs every 5th frame (the last result is drawn in between), which keeps
the stream at roughly 10 fps instead of stalling on every frame.

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
