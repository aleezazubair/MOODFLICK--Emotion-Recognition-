# Emotion Recognition from Facial Expressions Using a Convolutional Neural Network
PRESENTED BY ALEEZA ZUBAIR, SEHAR MEHMOOD, ANEEB BAARI



## Abstract

This project implements an end-to-end facial emotion recognition system capable of
classifying a human face into one of seven emotion categories: *angry, disgust, fear,
happy, neutral, sad,* and *surprise*. A Convolutional Neural Network (CNN) was trained
from scratch on the FER-2013 dataset (35,887 grayscale 48×48 images) and achieves
**57.09% accuracy on the 7,178-image test set**, against a random baseline of 14.3% and a
majority-class baseline of 24.7%.

The trained model is deployed through a Flask backend that serves a browser-based web
application, which streams the webcam live with emotion labels drawn on detected faces in
real time. Face localisation is performed using a Haar Cascade classifier prior to
inference.

The report documents the methodology, presents a per-class analysis of the results, and
examines the two principal limitations identified: overfitting of the network and severe
class imbalance in the dataset.

---

## 1. Introduction

### 1.1 Problem Statement

Facial expressions are one of the primary channels of non-verbal human communication.
Automating their interpretation enables applications in human–computer interaction,
driver-attention monitoring, market research, mental-health screening and adaptive
learning systems. The task is to determine, from a single image of a face, which of a
fixed set of emotion categories the expression belongs to.

The task is non-trivial. Expressions vary across individuals, cultures and contexts; they
are often subtle or blended; and image conditions such as lighting, pose and occlusion
vary widely. Furthermore, the ground-truth labels themselves are subjective — human
annotators agree on FER-2013 labels only about 65% of the time, which places a practical
ceiling on achievable accuracy.

### 1.2 Objectives

1. Build and train a CNN classifier for seven-class facial emotion recognition.
2. Evaluate the model rigorously using accuracy, precision, recall, F1-score and a
   confusion matrix.
3. Deploy the model in a real-time application driven by a live camera feed.
4. Analyse the failure modes of the system and identify concrete improvements.

### 1.3 Scope

The system classifies the single largest detected face per frame. Multi-face tracking and
emotion-intensity regression were considered but fall outside the scope of this
implementation.

---

## 2. Background

### 2.1 The FER-2013 Dataset

FER-2013 was introduced for the ICML 2013 Challenges in Representation Learning. It
consists of 48×48 pixel grayscale face images collected via automated web search and
labelled into seven emotion categories. It is a standard benchmark for this task. Its
known difficulties — low resolution, label noise and heavy class imbalance — are
reflected directly in the results reported in Section 6.

For reference, the winning entry of the original 2013 competition achieved approximately
71% accuracy, and human performance on the dataset is estimated at roughly 65 ± 5%.

### 2.2 Convolutional Neural Networks

A CNN learns a hierarchy of spatial features directly from pixels. Early convolutional
layers respond to edges and simple textures; deeper layers compose these into parts
(eyes, mouth corners, brow furrows) that are discriminative for expression. Weight
sharing makes the representation translation-tolerant and drastically reduces the
parameter count relative to a fully connected network of comparable capacity. This makes
CNNs the natural architecture family for this problem.

---

## 3. Dataset

### 3.1 Composition

The dataset is organised as one directory per emotion under `dataset/train/` and
`dataset/test/`.

| Emotion | Train images | Test images | Train share |
|---|---:|---:|---:|
| Angry | 3,995 | 958 | 13.9% |
| Disgust | 436 | 111 | 1.5% |
| Fear | 4,097 | 1,024 | 14.3% |
| Happy | 7,215 | 1,774 | 25.1% |
| Neutral | 4,965 | 1,233 | 17.3% |
| Sad | 4,830 | 1,247 | 16.8% |
| Surprise | 3,171 | 831 | 11.0% |
| **Total** | **28,709** | **7,178** | **100%** |

### 3.2 Class Imbalance

The distribution is markedly skewed. *Happy* contains 7,215 training images while
*disgust* contains only 436 — a ratio of **16.5 : 1**. Because the network is trained with
an unweighted cross-entropy loss, it can reduce the loss more effectively by learning the
frequent classes well and effectively ignoring the rare ones. Section 6.3 shows that this
is precisely what happened.

---

## 4. Methodology

### 4.1 Preprocessing

Implemented in `training/preprocessing.py`:

1. Each image is read in grayscale, discarding colour (expression is conveyed by shape
   and shading, and grayscale reduces the input dimensionality threefold).
2. Images are resized to a uniform 48×48.
3. Pixel values are normalised from `[0, 255]` to `[0, 1]`, which keeps gradients
   well-scaled during optimisation.
4. A channel dimension is added, giving each sample the shape `(48, 48, 1)`.
5. The processed arrays are cached to `X_data.npy` and `y_labels.npy` so that training
   runs do not repeat the decode step.

Class labels are assigned by the index of the emotion in the ordered list
`[angry, disgust, fear, happy, neutral, sad, surprise]`. **This ordering is
load-bearing**: every component that maps a model output index back to a human-readable
name must use exactly this order. A mismatch here silently produces confident but wrong
labels (see Section 7.3).

### 4.2 Network Architecture

Defined in `training/model.py` and `training/train.py`:

```
Input (48 × 48 × 1)
    ↓
Conv2D(32, 3×3, ReLU)  →  MaxPooling2D(2×2)
    ↓
Conv2D(64, 3×3, ReLU)  →  MaxPooling2D(2×2)
    ↓
Conv2D(128, 3×3, ReLU) →  MaxPooling2D(2×2)
    ↓
Flatten
    ↓
Dense(128, ReLU)  →  Dropout(0.5)
    ↓
Dense(7, Softmax)
```

**Total trainable parameters: 355,847.**

Filter counts double at each stage (32 → 64 → 128) while pooling halves the spatial
resolution, a standard design that trades spatial detail for representational depth. The
single `Dropout(0.5)` before the classifier is the only explicit regulariser in the
network — a point returned to in Section 7.1.

### 4.3 Training Configuration

| Setting | Value |
|---|---|
| Optimiser | Adam (default learning rate 0.001) |
| Loss | Categorical cross-entropy |
| Batch size | 64 |
| Maximum epochs | 50 |
| Validation split | 20% (stratified, `random_state=42`) |
| Early stopping | Monitor `val_loss`, patience 10, restore best weights |
| LR reduction | Factor 0.5, patience 5, minimum 1e-5 |
| Checkpointing | Save best model by `val_accuracy` |

The stratified split preserves the class proportions of the full dataset in both the
training and validation partitions, so validation metrics are not distorted by sampling.

---

## 5. System Implementation

### 5.1 Architecture

```
┌──────────────────┐        ┌──────────────────────────────┐
│  Web browser     │◄──────►│  Flask server (app.py)       │
│  (live UI)       │  HTTP  │                              │
└──────────────────┘        │  ┌────────────────────────┐  │
                            │  │ Haar Cascade detector  │  │
                            │  └───────────┬────────────┘  │
                            │              ▼               │
                            │  ┌────────────────────────┐  │
                            │  │ CNN (emotion_model.h5) │  │
                            │  └────────────────────────┘  │
                            └──────────────┬───────────────┘
                                           ▼
                                        Webcam
```

### 5.2 Inference Pipeline

The web interface uses the following prediction path:

1. **Face localisation.** A Haar Cascade (`haarcascade_frontalface_default.xml`) locates
   faces in the grayscale frame. When several are found, the one with the largest
   bounding-box area is taken as the subject.
2. **Crop.** The face region is cropped. This step is essential — the network was trained
   on tightly cropped faces, so presenting it with a full scene shifts the input far off
   the training distribution and degrades accuracy substantially.
3. **Normalise.** The crop is resized to 48×48 and scaled to `[0, 1]`, matching the
   preprocessing applied during training exactly.
4. **Predict.** The softmax layer yields a probability for each of the seven classes; the
   argmax is reported as the predicted emotion together with its confidence.

### 5.3 Web Application (Live Camera)

The primary interface is a single-page web application served at `/`.

The server captures webcam frames with OpenCV, runs the pipeline above, draws the
bounding box and label onto the frame, JPEG-encodes it, and pushes it to the browser as
an **MJPEG stream** (`multipart/x-mixed-replace`). The browser renders this with a plain
`<img>` element, so no client-side video or machine-learning code is required.

A measured optimisation was necessary here: running the CNN on every captured frame
stalls the stream. The implementation therefore runs inference on **every fifth frame**
and reuses the most recent prediction for the intervening frames. Face detection and
box drawing still occur every frame, so the overlay continues to track the face smoothly.
This sustains approximately **10 frames per second**, measured at 84 frames over an
8-second capture.

The browser polls a separate `/status` endpoint four times per second to update the
emotion readout and the confidence bars for all seven classes.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web user interface |
| `/video_feed` | GET | MJPEG stream with annotated frames |
| `/status` | GET | Latest prediction and all confidence scores |
| `/camera/start`, `/camera/stop` | POST | Acquire / release the webcam |
| `/api/health` | GET | Service and model status |

---

## 6. Results

All figures in this section were produced by evaluating the final trained model on the
complete 7,178-image test set, which is disjoint from both the training and validation
data.

### 6.1 Overall Performance

| Metric | Value |
|---|---|
| **Test accuracy** | **57.09%** |
| Macro-averaged F1 | 0.514 |
| Weighted-averaged F1 | 0.566 |
| Random baseline (1/7) | 14.29% |
| Majority-class baseline (always *happy*) | 24.71% |

The model performs well above both baselines, confirming that it has learned genuine
discriminative structure rather than exploiting the class prior.

### 6.2 Training Behaviour

Training halted at **epoch 19** of a permitted 50, triggered by early stopping, with the
best weights restored.

The curves in `training/training_history.png` show a clear and instructive pattern:

- Training and validation accuracy track each other closely for the first four epochs.
- They then diverge. Training accuracy continues climbing to roughly **72%**, while
  validation accuracy plateaus near **56%** and stays flat.
- Validation loss reaches its minimum (≈1.20) around epoch 8 and then **rises** to ≈1.29,
  even as training loss falls steadily to ≈0.76.

A rising validation loss alongside a falling training loss is the textbook signature of
**overfitting**: beyond roughly epoch 8 the network is memorising training examples rather
than learning generalisable features. The ~16-point gap between final training and
validation accuracy quantifies the effect.

Encouragingly, validation accuracy (≈56%) and test accuracy (57.09%) agree closely, which
indicates the validation split was representative and the reported test figure is
trustworthy.

### 6.3 Per-Class Performance

| Emotion | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Angry | 0.494 | 0.478 | 0.486 | 958 |
| Disgust | 0.792 | 0.171 | 0.281 | 111 |
| Fear | 0.422 | 0.294 | 0.346 | 1,024 |
| Happy | 0.767 | 0.796 | 0.781 | 1,774 |
| Neutral | 0.496 | 0.606 | 0.545 | 1,233 |
| Sad | 0.423 | 0.477 | 0.448 | 1,247 |
| Surprise | 0.747 | 0.681 | 0.712 | 831 |

Performance is highly uneven, and the pattern is systematic rather than random:

- **Happy (F1 0.781)** and **surprise (F1 0.712)** are recognised reliably. Both involve
  large, unambiguous geometric deformations — a raised mouth curve, a wide-open mouth and
  lifted brows — that survive downsampling to 48×48.
- **Fear (F1 0.346)** and **sad (F1 0.448)** are recognised poorly. These are subtle,
  low-contrast expressions whose distinguishing cues are fine-grained and largely
  destroyed at this resolution.
- **Disgust** exhibits the most revealing result: **precision 0.792 but recall 0.171**.
  When the model does predict *disgust* it is usually correct, but it only makes that
  prediction 24 times across the entire test set, correctly identifying just 19 of 111
  actual cases. This is a direct consequence of the 1.5% training share documented in
  Section 3.2 — the network has learned that predicting *disgust* is almost never worth
  the risk. **Accuracy alone conceals this failure entirely**, which is precisely why
  per-class recall was measured.

### 6.4 Confusion Analysis

The confusion matrix is saved at `training/confusion_matrix.png`. The dominant
misclassifications are:

| True → Predicted | Count | % of true class |
|---|---:|---:|
| Disgust → Angry | 40 | 36.0% |
| Fear → Sad | 279 | 27.2% |
| Sad → Neutral | 279 | 22.4% |
| Neutral → Sad | 234 | 19.0% |
| Angry → Sad | 174 | 18.2% |
| Surprise → Fear | 106 | 12.8% |

These errors are not arbitrary. Every dominant confusion pairs emotions that are
genuinely similar in facial geometry and in affective valence:

- *Disgust* and *anger* share the lowered brow and raised upper lip; with only 436
  training examples the model has no basis for separating them.
- *Fear* and *surprise* share widened eyes and raised brows; they differ mainly in mouth
  tension, a cue that is largely lost at 48×48.
- *Sad*, *neutral* and *angry* form a mutually confusable cluster of low-arousal, low-motion
  expressions — the bidirectional sad ↔ neutral confusion (279 and 234) shows the model
  cannot reliably separate a downturned mouth from a relaxed one.

The model's errors therefore mirror the ambiguities that human annotators also face,
rather than indicating a defect in the implementation.

---

## 7. Discussion and Limitations

### 7.1 Overfitting

The 16-point train–validation gap is the single largest limitation. Its causes are
identifiable in the design:

- **No data augmentation.** The network sees each of the 28,709 images in an identical
  form every epoch. Random horizontal flips, small rotations, zooms and brightness shifts
  would multiply the effective dataset size at zero labelling cost. This is the highest
  value-to-effort improvement available.
- **Minimal regularisation.** A single dropout layer before the classifier is the only
  explicit regulariser. Batch normalisation after each convolutional block would stabilise
  training and add a mild regularising effect; additional dropout between blocks and L2
  weight decay would further constrain the network.

### 7.2 Class Imbalance

Training with an unweighted loss allows the network to abandon the rare classes, as the
*disgust* recall of 0.171 demonstrates. Passing `class_weight` to `model.fit()` — inversely
proportional to class frequency — would penalise errors on rare classes proportionally
more, at some cost to majority-class accuracy. Oversampling the minority classes or
applying focal loss are alternative remedies.

### 7.3 Label-Order Consistency (Defect Found and Corrected)

During development the API was found to be returning emotion labels in the order
`[Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral]`, whereas the model was trained
against `[Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise]`. Indices 4, 5 and 6 were
therefore permuted, so every prediction falling into those three classes was reported
under the wrong name.

The defect was silent: the API returned well-formed responses with high confidence
values, and nothing in the output indicated an error. It was quantified by evaluating both
orderings against the same held-out images, which yielded **50.0% accuracy under the
correct ordering versus 27.6% under the incorrect one** — the fix nearly doubled measured
accuracy.

Two lessons follow. First, the class ordering is a contract between training and inference
and should be defined once and imported everywhere, not restated per file. Second, a
defect of this kind is invisible to unit-level checks and surfaces only when predictions
are compared against known labels end to end.

### 7.4 Other Constraints

- **Frontal faces only.** The Haar Cascade detects frontal faces; profile views and
  strong occlusion (hands, masks) cause detection to fail, and no face means no
  prediction.
- **Server-side camera.** The web application captures from the machine running the
  server, which suits a single-machine demonstration but not remote access.
- **Lighting sensitivity.** The training data offers limited illumination diversity, so
  accuracy degrades in low light and under strong directional lighting.
- **Single-face assumption.** Only the largest detected face is classified per frame.

---

## 8. Future Work

Ordered by expected benefit relative to effort:

1. **Data augmentation** — random flips, rotations, zooms and brightness jitter via
   `ImageDataGenerator`. Directly addresses the primary limitation; expected to add
   several points of accuracy.
2. **Class weighting** — recover usable recall on *disgust* and *fear*.
3. **Batch normalisation and deeper regularisation** — faster, more stable convergence and
   reduced overfitting.
4. **Transfer learning** — fine-tune a VGG-16 or ResNet-50 backbone pretrained on
   ImageNet. Published results using this approach reach approximately 70% on FER-2013.
5. **Better face detection** — replace the Haar Cascade with an MTCNN or DNN-based
   detector for robustness to pose and lighting.
6. **Temporal smoothing** — average predictions over a short window of frames to suppress
   the flicker caused by per-frame independent classification.

---

## 9. Conclusion

A complete facial emotion recognition system was designed, trained, evaluated and
deployed. The CNN attains **57.09% accuracy across seven classes on 7,178 unseen test
images**, substantially exceeding the 14.29% random and 24.71% majority-class baselines,
and comparable to typical results for a network of this size trained from scratch on
FER-2013 without augmentation.

Beyond the headline figure, the evaluation established *where* the model succeeds and
fails. High-signal expressions such as *happy* (F1 0.781) and *surprise* (F1 0.712) are
recognised reliably, while subtle expressions and under-represented classes are not —
*disgust* recall of 0.171 traces directly to its 1.5% share of the training data. The
observed confusions cluster along genuine perceptual similarities rather than appearing
at random, and the training curves locate the onset of overfitting at approximately
epoch 8, identifying data augmentation and class weighting as the concrete next steps.

The model was successfully deployed as a browser application streaming live annotated
webcam video at roughly 10 fps. The development process also demonstrated the value of
end-to-end verification — a label-ordering defect that no amount of code inspection had
revealed was caught by comparing predictions against known labels, and correcting it
nearly doubled measured accuracy.

---

## References

1. Goodfellow, I. J. et al. (2013). *Challenges in Representation Learning: A Report on
   Three Machine Learning Contests.* ICML Workshop on Representation Learning.
2. Viola, P. and Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of
   Simple Features.* CVPR.
3. Krizhevsky, A., Sutskever, I. and Hinton, G. (2012). *ImageNet Classification with Deep
   Convolutional Neural Networks.* NeurIPS.
4. Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from
   Overfitting.* JMLR 15(1).
5. Kingma, D. P. and Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR.
6. TensorFlow / Keras documentation — https://www.tensorflow.org/api_docs
7. OpenCV documentation — https://docs.opencv.org

---

## Appendix A — Project Structure

```
Emotion_Recognition_Project/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── training/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate_csv.py
│   ├── evaluate_analysis.py
│   ├── run_emotion_detection.py
│   ├── emotion_model.h5
│   ├── confusion_matrix.png
│   └── training_history.png
│
├── app.py
├── templates/
│   └── index.html
│
├── requirements.txt
├── README.md
└── REPORT.md
```

## Appendix B — Reproducing the Results

```bash
pip install -r requirements.txt

cd training
python preprocessing.py

python train.py

python evaluate_csv.py
python evaluate_analysis.py

cd ..
python app.py
```

The steps install dependencies, preprocess the dataset (regenerating `X_data.npy` and
`y_labels.npy`), train the model (about 19 epochs with early stopping), generate test-set
predictions and metrics, and launch the live web application at `http://localhost:5000`.

**Note.** Training is stochastic — GPU non-determinism and the shuffled batch order mean a
retrained model will not reproduce 57.09% exactly, though it should land within roughly
±1.5 points. The figures reported here correspond to the `emotion_model.h5` committed with
this project.

## Appendix C — Environment

| Component | Version / Detail |
|---|---|
| Python | 3.10 |
| TensorFlow / Keras | ≥ 2.10 |
| OpenCV | `opencv-python` |
| Flask | with `flask-cors` |
| scikit-learn | metrics and stratified splitting |
| Matplotlib / Seaborn | figure generation |
