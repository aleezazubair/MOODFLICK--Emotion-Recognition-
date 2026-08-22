import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Get the training directory
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(TRAINING_DIR, "test_predictions.csv")

# Check if CSV exists
if not os.path.exists(csv_file):
    print(f"Error: CSV file not found at {csv_file}")
    print("Please run evaluate_csv.py first to generate predictions")
    exit()

# Load CSV
df = pd.read_csv(csv_file)

# Get true and predicted labels
y_true = df['True Label']
y_pred = df['Predicted Label']

# Corrected emotion order
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Calculate accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=EMOTIONS, target_names=EMOTIONS))

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=EMOTIONS,
            yticklabels=EMOTIONS,
            cmap="Blues",
            cbar_kws={'label': 'Number of Predictions'})
plt.xlabel("Predicted Emotion", fontsize=12)
plt.ylabel("True Emotion", fontsize=12)
plt.title(f"Confusion Matrix - Overall Accuracy: {acc*100:.2f}%", fontsize=14, fontweight='bold')
plt.tight_layout()

# Save confusion matrix
save_path = os.path.join(TRAINING_DIR, "confusion_matrix.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nConfusion Matrix saved at: {save_path}")
plt.show()
