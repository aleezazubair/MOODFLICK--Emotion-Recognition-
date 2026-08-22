import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Get the training directory
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))

# Load preprocessed data
X = np.load(os.path.join(TRAINING_DIR, "X_data.npy"))
y = np.load(os.path.join(TRAINING_DIR, "y_labels.npy"))

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y, num_classes=7)

# 3️⃣ Build CNN Model
model = Sequential()

# Convolution + Pooling Layer 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D((2,2)))

# Convolution + Pooling Layer 2
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

# Convolution + Pooling Layer 3
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 emotions

# 4️⃣ Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("CNN Model Created Successfully!")
model.summary()
