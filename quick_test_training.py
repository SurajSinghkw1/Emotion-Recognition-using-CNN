"""
Quick test script to verify the training setup works correctly.
This trains for only 3 epochs to test the pipeline.
"""

import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.utils as utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

print("Quick Test Training - 3 epochs only")
print("=" * 50)

# Load data
data_path = 'dataset/test'
data_dir_list = os.listdir(data_path)

emotion_to_class = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 
                    'neutral': 4, 'sad': 5, 'surprise': 6}
label_text = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 
              4:'neutral', 5:'sad', 6:'surprise'}

img_data = []
labels = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    class_idx = emotion_to_class[dataset]
    for img in img_list:
        img_path = data_path + '/' + dataset + '/' + img
        img = cv2.imread(img_path)
        if img is not None:
            img_data.append(cv2.resize(img, (48, 48)))
            labels.append(class_idx)

img_data = np.array(img_data).astype('float32') / 255.0
labels = np.array(labels, dtype='int64')

Y = utils.to_categorical(labels, len(label_text))
x, y = shuffle(img_data, Y, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights))

# Simple model
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(48, 48, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.001))

# Data augmentation
aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.15, 
                        height_shift_range=0.15, horizontal_flip=True, fill_mode='nearest')
train_gen = aug.flow(X_train, y_train, batch_size=32, shuffle=True)

print(f"Training on {len(X_train)} samples for 3 epochs...")
model.fit(train_gen, steps_per_epoch=len(X_train)//32, epochs=3, verbose=1, 
          class_weight=class_weight_dict)

# Quick test
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nQuick Test Results:")
print(f"Test Accuracy: {test_acc:.4f}")

# Check predictions
y_pred = np.argmax(model.predict(X_test[:10], verbose=0), axis=1)
y_true = np.argmax(y_test[:10], axis=1)
print(f"\nSample predictions (first 10):")
for i in range(10):
    print(f"  True: {label_text[y_true[i]]}, Predicted: {label_text[y_pred[i]]}")

print("\n✓ Training pipeline works! Now run retrain_model.py for full training.")

