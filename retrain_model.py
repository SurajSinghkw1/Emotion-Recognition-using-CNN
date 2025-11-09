"""
Script to retrain the emotion recognition model with all fixes applied.
This ensures correct label assignment and improved model architecture.
"""

import pandas as pd
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow.keras.utils as utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D, Conv2D, BatchNormalization
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Nadam

print("=" * 60)
print("Starting Model Retraining with Fixed Code")
print("=" * 60)

# Define the data path
data_path = 'dataset/test'

# Check if directory exists
if not os.path.exists(data_path):
    print(f"Error: The directory {data_path} does not exist.")
    exit(1)

print(f"Successfully found the dataset at {data_path}")
categories = os.listdir(data_path)
print("\nAvailable emotion categories:")
for cat in categories:
    num_images = len(os.listdir(os.path.join(data_path, cat)))
    print(f"- {cat}: {num_images} images")

data_dir_list = os.listdir(data_path)
print(f"\n{len(data_dir_list)} classes: {data_dir_list}")

# Load images and assign labels correctly
print("\nLoading images and assigning labels...")
img_data = []
labels = []

# Map emotion names to class indices
emotion_to_class = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

label_text = {0:'angry', 1:'disgust', 2:'fear',
              3:'happy', 4:'neutral', 5:'sad', 
              6:'surprise'}

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print(f'Loading {len(img_list)} images from {dataset}...')
    class_idx = emotion_to_class[dataset]
    for img in img_list:
        img_path = data_path + '/' + dataset + '/' + img
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (48, 48))
            img_data.append(img)
            labels.append(class_idx)

# Convert to numpy arrays
img_data = np.array(img_data)
labels = np.array(labels, dtype='int64')

print(f"\nTotal images loaded: {len(img_data)}")
print(f"Image shape: {img_data[0].shape}")
print(f"\nLabel distribution:")
for i, emotion in label_text.items():
    count = np.sum(labels == i)
    print(f"  {emotion}: {count} images")

# Normalize images
print("\nNormalizing images...")
img_data = img_data.astype('float32')
img_data = img_data / 255.0

# Convert labels to categorical
num_classes = len(label_text)
Y = utils.to_categorical(labels, num_classes)
print(f"\nCategorical labels shape: {Y.shape}")

# Shuffle data
print("\nShuffling data...")
x, y = shuffle(img_data, Y, random_state=3)
print(f"Shuffled data shape: {x.shape}")
print(f"Shuffled labels shape: {y.shape}")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Compute class weights
print("\nComputing class weights to handle imbalance...")
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:")
for i, emotion in label_text.items():
    print(f"  {emotion}: {class_weight_dict[i]:.3f}")

# Create data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

# Create improved model
def create_model(optim):
    input_shape = (48, 48, 3)
    
    model = Sequential()
    
    # First Conv Block
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second Conv Block
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Third Conv Block
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Dense Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optim)
    
    return model

# Create callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,
    patience=8,  # Reduced patience for faster training
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    early_stopping,
    lr_scheduler,
]

# Training parameters
batch_size = 32
epochs = 50  # Reduced for faster training (early stopping will likely stop earlier)
optimizer = Adam(learning_rate=0.001)

# Create model
print("\n" + "=" * 60)
print("Creating model with improved architecture...")
print("=" * 60)
model = create_model(optimizer)
model.summary()

# Prepare data generators
val_size = int(len(X_train) * 0.1)
X_val = X_train[-val_size:]
y_val = y_train[-val_size:]
X_train_final = X_train[:-val_size]
y_train_final = y_train[:-val_size]

train_generator = aug.flow(X_train_final, y_train_final, batch_size=batch_size, shuffle=True)
val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)

print(f"\nTraining samples: {len(X_train_final)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Train model
print("\n" + "=" * 60)
print("Starting Training...")
print("=" * 60)
print("This may take a while. Please be patient...\n")

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train_final) // batch_size,
    validation_data=val_generator,
    validation_steps=len(X_val) // batch_size,
    epochs=epochs,
    verbose=2,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# Evaluate on test set
print("\n" + "=" * 60)
print("Evaluating on test set...")
print("=" * 60)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate classification report
from sklearn.metrics import confusion_matrix, classification_report
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test, batch_size=64, verbose=0), axis=1)

print("\n" + "=" * 60)
print("Classification Report")
print("=" * 60)
label_names = [label_text[i] for i in range(len(label_text))]
print(classification_report(y_true, y_pred, target_names=label_names))

# Save model
print("\n" + "=" * 60)
print("Saving model...")
print("=" * 60)
if os.path.isfile('model.h5'):
    # Backup old model
    import shutil
    shutil.copy('model.h5', 'model.h5.backup')
    print("Backed up old model to model.h5.backup")

model.save('model.h5')
print("Model saved successfully to model.h5")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Final Test Accuracy: {test_accuracy:.4f}")
print("Model has been saved to model.h5")
print("You can now use this model with app.py")

