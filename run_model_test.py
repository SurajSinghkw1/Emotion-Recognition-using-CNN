import os
import sys
import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Small test harness to verify model.h5 loads and runs an inference
# Usage:
#   python run_model_test.py           # runs on a synthetic image
#   python run_model_test.py --image path\to\face.jpg

EMOTION_MAP = {0: 'anger', 1: 'neutral', 2: 'disgust',
               3: 'fear', 4: 'happiness', 5: 'sadness', 6: 'surprise'}


def preprocess_image_for_model(img):
    """Take a single-channel or 3-channel image (numpy array),
    convert/resize to 48x48 grayscale normalized to [0,1], and
    return shape (1,48,48,1) ready for model.predict.
    """
    if img is None:
        raise ValueError("Input image is None")

    # If image has 3 channels, convert to grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 48x48
    img_resized = cv2.resize(img, (48, 48))

    arr = img_resized.astype('float32') / 255.0
    arr = arr.reshape(1, 48, 48, 1)
    return arr


def run_on_image(model, img):
    arr = preprocess_image_for_model(img)
    probs = model.predict(arr)
    # probs shape: (1, n_classes)
    idx = int(np.argmax(probs, axis=-1)[0])
    return idx, probs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', help='Path to an image file (optional)')
    parser.add_argument('--model', '-m', default='model.h5', help='Path to model.h5')
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: model file not found at '{model_path}'. Please place model.h5 in the project root or pass --model path.")
        sys.exit(2)

    print(f"Loading model from: {model_path}")
    try:
        model = load_model(model_path)
    except Exception as e:
        print("Failed to load model:", e)
        sys.exit(3)

    if args.image:
        if not os.path.exists(args.image):
            print(f"Image file not found: {args.image}")
            sys.exit(4)
        img = cv2.imread(args.image)
        if img is None:
            print("cv2 failed to read the image. Is it a valid image file?")
            sys.exit(5)
        print(f"Running inference on provided image: {args.image}")
    else:
        # Create a synthetic test image (48x48 gray with small random noise)
        print("No image provided — creating a synthetic test input (48x48) to validate model.")
        img = (np.random.rand(48, 48) * 255).astype('uint8')

    idx, probs = run_on_image(model, img)
    emotion = EMOTION_MAP.get(idx, f'unknown({idx})')
    print(f"Predicted class index: {idx}")
    print(f"Predicted emotion: {emotion}")
    print("Class probabilities:")
    for i, p in enumerate(probs):
        label = EMOTION_MAP.get(i, str(i))
        print(f"  {i} ({label}): {p:.4f}")


if __name__ == '__main__':
    main()
