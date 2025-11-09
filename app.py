import os
import io
import base64
import tempfile
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import traceback


app = Flask(__name__)


# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')
model = load_model(MODEL_PATH)


# IMPORTANT: This must match the label_text in model_training.ipynb
# Training uses: {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad', 6:'surprise'}
EMOTION_MAP = {0: 'angry', 1: 'disgust', 2: 'fear',
               3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
EMOJI_MAP = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😮'
}


# Helpers


def preprocess_image_for_model(img: np.ndarray):
    """Preprocess image to the model's expected input shape.

    The function inspects the loaded model's input shape and converts the provided
    OpenCV image (BGR or grayscale) to the correct channel layout (grayscale or RGB),
    resizes to 48x48, normalizes, and returns a batch array.
    """
    if img is None:
        raise ValueError("Image is None")

    # Determine expected number of channels from model input if possible
    try:
        expected_channels = int(model.input_shape[-1])
    except Exception:
        # Default to 1 (grayscale) if model shape not available
        expected_channels = 1

    # Resize first to avoid color-conversion issues
    img_resized = cv2.resize(img, (48, 48))

    # If model expects single channel (grayscale)
    if expected_channels == 1:
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        elif len(img_resized.shape) == 2:
            img_gray = img_resized
        else:
            raise ValueError("Unsupported image shape for grayscale conversion")
        arr = img_gray.astype('float32') / 255.0
        arr = arr.reshape(1, 48, 48, 1)
        return arr

    # If model expects 3 channels (RGB)
    if expected_channels == 3:
        if len(img_resized.shape) == 2:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Unsupported image shape for RGB conversion")
        arr = img_rgb.astype('float32') / 255.0
        arr = arr.reshape(1, 48, 48, 3)
        return arr

    # Fallback: try grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) if len(img_resized.shape) == 3 else img_resized
    arr = img_gray.astype('float32') / 255.0
    arr = arr.reshape(1, 48, 48, 1)
    return arr


def predict_from_image(img: np.ndarray):
    """Predict emotion from image with test-time augmentation."""
    try:
        # Log input image shape
        app.logger.info(f"Input image shape: {img.shape}")
        
        # Preprocess original image
        arr = preprocess_image_for_model(img)
        app.logger.info(f"Preprocessed shape: {arr.shape}")

        # Get prediction on original
        preds = model.predict(arr, verbose=0)

        # Test-time augmentation: predict on horizontal flip and average
        try:
            flipped = cv2.flip(img, 1)
            arr_f = preprocess_image_for_model(flipped)
            preds_f = model.predict(arr_f, verbose=0)
            # Average predictions for more robust results
            preds = (preds + preds_f) / 2.0
        except Exception:
            # if flip fails, use original prediction
            pass

        # Get predicted emotion
        idx = int(np.argmax(preds, axis=-1)[0])
        emotion = EMOTION_MAP.get(idx, 'unknown')
        score = float(np.max(preds))
        
        # Log all probabilities for debugging
        probs_dict = {EMOTION_MAP.get(i, 'unknown'): float(preds[0][i]) for i in range(len(EMOTION_MAP))}
        app.logger.info(f"Predicted emotion: {emotion} (score: {score:.3f})")
        app.logger.info(f"All probabilities: {probs_dict}")
        
        return emotion, score, preds[0].tolist()
        
    except Exception as e:
        app.logger.error(f"Error in predict_from_image: {str(e)}")
        raise


def _decode_image_bytes(in_memory: bytes):
    """Try decode with OpenCV; if it fails fall back to Pillow and return BGR ndarray or None."""
    # First attempt: OpenCV
    try:
        nparr = np.frombuffer(in_memory, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception as e:
        app.logger.debug("cv2.imdecode raised: %s\n%s", e, traceback.format_exc())

    # Fallback: PIL (handles more formats and EXIF orientation)
    try:
        pil = Image.open(io.BytesIO(in_memory))
        pil = ImageOps.exif_transpose(pil)  # handle EXIF orientation if present
        pil = pil.convert('RGB')
        arr = np.asarray(pil)  # RGB
        # convert RGB -> BGR for downstream OpenCV operations
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        app.logger.debug("PIL fallback failed: %s\n%s", e, traceback.format_exc())
        return None


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    """Receive a base64-encoded image from the client (webcam snapshot)."""
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'no image supplied'}), 400
    b64 = data['image']
    # Cut off header if present
    if ',' in b64:
        b64 = b64.split(',', 1)[1]
    try:
        b = base64.b64decode(b64)
        nparr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': 'failed to decode image', 'detail': str(e)}), 400

    if img is None:
        return jsonify({'error': 'failed to decode image (webcam payload)'}), 400

    name = data.get('name', 'Unknown')
    emotion, score, probs = predict_from_image(img)
    return jsonify({'name': name, 'emotion': emotion, 'emoji': EMOJI_MAP.get(emotion, ''), 'score': score, 'probs': probs})


@app.route('/predict_image', methods=['POST'])
def predict_image():
    """Handle an uploaded image file via form-data."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read raw bytes (do not rely on filename extension)
        in_memory = f.read()
        app.logger.info("Received file '%s' mimetype=%s size=%d", f.filename, f.mimetype, len(in_memory) if in_memory else 0)

        # Try decoding robustly
        img = _decode_image_bytes(in_memory)
        if img is None:
            # Last-ditch: maybe the client sent a data URL inside the file content
            try:
                s = in_memory.decode('utf-8', errors='ignore')
                if s.startswith('data:') and ',' in s:
                    b64 = s.split(',', 1)[1]
                    b = base64.b64decode(b64)
                    img = _decode_image_bytes(b)
            except Exception:
                pass

        if img is None:
            app.logger.debug("Failed to decode uploaded image '%s' (size=%d)", f.filename, len(in_memory))
            return jsonify({'error': 'Failed to decode image. Please ensure it is a valid image file (supported: PNG, JPEG, BMP, GIF, etc.)'}), 400
        
        # Get prediction
        name = request.form.get('name', 'Unknown')
        emotion, score, probs = predict_from_image(img)
        
        return jsonify({
            'name': name,
            'emotion': emotion,
            'emoji': EMOJI_MAP.get(emotion, ''),
            'score': score,
            'probs': probs
        })
        
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 400


@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Accept an uploaded video, sample frames and return aggregated emotion counts."""
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'no selected file'}), 400
    name = request.form.get('name', 'Unknown')

    # Save to a temporary file to let OpenCV read it
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(f.filename)[1])
    try:
        with os.fdopen(tmp_fd, 'wb') as tmp:
            tmp.write(f.read())

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({'error': 'failed to open video file'}), 400
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        # Improved sampling: sample roughly twice per second (every 0.5s) to increase observations
        SAMPLE_SECONDS = 0.5
        sample_step = max(1, int(round(fps * SAMPLE_SECONDS)))

        # Use Haar cascade to detect faces
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

        counts = {}
        probs_accum = None
        weight_sum = 0.0
        frames_considered = 0
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # sample every sample_step frames
            if i % sample_step != 0:
                i += 1
                continue

            try:
                # detect faces on a grayscale copy for speed
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

                # For each detected face, predict emotion
                for (x, y, w, h) in faces:
                    pad = int(0.15 * w)
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(frame.shape[1], x + w + pad)
                    y2 = min(frame.shape[0], y + h + pad)
                    face_roi = frame[y1:y2, x1:x2]

                    emotion, score, probs = predict_from_image(face_roi)
                    if score < 0.30:
                        continue

                    counts[emotion] = counts.get(emotion, 0) + 1
                    p = np.array(probs, dtype=float)
                    w_conf = float(score)
                    if probs_accum is None:
                        probs_accum = p * w_conf
                    else:
                        probs_accum += p * w_conf
                    weight_sum += w_conf
                    frames_considered += 1
            except Exception:
                pass
            i += 1
        cap.release()

        total = frames_considered
        if total == 0:
            return jsonify({'name': name, 'counts': counts, 'message': 'no faces/valid frames detected'})

        if probs_accum is not None and weight_sum > 0:
            avg_probs_arr = (probs_accum / weight_sum)
            avg_probs = avg_probs_arr.tolist()
            top_idx = int(np.argmax(avg_probs_arr))
        else:
            avg_probs = None
            top_idx = None
        top_emotion = EMOTION_MAP.get(top_idx, None) if top_idx is not None else None

        response = {
            'name': name,
            'counts': counts,
            'frames_considered': int(total),
            'top_emotion': top_emotion,
            'emoji': EMOJI_MAP.get(top_emotion, '') if top_emotion else '',
            'avg_probs': avg_probs
        }
        return jsonify(response)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)