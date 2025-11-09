import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import os

# Initialize emotion tracking
emotion_counts = {
    'anger': 0, 'neutral': 0, 'disgust': 0,
    'fear': 0, 'happiness': 0, 'sadness': 0, 'surprise': 0
}


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('model.h5')
emotion_dict = {0: 'anger', 1: 'neutral', 2: 'disgust',
                3: 'fear', 4: 'happiness',
                5: 'sadness', 6: 'surprise'}

j = 0

print('\n Select your choice: ')
print('\n 1) Capture live feed using webcam')
print('\n 2) Select a video file ')
print('\n 3) Select a image file ')
print('\n Enter Your Choice :')

choice = int(input('Choice: \n'))

if choice == 1:
    print("Attempting to access webcam...")

    def try_open_camera(max_index=3):
        """Try to open camera indices 0..max_index using several API backends.
        Returns an open cv2.VideoCapture or None.
        """
        backends = [None, cv2.CAP_DSHOW, cv2.CAP_MSMF]
        for idx in range(0, max_index + 1):
            for backend in backends:
                try:
                    if backend is None:
                        cap = cv2.VideoCapture(idx)
                    else:
                        cap = cv2.VideoCapture(idx, backend)

                    if not cap or not cap.isOpened():
                        if cap:
                            cap.release()
                        continue

                    # Try reading a test frame
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Opened camera index {idx} using backend {backend}")
                        return cap
                    else:
                        cap.release()
                except Exception:
                    # ignore and try next
                    try:
                        if cap:
                            cap.release()
                    except Exception:
                        pass
        return None

    cap = try_open_camera(max_index=3)
    if cap is None or not cap.isOpened():
        print("\nError: Could not open any webcam device using OpenCV.")
        print("Quick suggestions:")
        print(" - Close other apps that may be using the camera (Zoom/Teams/browser/Camera app)")
        print(" - Check Windows Settings -> Privacy & Security -> Camera and allow desktop apps to access the camera")
        print(" - Reboot the machine or try running the script as Administrator")
        print(" - Run 'camera_diag.py' (included) to collect detailed backend/index diagnostics")
        print("If you'd like, install ffmpeg and I can add an ffmpeg-based fallback to capture frames when OpenCV fails.")
        exit()

elif choice == 2:
    print("Opening file selection dialog...")
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Make the dialog appear on top
    
    filetypes = [
        ('Video files', '*.mp4 *.avi *.mov *.mkv'),
        ('All files', '*.*')
    ]
    
    path = filedialog.askopenfilename(
        title='Select a video file',
        filetypes=filetypes,
        initialdir=os.path.expanduser('~')  # Start in user's home directory
    )
    
    root.destroy()
    
    if path and os.path.exists(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            exit()
    else:
        print("No file selected")
        exit()

elif choice == 3:
    j = 1
    pass

else:
    print("Invalid choice")
    exit()


def convert_image(image):
    """Normalize and shape an ROI for model prediction.
    Accepts grayscale or BGR images and returns predicted class index.
    """
    # If color image, convert to grayscale
    if image is None:
        raise ValueError("convert_image got None image")
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    pic = cv2.resize(image, (48, 48))
    pic = pic.astype('float32') / 255.0
    pic = pic.reshape(1, 48, 48, 1)
    preds = model.predict(pic)
    ans = int(np.argmax(preds, axis=-1)[0])
    return ans


if j == 0:
    print("\nPress ESC to exit")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Flip for a mirror-like view and convert to grayscale for face detection
            frame_flipped = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                # Draw rectangle on the colored frame (so label is visible)
                cv2.rectangle(frame_flipped, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]

                try:
                    prediction = int(convert_image(roi_gray))
                    emotion = emotion_dict.get(prediction, 'unknown')
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    cv2.putText(frame_flipped, emotion, (x + 20, y - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2, cv2.LINE_AA)
                except Exception as e:
                    # Don't crash the loop on prediction errors
                    print(f"Warning: failed to predict emotion for a face: {e}")

            cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Video', frame_flipped)
            cv2.resizeWindow('Video', 1000, 600)

            if cv2.waitKey(1) == 27:  # press ESC to break
                cap.release()
                cv2.destroyAllWindows()
                break
        else:
            break

else:  # Image mode
    print("Opening file selection dialog...")
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Make the dialog appear on top
    
    filetypes = [
        ('Image files', '*.png *.jpg *.jpeg *.bmp'),
        ('All files', '*.*')
    ]
    
    path = filedialog.askopenfilename(
        title='Select an image file',
        filetypes=filetypes,
        initialdir=os.path.expanduser('~')  # Start in user's home directory
    )
    
    root.destroy()
    
    if path and os.path.exists(path):
        print(f"Selected image: {path}")
        print("Reading image file...")
        image = cv2.imread(path)
        
        if image is None:
            print("Error: Could not read the image file. Please make sure it's a valid image format.")
            exit()
        
        print("Converting image to grayscale...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print("Detecting faces...")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        if len(faces) == 0:
            print("No faces detected in the image. Try a different image with clear, front-facing faces.")
            # Still show the image even if no faces are detected
            cv2.namedWindow('Image', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Image', image)
            cv2.resizeWindow('Image', 1000, 600)
            print("Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            exit()
        
        print(f"Found {len(faces)} faces in the image")
        for (x, y, w, h) in faces:
            print(f"Processing face at position ({x}, {y})")
            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract face region
            roi_gray = gray[y:y + h, x:x + w]
            
            try:
                # Predict emotion
                prediction = int(convert_image(roi_gray))
                emotion = emotion_dict[prediction]
                emotion_counts[emotion] += 1
                
                # Add emotion label
                cv2.putText(image, emotion, (x + 20, y - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                           2, cv2.LINE_AA)
                print(f"Detected emotion: {emotion}")
            except Exception as e:
                print(f"Error processing face: {str(e)}")
        
        print("\nDisplaying image with detected emotions...")
        cv2.namedWindow('Image', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Image', image)
        cv2.resizeWindow('Image', 1000, 600)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No file selected or file selection cancelled")
        exit()

# Print emotion detection summary
print("\nEmotion Detection Summary:")
total_detections = sum(emotion_counts.values())

if total_detections > 0:
    print("\nDetected emotions and their counts:")
    for emotion, count in emotion_counts.items():
        if count > 0:
            percentage = (count / total_detections) * 100
            print(f"{emotion}: {count} times ({percentage:.2f}%)")
else:
    print("No emotions were detected")
