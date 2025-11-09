import cv2
import time

def test_camera_with_index(index, label):
    print(f"\nTrying {label} (index {index})...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use DirectShow for better compatibility on Windows
    if not cap.isOpened():
        print(f"Failed to open camera at index {index}")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for attempt in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"Success with {label}!")
            cv2.imshow(f'{label} Frame', frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            cap.release()
            return True
        time.sleep(0.5)

    print(f"Failed to read frame from {label}")
    cap.release()
    return False

def test_builtin_camera():
    print("Testing built-in webcam access...")

    # Try common indices
    for i in range(3):
        if test_camera_with_index(i, f"Camera Index {i}"):
            return True

    print("\nBuilt-in webcam test failed. Please check:")
    print("1. Press Windows + I")
    print("2. Go to Privacy & Security -> Camera")
    print("3. Make sure Camera access is ON")
    print("4. Make sure Let apps access your camera is ON")
    print("5. Try restarting your computer")
    return False

if __name__ == "__main__":
    test_builtin_camera()
