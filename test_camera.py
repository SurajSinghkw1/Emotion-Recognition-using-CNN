import cv2

def test_camera():
    print("Testing camera access...")
    
    # Try DirectShow first
    print("\nTrying DirectShow...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("Successfully accessed camera with DirectShow!")
            cap.release()
            return True
        cap.release()
    
    # Try default API
    print("\nTrying default API...")
    cap = cv2.VideoCapture(0)
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("Successfully accessed camera with default API!")
            cap.release()
            return True
        cap.release()
    
    print("\nCould not access camera. Please check:")
    print("1. Is your webcam connected?")
    print("2. Open Windows Camera app to test if it works there")
    print("3. Check Windows Settings > Privacy & Security > Camera")
    print("4. Check Device Manager for camera issues")
    return False

if __name__ == "__main__":
    test_camera()