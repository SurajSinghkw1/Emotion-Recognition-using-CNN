import cv2
import time

backends = [
    ("DEFAULT", None),
    ("CAP_DSHOW", cv2.CAP_DSHOW),
    ("CAP_MSMF", cv2.CAP_MSMF),
    ("CAP_ANY", cv2.CAP_ANY),
]

indices = range(0, 6)

print("OpenCV version:", cv2.__version__)
print()

# Print a short summary of video I/O support from build info
try:
    info = cv2.getBuildInformation()
    print("\nVideo I/O and GUI support:")
    for line in info.splitlines():
        if line.strip().startswith("Video I/O") or line.strip().startswith("GUI: "):
            print(line)
except Exception as e:
    print("Could not get build info:", e)

print('\nTesting camera indices and backends (this may take ~20s)...\n')

results = []
for name, backend in backends:
    for idx in indices:
        cap = None
        try:
            print(f"Trying Backend={name:8} Index={idx}...")
            if backend is None:
                cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx, backend)

            time.sleep(0.3)
            opened = cap.isOpened()
            frame_ok = False
            if opened:
                for _ in range(3):  # Try multiple reads
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame_ok = True
                        break

            results.append((name, idx, opened, frame_ok))

            print(f"  opened={opened}  frame_ok={frame_ok}")
            if frame_ok:
                cv2.imshow(f'{name} Index {idx}', frame)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append((name, idx, False, False))
        finally:
            if cap:
                cap.release()

print('\nSummary of working configurations:')
for name, idx, opened, frame_ok in results:
    if opened or frame_ok:
        print(f"✅ Backend={name:8} Index={idx} opened={opened} frame_ok={frame_ok}")

print('\nIf none were OK, try:')
print(" - Close other apps that might use the camera (Zoom, Teams, Camera app)")
print(" - Run the script from an elevated terminal (Run as Administrator)")
print(" - Ensure 'Let desktop apps access your camera' is allowed in privacy settings")
print(" - Check Device Manager for camera index and driver status")
print(" - Try updating OpenCV or using OpenCV compiled with Media Foundation support")
