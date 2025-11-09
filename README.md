
# Facial-Emotion-Recognition

This repository contains a pre-trained emotion recognition model (`model.h5`) and a small demo script `test.py` that can:

- Capture webcam feed (real-time emotion detection)
- Process a video file
- Process a single image

This README explains how to set up the environment on Windows, install dependencies, run the demo, and troubleshoot common issues (camera access problems).

## Prerequisites

- Windows 10/11
- Python 3.10 - 3.13 (3.13 was used in this workspace)
- At least 8 GB RAM recommended for TensorFlow workloads

## Quick setup (recommended)

Open a Command Prompt in the project folder (the folder containing `test.py`, `model.h5`, etc.) and run:

```cmd
# 1) Create a virtual environment (if you don't have one)
python -m venv .venv

# 2) Activate it (Windows cmd)
.venv\\Scripts\\activate

# 3) Upgrade pip
python -m pip install --upgrade pip

# 4) Install required packages
python -m pip install -r requirements.txt
```

### Notes on versions and compatibility

- TensorFlow: This project used TensorFlow 2.20 on the current Python build. Older TensorFlow (2.3.x) from the original project will not install on newer Python versions; if you need that exact TensorFlow, create the venv using Python 3.8/3.9/3.10 and install the original versions.
- If you need plotting/visualization (`matplotlib`/`seaborn`), installing `matplotlib` on Windows may require Microsoft C++ Build Tools. The demo script has been simplified to avoid mandatory plotting dependencies.

## Optional tools

- ffmpeg (optional): Useful fallback if OpenCV cannot open your camera. Download from https://ffmpeg.org/ and add `ffmpeg.exe` to your PATH.
- Microsoft Visual C++ Build Tools: required only if you want to compile packages like some `matplotlib` wheels locally. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Running the demo

Run the demo script using the venv Python:

```cmd
.venv\\Scripts\\python.exe test.py
```

When prompted choose:
1) Capture live feed using webcam
2) Select a video file
3) Select an image file

- Webcam (option 1): The script attempts to open your system camera. If OpenCV cannot open the camera on your system, see the Troubleshooting section below.
- Video (option 2): A file dialog will open; choose a video file and the script will process it.
- Image (option 3): A file dialog will open; choose an image and the script will process it and display results.

## Troubleshooting — camera access

If the webcam option fails with errors like "Camera index out of range" or OpenCV can't open the camera, try the following steps:

1) Simple checks
- Make sure the Camera app (Windows) can access the camera.
- Close other apps that may use the camera (Teams, Zoom, browsers, Camera app).
- Reboot the machine.
- In Settings → Privacy & security → Camera, ensure:
	- "Camera access" is ON
	- "Let desktop apps access your camera" is ON

2) Run the provided diagnostic script
- We included `camera_diag.py` which tests multiple OpenCV backends and indices. Run it with the same venv Python:

```cmd
.venv\\Scripts\\python.exe camera_diag.py
```

Check the output. If none of the indices open, OpenCV cannot access your camera via UVC. In that case try the ffmpeg test below.

3) Test with ffmpeg (if installed)
- List DirectShow devices:

```cmd
ffmpeg -list_devices true -f dshow -i dummy
```

- Capture one frame with the device name shown (replace the device name accordingly):

```cmd
ffmpeg -f dshow -i video="Integrated Camera" -frames:v 1 out.jpg
```

If ffmpeg can capture, we can implement an ffmpeg-based fallback to the script.

4) Run as Administrator
- Open an elevated command prompt (Run as Administrator) and re-run `camera_diag.py` and `test.py` — permissions sometimes matter for certain drivers.

5) Use an external USB webcam
- External UVC-compliant webcams often work without the driver/middleware complications built-in cameras have.

## Files added/edited

- `test.py` — demo script (updated to be more robust and to avoid optional plotting libs)
- `camera_diag.py` — diagnostic script to test camera indices and backends
- `test_camera.py` and `test_builtin_camera.py` — small helper scripts used during debugging (can be removed)
- `requirements.txt` — list of Python packages used

## If something doesn't work

Tell me the exact error output (copy/paste the terminal output). If camera access is the blocker we can either add a ffmpeg fallback to `test.py` or you can continue testing the model using a video/image file.

## Quick next steps I can do for you

- Add an ffmpeg fallback to `test.py` that uses ffmpeg frames when OpenCV can't open the camera. This requires ffmpeg installed locally.
- Help you run `camera_diag.py` from an elevated prompt and interpret the results.

Thank you — tell me which follow-up you want (ffmpeg fallback, re-run diag as admin, or continue with image/video testing).
 
## Simple README (short)

Facial-Emotion-Recognition — simple demo that detects emotions from faces using a pre-trained Keras model.

Quick steps (Windows cmd):

1) Create & activate a virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2) Install requirements

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3) Run demo (menu: webcam / video / image)

```cmd
python test.py
```

4) Quick model load test (no camera needed)

```cmd
python run_model_test.py
```

If webcam doesn't open on your machine, use the image or video options, run `camera_diag.py` for diagnostics, or install `ffmpeg` and ask for an ffmpeg fallback.

That's it — this README is intentionally minimal. If you want the longer troubleshooting guide again, say so and I'll restore it.
