# Real-Time Door Open Monitoring and Alert System

## Overview

This project implements a computer-vision-based door monitoring system using **YOLO object detection, OpenCV, PyTorch, and an RTSP CCTV camera stream**.

The system detects whether a monitored door is open or closed. It includes a real-time monitoring script that tracks how long the door remains open and triggers an alert when the confirmed open duration reaches **20 seconds**.

The project also includes two video-testing scripts that run the trained YOLO model on a previously recorded CCTV video:

- `alert.py` — Live RTSP monitoring with door-open duration tracking and alerts.
- `speed.py` — Door detection on a recorded video after resizing frames to 960 × 540 to reduce processing overhead.
- `record.py` — Door detection on the same recorded video using the original frame resolution.

This setup makes it possible to test the model on recorded footage, compare the effect of frame resizing, and run the alert system on a live camera stream.

## Features

- **Live CCTV monitoring:** Processes frames from an RTSP camera stream using `alert.py`.
- **Recorded-video testing:** Runs detection on `door_record.mp4` using `speed.py` and `record.py`.
- **Door-state detection:** Uses custom-trained YOLO weights to detect `door_open` and `door_closed`.
- **20-second alert:** Triggers an alert when the confirmed door-open duration reaches the configured threshold.
- **State stabilization:** In `alert.py`, requires multiple consecutive detections before confirming a state change.
- **Open-duration tracking:** Tracks how long the confirmed door-open state persists.
- **Automatic timer reset:** Resets the timer when the door-closed state is confirmed.
- **Live visualization:** Displays detection boxes and class labels on the video feed.
- **Frame resizing:** `speed.py` resizes video frames to 960 × 540 to reduce the number of pixels processed and displayed.
- **Resolution comparison:** `record.py` processes frames without explicitly resizing them before YOLO inference.
- **CPU/GPU support:** `speed.py` checks for CUDA availability and selects a GPU when available, otherwise using the CPU.

## Project Structure

```text
rtsp_access/
├── alert-1.py          # Live RTSP monitoring and 20-second alert logic
├── speed.py          # Detection on recorded video with resized frames
├── record.py         # Detection on recorded video at original frame resolution
├── door-1.pt         # Trained YOLO model weights (local file)
├── door_record.mp4   # Recorded CCTV video used for offline testing
├── README.md         # Project documentation
└── .venv/            # Optional Python virtual environment
```

The model weights, recorded video, camera credentials, and virtual environment may be kept locally rather than committed to GitHub. Only share these files if you have permission and intend to distribute them.

## Technology Stack

- Python
- Ultralytics YOLO
- OpenCV
- PyTorch
- RTSP video streaming
- Recorded MP4 video processing
- Python `time` module for duration tracking in `alert.py`

## Prerequisites

- Python 3.10 or a version compatible with your installed dependencies.
- A trained YOLO model with the expected classes `door_open` and `door_closed`.
- An accessible RTSP-enabled CCTV camera for live monitoring.
- A recorded video named `door_record.mp4` for offline testing with `speed.py` and `record.py`.
- An NVIDIA GPU is optional; CPU inference is supported.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_PRIVATE_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your private repository URL and folder name.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install ultralytics opencv-python torch
```

For NVIDIA GPU acceleration, install a PyTorch version compatible with your CUDA environment using the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Configuration

Before running the scripts, ensure that the model and video paths point to the correct files on your computer.

| Setting | Purpose |
|---|---|
| `MODEL_PATH` or the model path in `YOLO()` | Location of the trained weights, `door-1.pt` |
| `RTSP_URL` | Live camera address used by `alert.py` |
| `video_path` | Recorded MP4 file used by `speed.py` and `record.py` |
| `ALERT_TIME` | Door-open alert threshold in `alert.py`; currently 20 seconds |
| `CONFIDENCE` | Detection confidence threshold in `alert.py`, if configured; the video scripts use `conf=0.35` |
| `DISPLAY_WIDTH` | Resized frame width in `speed.py`: 960 pixels |
| `DISPLAY_HEIGHT` | Resized frame height in `speed.py`: 540 pixels |
| `REQUIRED_CONSECUTIVE_FRAMES` | Consecutive detections needed to confirm a state in `alert.py`; currently 3 |

**Important:** The class names in the trained model must match the names expected by the scripts. Check `model.names` if detections are not being interpreted correctly.

The example Windows paths in the scripts should be replaced with the paths on your own machine. Using relative paths can make the project easier to run on different computers.

## Script 1: `alert.py` — Live Monitoring with Alerts

`alert.py` is the main monitoring script. It connects to a live RTSP CCTV stream and continuously analyzes incoming frames.

### How it works

1. Connects to the configured RTSP camera.
2. Loads the trained YOLO model.
3. Processes incoming frames and detects the door state.
4. Requires three consecutive detections of a candidate state before confirming a state change.
5. Starts the timer when `door_open` is confirmed.
6. Tracks the elapsed time while the door remains confirmed open.
7. Prints an alert in the terminal and displays it on the video when the open duration reaches 20 seconds.
8. Resets the timer when `door_closed` is confirmed.

### Run

```bash
python alert-1.py
```

The program displays the live camera feed with the detected door state, elapsed open time, alert status, and processing FPS, depending on the annotations implemented in the script.

Press **Q** while the video window is focused to exit.

## Script 2: `speed.py` — Faster Detection on Recorded Video

`speed.py` runs the trained YOLO model on the recorded file `door_record.mp4`. Unlike `alert.py`, it does not connect to a live RTSP stream and does not implement the 20-second alert timer.

The script resizes every frame to **960 × 540** before passing it to the model and displaying the results.

### Why resize the frames?

The recorded CCTV video has an original resolution of 2560 × 1440. Resizing it to 960 × 540 reduces the number of pixels in each frame substantially.

| Property | Original frame | Resized frame |
|---|---:|---:|
| Width | 2560 px | 960 px |
| Height | 1440 px | 540 px |
| Total pixels | 3,686,400 | 518,400 |

The resized frame contains approximately **86% fewer pixels** than the original frame.

This can reduce the overhead associated with frame processing and display. However, actual speed depends on the hardware, YOLO inference settings, model architecture, and video decoding performance. Resizing alone does not guarantee a particular FPS improvement.

### Processing pipeline

1. Loads `door-1.pt`.
2. Opens `door_record.mp4`.
3. Checks whether CUDA is available and selects the inference device.
4. Resizes each frame to 960 × 540.
5. Runs YOLO inference with `conf=0.35` and `imgsz=640`.
6. Draws bounding boxes and class labels.
7. Displays the annotated video.
8. Continues until the video ends or the user presses Q.

### Run

```bash
python speed.py
```

Use this script when you want to test door detection on recorded footage while reducing the displayed and processed frame dimensions.

## Script 3: `record.py` — Detection at Original Frame Resolution

`record.py` also processes `door_record.mp4`, but it does **not explicitly resize frames before inference**.

The script reads each frame at its original decoded resolution, passes it to YOLO, and displays the annotated result.

### Processing pipeline

1. Loads the trained YOLO weights.
2. Opens `door_record.mp4`.
3. Reads each frame at its original resolution.
4. Runs YOLO inference with `conf=0.35` and `imgsz=640`.
5. Draws bounding boxes and labels.
6. Displays the annotated video.
7. Stops when the video ends or the user presses Q.

### Run

```bash
python record.py
```

### Important distinction

Although `record.py` passes the original frame to YOLO, the setting `imgsz=640` means Ultralytics normally resizes and letterboxes the input internally for inference. Therefore, `record.py` does not necessarily run the model at the full 2560 × 1440 inference resolution.

The main difference is that `speed.py` first resizes the frame to 960 × 540, while `record.py` supplies the original frame to the inference pipeline.

## Comparing the Three Scripts

| Feature | `alert.py` | `speed.py` | `record.py` |
|---|---|---|---|
| Video source | Live RTSP stream | Recorded MP4 | Recorded MP4 |
| Input frame resizing | As configured in the script | 960 × 540 | No explicit resizing |
| YOLO inference | Yes | Yes | Yes |
| Door-state stabilization | Yes, if configured | No | No |
| Open-duration timer | Yes | No | No |
| 20-second alert | Yes | No | No |
| CUDA device selection | Depends on implementation | Explicitly checks CUDA | Uses Ultralytics default device |
| Primary purpose | Continuous monitoring | Faster offline testing | Offline detection comparison |

The three scripts serve different purposes. Use `alert.py` for live door monitoring and threshold-based alerts. Use `speed.py` and `record.py` to evaluate detections on the same recorded video and compare the impact of resizing.

## Understanding the Inference Settings

Both offline scripts use:

```python
conf=0.35
imgsz=640
```

- **`conf=0.35`:** Filters out detections below the selected confidence threshold. Lowering the threshold may reveal additional detections but can also increase false positives.
- **`imgsz=640`:** Sets the target inference image size used by YOLO. It is distinct from the dimensions used to display the video.

When evaluating detection quality, compare missed detections and false positives alongside FPS. A faster script is not necessarily better if it misses doors or misclassifies their states.

## Timer and Alert Logic in `alert.py`

The timer applies to the confirmed door state, not to the two offline testing scripts.

- **Door closed:** The timer is inactive or reset.
- **Door open:** The timer starts when the state is confirmed.
- **Open for less than 20 seconds:** Monitoring continues without a threshold alert.
- **Open for 20 seconds or longer:** The alert is triggered.
- **Door closes:** The timer resets for the next opening.

The timer uses elapsed wall-clock time. Because the state is confirmed only after consecutive detections, the timer starts after confirmation rather than necessarily at the exact physical moment the door begins opening.

## Current Limitations

- Detection quality depends on the model, training data, camera angle, lighting, and visibility of the door.
- Resizing may remove visual details needed to detect small or partially visible doors.
- Processing speed depends on hardware, inference settings, and video decoding overhead.
- `speed.py` and `record.py` are offline detection scripts; they do not trigger the 20-second door-open alert.
- The consecutive-frame rule in `alert.py` may delay confirmation of a state change.
- The alert currently appears in the terminal and live video window. Email, SMS, and external notifications are not implemented.
- The system does not verify access-card events or determine whether a door opening was authorized.
- The system has not been established as a certified security or life-safety system.

## Future Improvements

- Evaluate model accuracy using labeled frames from the target CCTV camera.
- Measure and compare FPS for `speed.py` and `record.py`.
- Log door-state transitions and alert timestamps.
- Add configurable alert thresholds.
- Integrate access-control records to distinguish authorized from potentially unauthorized openings.
- Add email, SMS, or dashboard notifications.
- Improve RTSP reconnection, frame buffering, and continuous-monitoring reliability.
- Consider exporting the model to a supported optimized inference format if additional performance is required.

## Security and Privacy

Do not commit private RTSP URLs, camera usernames or passwords, access credentials, or sensitive CCTV footage to GitHub.

Keep `door-1.pt` and `door_record.mp4` private if they contain proprietary model weights or confidential footage. If you need to share a demonstration, consider using an appropriately anonymized or non-sensitive sample video.

## Disclaimer

This project is a prototype for computer-vision-based door-state monitoring. Its performance and reliability should be evaluated under real operating conditions before it is used in a security-critical environment.

