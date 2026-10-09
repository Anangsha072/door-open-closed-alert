import cv2
import torch
from ultralytics import YOLO

# ==========================================================
# 1. Check GPU
# ==========================================================

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    device = 0
else:
    print("⚠️ Running on CPU")
    device = "cpu"


# ==========================================================
# 2. Load YOLO model
# ==========================================================

model = YOLO(
    r"C:\Users\HP\OneDrive\Desktop\rtsp_access\door-1.pt"
)


# ==========================================================
# 3. Video / RTSP source
# ==========================================================

video_path = r"C:\Users\HP\OneDrive\Desktop\rtsp_access\door_record.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Could not open video")
    exit()

print("✅ Video opened successfully")


# ==========================================================
# 4. Target resolution
# ==========================================================

# Original video:
# 2560 × 1440
#
# We resize it to:
# 1280 × 720

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


# ==========================================================
# 5. Process video
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("✅ Video finished")
        break

    # ------------------------------------------------------
    # Resize frame
    # ------------------------------------------------------

    frame = cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    )

    # ------------------------------------------------------
    # YOLO detection
    # ------------------------------------------------------

    results = model.predict(
        source=frame,
        conf=0.35,
        imgsz=640,
        device=device,
        verbose=False
    )

    # ------------------------------------------------------
    # Draw detections
    # ------------------------------------------------------

    annotated_frame = results[0].plot()

    # ------------------------------------------------------
    # Display
    # ------------------------------------------------------

    cv2.imshow(
        "Door Detection",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================================
# 6. Cleanup
# ==========================================================

cap.release()
cv2.destroyAllWindows()

print("Program finished")