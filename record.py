import cv2
from ultralytics import YOLO

# --------------------------------------------------
# 1. Load your trained door detection model
# --------------------------------------------------

model = YOLO(
    r"C:\Users\HP\OneDrive\Desktop\rtsp_access\door-1.pt"
)

# --------------------------------------------------
# 2. Recorded RTSP video
# --------------------------------------------------

video_path = r"C:\Users\HP\OneDrive\Desktop\rtsp_access\door_record.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Could not open video")
    exit()

print("✅ Video opened successfully")

# --------------------------------------------------
# 3. Read and process the video
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("✅ Video finished")
        break

    # --------------------------------------------------
    # Run YOLO
    # --------------------------------------------------

    results = model.predict(
        source=frame,
        conf=0.35,
        imgsz=640,
        verbose=False
    )

    # --------------------------------------------------
    # Draw detections
    # --------------------------------------------------

    annotated_frame = results[0].plot()

    # --------------------------------------------------
    # Show video
    # --------------------------------------------------

    cv2.imshow("Door Detection", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# 4. Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Program finished")