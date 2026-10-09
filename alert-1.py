
import cv2
import torch
import time
from ultralytics import YOLO


# ==========================================================
# 1. SETTINGS
# ==========================================================

MODEL_PATH = r"C:\Users\HP\OneDrive\Desktop\rtsp_access\door-1.pt"

# Put your actual RTSP URL here
RTSP_URL = "your-link"

# Alert after door remains open for 20 seconds
ALERT_TIME = 20

# YOLO confidence threshold
CONFIDENCE = 0.35

# Resize camera frame
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# Number of consecutive detections needed
# before changing the confirmed state
REQUIRED_CONSECUTIVE_FRAMES = 3


# ==========================================================
# 2. CHECK GPU
# ==========================================================

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    device = 0

else:

    print("⚠️ Running on CPU")

    device = "cpu"


# ==========================================================
# 3. LOAD YOLO MODEL
# ==========================================================

print()
print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("✅ Model loaded")

print("Model classes:", model.names)


# ==========================================================
# 4. OPEN RTSP STREAM
# ==========================================================

print()
print("Connecting to RTSP camera...")

cap = cv2.VideoCapture(RTSP_URL)

# Reduce buffering where supported
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


if not cap.isOpened():

    print("❌ Could not open RTSP stream")

    exit()


print("✅ RTSP stream connected")


# ==========================================================
# 5. DOOR STATE VARIABLES
# ==========================================================

# Confirmed state
confirmed_state = "unknown"

# Candidate state used for stabilization
candidate_state = None

# Number of consecutive frames with candidate state
candidate_count = 0

# Time when confirmed door-open state began
door_open_start = None

# Prevent repeated alerts
alert_sent = False


# ==========================================================
# 6. FPS MEASUREMENT
# ==========================================================

fps_start_time = time.time()
fps_frame_count = 0
display_fps = 0


# ==========================================================
# 7. MAIN LOOP
# ==========================================================

while True:

    # ------------------------------------------------------
    # Read latest camera frame
    # ------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("❌ Failed to read RTSP frame")

        break


    # ------------------------------------------------------
    # Resize frame
    # ------------------------------------------------------

    frame = cv2.resize(
        frame,
        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        )
    )


    # ------------------------------------------------------
    # YOLO DETECTION
    # ------------------------------------------------------

    results = model.predict(

        source=frame,

        conf=CONFIDENCE,

        imgsz=640,

        device=device,

        verbose=False
    )

    result = results[0]


    # ======================================================
    # 8. DETERMINE DOOR STATE
    # ======================================================

    detected_state = None


    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = model.names[class_id]


            # ------------------------------------------------
            # Door open
            # ------------------------------------------------

            if class_name == "door_open":

                detected_state = "door_open"

                break


            # ------------------------------------------------
            # Door closed
            # ------------------------------------------------

            elif class_name == "door_closed":

                detected_state = "door_closed"

                break


    # ======================================================
    # 9. STATE STABILIZATION
    # ======================================================

    if detected_state is not None:

        if detected_state != candidate_state:

            # New candidate state
            candidate_state = detected_state

            candidate_count = 1

        else:

            # Same state again
            candidate_count += 1


        # --------------------------------------------------
        # Confirm state after several consecutive detections
        # --------------------------------------------------

        if candidate_count >= REQUIRED_CONSECUTIVE_FRAMES:

            if confirmed_state != candidate_state:

                confirmed_state = candidate_state


                # ==========================================
                # DOOR OPENED
                # ==========================================

                if confirmed_state == "door_open":

                    door_open_start = time.time()

                    alert_sent = False

                    print()
                    print("🟠 DOOR OPEN")
                    print("⏱️ 20-second timer started")


                # ==========================================
                # DOOR CLOSED
                # ==========================================

                elif confirmed_state == "door_closed":

                    door_open_start = None

                    alert_sent = False

                    print()
                    print("🟢 DOOR CLOSED")
                    print("⏱️ Timer reset")


    # ======================================================
    # 10. CALCULATE OPEN DURATION
    # ======================================================

    open_duration = 0


    if (
        confirmed_state == "door_open"
        and door_open_start is not None
    ):

        open_duration = (
            time.time()
            - door_open_start
        )


        # --------------------------------------------------
        # Print duration
        # --------------------------------------------------

        print(
            f"\rDoor open: "
            f"{open_duration:.1f} seconds",
            end=""
        )


        # ==================================================
        # 11. 20-SECOND ALERT
        # ==================================================

        if (
            open_duration >= ALERT_TIME
            and not alert_sent
        ):

            print()
            print()
            print(
                "🚨🚨🚨 ALERT 🚨🚨🚨"
            )

            print(
                "Door has been open "
                "for more than 20 seconds!"
            )

            alert_sent = True


    # ======================================================
    # 12. DRAW YOLO RESULTS
    # ======================================================

    annotated_frame = result.plot()


    # ======================================================
    # 13. DISPLAY DOOR STATE
    # ======================================================

    if confirmed_state == "door_open":

        state_text = "DOOR OPEN"

    elif confirmed_state == "door_closed":

        state_text = "DOOR CLOSED"

    else:

        state_text = "UNKNOWN"


    cv2.putText(

        annotated_frame,

        state_text,

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0, 255, 255),

        2
    )


    # ======================================================
    # 14. DISPLAY TIMER
    # ======================================================

    if confirmed_state == "door_open":

        timer_text = (
            f"Open: {open_duration:.1f}s"
        )

    else:

        timer_text = "Open: 0.0s"


    cv2.putText(

        annotated_frame,

        timer_text,

        (20, 80),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2
    )


    # ======================================================
    # 15. DISPLAY ALERT
    # ======================================================

    if alert_sent:

        cv2.putText(

            annotated_frame,

            "!!! DOOR OPEN > 20 SECONDS !!!",

            (20, 125),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 0, 255),

            3
        )


    # ======================================================
    # 16. CALCULATE PROCESSING FPS
    # ======================================================

    fps_frame_count += 1

    elapsed = (
        time.time()
        - fps_start_time
    )


    if elapsed >= 1.0:

        display_fps = (
            fps_frame_count / elapsed
        )

        fps_frame_count = 0

        fps_start_time = time.time()


    # ======================================================
    # 17. DISPLAY FPS
    # ======================================================

    cv2.putText(

        annotated_frame,

        f"Processing FPS: {display_fps:.1f}",

        (20, 165),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    # ======================================================
    # 18. SHOW LIVE VIDEO
    # ======================================================

    cv2.imshow(

        "LIVE Door Monitoring",

        annotated_frame
    )


    # ======================================================
    # 19. QUIT
    # ======================================================

    # Press Q to stop

    if cv2.waitKey(1) & 0xFF == ord("q"):

        print()
        print("Stopping...")

        break


# ==========================================================
# 20. CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()

print()
print("==========================================")
print("Live monitoring stopped")
print("==========================================")