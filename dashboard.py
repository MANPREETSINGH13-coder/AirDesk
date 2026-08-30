import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp


# ============================================================
# AirDesk AI - Professional Dashboard
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"


# ============================================================
# Gesture Detection
# ============================================================

def distance(a, b):
    import math

    return math.hypot(
        a.x - b.x,
        a.y - b.y
    )


def finger_extended(landmarks, tip, pip):

    wrist = landmarks[0]

    tip_distance = distance(
        landmarks[tip],
        wrist
    )

    pip_distance = distance(
        landmarks[pip],
        wrist
    )

    return tip_distance > pip_distance * 1.12


def detect_gesture(landmarks):

    index = finger_extended(
        landmarks, 8, 6
    )

    middle = finger_extended(
        landmarks, 12, 10
    )

    ring = finger_extended(
        landmarks, 16, 14
    )

    pinky = finger_extended(
        landmarks, 20, 18
    )

    extended = sum([
        index,
        middle,
        ring,
        pinky
    ])

    # Pinch

    thumb_index = distance(
        landmarks[4],
        landmarks[8]
    )

    palm = distance(
        landmarks[0],
        landmarks[9]
    )

    if palm > 0:

        if thumb_index / palm < 0.55:
            return "PINCH"

    # Thumbs Up

    if (
        landmarks[4].y < landmarks[3].y
        and landmarks[4].y < landmarks[0].y
        and extended == 0
    ):
        return "THUMBS_UP"

    # Open Palm

    if extended == 4:
        return "OPEN_PALM"

    # Fist

    if extended == 0:
        return "FIST"

    # Peace

    if (
        index
        and middle
        and not ring
        and not pinky
    ):
        return "PEACE"

    # Point

    if (
        index
        and not middle
        and not ring
        and not pinky
    ):
        return "POINT"

    return "UNKNOWN"


# ============================================================
# Model
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)


# ============================================================
# Camera
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    raise RuntimeError(
        "Could not access the camera."
    )


# ============================================================
# State
# ============================================================

history = deque(maxlen=8)

active_gesture = "NO_HAND"

control_enabled = False

start_time = time.perf_counter()

frame_count = 0


# ============================================================
# Dashboard
# ============================================================

print()
print("==========================================")
print("             AIRDESK AI")
print("        Intelligent Air Control")
print("==========================================")
print()
print("Dashboard started.")
print("Press Q to quit.")
print()
print("==========================================")


try:

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            # ----------------------------------------------------
            # MediaPipe
            # ----------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(
                (
                    time.perf_counter()
                    - start_time
                ) * 1000
            )

            result = landmarker.detect_for_video(
                image,
                timestamp
            )

            # ----------------------------------------------------
            # Gesture
            # ----------------------------------------------------

            active_gesture = "NO_HAND"

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                raw = detect_gesture(
                    landmarks
                )

                history.append(raw)

                if len(history) >= 5:

                    latest = history[-1]

                    if all(
                        item == latest
                        for item in history
                    ):

                        active_gesture = latest

                else:

                    active_gesture = raw

                # ------------------------------------------------
                # Control state
                # ------------------------------------------------

                if active_gesture == "THUMBS_UP":

                    control_enabled = True

                elif active_gesture in (
                    "OPEN_PALM",
                    "FIST"
                ):

                    control_enabled = False

                # ------------------------------------------------
                # Draw hand
                # ------------------------------------------------

                connections = (
                    mp.tasks.vision
                    .HandLandmarksConnections
                    .HAND_CONNECTIONS
                )

                for landmark in landmarks:

                    x = int(
                        landmark.x * width
                    )

                    y = int(
                        landmark.y * height
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

                for connection in connections:

                    start = landmarks[
                        connection.start
                    ]

                    end = landmarks[
                        connection.end
                    ]

                    p1 = (
                        int(start.x * width),
                        int(start.y * height)
                    )

                    p2 = (
                        int(end.x * width),
                        int(end.y * height)
                    )

                    cv2.line(
                        frame,
                        p1,
                        p2,
                        (255, 255, 255),
                        2
                    )

            else:

                history.clear()
                control_enabled = False

            # ====================================================
            # FPS
            # ====================================================

            frame_count += 1

            elapsed = (
                time.perf_counter()
                - start_time
            )

            fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0
            )

            # ====================================================
            # Dashboard Overlay
            # ====================================================

            # Header

            cv2.rectangle(
                frame,
                (0, 0),
                (width, 170),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                frame,
                "AIRDESK AI",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Intelligent Air Control",
                (27, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (190, 190, 190),
                1
            )

            # Gesture

            cv2.putText(
                frame,
                f"GESTURE: {active_gesture}",
                (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0),
                2
            )

            # Status

            status = (
                "CONTROL ACTIVE"
                if control_enabled
                else "SAFE / PAUSED"
            )

            cv2.putText(
                frame,
                f"STATUS: {status}",
                (25, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

            # FPS

            cv2.putText(
                frame,
                f"FPS {fps:.1f}",
                (width - 120, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # ----------------------------------------------------
            # Bottom controls panel
            # ----------------------------------------------------

            panel_top = height - 170

            cv2.rectangle(
                frame,
                (0, panel_top),
                (width, height),
                (15, 15, 15),
                -1
            )

            cv2.putText(
                frame,
                "GESTURE CONTROLS",
                (25, panel_top + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

            controls = [
                "POINT   -> Cursor",
                "PINCH   -> Click",
                "PEACE   -> Scroll",
                "OPEN    -> Pause",
                "FIST    -> Stop",
                "THUMB   -> Enable",
            ]

            y = panel_top + 58

            for item in controls:

                cv2.putText(
                    frame,
                    item,
                    (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (210, 210, 210),
                    1
                )

                y += 22

            # ----------------------------------------------------
            # Camera
            # ----------------------------------------------------

            cv2.imshow(
                "AirDesk AI - Dashboard",
                frame
            )

            # Q

            if cv2.waitKey(1) & 0xFF == ord("q"):

                break


finally:

    camera.release()

    cv2.destroyAllWindows()

    print()
    print("AirDesk AI Dashboard stopped.")
