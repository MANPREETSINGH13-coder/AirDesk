import math
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import pyautogui


# ============================================================
# AIRDESK AI - STEP 6
# Unified Integration
# Camera + Hand Tracking + Gestures + Controls + Dashboard
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# ============================================================
# Screen
# ============================================================

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

cursor_x = SCREEN_WIDTH / 2
cursor_y = SCREEN_HEIGHT / 2


# ============================================================
# Safety / Timing
# ============================================================

SMOOTHING_FRAMES = 5
CLICK_COOLDOWN = 0.8
SCROLL_COOLDOWN = 0.15

last_click_time = 0
last_scroll_time = 0

control_enabled = False

gesture_history = deque(
    maxlen=SMOOTHING_FRAMES
)

active_gesture = "NO_HAND"


# ============================================================
# Gesture Utilities
# ============================================================

def distance(a, b):
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

    # PINCH
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

    # THUMBS UP
    if (
        landmarks[4].y < landmarks[3].y
        and landmarks[4].y < landmarks[0].y
        and extended == 0
    ):
        return "THUMBS_UP"

    # OPEN PALM
    if extended == 4:
        return "OPEN_PALM"

    # FIST
    if extended == 0:
        return "FIST"

    # PEACE
    if (
        index
        and middle
        and not ring
        and not pinky
    ):
        return "PEACE"

    # POINT
    if (
        index
        and not middle
        and not ring
        and not pinky
    ):
        return "POINT"

    return "UNKNOWN"


def smooth_gesture(raw_gesture):

    global active_gesture

    gesture_history.append(raw_gesture)

    if len(gesture_history) == SMOOTHING_FRAMES:

        latest = gesture_history[-1]

        if all(
            item == latest
            for item in gesture_history
        ):
            active_gesture = latest

    return active_gesture


# ============================================================
# MediaPipe
# ============================================================

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
# Startup
# ============================================================

print()
print("==========================================")
print("              AIRDESK AI")
print("        Unified Vision Controller")
print("==========================================")
print()
print(f"MediaPipe : {mp.__version__}")
print(f"Screen    : {SCREEN_WIDTH} x {SCREEN_HEIGHT}")
print()
print("Gestures:")
print("  THUMBS_UP -> Enable controls")
print("  OPEN_PALM -> Safety pause")
print("  FIST      -> Stop controls")
print("  POINT     -> Move cursor")
print("  PINCH     -> Left click")
print("  PEACE     -> Scroll")
print()
print("Press Q to quit.")
print("==========================================")
print()


start_time = time.perf_counter()
frame_count = 0


# ============================================================
# Main Application
# ============================================================

try:

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            # ------------------------------------------------
            # MediaPipe image
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Detection
            # ------------------------------------------------

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                raw_gesture = detect_gesture(
                    landmarks
                )

                gesture = smooth_gesture(
                    raw_gesture
                )

                # --------------------------------------------
                # Safety state
                # --------------------------------------------

                if gesture == "THUMBS_UP":

                    control_enabled = True

                elif gesture in (
                    "OPEN_PALM",
                    "FIST"
                ):

                    control_enabled = False

                # --------------------------------------------
                # Cursor
                # --------------------------------------------

                if (
                    control_enabled
                    and gesture == "POINT"
                ):

                    index_tip = landmarks[8]

                    target_x = (
                        index_tip.x
                        * SCREEN_WIDTH
                    )

                    target_y = (
                        index_tip.y
                        * SCREEN_HEIGHT
                    )

                    target_x = (
                        SCREEN_WIDTH
                        - target_x
                    )

                    cursor_x += (
                        target_x - cursor_x
                    ) * 0.35

                    cursor_y += (
                        target_y - cursor_y
                    ) * 0.35

                    cursor_x = max(
                        0,
                        min(
                            SCREEN_WIDTH - 1,
                            cursor_x
                        )
                    )

                    cursor_y = max(
                        0,
                        min(
                            SCREEN_HEIGHT - 1,
                            cursor_y
                        )
                    )

                    pyautogui.moveTo(
                        int(cursor_x),
                        int(cursor_y),
                        duration=0
                    )

                # --------------------------------------------
                # Pinch click
                # --------------------------------------------

                if (
                    control_enabled
                    and gesture == "PINCH"
                ):

                    now = time.perf_counter()

                    if (
                        now - last_click_time
                        > CLICK_COOLDOWN
                    ):

                        pyautogui.click()

                        last_click_time = now

                # --------------------------------------------
                # Peace scroll
                # --------------------------------------------

                if (
                    control_enabled
                    and gesture == "PEACE"
                ):

                    now = time.perf_counter()

                    if (
                        now - last_scroll_time
                        > SCROLL_COOLDOWN
                    ):

                        wrist_y = landmarks[0].y
                        middle_y = landmarks[12].y

                        difference = (
                            wrist_y - middle_y
                        )

                        if difference > 0.08:

                            pyautogui.scroll(-1)

                        elif difference < -0.08:

                            pyautogui.scroll(1)

                        last_scroll_time = now

                # --------------------------------------------
                # Hand landmarks
                # --------------------------------------------

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

                    start_point = landmarks[
                        connection.start
                    ]

                    end_point = landmarks[
                        connection.end
                    ]

                    p1 = (
                        int(start_point.x * width),
                        int(start_point.y * height)
                    )

                    p2 = (
                        int(end_point.x * width),
                        int(end_point.y * height)
                    )

                    cv2.line(
                        frame,
                        p1,
                        p2,
                        (255, 255, 255),
                        2
                    )

            else:

                gesture_history.clear()
                active_gesture = "NO_HAND"
                control_enabled = False

            # =================================================
            # FPS
            # =================================================

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

            # =================================================
            # Dashboard
            # =================================================

            status = (
                "CONTROL ACTIVE"
                if control_enabled
                else "SAFE / PAUSED"
            )

            cv2.rectangle(
                frame,
                (0, 0),
                (width, 165),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                frame,
                "AIRDESK AI",
                (25, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Unified Vision Controller",
                (27, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (190, 190, 190),
                1
            )

            cv2.putText(
                frame,
                f"GESTURE: {active_gesture}",
                (25, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"STATUS: {status}",
                (25, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (width - 120, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # =================================================
            # Bottom controls
            # =================================================

            panel_top = height - 150

            cv2.rectangle(
                frame,
                (0, panel_top),
                (width, height),
                (15, 15, 15),
                -1
            )

            cv2.putText(
                frame,
                "CONTROLS",
                (25, panel_top + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            controls = [
                "POINT -> Cursor",
                "PINCH -> Click",
                "PEACE -> Scroll",
                "OPEN -> Pause",
                "FIST -> Stop",
                "THUMB -> Enable"
            ]

            y = panel_top + 58

            for item in controls:

                cv2.putText(
                    frame,
                    item,
                    (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (210, 210, 210),
                    1
                )

                y += 21

            # =================================================
            # Window
            # =================================================

            cv2.imshow(
                "AirDesk AI - Unified Controller",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


finally:

    camera.release()

    cv2.destroyAllWindows()

    print()
    print("==========================================")
    print("AirDesk AI stopped safely.")
    print("==========================================")

