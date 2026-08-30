import math
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import pyautogui


# ============================================================
# AirDesk AI - Air Control
# Step 4: Gesture -> Computer Control
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

# Screen size
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Safety settings
SMOOTHING_FRAMES = 5
CLICK_COOLDOWN = 0.8
SCROLL_COOLDOWN = 0.15

# Cursor smoothing
cursor_x = SCREEN_WIDTH / 2
cursor_y = SCREEN_HEIGHT / 2

last_click_time = 0
last_scroll_time = 0

gesture_history = deque(
    maxlen=SMOOTHING_FRAMES
)

active_gesture = "UNKNOWN"


# ============================================================
# Utility
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

        pinch_ratio = thumb_index / palm

        if pinch_ratio < 0.55:
            return "PINCH"

    # Thumbs up
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    wrist = landmarks[0]

    if (
        thumb_tip.y < thumb_ip.y
        and thumb_tip.y < wrist.y
        and extended == 0
    ):
        return "THUMBS_UP"

    # Open palm
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


def smooth_gesture(gesture):

    global active_gesture

    gesture_history.append(gesture)

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
# Startup
# ============================================================

print()
print("==========================================")
print("          AirDesk AI Control")
print("==========================================")
print(f"Screen: {SCREEN_WIDTH} x {SCREEN_HEIGHT}")
print()
print("Controls:")
print("  THUMBS_UP  -> Enable control")
print("  OPEN_PALM  -> Safety pause")
print("  POINT      -> Move cursor")
print("  PINCH      -> Left click")
print("  PEACE      -> Scroll")
print("  FIST       -> Stop controls")
print()
print("Press Q in the camera window to quit.")
print("==========================================")
print()


control_enabled = False

start_time = time.perf_counter()
frame_count = 0


# ============================================================
# Main Loop
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

            gesture = "NO_HAND"

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                raw_gesture = detect_gesture(
                    landmarks
                )

                gesture = smooth_gesture(
                    raw_gesture
                )

                # =================================================
                # Control State
                # =================================================

                if gesture == "THUMBS_UP":

                    control_enabled = True

                elif gesture == "OPEN_PALM":

                    control_enabled = False

                elif gesture == "FIST":

                    control_enabled = False

                # =================================================
                # Cursor Control
                # =================================================

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

                    # Mirror compensation
                    target_x = SCREEN_WIDTH - target_x

                    # Smooth cursor movement
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

                # =================================================
                # Pinch -> Click
                # =================================================

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

                # =================================================
                # Peace -> Scroll
                # =================================================

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

                # =================================================
                # Draw landmarks
                # =================================================

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

                gesture_history.clear()
                active_gesture = "UNKNOWN"

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
            # UI
            # ====================================================

            mode = (
                "CONTROL ACTIVE"
                if control_enabled
                else "SAFE / PAUSED"
            )

            cv2.putText(
                frame,
                "AirDesk AI",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Mode: {mode}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "AirDesk AI - Air Control",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


finally:

    camera.release()
    cv2.destroyAllWindows()

    # Always release mouse state safely
    pyautogui.PAUSE = 0

    print()
    print("AirDesk AI Control stopped.")
