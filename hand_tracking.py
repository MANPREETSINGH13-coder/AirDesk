import math
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"


class GestureEngine:
    def __init__(self, smoothing_frames=5):
        self.history = deque(maxlen=smoothing_frames)
        self.current_gesture = "UNKNOWN"

    @staticmethod
    def distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def finger_extended(landmarks, tip, pip):
        wrist = landmarks[0]

        tip_distance = GestureEngine.distance(
            landmarks[tip], wrist
        )

        pip_distance = GestureEngine.distance(
            landmarks[pip], wrist
        )

        return tip_distance > pip_distance * 1.12

    def detect_raw(self, landmarks):

        index = self.finger_extended(landmarks, 8, 6)
        middle = self.finger_extended(landmarks, 12, 10)
        ring = self.finger_extended(landmarks, 16, 14)
        pinky = self.finger_extended(landmarks, 20, 18)

        extended = sum([index, middle, ring, pinky])

        thumb_index = self.distance(
            landmarks[4], landmarks[8]
        )

        palm = self.distance(
            landmarks[0], landmarks[9]
        )

        if palm > 0 and thumb_index / palm < 0.55:
            return "PINCH"

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        wrist = landmarks[0]

        if (
            thumb_tip.y < thumb_ip.y
            and thumb_tip.y < wrist.y
            and extended == 0
        ):
            return "THUMBS_UP"

        if extended == 4:
            return "OPEN_PALM"

        if extended == 0:
            return "FIST"

        if index and middle and not ring and not pinky:
            return "PEACE"

        if index and not middle and not ring and not pinky:
            return "POINT"

        return "UNKNOWN"

    def update(self, landmarks):

        gesture = self.detect_raw(landmarks)

        self.history.append(gesture)

        if len(self.history) == self.history.maxlen:
            latest = self.history[-1]

            if all(
                item == latest
                for item in self.history
            ):
                self.current_gesture = latest

        return self.current_gesture

    def stability(self):

        if not self.history:
            return 0

        matches = sum(
            item == self.current_gesture
            for item in self.history
        )

        return int(
            matches / len(self.history) * 100
        )


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not access the camera.")


gesture_engine = GestureEngine(5)

start_time = time.perf_counter()
frame_count = 0


print("================================")
print("       AirDesk AI")
print("================================")
print(f"MediaPipe: {mp.__version__}")
print("Gesture engine started.")
print("Press Q to quit.")
print("================================")


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
                (time.perf_counter() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                image,
                timestamp
            )

            gesture = "NO_HAND"
            stability = 0

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                gesture = gesture_engine.update(
                    landmarks
                )

                stability = gesture_engine.stability()

                connections = (
                    mp.tasks.vision
                    .HandLandmarksConnections
                    .HAND_CONNECTIONS
                )

                for hand in result.hand_landmarks:

                    for landmark in hand:

                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 255, 0),
                            -1
                        )

                    for connection in connections:

                        start = hand[connection.start]
                        end = hand[connection.end]

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

            frame_count += 1

            elapsed = time.perf_counter() - start_time

            fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0
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
                f"FPS: {fps:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Stability: {stability}%",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "AirDesk AI - Vision Engine",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("AirDesk AI stopped.")
