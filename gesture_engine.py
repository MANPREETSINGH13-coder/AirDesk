import math
from collections import deque


class GestureEngine:
    """
    Converts MediaPipe hand landmarks into stable gestures.

    Supported gestures:
        OPEN_PALM
        FIST
        POINT
        PEACE
        THUMBS_UP
        PINCH
        UNKNOWN
    """

    def __init__(self, smoothing_frames=5):
        self.history = deque(maxlen=smoothing_frames)
        self.current_gesture = "UNKNOWN"

    @staticmethod
    def distance(a, b):
        """Euclidean distance between two landmarks."""
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def finger_extended(landmarks, tip, pip):
        """
        Basic finger-extension test.

        A finger is considered extended when its tip is farther
        from the wrist than its PIP joint.
        """
        wrist = landmarks[0]

        tip_distance = GestureEngine.distance(landmarks[tip], wrist)
        pip_distance = GestureEngine.distance(landmarks[pip], wrist)

        return tip_distance > pip_distance * 1.12

    def detect_raw(self, landmarks):
        """Detect gesture from one frame."""

        # Finger landmark indexes:
        # Thumb: 4
        # Index: 8
        # Middle: 12
        # Ring: 16
        # Pinky: 20

        index_extended = self.finger_extended(landmarks, 8, 6)
        middle_extended = self.finger_extended(landmarks, 12, 10)
        ring_extended = self.finger_extended(landmarks, 16, 14)
        pinky_extended = self.finger_extended(landmarks, 20, 18)

        extended_count = sum([
            index_extended,
            middle_extended,
            ring_extended,
            pinky_extended,
        ])

        # --------------------------------------------------
        # PINCH
        # --------------------------------------------------

        thumb_index_distance = self.distance(
            landmarks[4],
            landmarks[8],
        )

        palm_size = self.distance(
            landmarks[0],
            landmarks[9],
        )

        if palm_size > 0:
            normalized_pinch = thumb_index_distance / palm_size

            if normalized_pinch < 0.55:
                return "PINCH"

        # --------------------------------------------------
        # THUMBS UP
        # --------------------------------------------------

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        wrist = landmarks[0]

        thumb_up = (
            thumb_tip.y < thumb_ip.y
            and thumb_tip.y < wrist.y
        )

        other_fingers_folded = extended_count == 0

        if thumb_up and other_fingers_folded:
            return "THUMBS_UP"

        # --------------------------------------------------
        # OPEN PALM
        # --------------------------------------------------

        if extended_count == 4:
            return "OPEN_PALM"

        # --------------------------------------------------
        # FIST
        # --------------------------------------------------

        if extended_count == 0:
            return "FIST"

        # --------------------------------------------------
        # PEACE
        # --------------------------------------------------

        if (
            index_extended
            and middle_extended
            and not ring_extended
            and not pinky_extended
        ):
            return "PEACE"

        # --------------------------------------------------
        # POINT
        # --------------------------------------------------

        if (
            index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended
        ):
            return "POINT"

        return "UNKNOWN"

    def update(self, landmarks):
        """
        Add the raw gesture to history and return the
        temporally-smoothed gesture.
        """

        raw_gesture = self.detect_raw(landmarks)

        self.history.append(raw_gesture)

        # Require the same gesture repeatedly before activating it.
        if len(self.history) >= self.history.maxlen:

            latest = self.history[-1]

            if all(
                gesture == latest
                for gesture in self.history
            ):
                self.current_gesture = latest

        return self.current_gesture

    def stability(self):
        """Return stability percentage for current gesture."""

        if not self.history:
            return 0

        matches = sum(
            gesture == self.current_gesture
            for gesture in self.history
        )

        return int(
            (matches / len(self.history)) * 100
        )