from __future__ import annotations

import argparse
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

# Suppress annoying TensorFlow info/warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_PATH = SCRIPT_DIR / "landmark_gesture_recognition.keras"
ANNOTATIONS_DIR = PROJECT_DIR / "gesture_dataset_sample" / "_annotations"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "captures" / f"selfie-{time.strftime('%Y%m%d-%H%M%S')}.jpg"

# Gesture Constants
START_GESTURES = {"like"}
FILTER_GESTURES = {"peace"}
PORTRAIT_GESTURES = {"rock"}
ZOOM_GESTURES = {"two_up"}
STOP_GESTURES = {"stop"}
CANNY_GESTURES = {"dislike"}
INVERT_GESTURES = {"three"}
GRAYSCALE_GESTURES = {"ok"}
NEUTRAL_GESTURE = "fist"


@dataclass(slots=True)
class AppConfig:
    camera_id: int
    output_path: Path
    countdown_seconds: int
    confidence_threshold: float
    history_size: int
    stable_votes: int
    mirror: bool


def load_label_names() -> list[str]:
    label_names = [path.stem for path in sorted(ANNOTATIONS_DIR.glob("*.json"))]
    if not label_names:
        raise FileNotFoundError(f"No annotation files found in {ANNOTATIONS_DIR}")
    return label_names


def normalize_landmarks(landmarks) -> list[float]:
    """Converts 3D MediaPipe landmarks into relative, normalized coordinates.""" 
    #Partially ai generated, but the idea of using the z coordinate and normalizing was my own idea.
    coords = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    base_x, base_y, base_z = coords[0]
    translated = [(x - base_x, y - base_y, z - base_z) for x, y, z in coords]
    max_val = max(max(abs(x), abs(y), abs(z)) for x, y, z in translated)
    max_val = max_val if max_val > 0 else 1.0
    normalized = []
    for x, y, z in translated:
        normalized.extend([x / max_val, y / max_val, z / max_val])
    return normalized


class HandTracker:
    def __init__(self) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_features(self, frame: np.ndarray) -> list[float] | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None
        return normalize_landmarks(results.multi_hand_landmarks[0])

    def close(self) -> None:
        self._hands.close()


# --- Image Filters --- (partially ai generated)
def apply_sepia(frame: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [0.131, 0.534, 0.272],
            [0.168, 0.686, 0.349],
            [0.189, 0.769, 0.393],
        ],
        dtype=np.float32,
    )
    return np.clip(cv2.transform(frame, kernel), 0, 255).astype(np.uint8)

def apply_canny(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_invert(frame: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(frame)

def apply_canny(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lower, upper)
    neon_canvas = np.zeros_like(frame)
    neon_canvas[edges > 0] = [255, 255, 0]
    color_accents = cv2.bitwise_and(frame, frame, mask=edges)
    return cv2.addWeighted(color_accents, 0.4, neon_canvas, 0.6, 0)

def apply_portrait_mode(frame: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (0, 0), 21)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    radius_x = max(1, int(frame.shape[1] * 0.22))
    radius_y = max(1, int(frame.shape[0] * 0.30))
    cv2.ellipse(mask, center, (radius_x, radius_y), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 25) / 255.0
    mask_3 = cv2.merge([mask, mask, mask])
    return np.clip(frame * mask_3 + blurred * (1.0 - mask_3), 0, 255).astype(np.uint8)

def apply_zoom(frame: np.ndarray, zoom_factor: float) -> np.ndarray:
    if zoom_factor <= 1.0:
        return frame
    h, w = frame.shape[:2]
    nw, nh = int(w / zoom_factor), int(h / zoom_factor)
    sx, sy = (w - nw) // 2, (h - nh) // 2
    cropped = frame[sy : sy + nh, sx : sx + nw]
    if cropped.size == 0:
        return frame
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def draw_status(frame: np.ndarray, lines: list[str]) -> np.ndarray: # mostly ai generated
    output = frame.copy()
    panel_height = 32 + 24 * len(lines)
    overlay = output.copy()
    cv2.rectangle(overlay, (12, 12), (min(output.shape[1] - 12, 520), 12 + panel_height), (0, 0, 0), -1)
    output = cv2.addWeighted(overlay, 0.4, output, 0.6, 0)
    for i, line in enumerate(lines):
        position = (24, 40 + i * 24)
        cv2.putText(output, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def resolve_output_path(raw_path: Path) -> Path:
    if raw_path.suffix:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        return raw_path
    raw_path.mkdir(parents=True, exist_ok=True)
    return raw_path / f"selfie-{time.strftime('%Y%m%d-%H%M%S')}.jpg"


def parse_args() -> AppConfig: # ai generated
    parser = argparse.ArgumentParser(description="Gesture-controlled selfie camera")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device index")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Image file or output directory")
    parser.add_argument("--countdown", type=int, default=5, help="Countdown length in seconds")
    parser.add_argument("--threshold", type=float, default=0.75, help="Minimum model confidence for a gesture")
    parser.add_argument("--history-size", type=int, default=12, help="How many predictions to keep for smoothing")
    parser.add_argument("--stable-votes", type=int, default=7, help="Votes required for a stable gesture")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirror view")
    args = parser.parse_args()
    return AppConfig(
        camera_id=args.camera_id,
        output_path=resolve_output_path(args.output),
        countdown_seconds=max(1, args.countdown),
        confidence_threshold=min(max(args.threshold, 0.0), 1.0),
        history_size=max(3, args.history_size),
        stable_votes=max(1, args.stable_votes),
        mirror=not args.no_mirror,
    )


class GestureCameraApp: # camera window and lists for filter generated
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.label_names = load_label_names()
        self.model = load_model(MODEL_PATH, compile=False)
        self._validate_model_shape()
        
        self.hand_tracker = HandTracker()
        self.cap = cv2.VideoCapture(self.config.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.config.camera_id}")
            
        self.history: deque[tuple[str, float]] = deque(maxlen=self.config.history_size)
        self.last_gesture = "none"
        self.zoom_levels = [1.0, 1.25, 1.5]
        self.effects = {
            "sepia": False,
            "portrait": False,
            "canny": False,
            "invert": False,
            "gray": False,
            "zoom_idx": 0,
        }
        self.countdown_ends_at: float | None = None
        self.action_cooldown_until = 0.0
        self.last_saved_path: Path | None = None

    def _validate_model_shape(self) -> None:
        output_classes = int(self.model.output_shape[-1])
        if output_classes != len(self.label_names):
            raise ValueError(
                f"Model outputs {output_classes} classes but found {len(self.label_names)} labels"
            )

    def _classify(self, frame: np.ndarray) -> tuple[str, float]:
        features = self.hand_tracker.get_features(frame)
        if features is None:
            return "none", 0.0
        prediction = self.model.predict(np.asarray([features], dtype=np.float32), verbose=0)[0]
        index = int(np.argmax(prediction))
        return self.label_names[index], float(prediction[index])

    def _stable_gesture(self) -> tuple[str, float] | None: # my idea but the implementation was mostly ai generated
        valid = [(l, c) for l, c in self.history if c >= self.config.confidence_threshold]
        if len(valid) < self.config.stable_votes:
            return None
        label, count = Counter(l for l, c in valid).most_common(1)[0]
        if count < self.config.stable_votes:
            return None
        confidence = max(c for current_label, c in valid if current_label == label)
        return label, confidence

    def _handle_gesture(self, label: str) -> None:
        now = time.monotonic()
        if label == self.last_gesture or now < self.action_cooldown_until:
            return
            
        self.last_gesture = label
        self.action_cooldown_until = now + 0.8
        
        if label in START_GESTURES:
            if self.countdown_ends_at is None:
                self.countdown_ends_at = now + self.config.countdown_seconds
        elif label in FILTER_GESTURES:
            self.effects["sepia"] = not self.effects["sepia"]
        elif label in PORTRAIT_GESTURES:
            self.effects["portrait"] = not self.effects["portrait"]
        elif label in ZOOM_GESTURES:
            self.effects["zoom_idx"] = (self.effects["zoom_idx"] + 1) % len(self.zoom_levels)
        elif label in CANNY_GESTURES:
            self.effects["canny"] = not self.effects["canny"]
        elif label in INVERT_GESTURES:
            self.effects["invert"] = not self.effects["invert"]
        elif label in GRAYSCALE_GESTURES:
            self.effects["gray"] = not self.effects["gray"]
        elif label in STOP_GESTURES:
            self.countdown_ends_at = None
        elif label == NEUTRAL_GESTURE:
            self.last_gesture = NEUTRAL_GESTURE

    def run(self) -> None:
        window_name = "Gesture Camera"
        cv2.namedWindow(window_name)
        try:
            while True:
                # check if the user clicked the 'X' button on the GUI window (ai generated)
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window manually closed. Exiting...")
                    return

                success, frame = self.cap.read()
                if not success:
                    raise RuntimeError("Failed to capture a frame from the camera")

                if self.config.mirror:
                    frame = cv2.flip(frame, 1)

                label, conf = self._classify(frame)
                self.history.append((label, conf))
                stable = self._stable_gesture()
                if stable:
                    self._handle_gesture(stable[0])

                # Apply Processing Pipeline
                img = apply_zoom(frame, self.zoom_levels[self.effects["zoom_idx"]])
                if self.effects["portrait"]:
                    img = apply_portrait_mode(img)
                if self.effects["sepia"]:
                    img = apply_sepia(img)
                if self.effects["canny"]:
                    img = apply_canny(img)
                if self.effects["invert"]:
                    img = apply_invert(img)
                if self.effects["gray"]:
                    img = apply_grayscale(img)

                # Active UI text strings
                status_lines = [
                    f"Gesture: {label} ({conf:0.2f})",
                    f"Zoom: {self.zoom_levels[self.effects['zoom_idx']]:0.2f}x",
                ]

                # Timer Handling
                if self.countdown_ends_at is not None:
                    remaining = self.countdown_ends_at - time.monotonic()
                    if remaining <= 0:
                        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(self.config.output_path), img)
                        self.last_saved_path = self.config.output_path
                        print(f"Saved selfie to {self.config.output_path}")
                        self.countdown_ends_at = None
                    else:
                        status_lines.insert(0, f"Countdown: {remaining:0.1f}s")
                else:
                    path_display = self.last_saved_path if self.last_saved_path else self.config.output_path
                    status_lines.append(f"Output: {path_display.name}")

                img = draw_status(img, status_lines)
                cv2.imshow(window_name, img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return
        finally:
            self.cap.release()
            self.hand_tracker.close()
            cv2.destroyAllWindows()


def main() -> None:
    config = parse_args()
    app = GestureCameraApp(config)
    try:
        app.run()
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")


if __name__ == "__main__":
    main()