from __future__ import annotations #ai generated just to test my other model

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.models import load_model
import mediapipe as mp


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Pointing to your second trained model (VGG16 Transfer Learning)
MODEL_PATH = PROJECT_DIR / "03-camera-app" / "landmark_gesture_recognition.keras"
ANNOTATIONS_PATH = SCRIPT_DIR / "annot-name.json"
OUTPUT_PATH = PROJECT_DIR / "02-dataset" / "conf-matrix-v2.png"

IMG_SIZE = 64
# Robust mapping dictionary to automatically clean JSON spelling typos
LABEL_CLEANER = {
    "sspeace": "peace",
    "srock": "rock",
    "sdislike": "dislike"
}

def load_label_names() -> list[str]:
    annotation_dir = PROJECT_DIR / "gesture_dataset_sample" / "_annotations"
    label_names = [path.stem for path in sorted(annotation_dir.glob("*.json"))]
    if not label_names:
        raise FileNotFoundError(f"No annotation files found in {annotation_dir}")
    return label_names

def extract_landmarks_from_crop(crop_img: np.ndarray, hands_detector) -> list[float] | None:
    """Passes the image crop to MediaPipe and extracts wrist-normalized coordinates."""
    # MediaPipe requires RGB images
    img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return None
        
    # Take the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]
    wrist = hand_landmarks.landmark[0]
    
    landmarks = []
    for lm in hand_landmarks.landmark:
        # Normalize coordinates relative to the wrist position (matching Part 3 training)
        landmarks.append(lm.x - wrist.x)
        landmarks.append(lm.y - wrist.y)
        landmarks.append(lm.z - wrist.z)
        
    return landmarks

def main() -> None:
    annotations = json.loads(ANNOTATIONS_PATH.read_text())
    label_names = load_label_names()
    
    print(f"Loading Landmark Model from: {MODEL_PATH.name}...")
    model = load_model(MODEL_PATH)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=1, 
        min_detection_confidence=0.3
    )

    y_true: list[str] = []
    y_pred: list[str] = []
    skipped_count = 0

    for image_id, annotation in annotations.items():
        image_path = SCRIPT_DIR / f"{image_id}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠️ Warning: Could not read image {image_path}. Skipping.")
            continue

        for bbox, raw_label in zip(annotation["bboxes"], annotation["labels"]):
            label = LABEL_CLEANER.get(raw_label, raw_label)
            if label not in label_names:
                continue

            # Crop the bounding box region 
            x1 = max(0, int(bbox[0] * image.shape[1]))
            y1 = max(0, int(bbox[1] * image.shape[0]))
            x2 = min(image.shape[1], x1 + int(bbox[2] * image.shape[1]))
            y2 = min(image.shape[0], y1 + int(bbox[3] * image.shape[0]))
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                crop = image

            # Extract structural 63-feature landmarks instead of resizing pixels
            landmarks = extract_landmarks_from_crop(crop, hands_detector)
            
            # Fallback: If crop was tight, try processing the full image for tracking context
            if landmarks is None:
                landmarks = extract_landmarks_from_crop(image, hands_detector)

            if landmarks is None:
                print(f"❌ MediaPipe failed to locate hand landmarks in {image_id}. Skipping sample.")
                skipped_count += 1
                continue

            # Shape array to (1, 63) to fulfill Dense input layer expectations
            input_data = np.array(landmarks, dtype=np.float32).reshape(1, 63)
            
            prediction = model.predict(input_data, verbose=0)[0]
            predicted_label = label_names[int(np.argmax(prediction))]

            y_true.append(label)
            y_pred.append(predicted_label)

            print(f"📷 {image_id}: True = [{label}] ➡️ Predicted = [{predicted_label}]")

    hands_detector.close()

    if not y_true:
        print("❌ Error: No hand landmarks were detected across your entire dataset. Matrix cannot be generated.")
        return

    # Generate and plot confusion matrix
    matrix = confusion_matrix(y_true, y_pred, labels=label_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=label_names)
    
    display.plot(ax=ax, cmap="Purples", colorbar=False, values_format="d")
    ax.set_title("Landmark DNN Model - Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✅ Success! Saved matrix to {OUTPUT_PATH}")
    if skipped_count > 0:
        print(f"⚠️ Note: {skipped_count} bounding boxes were skipped due to lack of detectable hand landmarks.")

if __name__ == "__main__":
    main()