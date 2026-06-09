from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
from keras import callbacks, layers, losses, models, optimizers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "gesture_dataset_sample"
ANNOTATIONS_DIR = DATA_DIR / "_annotations"
OUTPUT_MODEL_PATH = SCRIPT_DIR / "landmark_gesture_recognition.keras"
OUTPUT_LABELS_PATH = SCRIPT_DIR / "landmark_gesture_labels.json"

# Now using x, y, AND z coordinates (21 landmarks * 3)
NUM_FEATURES = 63 
SEED = 42

def log_step(message: str) -> None:
    print(message, flush=True)

def normalize_landmarks(landmarks) -> list[float]:
    """Converts 3D MediaPipe landmarks into relative, normalized coordinates."""
    coords = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    
    # 1. Translate relative to wrist (point 0)
    base_x, base_y, base_z = coords[0]
    translated = [(x - base_x, y - base_y, z - base_z) for x, y, z in coords]
    
    # 2. Scale normalize using max distance from wrist
    max_val = max(max(abs(x), abs(y), abs(z)) for x, y, z in translated)
    max_val = max_val if max_val > 0 else 1.0
        
    normalized = []
    for x, y, z in translated:
        normalized.extend([x / max_val, y / max_val, z / max_val])
    return normalized

def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]: # mostly ai generated (using the MP Hands code and image alteration was my own idea!)
    label_names = sorted([path.stem for path in ANNOTATIONS_DIR.glob("*.json")])
    features_list, labels = [], []
    
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for label_index, label_name in enumerate(label_names):
            image_dir = DATA_DIR / label_name
            log_step(f"  -> Extracting landmarks for '{label_name}'...")
            
            for image_path in image_dir.glob("*.jpg"):
                image = cv2.imread(str(image_path))
                if image is None: continue
                
                # Data Augmentation: Original, Flipped, and Slightly Rotated
                for angle in [0, 10, -10]:
                    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1.0)
                    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                    
                    for img in [rotated, cv2.flip(rotated, 1)]:
                        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        if results.multi_hand_landmarks:
                            features_list.append(normalize_landmarks(results.multi_hand_landmarks[0]))
                            labels.append(label_index)

    return np.asarray(features_list, dtype=np.float32), np.asarray(labels, dtype=np.int32), label_names

def build_model(num_classes: int) -> models.Model: 
    model = models.Sequential([
        layers.Input(shape=(NUM_FEATURES,)),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a landmark-based gesture classifier")
    parser.add_argument("--epochs", type=int, default=150, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_step("[1/6] processing images through MediaPipe...")
    X, y, label_names = load_dataset()
    log_step(f"[1/6] dataset loaded: {len(X)} samples across {len(label_names)} classes")

    log_step("[2/6] splitting train/test data")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=SEED, stratify=y
    )
    log_step(f"[2/6] train samples: {len(x_train)}, test samples: {len(x_test)}")

    log_step("[3/6] converting labels to one-hot vectors")
    y_train_cat = to_categorical(y_train, num_classes=len(label_names))
    y_test_cat = to_categorical(y_test, num_classes=len(label_names))

    log_step("[4/6] building the model")
    model = build_model(len(label_names))
    model.summary()
    
    training_callbacks = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    log_step(f"[5/6] training model for up to {args.epochs} epoch(s)")
    model.fit(
        x_train,
        y_train_cat,
        validation_data=(x_test, y_test_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=training_callbacks,
        verbose=1, # <--- Turned the Keras progress bar back on!
        shuffle=True,
    )

    log_step("[6/6] evaluating and saving model")
    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    log_step(f"Final Test loss: {test_loss:.4f}")
    log_step(f"Final Test accuracy: {test_accuracy:.4f}")

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_MODEL_PATH)
    OUTPUT_LABELS_PATH.write_text(json.dumps(label_names, indent=2), encoding="utf-8")
    log_step(f"Saved model to {OUTPUT_MODEL_PATH}")
    log_step(f"Saved labels to {OUTPUT_LABELS_PATH}")


if __name__ == "__main__":
    main()