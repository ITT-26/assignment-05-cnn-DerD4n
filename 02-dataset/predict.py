from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.models import load_model


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_DIR / "01-hyperparameters" / "gesture_recognition.keras"
ANNOTATIONS_PATH = SCRIPT_DIR / "annot-name.json"
OUTPUT_PATH = PROJECT_DIR / "02-dataset" / "conf-matrix.png"

IMG_SIZE = 64


def load_label_names() -> list[str]:
	annotation_dir = PROJECT_DIR / "gesture_dataset_sample" / "_annotations"
	label_names = [path.stem for path in sorted(annotation_dir.glob("*.json"))]
	if not label_names:
		raise FileNotFoundError(f"No annotation files found in {annotation_dir}")
	return label_names


def preprocess_crop(image: np.ndarray, bbox: list[float]) -> np.ndarray:
	x1 = max(0, int(bbox[0] * image.shape[1]))
	y1 = max(0, int(bbox[1] * image.shape[0]))
	x2 = min(image.shape[1], x1 + int(bbox[2] * image.shape[1]))
	y2 = min(image.shape[0], y1 + int(bbox[3] * image.shape[0]))

	crop = image[y1:y2, x1:x2]
	if crop.size == 0:
		crop = image

	crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
	crop = crop.astype(np.float32) / 255.0
	return crop


def main() -> None:
	annotations = json.loads(ANNOTATIONS_PATH.read_text())
	label_names = load_label_names()
	model = load_model(MODEL_PATH)

	y_true: list[str] = []
	y_pred: list[str] = []

	for image_id, annotation in annotations.items():
		image_path = SCRIPT_DIR / f"{image_id}.jpg"
		image = cv2.imread(str(image_path))
		if image is None:
			raise FileNotFoundError(f"Could not read image {image_path}")

		for bbox, label in zip(annotation["bboxes"], annotation["labels"]):
			processed = preprocess_crop(image, bbox)
			prediction = model.predict(processed[np.newaxis, ...], verbose=0)[0]
			predicted_label = label_names[int(np.argmax(prediction))]

			y_true.append(label)
			y_pred.append(predicted_label)

			print(f"{image_id}: true={label} predicted={predicted_label}")

	matrix = confusion_matrix(y_true, y_pred, labels=label_names)

	fig, ax = plt.subplots(figsize=(10, 8))
	display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=label_names)
	display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
	ax.set_title("Gesture Classification Confusion Matrix")
	plt.xticks(rotation=45, ha="right")
	plt.tight_layout()
	fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
	plt.close(fig)

	print(f"Saved confusion matrix to {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
