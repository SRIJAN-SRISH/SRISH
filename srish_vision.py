"""Leaf verification and disease detection, with no UI attached.

Two YOLO models, in order:

  gatekeeper  a classifier that answers "is this even a tomato leaf?", so the
              disease model is never asked to find lesions in a photo of a cat
  detector    the trained detector that boxes Leaf Miner damage

Both agentic_bot.py and admin_portal.py use this, so neither carries its own
copy of the thresholds.
"""

import io
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# Weights are gitignored (too large), so these are paths, not bundled files.
GATEKEEPER_PATH = os.environ.get("SRISH_GATEKEEPER_PATH", os.path.join(HERE, "keeper.pt"))
DETECTOR_PATH = os.environ.get("SRISH_DETECTOR_PATH", os.path.join(HERE, "yolo.pt"))

# Below this the gatekeeper is not confident enough to call it a tomato leaf.
LEAF_CONFIDENCE = float(os.environ.get("SRISH_LEAF_CONFIDENCE", "0.65"))
LEAF_CLASS = os.environ.get("SRISH_LEAF_CLASS", "Tomato_Leaf")


@lru_cache(maxsize=1)
def load_gatekeeper():
    """The leaf/not-leaf classifier. Raises if the weights are missing."""
    from ultralytics import YOLO

    if not os.path.exists(GATEKEEPER_PATH):
        raise FileNotFoundError(
            f"Gatekeeper weights not found at {GATEKEEPER_PATH}. "
            "Set SRISH_GATEKEEPER_PATH, or put keeper.pt next to this file."
        )
    return YOLO(GATEKEEPER_PATH)


@lru_cache(maxsize=1)
def load_detector():
    """The Leaf Miner detector. Raises if the weights are missing."""
    from ultralytics import YOLO

    if not os.path.exists(DETECTOR_PATH):
        raise FileNotFoundError(
            f"Detector weights not found at {DETECTOR_PATH}. "
            "Set SRISH_DETECTOR_PATH, or put yolo.pt next to this file."
        )
    return YOLO(DETECTOR_PATH)


def models_status() -> dict:
    """Which weights are present, for a health panel. Never raises."""
    out = {}
    for name, path in (("gatekeeper", GATEKEEPER_PATH), ("detector", DETECTOR_PATH)):
        exists = os.path.exists(path)
        out[name] = {
            "path": path,
            "present": exists,
            "size_mb": round(os.path.getsize(path) / 1e6, 1) if exists else None,
        }
    return out


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bytes_to_pil(raw_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw_bytes))


def check_if_leaf(frame) -> tuple[bool, float, str]:
    """Is this a tomato leaf? Returns (is_leaf, confidence, class_name)."""
    results = load_gatekeeper().predict(source=frame, verbose=False)
    probs = results[0].probs
    top = probs.top1
    confidence = float(probs.top1conf.item())
    class_name = results[0].names[top]
    return (class_name == LEAF_CLASS and confidence >= LEAF_CONFIDENCE), confidence, class_name


def detect_disease(frame) -> dict:
    """Run the detector.

    Returns {found, count, detections, annotated}, where annotated is a BGR
    image with boxes drawn and detections is a list of {label, confidence, box}.
    """
    results = load_detector().predict(source=frame, verbose=False)
    r = results[0]

    detections = []
    for box in r.boxes:
        cls = int(box.cls.item())
        detections.append({
            "label": r.names.get(cls, str(cls)),
            "confidence": round(float(box.conf.item()), 3),
            "box": [round(v, 1) for v in box.xyxy[0].tolist()],
        })

    return {
        "found": bool(detections),
        "count": len(detections),
        "detections": detections,
        "annotated": r.plot(),  # BGR, boxes drawn
    }
