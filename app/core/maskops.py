"""
Turning SAM's masks into the outlines a YOLO segmentation label needs.

SAM returns a per-instance bitmap; a YOLO seg label is a single normalised
polygon.  Everything that bridges the two lives here — the live auto-labelling
path, the snap-to-object tool and the offline dataset converter all share it,
so an outline means the same thing wherever it came from.

Nothing in here knows about BBox, which keeps it importable from every layer.
"""

from __future__ import annotations

import cv2
import numpy as np

# How closely an outline follows the raw contour, as a fraction of its
# perimeter.  Panels and other straight-edged things collapse to a handful of
# points at "medium"; "fine" keeps ragged edges such as cracks.
DETAIL_LEVELS = {
    "coarse": 0.012,
    "medium": 0.004,
    "fine": 0.0012,
}
DEFAULT_DETAIL = "coarse"

# A YOLO seg label needs at least a triangle.
MIN_POINTS = 3


def detail_epsilon(detail: str | float) -> float:
    if isinstance(detail, (int, float)):
        return float(detail)
    return DETAIL_LEVELS.get(str(detail), DETAIL_LEVELS[DEFAULT_DETAIL])


def to_bool_mask(raw, width: int, height: int) -> np.ndarray | None:
    """Normalise whatever the model handed back into an (H, W) bool array."""
    if raw is None:
        return None
    mask = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)
    mask = mask.astype(bool)
    while mask.ndim > 2:
        mask = mask.squeeze(0)
    if mask.ndim != 2:
        return None
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return mask


def mask_to_polygon(
    mask: np.ndarray,
    width: int,
    height: int,
    *,
    detail: str | float = DEFAULT_DETAIL,
    min_area: float = 4.0,
) -> list[float] | None:
    """
    Outline of the largest blob in *mask*, as flat normalised [x1, y1, x2, y2, …].

    Only the largest contour is kept: a YOLO seg label holds one polygon per
    instance, so a mask that came back in pieces contributes its main piece.
    Returns None when there is nothing worth outlining.
    """
    if mask is None or not mask.any():
        return None

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    epsilon = detail_epsilon(detail) * cv2.arcLength(largest, True)
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    # Simplifying can over-shoot on small or thin shapes; fall back to the
    # raw contour rather than returning something degenerate.
    points = simplified if len(simplified) >= MIN_POINTS else largest
    if len(points) < MIN_POINTS:
        return None

    pts = points.reshape(-1, 2).astype(float)
    pts[:, 0] /= max(width, 1)
    pts[:, 1] /= max(height, 1)
    return pts.clip(0.0, 1.0).flatten().tolist()


def polygon_bounds(polygon: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    """Pixel-space bounding box of a normalised polygon."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    return (min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height)


def mask_iou_with_box(mask: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """
    How well a mask lines up with a drawn box, 0–1.

    Used by snap-to-object: SAM answers a box prompt with every instance it
    thinks fits, and this picks the one the person actually drew around.
    """
    height, width = mask.shape[:2]
    x1 = max(0, min(width, int(box[0])))
    y1 = max(0, min(height, int(box[1])))
    x2 = max(0, min(width, int(box[2])))
    y2 = max(0, min(height, int(box[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    inside = float(mask[y1:y2, x1:x2].sum())
    total = float(mask.sum())
    drawn = float((x2 - x1) * (y2 - y1))
    union = total + drawn - inside
    return inside / union if union > 0 else 0.0
