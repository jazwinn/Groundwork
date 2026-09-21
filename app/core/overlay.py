"""Drawing labels onto a frame, so a picture of the work can be looked at.

The browser draws its own overlay on a canvas, which is no use to anything that
is not a browser.  Judging a label is a visual job — whether an outline follows
the object, whether one object has been cut in two — and a list of corner
coordinates does not answer it.  This draws the same thing server-side, so a
script, a terminal, or an agent reading the picture back can see what a person
looking at the editor would see.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.sam3_handler import BBox, FrameAnnotation

# The class palette the editor uses, in the order it uses it.  Kept the same so
# a rendered frame and the browser agree about which colour means which class.
CLASS_COLORS = [
    "#7c9cff", "#f0a03c", "#3fb27f", "#e2564d", "#c78bff",
    "#39c0c7", "#e5d15b", "#ef7fae", "#8fce5a", "#b08968",
]
HELD_OUT_COLOR = "#00e676"


def _bgr(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def class_color(class_id: int) -> tuple[int, int, int]:
    return _bgr(CLASS_COLORS[class_id % len(CLASS_COLORS)])


def _points(box: BBox, width: int, height: int) -> np.ndarray | None:
    if not box.polygon or len(box.polygon) < 6:
        return None
    flat = box.polygon
    pts = [(flat[i] * width, flat[i + 1] * height) for i in range(0, len(flat) - 1, 2)]
    return np.array(pts, dtype=np.int32)


def render(
    ann: FrameAnnotation,
    class_names: list[str],
    *,
    labels: bool = True,
    thickness: int = 0,
) -> np.ndarray | None:
    """The frame with its labels drawn on, as a BGR image.

    Line width follows the size of the picture rather than being fixed, so a
    4000px frame does not come back hairlined and a 640px one is not buried.
    """
    img = cv2.imread(ann.image_path)
    if img is None:
        return None
    height, width = img.shape[:2]
    if thickness <= 0:
        thickness = max(1, round(min(width, height) / 400))
    scale = max(0.4, min(width, height) / 1400)

    for index, box in enumerate(ann.boxes):
        color = class_color(box.class_id)
        pts = _points(box, width, height)
        if pts is not None:
            cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)),
                          color, thickness, cv2.LINE_AA)

    if not labels:
        return img

    # Chips go on after every outline, so a later outline cannot draw through
    # one, and each is nudged clear of the chips already down.  Objects in a row
    # share a top edge, which puts all their labels in the same place and leaves
    # a stack of unreadable smears exactly where the labels matter most.
    taken: list[tuple[int, int, int, int]] = []
    if ann.held_out:
        # Held out to be measured against — worth seeing at a glance, because
        # nothing is going to change this frame however wrong it looks.
        edge = _bgr(HELD_OUT_COLOR)
        cv2.rectangle(img, (0, 0), (width - 1, height - 1), edge, thickness * 3)
        _chip(img, "HELD OUT", (thickness * 4, thickness * 4), edge, scale, thickness, taken)

    for index, box in enumerate(ann.boxes):
        name = class_names[box.class_id] if 0 <= box.class_id < len(class_names) else "?"
        text = f"{index} {name}"
        if box.score is not None:
            text += f" {box.score * 100:.0f}%"
        _chip(img, text, (int(box.x1), int(box.y1)), class_color(box.class_id),
              scale, thickness, taken)
    return img


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _chip(img, text: str, at: tuple[int, int], color, scale: float, thickness: int,
          taken: list[tuple[int, int, int, int]] | None = None) -> None:
    """A filled label with the text knocked out of it, pinned inside the frame.

    Where one is already sitting, it steps down until it finds clear space, and
    gives up rather than walking off the bottom of the picture.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, max(1, thickness))
    pad = max(2, round(scale * 4))
    box_w, box_h = tw + 2 * pad, th + base + pad

    x = min(max(at[0], 0), max(0, img.shape[1] - box_w))
    y = max(at[1], box_h)
    if taken is not None:
        step = box_h + max(1, pad // 2)
        while y + base - pad < img.shape[0]:
            rect = (x, y - th - 2 * pad, x + box_w, y + base - pad)
            if not any(_overlaps(rect, other) for other in taken):
                break
            y += step
        else:
            return
        taken.append((x, y - th - 2 * pad, x + box_w, y + base - pad))

    cv2.rectangle(img, (x, y - th - 2 * pad), (x + box_w, y + base - pad), color, -1)
    cv2.putText(img, text, (x + pad, y - pad), font, scale, (20, 20, 20),
                max(1, thickness), cv2.LINE_AA)


def encode_png(img: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("The rendered frame could not be encoded.")
    return buffer.tobytes()
