"""Remove boxes below a size.

Auto-labelling finds distant and half-hidden objects along with the ones you
care about. A handful of pixels teaches a model very little and costs it a
lot, so the usual first pass over a fresh session is to throw the smallest away.
"""

NAME = "Drop small boxes"

PARAMS = [
    {"key": "min_side", "label": "Smallest side", "default": 24,
     "min": 1, "max": 400, "step": 1, "integer": True, "format": "pixels",
     "hint": "A box goes when either of its sides is shorter than this."},
    {"key": "by_area", "label": "Measure by area instead", "type": "bool", "default": False,
     "hint": "Keeps a long thin box that a shortest-side rule would throw away."},
]


def apply(frame, p):
    keep = []
    for box in frame.boxes:
        width = box.x2 - box.x1
        height = box.y2 - box.y1
        if p.by_area:
            big_enough = width * height >= p.min_side ** 2
        else:
            big_enough = width >= p.min_side and height >= p.min_side
        if big_enough:
            keep.append(box)
    return keep
