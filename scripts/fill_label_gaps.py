"""Copy labels into unlabelled frames from the labelled frame before them.

Frames pulled from a video change very little from one to the next, so a short
gap in the middle of a labelled run is usually a frame nobody got to rather than
a frame with nothing in it. This fills those gaps in and leaves the long ones
alone, since by then the picture has usually moved on.

It is also the example to copy when you need a script that can see frames
together: `apply_all` gets them all, in frame order, and edits `frame.boxes`.
"""

from copy import deepcopy

NAME = "Fill gaps from the frame before"

PARAMS = [
    {"key": "max_gap", "label": "Longest gap to fill", "default": 5,
     "min": 1, "max": 60, "step": 1, "integer": True, "format": "frames"},
    {"key": "only_between", "label": "Only where both sides are labelled",
     "type": "bool", "default": True,
     "hint": "On, an unlabelled run at the very start or end is left empty."},
]


def apply_all(frames, p):
    labelled = [i for i, frame in enumerate(frames) if frame.boxes]
    if not labelled:
        print("Nothing is labelled yet, so there is nothing to copy from.")
        return

    filled = 0
    for start, end in zip(labelled, labelled[1:]):
        if not 0 < end - start - 1 <= p.max_gap:
            continue
        for i in range(start + 1, end):
            frames[i].boxes = [deepcopy(box) for box in frames[start].boxes]
            filled += 1

    if not p.only_between:
        first, last = labelled[0], labelled[-1]
        for i in range(max(0, first - p.max_gap), first):
            frames[i].boxes = [deepcopy(box) for box in frames[first].boxes]
            filled += 1
        for i in range(last + 1, min(len(frames), last + 1 + p.max_gap)):
            frames[i].boxes = [deepcopy(box) for box in frames[last].boxes]
            filled += 1

    print(f"{filled} frames filled from their neighbours.")
