"""Move every box of one class onto another.

Class lists rarely survive first contact: two classes turn out to be the same
thing, or one you split early is not worth splitting. Renaming the class in the
list leaves the boxes where they were; this moves them.
"""

NAME = "Move boxes to another class"

PARAMS = [
    {"key": "source", "label": "Move boxes that are", "type": "class", "default": 0},
    {"key": "target", "label": "…so they become", "type": "class", "default": 0},
]


def apply(frame, p):
    if p.source == p.target:
        return frame.boxes          # nothing to do, and nothing counts as changed
    for box in frame.boxes:
        if box.class_id == p.source:
            box.class_id = p.target
    return frame.boxes
