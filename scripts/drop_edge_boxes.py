"""Remove boxes that run off the edge of the picture.

An object only half in shot is still a real object, but a box drawn around the
visible half says "this shape is what you are looking for" — which is how a
model learns to report half an object as a whole one.
"""

NAME = "Drop boxes at the edge"

PARAMS = [
    {"key": "margin", "label": "Count as the edge within", "default": 2,
     "min": 0, "max": 100, "step": 1, "integer": True, "format": "pixels"},
    {"key": "sides", "label": "Which edges", "type": "choice", "default": "all",
     "options": [["all", "All four"],
                 ["lr", "Left and right only"],
                 ["tb", "Top and bottom only"]],
     "hint": "Pictures panned left to right often only need the sides."},
]


def apply(frame, p):
    width, height = frame.size
    keep = []
    for box in frame.boxes:
        on_side = box.x1 <= p.margin or box.x2 >= width - p.margin
        on_top_or_bottom = box.y1 <= p.margin or box.y2 >= height - p.margin
        if p.sides == "lr":
            touching = on_side
        elif p.sides == "tb":
            touching = on_top_or_bottom
        else:
            touching = on_side or on_top_or_bottom
        if not touching:
            keep.append(box)
    return keep
