# Writing a script

**Tidy up boxes** is one rule, fixed: fold overlapping boxes together. A script is the
same shape with the rule left to you. Put a `.py` file in `scripts/`, and it appears in
the Label panel under **Run a script** with its own controls and both buttons — *Run on
this frame* and *Run on all N frames*.

Scripts change labels: boxes, outlines, keypoints and classes. They never rewrite the
pictures, and they cannot add or remove frames.

---

## The smallest script

```python
NAME = "Drop tiny boxes"


def apply(frame, p):
    return [b for b in frame.boxes if b.x2 - b.x1 >= 24]
```

Save it as `scripts/drop_tiny.py`. That is the whole contract: `apply` is handed one
frame and returns the boxes that frame should end up with.

Editing the file is enough — the panel picks the new version up on its next run, with no
restart. **Reload** in the panel re-reads the folder if you have added a file while it
was open.

---

## What `apply` may return

| Returned | Meaning |
|----------|---------|
| a list of boxes | the frame ends up with exactly those |
| a single box | the same, unwrapped for you |
| `None` | whatever is in `frame.boxes` is the answer — edit it in place if you prefer |

These three are the same script:

```python
def apply(frame, p):
    return [b for b in frame.boxes if b.class_id != 3]

def apply(frame, p):
    frame.boxes = [b for b in frame.boxes if b.class_id != 3]

def apply(frame, p):
    frame.boxes[:] = [b for b in frame.boxes if b.class_id != 3]
```

A frame is only written when the labels actually differ, so a script that decides to
change nothing costs nothing.

---

## Seeing frames together

Some rules need neighbours — filling a gap, checking that a count holds steady, carrying
a class forward. Define `apply_all` instead, which receives every frame in frame order:

```python
NAME = "Fill gaps"

from copy import deepcopy


def apply_all(frames, p):
    for i, frame in enumerate(frames):
        if not frame.boxes and i:
            frame.boxes = [deepcopy(b) for b in frames[i - 1].boxes]
```

A script with only `apply_all` runs across every frame, and *Run on this frame* is
greyed out — running it on one frame would not mean anything. A script may define both,
in which case `apply` handles one frame and `apply_all` handles the whole session.

---

## The frame

| | |
|---|---|
| `frame.boxes` | the labels — a copy, yours to edit |
| `frame.width`, `frame.height`, `frame.size` | the picture in pixels |
| `frame.index` | which frame this is |
| `frame.classes` | the session class names, in order |
| `frame.task` | `"detect"`, `"segment"` or `"pose"` |
| `frame.status` | `"pending"`, `"verified"` or `"exported"` |
| `frame.path` | the frame on disk |
| `frame.box(x1, y1, x2, y2, class_id=0)` | a new box |
| `frame.class_id("panel")` | a class number from its name |
| `frame.class_name(0)` | the other way round |
| `frame.image()` | the picture as a PIL image, to read |

## A box

| | |
|---|---|
| `x1`, `y1`, `x2`, `y2` | corners, **in pixels** |
| `class_id` | index into `frame.classes` |
| `polygon` | the outline, or `None` — a flat `[x1, y1, x2, y2, …]`, **0–1** across the picture |
| `keypoints` | pose points, or `None` — `(x, y)` pairs, **0–1**; an entry may be `None` |
| `score` | model confidence where there was one |
| `source` | `"sam"`, `"manual"` or `"dataset"` |

Corners are pixels and outlines are fractions. That is how they are stored and exported,
and mixing them up is the easiest mistake to make here.

---

## Settings

Declaring `PARAMS` puts controls above the buttons and passes their values in as `p`:

```python
PARAMS = [
    {"key": "min_side", "label": "Smallest side", "default": 24,
     "min": 1, "max": 400, "step": 1, "integer": True, "format": "pixels",
     "hint": "A box goes when either of its sides is shorter than this."},
]


def apply(frame, p):
    return [b for b in frame.boxes
            if b.x2 - b.x1 >= p.min_side and b.y2 - b.y1 >= p.min_side]
```

| `type` | Control | Value you get |
|--------|---------|---------------|
| `number` (the default) | a slider with `min` and `max`; a typing box without | `float`, or `int` with `"integer": True` |
| `bool` | a checkbox | `True` / `False` |
| `text` | a text box — takes `placeholder` | `str` |
| `choice` | a dropdown — takes `options`, either values or `[value, label]` pairs | `str` |
| `class` | a dropdown of this session's classes | `int`, always a class that exists |

Every entry may also carry `label`, `hint` and `format`. `format` is the unit shown
beside a slider, so `"format": "pixels"` reads *24 pixels · default*.

Values are settled before your script sees them: out of range is clamped, unusable falls
back to your default, and a `class` value always names a class that exists. `p` never
needs defending against.

Asking for a parameter you did not declare says so by name rather than failing
mysteriously later.

---

## When something goes wrong

A script that raises leaves the frame exactly as it was, and the traceback appears in the
panel, starting at your line:

```
Frame 12
Traceback (most recent call last):
  File "…/scripts/drop_tiny.py", line 7, in apply
    total += box.wdith
             ^^^^^^^^^
AttributeError: 'BBox' object has no attribute 'wdith'
```

Under `apply`, one bad frame costs only that frame and the rest of the run continues.
Under `apply_all`, an exception abandons the whole run and nothing is kept — there is no
telling how far it got.

That difference shows up in the Stop button too. A run across every frame using `apply`
can be stopped between frames, and what it had already done is kept. `apply_all` is a
single call into your script, so there is nothing to stop between — those runs have no
Stop button rather than one that would not do anything.

`print()` works. Output shows under the buttons, and in the job bar while a run across
every frame is going.

A file that will not import at all is still listed, with its error, so a typo in one
script never hides the others.

Refusing a frame on purpose, with a message rather than a traceback:

```python
from app.core.scripts import ScriptError


def apply(frame, p):
    if frame.task != "segment":
        raise ScriptError("This one only makes sense on a segmentation session.")
    ...
```

---

## Undo

*Run on this frame* is undone with `Ctrl+Z`, like any other edit.

*Run on all N frames* is not — undo only ever covered the frame on screen. So a copy of
the session is saved to `.groundwork/session/backups/` first, and the panel shows you the
path. Putting that file back over `.groundwork/session/session.json` (with the app
closed) restores everything as it was. The twenty most recent are kept.

---

## Notes

- **Scripts are ordinary Python.** They can import anything installed, including numpy,
  OpenCV and torch. There is no sandbox and no attempt at one — these are your files.
  What is guaranteed is that a script cannot leave the session half-edited.
- **Shared helpers** go in a file starting with `_`, which is importable but not listed:
  `_geometry.py` next to your scripts, then `from _geometry import ...`.
- **Work at the top of the file** runs when the folder is read, not when you press Run.
  Load a model inside `apply` (and cache it in a module-level variable), not at import.
- **Boxes are checked before they are kept.** Corners are put in order and clipped to the
  picture, a box with nothing left inside it is dropped and counted, and a class that
  does not exist is refused with a message rather than silently becoming class 0.
- **`NAME` and the docstring** become the name in the dropdown and the line under it. Without
  `NAME`, the filename is used.
