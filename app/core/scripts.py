"""
User scripts — label behaviour you write yourself.

"Tidy up boxes" is one rule baked into the app.  This is the same idea with the
rule left open: drop a .py file into the `scripts` folder and it appears in the
Label panel with its own controls, ready to run on the frame you are looking at
or on every frame at once.

The smallest useful script is a name and a function::

    NAME = "Drop tiny boxes"

    def apply(frame, p):
        return [b for b in frame.boxes if b.x2 - b.x1 >= 24]

`apply` is handed one frame at a time and returns the boxes that frame should
end up with — or it can edit `frame.boxes` in place and return nothing.  A
script that needs to see frames together defines `apply_all(frames, p)` instead,
which receives them all and edits `frame.boxes` as it goes.

Declaring PARAMS puts controls above the Run buttons and passes their settings
in as `p`, so a script can be adjusted without opening it again::

    PARAMS = [
        {"key": "min_side", "label": "Smallest side", "default": 24,
         "min": 1, "max": 400, "step": 1, "format": "pixels"},
    ]

These are your own scripts and they run as ordinary Python: they can import
anything, and nothing here pretends to be a sandbox.  What is guaranteed is
that a script cannot leave the session half-edited.  Every frame hands over a
*copy* of its labels, everything handed back is checked before it is kept, and
a frame whose script raised keeps exactly the labels it already had.
"""

from __future__ import annotations

import importlib.util
import inspect
import io
import json
import math
import os
import sys
import traceback
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.sam3_handler import BBox, FrameAnnotation
from app.core.yolo_trainer import _PROJECT_ROOT

SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")

# Scripts that want the class by name rather than through frame.box():
#     from app.core.scripts import Box
Box = BBox

# What a script prints is shown back in the panel.  A runaway loop printing on
# every frame would otherwise fill the browser, so only the first of them are kept.
MAX_OUTPUT_LINES = 300

PARAM_TYPES = ("number", "bool", "text", "choice", "class")


class ScriptError(Exception):
    """A script is unusable, or refused a frame.  The message is shown as-is,
    so it is written for the person reading the panel, not for a log."""


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def _normalise_param(spec: Any, where: str) -> dict:
    """Check one PARAMS entry and fill in what it left out."""
    if not isinstance(spec, dict):
        raise ScriptError(f"{where}: every PARAMS entry must be a dict, not {type(spec).__name__}.")
    key = str(spec.get("key") or "").strip()
    if not key or not key.isidentifier():
        raise ScriptError(f'{where}: PARAMS entries need a "key" that works as a Python name.')

    kind = str(spec.get("type") or ("choice" if spec.get("options") else "number"))
    if kind not in PARAM_TYPES:
        raise ScriptError(f'{where}: parameter "{key}" has type "{kind}"; '
                          f"use one of {', '.join(PARAM_TYPES)}.")

    out: dict = {
        "key": key,
        "type": kind,
        "label": str(spec.get("label") or key.replace("_", " ").capitalize()),
        "hint": str(spec.get("hint") or ""),
        "format": str(spec.get("format") or ""),
    }

    if kind == "number":
        integer = bool(spec.get("integer"))
        default = float(spec.get("default", 0))
        low, high = spec.get("min"), spec.get("max")
        out["min"] = None if low is None else float(low)
        out["max"] = None if high is None else float(high)
        if out["min"] is not None and out["max"] is not None and out["min"] > out["max"]:
            raise ScriptError(f'{where}: parameter "{key}" has min above max.')
        # A number with both ends becomes a slider; without them, a typing box.
        out["step"] = float(spec.get("step", 1 if integer else 0.01))
        out["integer"] = integer
        out["default"] = round(default) if integer else default
    elif kind == "bool":
        out["default"] = bool(spec.get("default", False))
    elif kind == "text":
        out["default"] = str(spec.get("default", ""))
        out["placeholder"] = str(spec.get("placeholder") or "")
    elif kind == "choice":
        options = []
        for option in spec.get("options") or []:
            if isinstance(option, (list, tuple)) and len(option) == 2:
                options.append([str(option[0]), str(option[1])])
            else:
                options.append([str(option), str(option)])
        if not options:
            raise ScriptError(f'{where}: parameter "{key}" is a choice but lists no options.')
        out["options"] = options
        default = spec.get("default", options[0][0])
        out["default"] = str(default)
    elif kind == "class":
        out["default"] = int(spec.get("default", 0))

    return out


def normalise_params(specs: Any, where: str) -> list[dict]:
    if specs is None:
        return []
    if not isinstance(specs, (list, tuple)):
        raise ScriptError(f"{where}: PARAMS must be a list.")
    seen: set[str] = set()
    out = []
    for spec in specs:
        param = _normalise_param(spec, where)
        if param["key"] in seen:
            raise ScriptError(f'{where}: two parameters both use the key "{param["key"]}".')
        seen.add(param["key"])
        out.append(param)
    return out


def sanitise(specs: list[dict], values: dict | None, class_count: int = 1) -> dict:
    """Settle every parameter on a usable value.

    Anything the browser did not send, or sent as nonsense, falls back to the
    script's own default — a script never has to defend itself against `p`.
    """
    given = values if isinstance(values, dict) else {}
    out: dict = {}
    for spec in specs:
        key = spec["key"]
        raw = given.get(key, spec["default"])
        out[key] = _coerce(spec, raw, class_count)
    return out


def _coerce(spec: dict, raw: Any, class_count: int) -> Any:
    kind = spec["type"]
    try:
        if kind == "number":
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
            if spec["min"] is not None:
                value = max(spec["min"], value)
            if spec["max"] is not None:
                value = min(spec["max"], value)
            return int(round(value)) if spec["integer"] else value
        if kind == "bool":
            return bool(raw)
        if kind == "text":
            return str(raw)
        if kind == "choice":
            value = str(raw)
            return value if any(value == opt[0] for opt in spec["options"]) else spec["default"]
        if kind == "class":
            value = int(raw)
            return value if 0 <= value < max(1, class_count) else 0
    except (TypeError, ValueError):
        pass
    return spec["default"]


class ParamValues:
    """The `p` a script is handed.  Reads as attributes, and says so plainly
    when a script asks for a parameter it never declared."""

    def __init__(self, values: dict):
        object.__setattr__(self, "_values", dict(values))

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        known = ", ".join(sorted(values)) or "none"
        raise AttributeError(f"This script has no parameter called '{name}'. "
                             f"Declared in PARAMS: {known}.")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Parameters are the settings you chose in the panel; "
                             "they cannot be assigned to from inside a script.")

    def __getitem__(self, key: str) -> Any:
        return object.__getattribute__(self, "_values")[key]

    def get(self, key: str, default: Any = None) -> Any:
        return object.__getattribute__(self, "_values").get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in object.__getattribute__(self, "_values")

    def __iter__(self):
        return iter(object.__getattribute__(self, "_values"))

    def keys(self):
        return object.__getattribute__(self, "_values").keys()

    def items(self):
        return object.__getattribute__(self, "_values").items()

    def __repr__(self) -> str:
        return f"Params({object.__getattribute__(self, '_values')!r})"


# ---------------------------------------------------------------------------
# The frame a script sees
# ---------------------------------------------------------------------------

class FrameView:
    """One frame, handed to a script.

    `boxes` starts as a copy of the frame's labels, so editing it cannot damage
    anything until the run finishes and the result has been checked.  Corners
    are in pixels; polygons and keypoints are 0–1 across the picture, which is
    how they are stored and exported.
    """

    def __init__(self, ann: FrameAnnotation, classes: list[str], task: str):
        self.index = ann.frame_index
        self.path = ann.image_path
        self.width = ann.width
        self.height = ann.height
        self.status = ann.status
        self.classes = list(classes)
        self.task = task
        self.boxes: list[BBox] = [deepcopy(box) for box in ann.boxes]
        self._before = _fingerprint(ann.boxes)
        self._image = None

    # ── describing the frame ──────────────────────────────────────

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def __repr__(self) -> str:
        return f"<frame {self.index}: {len(self.boxes)} boxes, {self.width}x{self.height}>"

    # ── making and naming boxes ───────────────────────────────────

    def box(self, x1: float, y1: float, x2: float, y2: float,
            class_id: int = 0, **extra: Any) -> BBox:
        """A new box, in pixels.

        It counts as hand-made rather than machine-made, on the same grounds as
        a merge: a person asked for it.
        """
        extra.setdefault("source", "manual")
        return BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    class_id=int(class_id), **extra)

    def class_id(self, name: str) -> int:
        """The number a class name is stored as."""
        try:
            return self.classes.index(name)
        except ValueError:
            raise ScriptError(f"There is no class called '{name}'. This session has: "
                              f"{', '.join(self.classes)}.") from None

    def class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return f"class {class_id}"

    def image(self):
        """The picture itself, as a PIL image — for scripts that need to look
        at pixels before deciding.  Reading only: scripts change labels, and
        the frame on disk is never rewritten."""
        if self._image is None:
            from PIL import Image
            self._image = Image.open(self.path).convert("RGB")
        return self._image

    # ── used by the runner ────────────────────────────────────────

    def changed(self) -> bool:
        return _fingerprint(self.boxes) != self._before


def _fingerprint(boxes: list[BBox]) -> str:
    """A value to compare two sets of labels by.

    Rounded and coerced first, because every box a script hands back is rebuilt
    on the way in: an integer corner becomes a float, a tuple keypoint becomes
    a list.  Comparing the raw shapes would call that a change and rewrite the
    whole session every time a script ran.
    """
    return json.dumps([_canonical(box) for box in boxes], sort_keys=True)


def _canonical(box: BBox) -> dict:
    d = box.to_dict()
    for corner in ("x1", "y1", "x2", "y2"):
        d[corner] = round(float(d[corner]), 4)      # sub-pixel; below that is noise
    d["class_id"] = int(d["class_id"])
    d["source"] = str(d.get("source") or "")
    if d.get("polygon"):
        d["polygon"] = [round(float(v), 6) for v in d["polygon"]]
    if d.get("keypoints"):
        d["keypoints"] = [None if k is None else [round(float(k[0]), 6), round(float(k[1]), 6)]
                          for k in d["keypoints"]]
    if d.get("score") is not None:
        d["score"] = round(float(d["score"]), 6)
    return d


# ---------------------------------------------------------------------------
# Checking what a script hands back
# ---------------------------------------------------------------------------

def _clean_polygon(raw: Any, where: str) -> list[float] | None:
    if raw is None:
        return None
    try:
        flat = [float(v) for v in raw]
    except (TypeError, ValueError):
        raise ScriptError(f"{where}: the outline must be a flat list of numbers "
                          f"[x1, y1, x2, y2, ...], 0–1 across the picture.") from None
    if not flat:
        return None
    if len(flat) % 2 or len(flat) < 6:
        raise ScriptError(f"{where}: an outline needs at least three points, "
                          f"as pairs of numbers — got {len(flat)}.")
    if not all(math.isfinite(v) for v in flat):
        return None
    return [min(1.0, max(0.0, v)) for v in flat]


def _clean_keypoints(raw: Any, where: str) -> list | None:
    if raw is None:
        return None
    out: list = []
    for point in raw:
        if point is None:
            out.append(None)
            continue
        try:
            x, y = (float(point[0]), float(point[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            raise ScriptError(f"{where}: each keypoint must be an (x, y) pair "
                              f"or None, 0–1 across the picture.") from None
        if not (math.isfinite(x) and math.isfinite(y)):
            out.append(None)
            continue
        out.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
    return out or None


def _clean_box(raw: Any, frame: FrameView, position: int) -> BBox | None:
    """One box from a script, made safe to store — or None if nothing is left
    of it.  Anything genuinely wrong raises, so the frame is left alone rather
    than written with labels that do not mean what the script intended."""
    where = f"frame {frame.index}, box {position}"

    if isinstance(raw, dict):
        try:
            raw = BBox.from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ScriptError(f"{where}: {exc}") from None
    if not isinstance(raw, BBox):
        raise ScriptError(f"{where}: expected a box, got {type(raw).__name__}. "
                          f"Return boxes from frame.boxes, or make one with frame.box(...).")

    try:
        corners = [float(raw.x1), float(raw.y1), float(raw.x2), float(raw.y2)]
    except (TypeError, ValueError):
        raise ScriptError(f"{where}: the corners are not numbers.") from None
    if not all(math.isfinite(v) for v in corners):
        raise ScriptError(f"{where}: the corners are not finite numbers.")

    x1, x2 = sorted((corners[0], corners[2]))
    y1, y2 = sorted((corners[1], corners[3]))
    if frame.width and frame.height:
        x1, x2 = min(max(x1, 0.0), frame.width), min(max(x2, 0.0), frame.width)
        y1, y2 = min(max(y1, 0.0), frame.height), min(max(y2, 0.0), frame.height)
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None   # nothing of it is on the picture

    try:
        class_id = int(raw.class_id)
    except (TypeError, ValueError):
        raise ScriptError(f"{where}: the class is not a number.") from None
    if not 0 <= class_id < len(frame.classes):
        raise ScriptError(f"{where}: there is no class {class_id}. This session has "
                          f"{len(frame.classes)}: {', '.join(frame.classes)}.")

    score = raw.score
    if score is not None:
        try:
            score = float(score)
            score = None if not math.isfinite(score) else score
        except (TypeError, ValueError):
            score = None

    return BBox(
        x1=x1, y1=y1, x2=x2, y2=y2, class_id=class_id,
        source=str(raw.source or "manual"),
        polygon=_clean_polygon(raw.polygon, where),
        keypoints=_clean_keypoints(raw.keypoints, where),
        score=score,
    )


def _check(returned: Any, frame: FrameView) -> tuple[list[BBox], int]:
    """Turn whatever a script gave back into a list of storable boxes."""
    if returned is None:
        returned = frame.boxes          # it edited frame.boxes instead of returning
    if isinstance(returned, BBox):
        returned = [returned]           # one box, unwrapped
    if isinstance(returned, (str, bytes, dict)):
        raise ScriptError(f"frame {frame.index}: apply() returned "
                          f"{type(returned).__name__}, not a list of boxes.")
    try:
        items = list(returned)
    except TypeError:
        raise ScriptError(f"frame {frame.index}: apply() returned "
                          f"{type(returned).__name__}, not a list of boxes.") from None

    kept: list[BBox] = []
    dropped = 0
    for position, raw in enumerate(items):
        box = _clean_box(raw, frame, position)
        if box is None:
            dropped += 1
        else:
            kept.append(box)
    return kept, dropped


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_modules: dict[str, tuple[float, Any]] = {}


def _ensure_dir() -> None:
    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    # Appended, not prepended: a script called json.py should not become the
    # json every other part of the app imports.
    if SCRIPTS_DIR not in sys.path:
        sys.path.append(SCRIPTS_DIR)


def _module_for(path: str) -> Any:
    """Import a script, re-importing it whenever the file has been edited."""
    mtime = os.path.getmtime(path)
    cached = _modules.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    name = "groundwork_script_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ScriptError("Python could not read this file.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _modules[path] = (mtime, module)
    return module


@dataclass
class ScriptInfo:
    id: str
    name: str
    description: str
    path: str
    params: list[dict] = field(default_factory=list)
    per_frame: bool = False     # defines apply()
    across: bool = False        # defines apply_all()
    error: str = ""
    module: Any = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "params": self.params,
            # What the two buttons should do with it.
            "canFrame": self.per_frame and not self.error,
            "canAll": (self.per_frame or self.across) and not self.error,
            "acrossOnly": self.across and not self.per_frame and not self.error,
            "error": self.error,
        }


def _summary(module: Any) -> str:
    stated = getattr(module, "DESCRIPTION", "")
    if stated:
        return " ".join(str(stated).split())
    doc = inspect.getdoc(module) or ""
    first = doc.split("\n\n")[0] if doc else ""
    return " ".join(first.split())


def _trace(exc: BaseException, path: str) -> str:
    """The traceback, starting where the script itself comes into it — the
    frames above are this file's plumbing and only get in the way."""
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
    lines = text.splitlines()
    marker = os.path.basename(path)
    for i, line in enumerate(lines):
        if marker in line and line.lstrip().startswith("File "):
            return "\n".join(["Traceback (most recent call last):"] + lines[i:])
    return text


def _inspect(path: str) -> ScriptInfo:
    """Read one file's name, description and parameters, reporting rather than
    raising when it will not load — a broken script must not hide the others."""
    sid = os.path.splitext(os.path.basename(path))[0]
    fallback = sid.replace("_", " ").replace("-", " ").strip().capitalize()

    try:
        module = _module_for(path)
    except (Exception, SystemExit) as exc:
        return ScriptInfo(id=sid, name=fallback, description="", path=path,
                          error=_trace(exc, path))

    name = str(getattr(module, "NAME", "") or fallback)
    info = ScriptInfo(id=sid, name=name, description=_summary(module),
                      path=path, module=module)

    try:
        info.params = normalise_params(getattr(module, "PARAMS", None), name)
    except ScriptError as exc:
        info.error = str(exc)
        return info

    info.per_frame = callable(getattr(module, "apply", None))
    info.across = callable(getattr(module, "apply_all", None))
    if not info.per_frame and not info.across:
        info.error = ("This file has no apply(frame, p) and no apply_all(frames, p), "
                      "so there is nothing to run.")
    return info


def discover() -> list[ScriptInfo]:
    """Every script in the folder, in the order they should be listed."""
    _ensure_dir()
    found = []
    try:
        entries = sorted(os.listdir(SCRIPTS_DIR))
    except OSError:
        return []
    for entry in entries:
        # A leading underscore marks shared helpers, not scripts to run.
        if not entry.endswith(".py") or entry.startswith("_"):
            continue
        found.append(_inspect(os.path.join(SCRIPTS_DIR, entry)))
    found.sort(key=lambda s: s.name.lower())
    return found


def get(script_id: str) -> ScriptInfo:
    """One script by id, ready to run, or a ScriptError saying why not."""
    _ensure_dir()
    stem = os.path.basename(str(script_id or "")).removesuffix(".py")
    if not stem:
        raise ScriptError("No script was chosen.")
    path = os.path.join(SCRIPTS_DIR, f"{stem}.py")
    if not os.path.isfile(path):
        raise ScriptError(f"There is no script called '{stem}' in {SCRIPTS_DIR}.")
    info = _inspect(path)
    if info.error:
        raise ScriptError(info.error)
    return info


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

class _Output(io.TextIOBase):
    """Collects what a script prints, so it can be shown next to the buttons.

    Redirecting stdout is process-wide, so a line printed by some other thread
    during the few moments a script runs can land here instead of the console.
    That is the whole cost, and print() is far too useful while writing a script
    to give up over it.
    """

    def __init__(self, sink: list[str], on_line: Callable[[str], None] | None = None):
        self._sink = sink
        self._on_line = on_line
        self._partial = ""

    def write(self, text: str) -> int:
        self._partial += text
        while "\n" in self._partial:
            line, _, self._partial = self._partial.partition("\n")
            self._line(line)
        return len(text)

    def flush(self) -> None:
        if self._partial:
            self._line(self._partial)
            self._partial = ""

    def _line(self, line: str) -> None:
        line = line[:500]
        if len(self._sink) < MAX_OUTPUT_LINES:
            self._sink.append(line)
        elif len(self._sink) == MAX_OUTPUT_LINES:
            self._sink.append("… the rest of the output is not shown.")
        if self._on_line:
            self._on_line(line)


@dataclass
class RunReport:
    script: str = ""
    scanned: int = 0
    changed: int = 0
    before: int = 0
    after: int = 0
    dropped: int = 0
    stopped: bool = False
    errors: list[dict] = field(default_factory=list)
    output: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "script": self.script,
            "scanned": self.scanned,
            "changed": self.changed,
            "before": self.before,
            "after": self.after,
            "added": max(0, self.after - self.before),
            "removed": max(0, self.before - self.after),
            "dropped": self.dropped,
            "stopped": self.stopped,
            "errors": self.errors,
            "output": self.output,
        }


def run(
    info: ScriptInfo,
    views: list[FrameView],
    values: dict,
    *,
    scope: str = "frame",
    should_abort: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_line: Callable[[str], None] | None = None,
) -> RunReport:
    """Run *info* over *views*, leaving each view holding its final boxes.

    Nothing is written anywhere: the caller decides what to do with the views
    whose `changed()` says the script had something to say about them.
    """
    report = RunReport(script=info.name, scanned=len(views))
    report.before = sum(len(view.boxes) for view in views)
    params = ParamValues(values)
    sink = _Output(report.output, on_line)

    across = getattr(info.module, "apply_all", None)
    per_frame = getattr(info.module, "apply", None)

    if scope == "all" and callable(across):
        _run_across(info, across, views, params, sink, report)
    elif callable(per_frame):
        _run_each(info, per_frame, views, params, sink, report,
                  should_abort=should_abort, on_progress=on_progress)
    else:
        raise ScriptError(f"{info.name} only defines apply_all, so it runs across "
                          f"every frame rather than on one.")

    report.after = sum(len(view.boxes) for view in views)
    report.changed = sum(1 for view in views if view.changed())
    return report


def _run_across(info, across, views, params, sink, report) -> None:
    """One call, all the frames.  An exception here is about the run as a
    whole, so nothing is kept — there is no telling how far it got."""
    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            across(views, params)
            sink.flush()
    except ScriptError as exc:
        sink.flush()
        report.errors.append({"frame": None, "error": str(exc)})
        _revert(views)
        return
    except (Exception, SystemExit) as exc:
        sink.flush()
        report.errors.append({"frame": None, "error": _trace(exc, info.path)})
        _revert(views)
        return

    # Check every frame it touched, and keep the ones that came back sound.
    for view in views:
        try:
            view.boxes, dropped = _check(view.boxes, view)
            report.dropped += dropped
        except ScriptError as exc:
            report.errors.append({"frame": view.index, "error": str(exc)})
            _revert([view])


def _run_each(info, per_frame, views, params, sink, report, *,
              should_abort=None, on_progress=None) -> None:
    """Frame by frame, so one bad frame costs only that frame."""
    total = len(views)
    for position, view in enumerate(views):
        if should_abort and should_abort():
            report.stopped = True
            report.scanned = position
            break
        if on_progress:
            on_progress(position, total, f"Frame {view.index}")
        try:
            with redirect_stdout(sink), redirect_stderr(sink):
                returned = per_frame(view, params)
                sink.flush()
            view.boxes, dropped = _check(returned, view)
            report.dropped += dropped
        except ScriptError as exc:
            sink.flush()
            report.errors.append({"frame": view.index, "error": str(exc)})
            _revert([view])
        except (Exception, SystemExit) as exc:
            sink.flush()
            report.errors.append({"frame": view.index, "error": _trace(exc, info.path)})
            _revert([view])
    if on_progress:
        on_progress(report.scanned, total, "")


def _revert(views: list[FrameView]) -> None:
    """Put a frame back the way it was, so a failed script changes nothing."""
    for view in views:
        view.boxes = [BBox.from_dict(d) for d in json.loads(view._before)]
