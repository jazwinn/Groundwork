"""
YOLO training wrapper.

MODEL_REGISTRY maps UI display names to Ultralytics weight filenames.
DETECTION_MODELS / SEGMENTATION_MODELS / POSE_MODELS are pre-filtered lists
for the UI.  train_yolo() runs a training job and can be stopped mid-run.
"""

from __future__ import annotations

import os

from typing import Callable

# Project root = two levels up from app/core/yolo_trainer.py
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
RUNS_DIR = os.path.join(_PROJECT_ROOT, "runs")


def configure_ultralytics_dirs() -> str:
    """Force Ultralytics to read/write everything inside the project root.

    Ultralytics keeps global runs/weights/datasets directories in
    ``%APPDATA%/Ultralytics/settings.json``; out of the box these can point
    *outside* this project (e.g. ``C:/Users/<you>/Groundwork/runs``), which
    scatters generated files. Repoint them under the project root so training
    runs, downloaded weights and datasets all stay inside it. Returns RUNS_DIR.
    """
    try:
        from ultralytics import settings as _ul_settings
        wanted = {
            "runs_dir":     RUNS_DIR,
            "weights_dir":  os.path.join(_PROJECT_ROOT, "weights"),
            "datasets_dir": _PROJECT_ROOT,
        }
        # Only write when something actually differs (avoids needless disk I/O).
        if any(str(_ul_settings.get(k, "")) != v for k, v in wanted.items()):
            _ul_settings.update(wanted)
    except Exception:
        # Settings-API differences or read-only FS — call sites still pass an
        # absolute project path, which keeps outputs inside the root regardless.
        pass
    return RUNS_DIR


MODEL_REGISTRY: dict[str, str] = {
    # ── Detection ─────────────────────────────────────────────────────────
    "YOLOv8n":       "yolov8n.pt",
    "YOLOv8s":       "yolov8s.pt",
    "YOLOv8m":       "yolov8m.pt",
    "YOLOv8l":       "yolov8l.pt",
    "YOLOv8x":       "yolov8x.pt",
    "YOLO11n":       "yolo11n.pt",
    "YOLO11s":       "yolo11s.pt",
    "YOLO11m":       "yolo11m.pt",
    "YOLO11l":       "yolo11l.pt",
    "YOLO12n":       "yolo12n.pt",
    "YOLO12s":       "yolo12s.pt",
    "YOLO12m":       "yolo12m.pt",
    "YOLO12l":       "yolo12l.pt",
    "YOLO26n":       "yolo26n.pt",
    "YOLO26s":       "yolo26s.pt",
    "YOLO26m":       "yolo26m.pt",
    "YOLO26l":       "yolo26l.pt",
    # ── Segmentation ──────────────────────────────────────────────────────
    # v8 / v11 ship pretrained -seg.pt checkpoints (direct download).
    "YOLOv8n-seg":   "yolov8n-seg.pt",
    "YOLOv8s-seg":   "yolov8s-seg.pt",
    "YOLOv8m-seg":   "yolov8m-seg.pt",
    "YOLOv8l-seg":   "yolov8l-seg.pt",
    "YOLOv8x-seg":   "yolov8x-seg.pt",
    "YOLO11n-seg":   "yolo11n-seg.pt",
    "YOLO11s-seg":   "yolo11s-seg.pt",
    "YOLO11m-seg":   "yolo11m-seg.pt",
    "YOLO11l-seg":   "yolo11l-seg.pt",
    # v12 / v26 support the segment task but have no published -seg.pt weights;
    # these are built from the architecture YAML and transfer-load the detection
    # backbone (see SEG_TRANSFER_BASE / _build_model).
    "YOLO12n-seg":   "yolo12n-seg.yaml",
    "YOLO12s-seg":   "yolo12s-seg.yaml",
    "YOLO12m-seg":   "yolo12m-seg.yaml",
    "YOLO12l-seg":   "yolo12l-seg.yaml",
    "YOLO26n-seg":   "yolo26n-seg.yaml",
    "YOLO26s-seg":   "yolo26s-seg.yaml",
    "YOLO26m-seg":   "yolo26m-seg.yaml",
    "YOLO26l-seg":   "yolo26l-seg.yaml",
    # FastSAM — pretrained segmentation checkpoints (always seg-only).
    "FastSAM-s":     "FastSAM-s.pt",
    "FastSAM-x":     "FastSAM-x.pt",
    # ── Pose ──────────────────────────────────────────────────────────────
    # v8 / v11 ship pretrained -pose.pt checkpoints (direct download).
    "YOLOv8n-pose":  "yolov8n-pose.pt",
    "YOLOv8s-pose":  "yolov8s-pose.pt",
    "YOLOv8m-pose":  "yolov8m-pose.pt",
    "YOLOv8l-pose":  "yolov8l-pose.pt",
    "YOLOv8x-pose":  "yolov8x-pose.pt",
    "YOLO11n-pose":  "yolo11n-pose.pt",
    "YOLO11s-pose":  "yolo11s-pose.pt",
    "YOLO11m-pose":  "yolo11m-pose.pt",
    "YOLO11l-pose":  "yolo11l-pose.pt",
    "YOLO11x-pose":  "yolo11x-pose.pt",
}

# For seg models built from a .yaml (no pretrained seg weights), transfer the
# matching detection checkpoint's backbone for a warm start instead of training
# from scratch.  Maps the seg display name → detection .pt to .load().
SEG_TRANSFER_BASE: dict[str, str] = {
    "YOLO12n-seg": "yolo12n.pt",
    "YOLO12s-seg": "yolo12s.pt",
    "YOLO12m-seg": "yolo12m.pt",
    "YOLO12l-seg": "yolo12l.pt",
    "YOLO26n-seg": "yolo26n.pt",
    "YOLO26s-seg": "yolo26s.pt",
    "YOLO26m-seg": "yolo26m.pt",
    "YOLO26l-seg": "yolo26l.pt",
}


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

# Ultralytics augments every epoch whether or not you ask it to — these are the
# knobs behind that, carrying the values it uses when nothing is passed.  They
# live here rather than in the UI so the range the server accepts and the range
# the sliders offer cannot drift apart.
#
# Absent on purpose: `erasing` and `auto_augment` only take effect in
# classification training, so a slider for them would move nothing, and `bgr`
# swaps colour channels, which is meaningless on the greyscale imagery this is
# usually pointed at.
#
# `max` is the top of the slider, not the top of what Ultralytics accepts.
# Rotation past 180° repeats and shear past about 45° destroys the picture, so
# the sliders stop where the values stop being useful.
AUGMENTATIONS: list[dict] = [
    # Photometric
    {"key": "hsv_h", "label": "Hue", "group": "photometric", "format": "amount",
     "default": 0.015, "min": 0.0, "max": 0.2, "step": 0.005,
     "hint": "Colour images only — greyscale has no hue to shift."},
    {"key": "hsv_s", "label": "Saturation", "group": "photometric", "format": "amount",
     "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
     "hint": "Colour images only."},
    {"key": "hsv_v", "label": "Brightness", "group": "photometric", "format": "amount",
     "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
     "hint": "Works on any image, greyscale included."},

    # Geometric
    {"key": "degrees", "label": "Rotation", "group": "geometric", "format": "degrees",
     "default": 0.0, "min": 0.0, "max": 180.0, "step": 5.0,
     "hint": "Turn this up for pictures taken looking straight down, where there is no upright."},
    {"key": "translate", "label": "Shift", "group": "geometric", "format": "amount",
     "default": 0.1, "min": 0.0, "max": 0.9, "step": 0.05},
    {"key": "scale", "label": "Zoom", "group": "geometric", "format": "amount",
     "default": 0.5, "min": 0.0, "max": 0.9, "step": 0.05},
    {"key": "shear", "label": "Shear", "group": "geometric", "format": "degrees",
     "default": 0.0, "min": 0.0, "max": 45.0, "step": 1.0},
    {"key": "perspective", "label": "Perspective", "group": "geometric", "format": "amount",
     "default": 0.0, "min": 0.0, "max": 0.001, "step": 0.0001,
     "hint": "Tilts the picture as if the camera were off to one side. A little goes a long way."},
    {"key": "flipud", "label": "Flip top to bottom", "group": "geometric", "format": "chance",
     "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
     "hint": "Off by default because most photographs have an up. Overhead pictures do not, so this is free variety."},
    {"key": "fliplr", "label": "Flip left to right", "group": "geometric", "format": "chance",
     "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},

    # Composition
    {"key": "mosaic", "label": "Mosaic", "group": "composition", "format": "chance",
     "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
     "hint": "Stitches four pictures into one, so the model sees objects at more sizes and against more backgrounds."},
    {"key": "close_mosaic", "label": "Stop mosaic before the end", "group": "composition",
     "format": "epochs", "default": 10, "min": 0, "max": 50, "step": 1, "integer": True,
     "hint": "Trains the last few epochs on whole pictures, so the model finishes on images that look like the ones it will meet."},
    {"key": "mixup", "label": "Mix two pictures", "group": "composition", "format": "chance",
     "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
    {"key": "cutmix", "label": "Paste a patch of another", "group": "composition", "format": "chance",
     "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
    {"key": "copy_paste", "label": "Copy objects between pictures", "group": "composition",
     "format": "chance", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
     "hint": "Needs outline labels — it lifts objects out along their shape."},
]

AUGMENTATION_DEFAULTS: dict = {spec["key"]: spec["default"] for spec in AUGMENTATIONS}


def sanitise_augmentations(values: dict | None) -> dict:
    """Keep the settings we recognise, clamped to the range the sliders offer.

    Anything unknown is dropped rather than forwarded: Ultralytics takes a very
    wide keyword set, and passing it whatever arrives would turn a typo in the
    browser into a silent change of training behaviour.
    """
    if not values:
        return {}
    cleaned: dict = {}
    for spec in AUGMENTATIONS:
        raw = values.get(spec["key"])
        if raw is None:
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        number = min(max(number, spec["min"]), spec["max"])
        cleaned[spec["key"]] = int(round(number)) if spec.get("integer") else round(number, 5)
    return cleaned


def _is_seg_key(key: str) -> bool:
    return key.endswith("-seg") or key.startswith("FastSAM")

def _is_pose_key(key: str) -> bool:
    return key.endswith("-pose")

DETECTION_MODELS:    list[str] = [k for k in MODEL_REGISTRY if not _is_seg_key(k) and not _is_pose_key(k)]
SEGMENTATION_MODELS: list[str] = [k for k in MODEL_REGISTRY if _is_seg_key(k)]
POSE_MODELS:         list[str] = [k for k in MODEL_REGISTRY if _is_pose_key(k)]

# SAM 2 keys are handled by sam2_trainer; re-exported here for convenience.
def _is_sam2_key(key: str) -> bool:
    return key.startswith("SAM 2")


def _build_model(YOLO, model_key: str):
    """Construct the YOLO model for *model_key*.

    Most models load directly from a pretrained .pt.  Seg models that have no
    published -seg.pt (v12 / v26) are mapped to a .yaml architecture and
    transfer-load the matching detection backbone for a warm start.
    """
    weights = MODEL_REGISTRY[model_key]
    model = YOLO(weights)

    if weights.endswith(".yaml"):
        base = SEG_TRANSFER_BASE.get(model_key)
        if base:
            try:
                model = model.load(base)  # transfer detection backbone
            except Exception:
                # No compatible pretrained weights available — train the
                # architecture from scratch rather than failing outright.
                pass
    return model


def metric_label_for(model_key: str) -> str:
    if _is_pose_key(model_key):
        return "mAP50-pose"
    if _is_seg_key(model_key):
        return "mAP50-mask"
    return "mAP50"


def _epoch_metric(metrics: dict, model_key: str) -> float:
    if _is_pose_key(model_key):
        return float(metrics.get("metrics/mAP50(P)", 0.0) or metrics.get("mAP50", 0.0))
    if _is_seg_key(model_key):
        # Prefer mask mAP50; fall back to box mAP50
        return float(
            metrics.get("metrics/mAP50(M)", 0.0)
            or metrics.get("metrics/mAP50(B)", 0.0)
            or metrics.get("mAP50", 0.0)
        )
    return float(metrics.get("metrics/mAP50(B)", 0.0) or metrics.get("mAP50", 0.0))


def train_yolo(
    model_key: str,
    data_yaml: str,
    *,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int | float = -1,
    cache: bool | str = False,
    workers: int = 4,
    augment: dict | None = None,
    project: str | None = None,
    name: str = "exp",
    on_epoch: Callable[[int, int, float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """
    Train an Ultralytics model, reporting progress per epoch.

    ``should_stop`` is polled after every batch and epoch; when it returns True
    Ultralytics is asked to finish early and the best-so-far weights are kept,
    so pressing Stop never throws away completed epochs.

    ``augment`` overrides the image augmentation Ultralytics applies each epoch.
    Leaving it out keeps its defaults, which are themselves not "no augmentation"
    — see AUGMENTATIONS.
    """
    from ultralytics import YOLO

    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key!r}")

    configure_ultralytics_dirs()  # keep all outputs inside the project root
    model = _build_model(YOLO, model_key)
    project = project or os.path.join(RUNS_DIR, "train")
    stopped = {"value": False}

    def _check_stop(trainer) -> None:
        if should_stop and should_stop():
            stopped["value"] = True
            # BaseTrainer checks `stop` at the end of each epoch to break out.
            trainer.stop = True
            trainer.epochs = min(getattr(trainer, "epochs", epochs), trainer.epoch + 1)

    def _on_epoch_end(trainer) -> None:
        epoch = trainer.epoch + 1
        value = _epoch_metric(trainer.metrics or {}, model_key)
        if on_epoch:
            on_epoch(epoch, epochs, value)
        if on_log:
            on_log(f"epoch {epoch}/{epochs}  {metric_label_for(model_key)}={value:.4f}")
        _check_stop(trainer)

    model.add_callback("on_train_epoch_end", _on_epoch_end)
    model.add_callback("on_train_batch_end", _check_stop)

    settings = sanitise_augmentations(augment)
    if settings and on_log:
        changed = {k: v for k, v in settings.items() if v != AUGMENTATION_DEFAULTS[k]}
        on_log(f"augmentation: {changed or 'defaults'}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=name,
        exist_ok=True,
        verbose=False,
        batch=batch,
        cache=cache,
        workers=workers,
        **settings,
    )

    best = str(getattr(results, "best", "") or "")
    save_dir = str(getattr(results, "save_dir", "") or project)
    if not best:
        candidate = os.path.join(save_dir, "weights", "best.pt")
        if os.path.isfile(candidate):
            best = candidate
    return {"best": best, "save_dir": save_dir, "stopped_early": stopped["value"]}


def score_model(
    weights: str,
    data_yaml: str,
    *,
    imgsz: int = 640,
    split: str = "val",
) -> dict:
    """Run trained weights over the validation half and report what they scored.

    Training already prints a score per epoch, but that one is whatever the last
    epoch happened to produce and it moves with the run.  A gate needs one
    number it can put next to another number from a different run, so the best
    weights are re-measured here in a pass of their own.

    mAP50-95 leads because mAP50 saturates: once boxes are roughly right it sits
    near the top and stops separating a good change from a bad one, which is the
    only thing a gate is for.
    """
    from ultralytics import YOLO

    configure_ultralytics_dirs()
    metrics = YOLO(weights).val(data=data_yaml, imgsz=imgsz, split=split,
                                verbose=False, plots=False)
    results = getattr(metrics, "results_dict", {}) or {}

    def pick(*names: str) -> float:
        for name in names:
            if name in results:
                try:
                    return float(results[name])
                except (TypeError, ValueError):
                    continue
        return 0.0

    box = pick("metrics/mAP50-95(B)", "metrics/mAP50-95")
    mask = pick("metrics/mAP50-95(M)")
    pose = pick("metrics/mAP50-95(P)")
    # Segmentation and pose runs also report a box score; the shape-aware one is
    # what the label actually claims, so it wins where there is one.
    headline = mask or pose or box
    return {
        "score": round(headline, 5),
        "mAP50_95": round(headline, 5),
        "mAP50": round(pick("metrics/mAP50(M)", "metrics/mAP50(P)", "metrics/mAP50(B)"), 5),
        "box_mAP50_95": round(box, 5),
        "mask_mAP50_95": round(mask, 5) if mask else None,
        "pose_mAP50_95": round(pose, 5) if pose else None,
        "weights": weights,
        "split": split,
    }
