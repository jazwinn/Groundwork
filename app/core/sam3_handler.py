"""
SAM 3 / SAM 3.1 inference — Promptable Concept Segmentation (PCS).

Unlike SAM 2 (point-grid mask generation), SAM 3 segments by *concept*:
  - a TEXT phrase  (e.g. "car", "yellow school bus")  → segments ALL matches
  - POSITIVE box exemplars (label 1) → "find more things like this"
  - NEGATIVE box exemplars (label 0) → "but exclude things like this"
in a single forward pass, returning instance boxes for every matching object.

transformers integration loads through `facebook/sam3` (Sam3Model / Sam3Processor).
The `facebook/sam3.1` repo ships improved checkpoints but no transformers config,
so we load via `facebook/sam3` and fall back gracefully.

Everything here is Qt-free: long operations take callbacks for progress and a
`should_abort` predicate so the caller can stop them at any time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image

from app.core import maskops


# ---------------------------------------------------------------------------
# Shared data model
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0
    source: str = "sam"  # "sam" | "manual" | "dataset"
    polygon: list[float] | None = None   # flat [x1,y1,...] normalized 0–1 for seg labels
    keypoints: list[tuple[float, float] | None] | None = None  # normalized (x,y); None = removed corner
    score: float | None = None           # SAM confidence, when known

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "class_id": self.class_id, "source": self.source,
                "polygon": self.polygon,
                "keypoints": [list(k) if k is not None else None for k in self.keypoints]
                if self.keypoints is not None else None,
                "score": self.score}

    @staticmethod
    def from_dict(d: dict) -> "BBox":
        kpts = d.get("keypoints")
        if kpts is not None:
            kpts = [tuple(k) if k is not None else None for k in kpts]
        return BBox(x1=float(d["x1"]), y1=float(d["y1"]),
                    x2=float(d["x2"]), y2=float(d["y2"]),
                    class_id=int(d.get("class_id", 0)),
                    source=d.get("source", "sam"),
                    polygon=d.get("polygon"), keypoints=kpts,
                    score=d.get("score"))


@dataclass
class FrameAnnotation:
    frame_index: int
    image_path: str
    boxes: list[BBox] = field(default_factory=list)
    status: str = "pending"   # "pending" | "verified" — whether a person has checked it
    source_video: str = ""    # absolute path of the source video (empty for still images)
    source_image: str = ""    # absolute path of the source image (empty for video frames)
    width: int = 0
    height: int = 0
    # Held out to measure against: exported as the validation half, and refused
    # by everything that writes labels.  A frame cannot be both the ruler and
    # the thing being measured, so this is enforced rather than agreed to.
    held_out: bool = False
    # Whether this frame has been written into a dataset.  Kept apart from
    # `status`, which is about whether a person has checked the labels: those
    # are two different facts, and while "exported" was a third status value
    # exporting overwrote the answer to the first question with the answer to
    # the second, losing track of what had been reviewed every time.
    exported: bool = False

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "image_path": self.image_path,
            "boxes": [b.to_dict() for b in self.boxes],
            "status": self.status,
            "source_video": self.source_video,
            "source_image": self.source_image,
            "width": self.width,
            "height": self.height,
            "held_out": self.held_out,
            "exported": self.exported,
        }

    @staticmethod
    def from_dict(d: dict) -> "FrameAnnotation":
        # Sessions written while "exported" was a status carry no review state
        # for those frames — exporting had already overwritten it, and the
        # answer is not recoverable.  They come back as reviewed because that is
        # how the editor has always drawn them, so nothing changes appearance on
        # the way in; what changes is that exporting stops doing it again.
        status = d.get("status", "pending")
        exported = bool(d.get("exported", False))
        if status == "exported":
            status, exported = "verified", True
        return FrameAnnotation(
            frame_index=int(d["frame_index"]),
            image_path=d["image_path"],
            boxes=[BBox.from_dict(b) for b in d.get("boxes", [])],
            status=status,
            source_video=d.get("source_video", ""),
            source_image=d.get("source_image", ""),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
            held_out=bool(d.get("held_out", False)),
            exported=exported,
        )


AnnotationStore = dict[int, FrameAnnotation]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

SAM_MODELS = {
    "SAM 3.1 (facebook/sam3.1)": "facebook/sam3.1",
    "SAM 3 (facebook/sam3)":     "facebook/sam3",
}

# Transformers integration is published under facebook/sam3; sam3.1 is a
# checkpoint-only repo.  If a selected id can't be loaded we fall back to this.
_FALLBACK_MODEL_ID = "facebook/sam3"

# The video tracker is a distinct model (SAM2-style memory tracker) loaded from
# facebook/sam3 — it propagates the exact objects you box on one frame through
# the rest of an ordered video by visual memory.
_TRACKER_MODEL_ID = "facebook/sam3"


def best_device() -> str:
    """The device inference runs on."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_sam3(model_id: str, local_only: bool):
    """Load a SAM 3 model + processor, falling back to facebook/sam3."""
    from transformers import Sam3Model, Sam3Processor
    import torch

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def _try(mid: str):
        proc = Sam3Processor.from_pretrained(mid, local_files_only=local_only)
        mdl = Sam3Model.from_pretrained(mid, local_files_only=local_only, torch_dtype=dtype)
        return mdl, proc

    try:
        return _try(model_id)
    except Exception:
        if model_id != _FALLBACK_MODEL_ID:
            return _try(_FALLBACK_MODEL_ID)
        raise


def load_sam3(model_key: str):
    """Load (and move to the best device) the SAM 3 concept model."""
    device = best_device()
    model_id = SAM_MODELS.get(model_key, _FALLBACK_MODEL_ID)
    try:
        model, processor = _load_sam3(model_id, local_only=True)
    except Exception:
        model, processor = _load_sam3(model_id, local_only=False)
    return model.to(device).eval(), processor


def load_sam3_tracker():
    """Load the SAM 3 video memory tracker.

    Half precision on the GPU, as the concept model already does.  Encoding a
    frame drops from ~3.3 s to ~0.7 s and the mask is the same one (IoU 0.998
    against float32).  bf16 is deliberately not used: its shorter mantissa
    moves logits across the binarisation threshold and the outline falls apart.
    """
    import torch
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def _try(local_only: bool):
        processor = Sam3TrackerVideoProcessor.from_pretrained(
            _TRACKER_MODEL_ID, local_files_only=local_only
        )
        model = Sam3TrackerVideoModel.from_pretrained(
            _TRACKER_MODEL_ID, local_files_only=local_only, torch_dtype=dtype
        )
        return model, processor

    try:
        model, processor = _try(True)
    except Exception:
        model, processor = _try(False)

    device = best_device()
    return model.to(device).eval(), processor


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch
        return torch.device("cpu")


def _mask_to_bbox(mask) -> tuple[int, int, int, int] | None:
    """Tight bounding box (x1, y1, x2, y2) of a boolean mask, or None if empty."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return None
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    return int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])


def _to_device(inputs, model):
    device = _model_device(model)
    model_dtype = next(model.parameters()).dtype
    return {
        k: (v.to(device, dtype=model_dtype)
            if hasattr(v, "to") and getattr(v, "dtype", None) is not None
            and getattr(v.dtype, "is_floating_point", False)
            else v.to(device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }


def run_sam3(
    model,
    processor,
    pil_img: Image.Image,
    *,
    text: str | None = None,
    pos_boxes: list[tuple[float, float, float, float]] | None = None,
    neg_boxes: list[tuple[float, float, float, float]] | None = None,
    class_id: int = 0,
    threshold: float = 0.5,
    want_masks: bool = False,
    detail: str = maskops.DEFAULT_DETAIL,
) -> list[BBox]:
    """
    Run SAM 3 concept segmentation on a single image.

    Returns one BBox per matching instance.  Requires at least a text prompt
    or one positive exemplar box (otherwise SAM 3 has no concept to find).

    With *want_masks* each box also carries the outline of its mask, which is
    what a segmentation label needs.  The model computes those masks either
    way, so this only decides whether they are kept.
    """
    import torch

    W, H = pil_img.size

    proc_kwargs: dict = dict(images=pil_img, return_tensors="pt")
    if text:
        proc_kwargs["text"] = text

    boxes_xyxy: list[list[float]] = []
    labels: list[int] = []
    for b in (pos_boxes or []):
        boxes_xyxy.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        labels.append(1)
    for b in (neg_boxes or []):
        boxes_xyxy.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        labels.append(0)
    if boxes_xyxy:
        proc_kwargs["input_boxes"] = [boxes_xyxy]
        proc_kwargs["input_boxes_labels"] = [labels]

    if not text and not any(lbl == 1 for lbl in labels):
        return []  # nothing to segment

    inputs = _to_device(processor(**proc_kwargs), model)

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = inputs.get("original_sizes")
    if hasattr(target_sizes, "tolist"):
        target_sizes = target_sizes.tolist()
    if not target_sizes:
        target_sizes = [[H, W]]

    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5, target_sizes=target_sizes,
    )[0]

    return _results_to_boxes(results, W, H, class_id,
                             want_masks=want_masks, detail=detail)


def _results_to_boxes(
    results, W: int, H: int, class_id: int,
    *, want_masks: bool = False, detail: str = maskops.DEFAULT_DETAIL,
) -> list[BBox]:
    boxes = results.get("boxes")
    if boxes is None:
        return []
    scores = results.get("scores")
    masks = results.get("masks") if want_masks else None

    out: list[BBox] = []
    for i, box in enumerate(boxes):
        coords = box.tolist() if hasattr(box, "tolist") else list(box)
        x1, y1, x2, y2 = (float(coords[0]), float(coords[1]),
                          float(coords[2]), float(coords[3]))
        x1 = max(0.0, min(W, x1)); x2 = max(0.0, min(W, x2))
        y1 = max(0.0, min(H, y1)); y2 = max(0.0, min(H, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        score = None
        if scores is not None:
            try:
                score = round(float(scores[i]), 4)
            except Exception:
                score = None
        polygon = None
        if masks is not None and i < len(masks):
            polygon = maskops.mask_to_polygon(
                maskops.to_bool_mask(masks[i], W, H), W, H, detail=detail
            )
            if polygon:
                # The outline is the truth for a segmentation label, so keep the
                # box consistent with it rather than with the model's own box.
                x1, y1, x2, y2 = maskops.polygon_bounds(polygon, W, H)

        out.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2, class_id=class_id,
                        source="sam", score=score, polygon=polygon))
    return out


# ---------------------------------------------------------------------------
# Bulk operations (driven by callbacks — no threading here)
# ---------------------------------------------------------------------------

def run_concepts_over_frames(
    model,
    processor,
    frames: list[tuple[int, str]],
    concepts: list[tuple[str, int]],
    *,
    threshold: float = 0.5,
    want_masks: bool = False,
    detail: str = maskops.DEFAULT_DETAIL,
    on_boxes: Callable[[int, list[BBox]], None],
    on_progress: Callable[[int, int], None] | None = None,
    should_abort: Callable[[], bool] | None = None,
    on_error: Callable[[str], None] | None = None,
) -> int:
    """
    Search every frame for each (text, class_id) concept and report the hits.

    This is how both "find by description" and "copy this frame's labels
    everywhere" work: the concepts are either typed by the user or derived
    from the class names already present on a seed frame.

    Each concept gets its own forward pass.  Handing the processor several
    phrases at once does run, but the post-processed result carries no labels
    tying a box back to the phrase that matched it, so there would be no way to
    assign class ids — one pass per concept is the only correct option.
    """
    total = len(frames)
    processed = 0
    wanted = [(text, class_id) for text, class_id in concepts if text]

    for frame_index, png_path in frames:
        if should_abort and should_abort():
            break
        try:
            pil_img = Image.open(png_path).convert("RGB")
            boxes: list[BBox] = []
            for text, class_id in wanted:
                if should_abort and should_abort():
                    break
                boxes.extend(run_sam3(
                    model, processor, pil_img,
                    text=text, class_id=class_id, threshold=threshold,
                    want_masks=want_masks, detail=detail,
                ))
            on_boxes(frame_index, boxes)
        except Exception as exc:
            if on_error:
                on_error(f"Frame {frame_index}: {exc}")
        processed += 1
        if on_progress:
            on_progress(processed, total)
    return processed


# The object id used by the throwaway pre-warm prompt.  Nothing else uses it,
# and `reset_tracking_data` clears it before any real object is added.
_WARMUP_OBJ_ID = -1


class FrameEncodingCache:
    """Holds one encoded frame so that repeat snaps on it cost a decode.

    The tracker spends nearly all of its time turning the picture into vision
    features; matching a box against features that already exist is two orders
    of magnitude cheaper.  Outlining several objects on one frame is the normal
    way to work, so the session that holds those features is kept until the
    user moves to a different frame.

    Callers serialise their own access: snapping and pre-warming both drive the
    same model, so `Session` holds one lock across the pair rather than making
    every method here defensive about a race it is not in a position to resolve.
    """

    def __init__(self) -> None:
        self._key: object = None
        self._session = None

    def holds(self, key: object) -> bool:
        """True when *key* is already encoded — nothing to pre-warm."""
        return key is not None and key == self._key and self._session is not None

    def session_for(self, processor, pil_img: Image.Image, key: object, device: str):
        if key is not None and key == self._key and self._session is not None:
            # Drop the objects from the last snap but keep the vision features:
            # every snap is a fresh question about the same picture.
            self._session.reset_tracking_data()
            return self._session
        self.clear()
        session = processor.init_video_session(video=[pil_img], inference_device=device)
        self._key, self._session = key, session
        return session

    def clear(self) -> None:
        """Let go of the encoded frame — called when the frame changes."""
        self._session = None
        self._key = None


def encode_frame(model, processor, pil_img: Image.Image, cache: FrameEncodingCache,
                 key: object) -> None:
    """Make the tracker read a frame now, so a later snap on it is a decode.

    The encoder runs lazily inside the model's forward pass, not when the
    session is built, so there is no way to ask for features without asking a
    question.  A throwaway box in the corner is that question; the answer is
    dropped and only the features it forced are kept.  Costs one cheap decode
    on top of the encode, and in exchange it goes through exactly the same call
    path `segment_box` will take, rather than reaching into the session's cache.
    """
    import torch

    session = cache.session_for(processor, pil_img, key, best_device())
    processor.add_inputs_to_inference_session(
        inference_session=session, frame_idx=0, obj_ids=[_WARMUP_OBJ_ID],
        input_boxes=[[[0.0, 0.0, 16.0, 16.0]]],
    )
    with torch.inference_mode():
        model(session, frame_idx=0)
    session.reset_tracking_data()   # drop the throwaway object, keep the features


def segment_box(
    model,
    processor,
    pil_img: Image.Image,
    box: tuple[float, float, float, float],
    *,
    detail: str = maskops.DEFAULT_DETAIL,
    cache: "FrameEncodingCache | None" = None,
    cache_key: object = None,
) -> tuple[list[float] | None, tuple[float, float, float, float] | None]:
    """
    Outline the one object inside *box*.

    This uses the memory tracker rather than the concept model, because the two
    answer different questions: a box handed to the concept model means "find
    more things like this" and comes back with every loose match, while the
    tracker segments the object you actually drew around.  Run on a one-frame
    video, it is the classic click-and-snap segmenter.

    Pass a *cache* and a *cache_key* (the frame's image path will do) to keep
    the encoded frame between calls.  Encoding is nearly all of the cost, so
    outlining a second object on a frame you have already touched is a decode.

    Returns (polygon, bounds) in normalised / pixel form, or (None, None) when
    there is nothing to outline.
    """
    import torch

    device = best_device()
    width, height = pil_img.size

    if cache is not None:
        session = cache.session_for(processor, pil_img, cache_key, device)
    else:
        session = processor.init_video_session(video=[pil_img], inference_device=device)
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=0,
        obj_ids=[1],
        input_boxes=[[[float(box[0]), float(box[1]), float(box[2]), float(box[3])]]],
    )

    with torch.inference_mode():
        out = model(session, frame_idx=0)

    res = processor.post_process_masks(
        [out.pred_masks],
        original_sizes=[[session.video_height, session.video_width]],
        binarize=True,
    )[0]
    if res.shape[0] == 0:
        return None, None

    mask = res[0, 0].cpu().numpy().astype(bool)
    polygon = maskops.mask_to_polygon(mask, width, height, detail=detail)
    if not polygon:
        return None, None
    return polygon, maskops.polygon_bounds(polygon, width, height)


def track_through_frames(
    model,
    processor,
    frames: list[tuple[int, str]],
    seed_boxes: list[tuple[tuple[float, float, float, float], int]],
    start_frame_index: int,
    *,
    max_frames: int | None = None,
    want_masks: bool = False,
    detail: str = maskops.DEFAULT_DETAIL,
    on_boxes: Callable[[int, list[BBox]], None],
    on_progress: Callable[[int, int], None] | None = None,
    should_abort: Callable[[], bool] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> int:
    """
    Follow the exact objects boxed on the seed frame through the ordered
    frame list using SAM 3's memory tracker.  Each seed box becomes a tracked
    object carrying its class id.
    """
    import torch

    if not frames or not seed_boxes:
        return 0

    device = best_device()

    # Tracking only ever runs forward from the seed, and the tracker needs every
    # frame decoded up front — so decode just the window we are going to visit.
    # On a long video "track 200 frames" then costs 200 images of RAM, not all
    # 20,000 of them.
    pos_of = {fi: pos for pos, (fi, _) in enumerate(frames)}
    start_pos = pos_of.get(start_frame_index, 0)
    end_pos = len(frames) if not max_frames else min(len(frames), start_pos + max_frames)
    window = frames[start_pos:end_pos]
    if not window:
        return 0

    if on_status:
        on_status(f"Decoding {len(window)} frames for the tracker…")
    images = []
    for _, path in window:
        if should_abort and should_abort():
            return 0
        images.append(Image.open(path).convert("RGB"))

    if on_status:
        on_status("Starting tracker session…")
    session = processor.init_video_session(video=images, inference_device=device)

    # Seed every object on the start frame.
    obj_ids: list[int] = []
    input_boxes: list[list[float]] = []
    objid_to_class: dict[int, int] = {}
    for i, (box, class_id) in enumerate(seed_boxes, start=1):
        obj_ids.append(i)
        input_boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
        objid_to_class[i] = class_id

    # The session indexes the window, so the seed frame is position 0 in it.
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=0,
        obj_ids=obj_ids,
        input_boxes=[input_boxes],
    )

    total = len(window)
    done = 0

    with torch.inference_mode():
        for fpos in range(total):
            if should_abort and should_abort():
                break

            out = model(session, frame_idx=fpos)
            frame_index = window[fpos][0]
            res = processor.post_process_masks(
                [out.pred_masks],
                original_sizes=[[session.video_height, session.video_width]],
                binarize=True,
            )[0]
            scores = getattr(out, "object_score_logits", None)
            session_obj_ids = list(getattr(session, "obj_ids", []))

            boxes: list[BBox] = []
            for i in range(res.shape[0]):
                if scores is not None:
                    try:
                        if float(scores[i]) <= 0:
                            continue  # object not present in this frame
                    except Exception:
                        pass
                mask = res[i, 0].cpu().numpy().astype(bool)
                bb = _mask_to_bbox(mask)
                if bb is None:
                    continue
                oid = session_obj_ids[i] if i < len(session_obj_ids) else None
                x1, y1, x2, y2 = bb
                # The tracker already produces a mask per object, so tracking a
                # segmentation dataset costs nothing extra.
                polygon = None
                if want_masks:
                    height, width = mask.shape[:2]
                    polygon = maskops.mask_to_polygon(mask, width, height, detail=detail)
                    if polygon:
                        x1, y1, x2, y2 = maskops.polygon_bounds(polygon, width, height)
                boxes.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2, polygon=polygon,
                                  class_id=objid_to_class.get(oid, 0), source="sam"))

            on_boxes(frame_index, boxes)
            done += 1
            if on_progress:
                on_progress(done, total)

    return done
