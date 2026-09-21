"""
Exports verified annotations to the standard Ultralytics YOLO dataset format:

  output_dir/
    images/  *.png
    labels/  *.txt   (one line per box: class_id cx cy w h, normalized 0–1)
                     or for segmentation: class_id x1 y1 x2 y2 … xn yn
    data.yaml
"""

from __future__ import annotations

import os
import shutil

import cv2
import yaml

from app.core.sam3_handler import AnnotationStore, BBox


# ---------------------------------------------------------------------------
# YAML helpers — produces inline lists for kpt_shape: [4, 3]
# ---------------------------------------------------------------------------

class _InlineList(list):
    """Marker subclass so the custom representer can target it specifically."""


class _YAMLDumper(yaml.Dumper):
    pass


_YAMLDumper.add_representer(
    _InlineList,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    ),
)


def _bbox_to_yolo(bbox: BBox, img_w: int, img_h: int) -> str:
    cx = ((bbox.x1 + bbox.x2) / 2.0) / img_w
    cy = ((bbox.y1 + bbox.y2) / 2.0) / img_h
    w  = (bbox.x2 - bbox.x1) / img_w
    h  = (bbox.y2 - bbox.y1) / img_h
    # clamp to [0, 1] to handle edge cases from manual drawing
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))
    return f"{bbox.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _bbox_to_yolo_seg(bbox: BBox, img_w: int, img_h: int) -> str:
    """Return a YOLO segmentation label line for *bbox*.

    Uses bbox.polygon (flat normalized [x1,y1,x2,y2,...] list) when available.
    Falls back to the 4-corner rectangle of the bounding box so that manually
    drawn boxes and SAM3 boxes without mask data can still train segmentation
    models (FastSAM, YOLOv8-seg, etc.).
    """
    if bbox.polygon:
        # polygon is already normalized 0-1; clamp each value just in case
        coords = [max(0.0, min(1.0, v)) for v in bbox.polygon]
    else:
        # Build a rectangular polygon from the bbox corners (normalized)
        x1 = max(0.0, min(1.0, bbox.x1 / img_w))
        y1 = max(0.0, min(1.0, bbox.y1 / img_h))
        x2 = max(0.0, min(1.0, bbox.x2 / img_w))
        y2 = max(0.0, min(1.0, bbox.y2 / img_h))
        coords = [x1, y1, x2, y1, x2, y2, x1, y2]
    coords_str = " ".join(f"{v:.6f}" for v in coords)
    return f"{bbox.class_id} {coords_str}"


def _bbox_to_yolo_pose(bbox: BBox, img_w: int, img_h: int) -> str:
    """Return a YOLO pose label line for *bbox* with 4 corner keypoints.

    Keypoint order: TL (0), TR (1), BR (2), BL (3).

    Priority: stored bbox.keypoints (user may have dragged/removed corners) →
    extract_four_corners from polygon → axis-aligned bbox corners.
    None entries (removed corners) export as '0.0 0.0 0' (not labeled).
    """
    from app.core.yolo_pose_converter import extract_four_corners, bbox_corners_tl_tr_br_bl

    cx = max(0.0, min(1.0, ((bbox.x1 + bbox.x2) / 2.0) / img_w))
    cy = max(0.0, min(1.0, ((bbox.y1 + bbox.y2) / 2.0) / img_h))
    bw = max(0.0, min(1.0, (bbox.x2 - bbox.x1) / img_w))
    bh = max(0.0, min(1.0, (bbox.y2 - bbox.y1) / img_h))

    if bbox.keypoints is not None:
        parts = []
        for kpt in bbox.keypoints:
            if kpt is None:
                parts.append("0.000000 0.000000 0")
            else:
                parts.append(f"{kpt[0]:.6f} {kpt[1]:.6f} 1")
        kpt_parts = " ".join(parts)
    elif bbox.polygon:
        polygon_flat = [max(0.0, min(1.0, v)) for v in bbox.polygon]
        kpts = extract_four_corners(polygon_flat)
        kpt_parts = " ".join(f"{x:.6f} {y:.6f} 1" for x, y in kpts)
    else:
        x1 = max(0.0, min(1.0, bbox.x1 / img_w))
        y1 = max(0.0, min(1.0, bbox.y1 / img_h))
        x2 = max(0.0, min(1.0, bbox.x2 / img_w))
        y2 = max(0.0, min(1.0, bbox.y2 / img_h))
        kpts = bbox_corners_tl_tr_br_bl(x1, y1, x2, y2)
        kpt_parts = " ".join(f"{x:.6f} {y:.6f} 1" for x, y in kpts)

    return f"{bbox.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kpt_parts}"


def export_dataset(
    store: AnnotationStore,
    class_names: list[str],
    output_dir: str,
    skip_unverified: bool = False,
    is_seg: bool = False,
    is_pose: bool = False,
) -> dict:
    """
    Write YOLO dataset to output_dir.  Returns what went where, and a warning
    when the split cannot mean anything.
    Frames already exported are re-exported (idempotent).  Exporting records
    itself on the frame and leaves `status` alone: whether a person has
    checked a frame is not something exporting it can answer.
    If skip_unverified is True, frames still in "pending" status are skipped.
    If is_seg is True, labels use YOLO segmentation polygon format.
    If is_pose is True, labels use YOLO pose format with 4 corner keypoints
    (TL, TR, BR, BL) extracted from the segmentation polygon when available.

    Frames held out are listed in val.txt and every other frame in train.txt, so
    a score means something.  Pointing both halves at the same pictures — which
    is what this did until it learned to split — makes a model that has
    memorised the answers look like a model that has learned the job, and the
    better it memorises the better it scores.  With nothing held out that is
    still what happens, so it says so in the warning rather than reporting a
    number that reads like a real one.

    The split is two lists rather than two folders because everything else that
    reads an exported dataset — importing it back, converting it between tasks,
    the folder browser — looks for the pictures directly under images/, and
    moving them into subfolders would quietly empty all of it.
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    listed: dict[str, list[str]] = {"train": [], "val": []}

    for frame_index, annotation in sorted(store.items()):
        if skip_unverified and annotation.status == "pending":
            continue
        if not annotation.boxes:
            continue
        if not os.path.isfile(annotation.image_path):
            continue

        img = cv2.imread(annotation.image_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        split = "val" if annotation.held_out else "train"
        stem = f"frame_{frame_index:06d}"
        dst_img = os.path.join(images_dir, f"{stem}.png")
        dst_lbl = os.path.join(labels_dir, f"{stem}.txt")

        if annotation.image_path != dst_img:
            shutil.copy2(annotation.image_path, dst_img)

        if is_pose:
            lines = [_bbox_to_yolo_pose(b, img_w, img_h) for b in annotation.boxes]
        elif is_seg:
            lines = [_bbox_to_yolo_seg(b, img_w, img_h) for b in annotation.boxes]
        else:
            lines = [_bbox_to_yolo(b, img_w, img_h) for b in annotation.boxes]
        with open(dst_lbl, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        annotation.exported = True
        listed[split].append(f"./images/{stem}.png")

    counts = {split: len(names) for split, names in listed.items()}
    for split, names in listed.items():
        with open(os.path.join(output_dir, f"{split}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(names) + ("\n" if names else ""))

    # Write data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    names_list = list(class_names) if class_names else ["object"]
    if is_pose:
        task = "pose"
    elif is_seg:
        task = "segment"
    else:
        task = "detect"
    # With nothing held out there is no validation half to point at, so it falls
    # back to scoring against the training pictures — the old behaviour, kept so
    # that a session which has never held anything out still trains, and called
    # out so the number it produces is not mistaken for a measurement.
    warning = ""
    if counts["val"] == 0:
        val_list = "train.txt"
        warning = ("No frames are held out, so the score is measured on the same "
                   "pictures the model trains on and will read far better than "
                   "the model is. Hold some frames out to measure anything.")
    else:
        val_list = "val.txt"
    if counts["train"] == 0:
        warning = ("Every exported frame is held out, so there is nothing left to "
                   "train on. Put some frames back in play.")

    data_yaml: dict = {
        "path":  os.path.abspath(output_dir).replace("\\", "/"),
        "train": "train.txt",
        "val":   val_list,
        "nc":    len(names_list),
        "names": names_list,
        "task":  task,
    }
    if is_pose:
        data_yaml["kpt_shape"] = _InlineList([4, 3])
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.dump(data_yaml, fh, Dumper=_YAMLDumper, default_flow_style=False, sort_keys=False)

    return {
        "frames": counts["train"] + counts["val"],
        "train": counts["train"],
        "val": counts["val"],
        "warning": warning,
    }
