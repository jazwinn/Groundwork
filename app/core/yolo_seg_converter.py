"""Converts a YOLO detection dataset to instance segmentation format using SAM 3."""

from __future__ import annotations

import logging
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml
from PIL import Image

from app.core.yolo_pose_converter import (
    pairs_from_list as _pairs_from_list,
    _write_split_lists,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_log = logging.getLogger(__name__)


class YoloDatasetConverter:
    """Pure-Python pipeline — no Qt dependency."""

    def __init__(
        self,
        source_root: str,
        output_root: str,
        model_id: str = "facebook/sam3",
        progress_callback: Callable[[int, int], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
        model=None,
        processor=None,
        should_abort: Callable[[], bool] | None = None,
    ) -> None:
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.model_id = model_id
        self._progress_callback = progress_callback
        self._status_callback = status_callback
        self._model = model
        self._processor = processor
        self._should_abort = should_abort
        self._class_names: list[str] = []

    def _emit_status(self, msg: str) -> None:
        if self._status_callback:
            self._status_callback(msg)
        _log.info(msg)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def convert(self) -> dict:
        t0 = time.time()
        converted_items = 0
        fallback_items = 0
        failed_items = 0

        yaml_data = self._parse_yaml()
        self._class_names = yaml_data["names"]

        if self._model is None:
            self._emit_status("Loading SAM 3 model — this may take a minute…")
            self._load_model()
        self._emit_status("Model ready. Scanning dataset…")

        active_splits: list[str] = []
        all_pairs: list[tuple[str, Path, Path | None]] = []
        seen_img_paths: set[Path] = set()

        for split_name in ("train", "val", "test"):
            raw = yaml_data["splits"].get(split_name)
            if raw is None:
                continue
            pairs = self._collect_pairs(split_name, raw, yaml_data)
            added = 0
            for img_path, lbl_path in pairs:
                resolved = img_path.resolve()
                if resolved in seen_img_paths:
                    continue
                seen_img_paths.add(resolved)
                all_pairs.append((split_name, img_path, lbl_path))
                added += 1
            if added:
                active_splits.append(split_name)
                _log.info("Split %s: %d images", split_name, added)

        total = len(all_pairs)
        self._emit_status(f"Found {total} images across {len(active_splits)} split(s). Converting…")

        aborted = False
        listed: dict[str, list[str]] = {}
        for i, (split_name, img_path, lbl_path) in enumerate(all_pairs):
            if self._should_abort and self._should_abort():
                aborted = True
                break
            self._emit_status(f"{i + 1}/{total}  {img_path.name}")

            out_img_dir = self.output_root / "images"
            out_lbl_dir = self.output_root / "labels"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            try:
                c, f = self._process_image(img_path, lbl_path, out_img_dir, out_lbl_dir)
                converted_items += c
                fallback_items += f
                # Which half this picture came from, kept so the output can be
                # split the same way the input was.
                listed.setdefault(split_name, []).append(f"./images/{img_path.name}")
            except Exception as exc:
                _log.warning("Skipping %s: %s", img_path.name, exc)
                failed_items += 1

            if self._progress_callback:
                self._progress_callback(i + 1, total)

        self._write_yaml(self.output_root, yaml_data, active_splits, listed)

        return {
            "status": "cancelled" if aborted else "success",
            "output_dir": str(self.output_root),
            "converted_items": converted_items,
            "fallback_items": fallback_items,
            "failed_items": failed_items,
            "time_taken_sec": round(time.time() - t0, 2),
        }

    # ------------------------------------------------------------------
    # YAML parsing
    # ------------------------------------------------------------------

    def _parse_yaml(self) -> dict:
        yaml_path = self.source_root / "data.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(f"data.yaml not found in {self.source_root}")

        with open(yaml_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        names = raw.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        names = [str(n) for n in names]

        return {
            "names": names,
            "nc": raw.get("nc", len(names)),
            "splits": {
                "train": raw.get("train"),
                "val":   raw.get("val"),
                "test":  raw.get("test"),
            },
            "path":     raw.get("path"),
            "yaml_dir": yaml_path.parent,
            "_raw":     raw,
        }

    # ------------------------------------------------------------------
    # Pair collection
    # ------------------------------------------------------------------

    def _collect_pairs(
        self,
        split_name: str,
        split_path_raw: str,
        yaml_data: dict,
    ) -> list[tuple[Path, Path | None]]:
        candidate = Path(split_path_raw)

        if not candidate.is_absolute():
            if yaml_data["path"] is not None:
                candidate = Path(yaml_data["path"]) / split_path_raw
                if not candidate.exists():
                    candidate = yaml_data["yaml_dir"] / split_path_raw
            else:
                candidate = yaml_data["yaml_dir"] / split_path_raw

        if not candidate.exists():
            _log.warning("Split not found: %s", candidate)
            return []

        if candidate.is_file():
            return _pairs_from_list(candidate, yaml_data)

        images_dir = (candidate / "images") if (candidate / "images").is_dir() else candidate
        labels_dir = images_dir.parent / "labels"

        pairs: list[tuple[Path, Path | None]] = []
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            lbl = labels_dir / (img_path.stem + ".txt")
            pairs.append((img_path, lbl if lbl.is_file() else None))

        return pairs

    # ------------------------------------------------------------------
    # SAM 3 model
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return self._model, self._processor

        import torch
        from app.core.sam3_handler import _load_sam3

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._model, self._processor = _load_sam3(self.model_id, local_only=True)
        except Exception:
            self._model, self._processor = _load_sam3(self.model_id, local_only=False)

        # CRITICAL: _load_sam3 picks float16 when CUDA is available but never
        # moves the model off the CPU. Running fp16 on the CPU is pathologically
        # slow (no native fp16 compute) and looks like a hang. Move to the GPU,
        # or fall back to float32 if we're stuck on the CPU.
        if device == "cuda":
            self._model = self._model.to(device).eval()
        else:
            self._model = self._model.float().to(device).eval()

        self._emit_status(f"SAM 3 model ready on {device.upper()}.")
        return self._model, self._processor

    # ------------------------------------------------------------------
    # Image processing — ONE SAM 3 call per unique class per image
    # ------------------------------------------------------------------

    def _process_image(
        self,
        image_path: Path,
        label_path: Path | None,
        out_img_dir: Path,
        out_lbl_dir: Path,
    ) -> tuple[int, int]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"cv2 could not read {image_path}")
        img_h, img_w = img.shape[:2]

        shutil.copy2(image_path, out_img_dir / image_path.name)

        out_lbl_path = out_lbl_dir / (image_path.stem + ".txt")

        if label_path is None or not label_path.is_file():
            out_lbl_path.write_text("", encoding="utf-8")
            return 0, 0

        # --- Pass 1: parse all label lines ---------------------------------
        # Each annotation: dict with index, class_id, bbox coords, original line
        annotations: list[dict] = []
        passthrough_lines: list[tuple[int, str]] = []  # (index, line) for already-seg

        with open(label_path, encoding="utf-8") as fh:
            for raw_line in fh:
                parts = raw_line.strip().split()
                if not parts:
                    continue

                idx = len(annotations) + len(passthrough_lines)

                # Already segmentation format — keep unchanged
                if len(parts) > 5:
                    passthrough_lines.append((idx, raw_line.strip()))
                    continue

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])
                    cx, cy, bw, bh = (float(p) for p in parts[1:5])
                except ValueError:
                    continue

                xmin = (cx - bw / 2) * img_w
                ymin = (cy - bh / 2) * img_h
                xmax = (cx + bw / 2) * img_w
                ymax = (cy + bh / 2) * img_h
                annotations.append({
                    "order": idx,
                    "class_id": class_id,
                    "xmin": xmin, "ymin": ymin,
                    "xmax": xmax, "ymax": ymax,
                })

        if not annotations and not passthrough_lines:
            out_lbl_path.write_text("", encoding="utf-8")
            return 0, 0

        # --- Pass 2: SAM 3 inference grouped by class ----------------------
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Group annotation indices by class_id
        class_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, ann in enumerate(annotations):
            class_to_indices[ann["class_id"]].append(i)

        # For each unique class, run SAM 3 ONCE with all its boxes as exemplars
        # Then match every returned mask to its closest input bbox
        ann_polygons: dict[int, list[float] | None] = {}

        for class_id, ann_indices in class_to_indices.items():
            class_name = (
                self._class_names[class_id]
                if class_id < len(self._class_names)
                else "object"
            )
            bboxes = [
                [annotations[i]["xmin"], annotations[i]["ymin"],
                 annotations[i]["xmax"], annotations[i]["ymax"]]
                for i in ann_indices
            ]

            all_masks = self._get_masks_for_class(
                pil_img, class_name, bboxes, img_w, img_h
            )

            # Match each input bbox to its best mask by IoU
            for i, ann_idx in enumerate(ann_indices):
                ann = annotations[ann_idx]
                mask = self._select_best_mask(
                    all_masks,
                    ann["xmin"], ann["ymin"],
                    ann["xmax"], ann["ymax"],
                    img_w, img_h,
                )
                if mask is not None:
                    polygon = self._mask_to_polygon(
                        mask.astype(np.uint8) * 255, img_w, img_h
                    )
                else:
                    polygon = None
                ann_polygons[ann_idx] = polygon

        # --- Pass 3: assemble output lines in original order ---------------
        converted = 0
        fallback = 0
        out_lines: list[str] = []

        # Merge passthrough and newly computed lines sorted by original order
        pt_dict = {order: line for order, line in passthrough_lines}
        converted += len(passthrough_lines)

        # Rebuild in order: passthrough lines + annotation lines
        # Simpler: just append passthrough first, then annotations
        for _, line in passthrough_lines:
            out_lines.append(line)

        for ann_idx, ann in enumerate(annotations):
            class_id = ann["class_id"]
            polygon = ann_polygons.get(ann_idx)

            if polygon is None:
                polygon = self._bbox_to_fallback_polygon(
                    ann["xmin"], ann["ymin"],
                    ann["xmax"], ann["ymax"],
                    img_w, img_h,
                )
                fallback += 1
            else:
                converted += 1

            coords_str = " ".join(f"{v:.6f}" for v in polygon)
            out_lines.append(f"{class_id} {coords_str}")

        with open(out_lbl_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out_lines) + ("\n" if out_lines else ""))

        return converted, fallback

    # ------------------------------------------------------------------
    # SAM 3 inference — one call for all boxes of the same class
    # ------------------------------------------------------------------

    def _get_masks_for_class(
        self,
        pil_img: Image.Image,
        class_name: str,
        bboxes_pixel: list[list[float]],
        img_w: int,
        img_h: int,
    ) -> list[np.ndarray]:
        """Run SAM 3 with all bboxes of one class as positive exemplars.

        Returns a list of (H, W) bool numpy masks (may be empty on failure).
        """
        import torch

        model, processor = self._load_model()

        try:
            inputs = processor(
                images=pil_img,
                text=class_name,
                input_boxes=[bboxes_pixel],
                input_boxes_labels=[[1] * len(bboxes_pixel)],
                return_tensors="pt",
            )
        except Exception:
            # Fallback: text-only if processor rejects box kwargs
            try:
                inputs = processor(
                    images=pil_img,
                    text=class_name,
                    return_tensors="pt",
                )
            except Exception:
                return []

        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        inputs = {
            k: (v.to(device, dtype=model_dtype)
                if hasattr(v, "to") and v.dtype.is_floating_point
                else v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        try:
            with torch.inference_mode():
                outputs = model(**inputs)
        except Exception as exc:
            _log.warning("SAM 3 inference error: %s", exc)
            return []

        try:
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=[[img_h, img_w]],
            )[0]
        except Exception as exc:
            _log.warning("SAM 3 post-process error: %s", exc)
            return []

        raw_masks = results.get("masks")
        if raw_masks is None or (hasattr(raw_masks, "__len__") and len(raw_masks) == 0):
            return []

        out: list[np.ndarray] = []
        for raw in raw_masks:
            if hasattr(raw, "cpu"):
                m = raw.cpu().numpy().astype(bool)
            else:
                m = np.asarray(raw, dtype=bool)
            while m.ndim > 2:
                m = m.squeeze(0)
            if m.shape != (img_h, img_w):
                m = cv2.resize(
                    m.astype(np.uint8), (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            out.append(m)

        return out

    # ------------------------------------------------------------------
    # Mask utilities
    # ------------------------------------------------------------------

    def _select_best_mask(
        self,
        masks: list[np.ndarray],
        xmin: float, ymin: float,
        xmax: float, ymax: float,
        img_w: int, img_h: int,
    ) -> np.ndarray | None:
        if not masks:
            return None

        ref = np.zeros((img_h, img_w), dtype=bool)
        x0, y0 = max(0, int(xmin)), max(0, int(ymin))
        x1, y1 = min(img_w, int(xmax)), min(img_h, int(ymax))
        ref[y0:y1, x0:x1] = True

        best_iou = 0.0
        best: np.ndarray | None = None
        for m in masks:
            union = (m | ref).sum()
            if union == 0:
                continue
            iou = float((m & ref).sum()) / float(union)
            if iou > best_iou:
                best_iou = iou
                best = m

        return best if best_iou > 0.0 else None

    def _mask_to_polygon(
        self, mask: np.ndarray, img_w: int, img_h: int
    ) -> list[float] | None:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 1.0:
            return None
        pts = largest.reshape(-1, 2).astype(float)
        pts[:, 0] /= img_w
        pts[:, 1] /= img_h
        pts = pts.clip(0.0, 1.0)
        return pts.flatten().tolist()

    def _bbox_to_fallback_polygon(
        self,
        xmin: float, ymin: float,
        xmax: float, ymax: float,
        img_w: int, img_h: int,
    ) -> list[float]:
        return [
            xmin / img_w, ymin / img_h,
            xmax / img_w, ymin / img_h,
            xmax / img_w, ymax / img_h,
            xmin / img_w, ymax / img_h,
        ]

    # ------------------------------------------------------------------
    # Output YAML
    # ------------------------------------------------------------------

    def _write_yaml(
        self, output_root: Path, yaml_data: dict, splits: list[str],
        listed: dict[str, list[str]] | None = None,
    ) -> None:
        out: dict = {
            "path":  output_root.resolve().as_posix(),
            "nc":    yaml_data["nc"],
            "names": yaml_data["_raw"].get("names"),
            "task":  "segment",
        }
        # All splits share the same flat images/ folder at the dataset root, so
        # naming that folder for both halves would make the validation half the
        # training half — a model scored on what it memorised, reading better
        # the more it overfits.  The split the source came in with is written
        # back out as lists, so converting cannot silently undo it.  Falling
        # back to the folder is kept for a source that had no split to carry:
        # Ultralytics needs train and val to exist at all.
        written = _write_split_lists(output_root, listed or {})
        for split_name in ("train", "val", "test"):
            if split_name in written:
                out[split_name] = f"{split_name}.txt"
            elif split_name in splits or split_name in ("train", "val"):
                out[split_name] = "images"
        with open(output_root / "data.yaml", "w", encoding="utf-8") as fh:
            yaml.dump(out, fh, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Qt worker wrapper
# ---------------------------------------------------------------------------
