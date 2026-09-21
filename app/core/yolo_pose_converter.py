"""Converts a YOLO detection or segmentation dataset to YOLO pose format.

Extracts exactly 4 corner keypoints per instance in strict order:
  Index 0: Top-Left
  Index 1: Top-Right
  Index 2: Bottom-Right
  Index 3: Bottom-Left

For segmentation labels: the polygon is simplified to its 4 corners via
convex-hull → approxPolyDP → minAreaRect fallback.

For detection labels: the 4 bounding-box corners are used directly.

Output label format (one instance per line):
    class_id cx cy w h  x_tl y_tl 2  x_tr y_tr 2  x_br y_br 2  x_bl y_bl 2

data.yaml gains:
    task: pose
    kpt_shape: [4, 3]
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def pairs_from_list(list_file: Path, yaml_data: dict) -> list[tuple[Path, Path | None]]:
    """Read a data.yaml split that names a txt file rather than a folder.

    Both layouts are ordinary YOLO.  A dataset can put its splits in separate
    directories, or keep one images/ directory and name the members of each
    split in a text file — which is what Groundwork's own export writes, since
    holding a frame out changes which list it belongs to and not where its
    picture lives.  Reading only the first meant the app could not convert the
    datasets it had just written.
    """
    root = Path(yaml_data["path"]) if yaml_data.get("path") else yaml_data["yaml_dir"]
    pairs: list[tuple[Path, Path | None]] = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry:
            continue
        img_path = Path(entry)
        if not img_path.is_absolute():
            img_path = Path(os.path.normpath(root / entry))
        if img_path.suffix.lower() not in _IMAGE_EXTS or not img_path.is_file():
            continue
        lbl = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        pairs.append((img_path, lbl if lbl.is_file() else None))
    return pairs


def _write_split_lists(output_root: Path, listed: dict[str, list[str]]) -> set[str]:
    """Write train.txt / val.txt / test.txt beside a converted dataset.

    Returns the names actually written, so a split that produced nothing is
    left out rather than written empty — an empty val.txt trains fine and
    scores nothing, which looks like success.
    """
    written: set[str] = set()
    for split_name, names in listed.items():
        if not names:
            continue
        (output_root / f"{split_name}.txt").write_text(
            "\n".join(names) + "\n", encoding="utf-8")
        written.add(split_name)
    return written


_log = logging.getLogger(__name__)

# ── Shared corner-extraction helper ─────────────────────────────────────────

def extract_four_corners(
    polygon_flat: list[float],
) -> list[tuple[float, float]]:
    """Return the 4 corners of a quadrilateral in [TL, TR, BR, BL] order.

    Works in normalized (0-1) coordinate space.

    Pipeline:
      1. Convex hull of all polygon points.
      2. approxPolyDP to simplify hull to 4 vertices (epsilon auto-tuned).
      3. minAreaRect fallback if simplification fails.

    Ordering uses the sum/difference rule (robust for any rotation):
      TL = argmin(x+y),  TR = argmax(x-y),
      BR = argmax(x+y),  BL = argmin(x-y)
    """
    pts = np.array(polygon_flat, dtype=np.float32).reshape(-1, 2)

    if len(pts) < 3:
        xs, ys = pts[:, 0], pts[:, 1]
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()), float(ys.max())
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    # Scale to integer space for OpenCV (10 000 gives sub-pixel precision)
    SCALE = 10_000
    pts_int = (pts * SCALE).astype(np.int32)

    hull = cv2.convexHull(pts_int).reshape(-1, 2)

    if len(hull) == 4:
        corners_f = hull.astype(np.float32)
    elif len(hull) < 4:
        # Fewer than 4 hull points — use the bbox corners
        xs, ys = pts[:, 0], pts[:, 1]
        x1, y1 = xs.min() * SCALE, ys.min() * SCALE
        x2, y2 = xs.max() * SCALE, ys.max() * SCALE
        corners_f = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    else:
        # Simplify hull → 4 vertices
        hull_i32 = hull.astype(np.int32)
        perimeter = cv2.arcLength(hull_i32, True)
        corners_f = None
        for frac in (0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35):
            approx = cv2.approxPolyDP(hull_i32, frac * perimeter, True).reshape(-1, 2)
            if len(approx) == 4:
                corners_f = approx.astype(np.float32)
                break
        if corners_f is None:
            # Last resort: minimum area bounding rectangle
            rect = cv2.minAreaRect(pts_int.astype(np.float32))
            corners_f = cv2.boxPoints(rect)  # float32, 4 points

    # Normalize back to 0-1
    corners = np.clip(corners_f / SCALE, 0.0, 1.0)

    # Order TL, TR, BR, BL
    sums  = corners[:, 0] + corners[:, 1]
    diffs = corners[:, 0] - corners[:, 1]
    tl = corners[int(np.argmin(sums))]
    br = corners[int(np.argmax(sums))]
    tr = corners[int(np.argmax(diffs))]
    bl = corners[int(np.argmin(diffs))]

    return [
        (float(tl[0]), float(tl[1])),
        (float(tr[0]), float(tr[1])),
        (float(br[0]), float(br[1])),
        (float(bl[0]), float(bl[1])),
    ]


def bbox_corners_tl_tr_br_bl(
    x1: float, y1: float, x2: float, y2: float
) -> list[tuple[float, float]]:
    """Return the 4 axis-aligned bbox corners in TL, TR, BR, BL order."""
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


# ── Converter ───────────────────────────────────────────────────────────────

class YoloPoseConverter:
    """Pure-Python pipeline — no Qt dependency."""

    def __init__(
        self,
        source_root: str,
        output_root: str,
        edge_margin: float = 0.02,
        progress_callback: Callable[[int, int], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> None:
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.edge_margin = max(0.0, min(0.49, edge_margin))
        self._progress_callback = progress_callback
        self._status_callback = status_callback
        self._should_abort = should_abort

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

        total = len(all_pairs)
        self._emit_status(
            f"Found {total} images. Extracting 4 corner keypoints (TL, TR, BR, BL)…"
        )

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
            "n_keypoints": 4,
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
            return pairs_from_list(candidate, yaml_data)

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
    # Edge-margin filtering
    # ------------------------------------------------------------------

    def _filter_edge(
        self, kpts: list[tuple[float, float]]
    ) -> list[tuple[float, float] | None]:
        """Return None for any keypoint within edge_margin of any image border."""
        m = self.edge_margin
        result: list[tuple[float, float] | None] = []
        for x, y in kpts:
            if x < m or x > 1.0 - m or y < m or y > 1.0 - m:
                result.append(None)
            else:
                result.append((x, y))
        return result

    # ------------------------------------------------------------------
    # Image processing
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

        shutil.copy2(image_path, out_img_dir / image_path.name)

        out_lbl_path = out_lbl_dir / (image_path.stem + ".txt")

        if label_path is None or not label_path.is_file():
            out_lbl_path.write_text("", encoding="utf-8")
            return 0, 0

        out_lines: list[str] = []
        converted = 0
        fallback = 0

        with open(label_path, encoding="utf-8") as fh:
            for raw_line in fh:
                parts = raw_line.strip().split()
                if not parts:
                    continue

                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue

                n_vals = len(parts) - 1

                if n_vals == 4:
                    # Detection format: cx cy w h — use bbox corners directly
                    try:
                        cx, cy, bw, bh = (float(p) for p in parts[1:5])
                    except ValueError:
                        continue
                    x1 = max(0.0, cx - bw / 2)
                    y1 = max(0.0, cy - bh / 2)
                    x2 = min(1.0, cx + bw / 2)
                    y2 = min(1.0, cy + bh / 2)
                    kpts = self._filter_edge(bbox_corners_tl_tr_br_bl(x1, y1, x2, y2))
                    out_lines.append(self._make_pose_line(class_id, cx, cy, bw, bh, kpts))
                    fallback += 1

                elif n_vals >= 6 and n_vals % 2 == 0:
                    # Segmentation polygon: extract 4 corners
                    try:
                        polygon_flat = [float(p) for p in parts[1:]]
                    except ValueError:
                        continue
                    # Clamp polygon to [0,1] before deriving the bbox so that
                    # objects partially outside the image don't produce w/h > 1.
                    poly_clamped = [max(0.0, min(1.0, v)) for v in polygon_flat]
                    pts = np.array(poly_clamped).reshape(-1, 2)
                    xs, ys = pts[:, 0], pts[:, 1]
                    cx = float((xs.min() + xs.max()) / 2)
                    cy = float((ys.min() + ys.max()) / 2)
                    bw = float(xs.max() - xs.min())
                    bh = float(ys.max() - ys.min())
                    kpts = self._filter_edge(extract_four_corners(polygon_flat))
                    out_lines.append(self._make_pose_line(class_id, cx, cy, bw, bh, kpts))
                    converted += 1

                elif n_vals >= 5 and (n_vals - 4) % 3 == 0:
                    # Already pose format — pass through unchanged
                    out_lines.append(raw_line.strip())
                    converted += 1

        with open(out_lbl_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out_lines) + ("\n" if out_lines else ""))

        return converted, fallback

    @staticmethod
    def _make_pose_line(
        class_id: int,
        cx: float, cy: float, bw: float, bh: float,
        keypoints: list[tuple[float, float] | None],
    ) -> str:
        # Clamp bbox values so objects that extend outside the image don't
        # produce values > 1.0, which Ultralytics rejects as out-of-bounds.
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        bbox_str = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        # Use visibility=1 ("labeled") instead of 2 ("labeled+visible").
        # Both mean "include in loss". Using 1 avoids older Ultralytics
        # builds that check ALL label values against <= 1 (visibility=2 fails).
        parts = [
            "0.000000 0.000000 0" if kpt is None else f"{kpt[0]:.6f} {kpt[1]:.6f} 1"
            for kpt in keypoints
        ]
        return f"{bbox_str} {' '.join(parts)}"

    # ------------------------------------------------------------------
    # Output YAML
    # ------------------------------------------------------------------

    def _write_yaml(
        self,
        output_root: Path,
        yaml_data: dict,
        splits: list[str],
        listed: dict[str, list[str]] | None = None,
    ) -> None:
        out: dict = {
            "path":      output_root.resolve().as_posix(),
            "nc":        yaml_data["nc"],
            "names":     yaml_data["_raw"].get("names"),
            "task":      "pose",
            "kpt_shape": [4, 3],
        }
        # Every picture lands in one images/ directory, so pointing both halves
        # at that directory would make the validation half the training half —
        # a model scored on what it memorised, reading better the more it
        # overfits.  The split the source came in with is written back out as
        # lists instead, so converting a dataset cannot silently undo it.
        written = _write_split_lists(output_root, listed or {})
        for split_name in ("train", "val", "test"):
            if split_name in written:
                out[split_name] = f"{split_name}.txt"
            elif split_name in splits or split_name in ("train", "val"):
                out[split_name] = "images"
        # Use inline list for kpt_shape to match Ultralytics convention: [4, 3]
        class _Inline(list):
            pass
        class _D(yaml.Dumper):
            pass
        _D.add_representer(
            _Inline,
            lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True),
        )
        out["kpt_shape"] = _Inline(out["kpt_shape"])
        with open(output_root / "data.yaml", "w", encoding="utf-8") as fh:
            yaml.dump(out, fh, Dumper=_D, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Qt worker wrapper
# ---------------------------------------------------------------------------
