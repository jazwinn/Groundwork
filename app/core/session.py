"""
Session — the application brain.

Holds everything that used to live on MainWindow: the annotation store, the
class list, the loaded SAM 3 models, and the background jobs.  It is UI-free;
the web layer in app/server/api.py only translates HTTP/websocket traffic into
calls on this object.

Work is autosaved to <project>/.groundwork/session/session.json, so closing
the browser (or the server) no longer throws away an afternoon of labelling.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from typing import Callable, Iterable

from app.core import boxops, maskops, media_loader, scripts
from app.core.jobs import Job, JobManager
from app.core.sam3_handler import (
    AnnotationStore, BBox, FrameAnnotation, SAM_MODELS,
    FrameEncodingCache, encode_frame, load_sam3, load_sam3_tracker,
    run_concepts_over_frames, run_sam3, segment_box, track_through_frames,
)
from app.core.yolo_trainer import _PROJECT_ROOT

SESSION_ROOT = os.path.join(_PROJECT_ROOT, ".groundwork", "session")
FRAMES_DIR = os.path.join(SESSION_ROOT, "frames")
THUMBS_DIR = os.path.join(SESSION_ROOT, "thumbs")
STATE_FILE = os.path.join(SESSION_ROOT, "session.json")
# Copies of session.json taken before a sweeping change, so one can be undone
# by putting a file back.  Only the most recent are kept.
BACKUP_DIR = os.path.join(SESSION_ROOT, "backups")

# Sessions written before the project was renamed live here.  They are moved
# across on first load rather than read where they lie, so that a session has
# one home rather than two.
_LEGACY_ROOT = os.path.join(_PROJECT_ROOT, ".modeltrainer", "session")

THUMB_SIZE = 160

# How long a pre-warm waits before committing to an encode, so that flipping
# through frames does not queue one encode per frame passed through.
PREWARM_SETTLE = 0.25

# What to do with labels that already exist on a frame an auto-label run touches.
REPLACE = "replace"   # throw away what is there and use the new boxes
MERGE   = "merge"     # keep existing boxes and append the new ones
SKIP    = "skip"      # leave already-labelled frames completely alone

TASKS = ("detect", "segment", "pose")


def _adopt_legacy_session() -> None:
    """Move a session written before the rename into its new home, once.

    Failing is not fatal — `load_state` falls back to reading the old folder
    where it lies, and the frames are found by name either way.
    """
    if os.path.isfile(STATE_FILE) or not os.path.isdir(_LEGACY_ROOT):
        return
    try:
        if os.path.isdir(SESSION_ROOT):
            # Starting the renamed app creates this folder before there is
            # anything to put in it.  An empty one is just in the way; one with
            # files in it is somebody's work, so both are left where they are.
            if any(files for _, _, files in os.walk(SESSION_ROOT)):
                return
            shutil.rmtree(SESSION_ROOT, ignore_errors=True)
        os.makedirs(os.path.dirname(SESSION_ROOT), exist_ok=True)
        os.rename(_LEGACY_ROOT, SESSION_ROOT)
        old_parent = os.path.dirname(_LEGACY_ROOT)
        if os.path.isdir(old_parent) and not os.listdir(old_parent):
            os.rmdir(old_parent)
    except OSError:
        pass


# Windows refuses to rename onto a file that somebody else still has open, and
# on a machine where the project folder is synced or virus-scanned "somebody
# else" is routine — the save fails with a sharing violation through no fault
# of ours.  The other handle is held for a moment rather than a minute, so a
# few quick retries turn a failed save into a very slightly slower one.
_REPLACE_ATTEMPTS = 12
_REPLACE_PAUSE = 0.05


def _replace_atomically(tmp: str, path: str) -> None:
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_PAUSE)


def _locate_frame(image_path: str) -> str | None:
    """Find a frame whose recorded path no longer resolves.

    Paths are stored absolute, so renaming or moving the project folder would
    otherwise drop every frame on load without saying a word.  The pictures
    travel inside the session folder, so the file name is enough to find them
    again wherever that folder has ended up.
    """
    if os.path.isfile(image_path):
        return image_path
    for folder in (FRAMES_DIR, os.path.join(_LEGACY_ROOT, "frames")):
        moved = os.path.join(folder, os.path.basename(image_path))
        if os.path.isfile(moved):
            return moved
    return None


class Session:
    def __init__(self, emit: Callable[[dict], None] | None = None):
        self.store: AnnotationStore = {}
        self.class_names: list[str] = ["object"]
        self.task: str = "segment"
        self.sam_model_key: str = next(iter(SAM_MODELS))
        self.threshold: float = 0.5
        self.mask_detail: str = maskops.DEFAULT_DETAIL

        self._emit = emit
        self.jobs = JobManager()
        self._lock = threading.RLock()
        self._save_lock = threading.Lock()   # one writer at a time in save_state

        # Loaded models are cached for the life of the process
        self._sam_model = None
        self._sam_processor = None
        self._sam_loaded_key: str | None = None
        self._tracker_model = None
        self._tracker_processor = None
        self._snap_cache = FrameEncodingCache()
        self._snap_lock = threading.Lock()   # snapping and pre-warming share a model
        self._prewarm_want: object = None    # newest frame asked for, to skip stale work

        # source video path -> import stride used (drives the tracking warning)
        self.video_stride: dict[str, int] = {}

        self._dirty = False
        self._autosave_stop = threading.Event()
        # Before anything creates the new folder — an existing one is taken as
        # a sign there is nothing to bring across.
        _adopt_legacy_session()
        os.makedirs(FRAMES_DIR, exist_ok=True)
        os.makedirs(THUMBS_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Wiring
    # ──────────────────────────────────────────────────────────────

    def set_emitter(self, emit: Callable[[dict], None]) -> None:
        self._emit = emit
        self.jobs.set_emitter(emit)

    def emit(self, payload: dict) -> None:
        if self._emit:
            try:
                self._emit(payload)
            except Exception:
                pass

    def toast(self, message: str, level: str = "info") -> None:
        self.emit({"type": "toast", "level": level, "message": message})

    def start_autosave(self, interval: float = 4.0) -> None:
        def _loop() -> None:
            while not self._autosave_stop.wait(interval):
                if self._dirty:
                    try:
                        self.save_state()
                    except Exception:
                        pass

        threading.Thread(target=_loop, name="autosave", daemon=True).start()

    def shutdown(self) -> None:
        self._autosave_stop.set()
        self.jobs.cancel_all()
        try:
            self.save_state()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────

    def touch(self) -> None:
        self._dirty = True

    def save_state(self, path: str | None = None) -> str:
        """Write the session out, atomically, one writer at a time.

        Every save used to go through a single "session.json.tmp", and saves
        come from two places at once: the autosave thread, and whichever
        request thread just changed something.  Overlap them and one is still
        writing the temp file when the other renames it away — on Windows that
        is a PermissionError out of `os.replace`, which surfaced as a 500 on a
        request whose edit had *already* been applied in memory.  Serialising
        the whole save (snapshot included, so the newest one lands last) is
        what actually fixes that; the per-save temp name keeps two processes
        pointed at the same session folder from colliding the same way.
        """
        path = path or STATE_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self._save_lock:
            with self._lock:
                data = {
                    "version": 2,
                    "saved_at": time.time(),
                    "class_names": self.class_names,
                    "task": self.task,
                    "sam_model_key": self.sam_model_key,
                    "threshold": self.threshold,
                    "mask_detail": self.mask_detail,
                    "video_stride": self.video_stride,
                    "frames": [ann.to_dict() for ann in self.store.values()],
                }
                # Cleared against the snapshot, not after the write: anything
                # changed while this save is in flight re-dirties the session
                # and goes out with the next one instead of being marked saved.
                self._dirty = False
            tmp = f"{path}.{os.getpid()}.{threading.get_ident():x}.tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
                _replace_atomically(tmp, path)
            except BaseException:
                # Nothing reached session.json, so the session is still unsaved.
                self._dirty = True
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
        return path

    def flush_state(self, what: str = "") -> bool:
        """Save now if we can, and say so rather than throw if we cannot.

        For the callers that have *already* applied their change in memory.
        The change is not lost when this returns False — it is still in the
        store, still dirty, and the autosave thread will try again.
        """
        try:
            self.save_state()
            return True
        except Exception as exc:
            print(f"Could not save the session: {exc}", file=sys.stderr)
            self.toast(f"{what + ': ' if what else ''}the change was applied but "
                       f"could not be saved just now — retrying in the background.",
                       "warn")
            return False

    def load_state(self, path: str | None = None) -> int:
        if path is None:
            _adopt_legacy_session()
            path = STATE_FILE if os.path.isfile(STATE_FILE) else os.path.join(_LEGACY_ROOT, "session.json")
        if not os.path.isfile(path):
            return 0
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        frames: list[FrameAnnotation] = []
        repaired = 0
        for entry in data.get("frames", []):
            ann = FrameAnnotation.from_dict(entry)
            found = _locate_frame(ann.image_path)
            if found is None:
                continue          # the picture is genuinely gone
            if found != ann.image_path:
                ann.image_path = found
                repaired += 1
            frames.append(ann)

        with self._lock:
            self.class_names = data.get("class_names") or ["object"]
            self.task = data.get("task", "segment")
            self.sam_model_key = data.get("sam_model_key", self.sam_model_key)
            self.threshold = float(data.get("threshold", 0.5))
            self.mask_detail = data.get("mask_detail", maskops.DEFAULT_DETAIL)
            self.video_stride = {k: int(v) for k, v in (data.get("video_stride") or {}).items()}
            self.store = {f.frame_index: f for f in frames}
        # Corrected paths are written back on the next save rather than kept
        # only in memory, so the repair happens once and not on every start.
        self._dirty = bool(repaired)
        return len(frames)

    # ──────────────────────────────────────────────────────────────
    # Frame access
    # ──────────────────────────────────────────────────────────────

    def frame_meta(self, ann: FrameAnnotation) -> dict:
        return {
            "index": ann.frame_index,
            "status": ann.status,
            "boxes": len(ann.boxes),
            "width": ann.width,
            "height": ann.height,
            "source": os.path.basename(ann.source_video or ann.source_image or ""),
            "isVideo": bool(ann.source_video),
            "heldOut": ann.held_out,
            "exported": ann.exported,
        }

    def frames_meta(self) -> list[dict]:
        with self._lock:
            return [self.frame_meta(a) for _, a in sorted(self.store.items())]

    def get(self, index: int) -> FrameAnnotation | None:
        return self.store.get(index)

    def thumbnail(self, index: int) -> str | None:
        """Return a path to a cached thumbnail, creating it on first request."""
        ann = self.store.get(index)
        if ann is None or not os.path.isfile(ann.image_path):
            return None
        thumb = os.path.join(THUMBS_DIR, f"{index:06d}.jpg")
        if os.path.isfile(thumb):
            return thumb
        import cv2

        img = cv2.imread(ann.image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = THUMB_SIZE / max(h, w, 1)
        if scale < 1:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(thumb, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return thumb

    def is_held_out(self, index: int) -> bool:
        ann = self.store.get(index)
        return bool(ann is not None and ann.held_out)

    def set_held_out(self, indices: Iterable[int], held: bool) -> int:
        """Hold frames out to be measured against, or put them back in play.

        Holding a frame out does not change its labels; it stops anything else
        from changing them.  Whoever holds a frame out is saying its labels are
        the ones to trust, so this is the one thing an agent working on the
        session should not be doing to itself.
        """
        changed = 0
        for index in indices:
            ann = self.store.get(index)
            if ann is None or ann.held_out == bool(held):
                continue
            ann.held_out = bool(held)
            changed += 1
            self.emit({"type": "frame", "event": "updated", "frame": self.frame_meta(ann)})
        if changed:
            self.touch()
            self.save_state()
        return changed

    def set_boxes(self, index: int, boxes: list[BBox], status: str | None = None) -> dict | None:
        ann = self.store.get(index)
        if ann is None or ann.held_out:
            return None
        ann.boxes = boxes
        if status:
            ann.status = status
        elif boxes:
            ann.status = "verified"
        else:
            ann.status = "pending"
        self.touch()
        meta = self.frame_meta(ann)
        self.emit({"type": "frame", "event": "updated", "frame": meta})
        return meta

    def apply_auto_boxes(self, index: int, boxes: list[BBox], policy: str) -> None:
        """Merge SAM output into a frame according to the chosen label policy."""
        ann = self.store.get(index)
        if ann is None or ann.held_out:
            return
        if policy == SKIP and ann.boxes:
            return
        if not boxes:
            # "Replace" means replace with what was found, never erase on a miss:
            # a frame the model has nothing to say about keeps its existing labels.
            return
        if policy == MERGE:
            ann.boxes = list(ann.boxes) + list(boxes)
        else:
            ann.boxes = list(boxes)
        ann.status = "pending"   # machine output — waiting for you to look at it
        self.touch()
        self.emit({"type": "frame", "event": "updated", "frame": self.frame_meta(ann)})

    def set_status(self, index: int, status: str) -> dict | None:
        ann = self.store.get(index)
        if ann is None:
            return None
        ann.status = status
        self.touch()
        meta = self.frame_meta(ann)
        self.emit({"type": "frame", "event": "updated", "frame": meta})
        return meta

    # ──────────────────────────────────────────────────────────────
    # Merging
    # ──────────────────────────────────────────────────────────────

    def merge_boxes_on_frame(self, index: int, box_indices: list[int]) -> list[BBox]:
        """Replace the chosen boxes on one frame with a single box covering them."""
        ann = self.store.get(index)
        if ann is None:
            raise ValueError(f"Frame {index} is not loaded.")
        self._read_size(ann)
        merged = boxops.merge_selection(ann.boxes, box_indices, ann.width, ann.height)
        self.set_boxes(index, merged, status="verified")
        return merged

    def merge_overlapping(
        self,
        frames: Iterable[int] | None = None,
        *,
        threshold: float = 0.8,
        same_class_only: bool = True,
        should_abort: Callable[[], bool] | None = None,
    ) -> dict:
        """Fold stacked duplicate boxes together across the given frames."""
        targets = list(frames) if frames is not None else sorted(self.store)
        changed = 0
        removed = 0

        for index in targets:
            if should_abort and should_abort():
                break
            ann = self.store.get(index)
            if ann is None or ann.held_out or len(ann.boxes) < 2:
                continue
            self._read_size(ann)
            merged, dropped = boxops.merge_overlaps(
                ann.boxes, threshold=threshold, same_class_only=same_class_only,
                width=ann.width, height=ann.height,
            )
            if dropped:
                changed += 1
                removed += dropped
                self.set_boxes(index, merged, status=ann.status)

        if removed:
            # The boxes are already folded away in the store, and set_boxes has
            # marked the session dirty, so a save that fails here is "not on
            # disk yet" and not "not merged" — autosave comes back for it.
            # Raising would tell the caller the merge did not happen, which is
            # the one thing that is definitely untrue by this point, and would
            # leave the UI showing boxes that no longer exist.
            self.flush_state("Merge overlapping")
        return {"frames": changed, "removed": removed, "scanned": len(targets)}

    # ──────────────────────────────────────────────────────────────
    # Restore points
    # ──────────────────────────────────────────────────────────────

    def backup_state(self, tag: str = "") -> str:
        """Copy the saved session aside before something sweeping happens to it.

        Undo only ever covered the frame on screen, so a change that touches
        every frame needs a way back that is not undo.  Putting one of these
        files back over session.json is that way.
        """
        self.save_state()
        os.makedirs(BACKUP_DIR, exist_ok=True)
        safe = "".join(c for c in str(tag) if c.isalnum() or c in "-_")[:40] or "backup"
        path = os.path.join(BACKUP_DIR, f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}.json")
        shutil.copy2(STATE_FILE, path)
        self._prune_backups()
        return path

    @staticmethod
    def _prune_backups(keep: int = 20) -> None:
        try:
            saved = sorted(f for f in os.listdir(BACKUP_DIR) if f.endswith(".json"))
        except OSError:
            return
        for name in (saved[:-keep] if len(saved) > keep else []):
            try:
                os.remove(os.path.join(BACKUP_DIR, name))
            except OSError:
                pass

    # ──────────────────────────────────────────────────────────────
    # User scripts
    # ──────────────────────────────────────────────────────────────

    def script_list(self) -> list[dict]:
        """Every script in the folder, including any that will not load — a
        script with a syntax error is worth showing along with its error."""
        return [info.to_dict() for info in scripts.discover()]

    def run_script(
        self,
        script_id: str,
        frames: Iterable[int] | None = None,
        params: dict | None = None,
        *,
        job: Job | None = None,
    ) -> dict:
        """Run a user script over some frames, and keep what it hands back.

        The script works on copies throughout.  Only frames it actually changed
        are written, and only once the boxes it produced have been checked, so
        a script that fails partway leaves everything it did not reach alone.
        """
        info = scripts.get(script_id)
        values = scripts.sanitise(info.params, params, len(self.class_names))
        scope = "frame" if frames is not None else "all"

        wanted = list(frames) if frames is not None else sorted(self.store)
        views = []
        spared = 0
        for index in wanted:
            ann = self.store.get(index)
            if ann is None:
                continue
            if ann.held_out:
                spared += 1
                continue
            self._read_size(ann)
            views.append(scripts.FrameView(ann, self.class_names, self.task))

        report = scripts.run(
            info, views, values, scope=scope,
            should_abort=(lambda: job.is_cancelled) if job else None,
            on_progress=(lambda done, total, msg: job.set_progress(done, total, msg)) if job else None,
            on_line=job.log if job else None,
        )

        written = 0
        for view in views:
            if not view.changed():
                continue
            ann = self.store.get(view.index)
            if ann is None:
                continue
            # The status is carried over rather than set: a script is not a
            # person looking at the frame and deciding it is right.
            self.set_boxes(view.index, view.boxes, status=ann.status)
            written += 1
        if written:
            self.save_state()

        result = report.to_dict()
        result["changed"] = written
        result["heldOut"] = spared
        return result

    def delete_frames(self, indices: Iterable[int]) -> int:
        removed = 0
        with self._lock:
            for index in list(indices):
                held = self.store.get(index)
                if held is not None and held.held_out:
                    continue
                if self.store.pop(index, None) is not None:
                    removed += 1
                    thumb = os.path.join(THUMBS_DIR, f"{index:06d}.jpg")
                    if os.path.isfile(thumb):
                        try:
                            os.remove(thumb)
                        except OSError:
                            pass
        if removed:
            self.touch()
            self.emit({"type": "frames", "event": "removed", "indices": list(indices)})
        return removed

    def clear(self) -> None:
        with self._lock:
            self.store.clear()
            self.video_stride.clear()
        self._snap_cache.clear()  # the frame it encoded is about to be deleted
        for directory in (FRAMES_DIR, THUMBS_DIR):
            shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory, exist_ok=True)
        self.touch()
        self.emit({"type": "frames", "event": "cleared"})

    def set_classes(self, names: list[str]) -> list[str]:
        cleaned = [n.strip() for n in names if n and n.strip()]
        self.class_names = cleaned or ["object"]
        self.touch()
        self.emit({"type": "classes", "classes": self.class_names})
        return self.class_names

    def next_frame_index(self) -> int:
        return max(self.store.keys(), default=-1) + 1

    # ──────────────────────────────────────────────────────────────
    # Models
    # ──────────────────────────────────────────────────────────────

    @property
    def wants_masks(self) -> bool:
        """Segmentation labels need outlines; the other tasks do not."""
        return self.task == "segment"

    def cached_sam(self):
        """The already-loaded SAM 3 model, if any — never triggers a load."""
        return self._sam_model, self._sam_processor

    def ensure_sam(self, job: Job | None = None):
        """Load (once) the SAM 3 concept model for the selected model key."""
        if self._sam_model is not None and self._sam_loaded_key == self.sam_model_key:
            return self._sam_model, self._sam_processor
        if job:
            job.set_message(f"Loading {self.sam_model_key} — first run downloads weights…")
        self._sam_model, self._sam_processor = load_sam3(self.sam_model_key)
        self._sam_loaded_key = self.sam_model_key
        return self._sam_model, self._sam_processor

    def ensure_tracker(self, job: Job | None = None):
        if self._tracker_model is not None:
            return self._tracker_model, self._tracker_processor
        if job:
            job.set_message("Loading the SAM 3 video tracker…")
        self._tracker_model, self._tracker_processor = load_sam3_tracker()
        return self._tracker_model, self._tracker_processor

    # ──────────────────────────────────────────────────────────────
    # Media import
    # ──────────────────────────────────────────────────────────────

    def plan_import(self, path: str, mode: str) -> tuple[str, str | None, str]:
        """Work out what a path actually holds, before importing it.

        It can be a plain pile of pictures, a YOLO dataset root carrying its
        own classes, the images/ directory inside one, or a single video.  The
        browser and the command line have to make the same sense of it or the
        same folder would import two different ways, so both ask here.

        Returns (images_dir, labels_dir or None, task named by any data.yaml).
        """
        if not os.path.isdir(path) or mode not in ("replace", "dataset"):
            return path, None, "detect"

        images_dir, labels_dir, yaml_path = self.resolve_dataset_dirs(path)
        dataset_task = "detect"
        if mode == "dataset" and not labels_dir and not yaml_path:
            # A plain media folder chosen in dataset mode — import it as media.
            images_dir, labels_dir = path, None
        if yaml_path:
            dataset_task = self.apply_dataset_yaml(yaml_path)["task"]
        elif mode == "dataset":
            self.toast("No data.yaml found — importing images only, classes unchanged.",
                       "warn")
        if mode == "replace" and not labels_dir:
            images_dir = path
        return images_dir, labels_dir, dataset_task

    def start_import(
        self,
        path: str,
        *,
        mode: str = "replace",          # replace | add | dataset
        stride: int = 1,
        labels_dir: str | None = None,
        dataset_task: str = "detect",
    ) -> Job:
        if mode == "replace" or mode == "dataset":
            self.clear()

        offset = self.next_frame_index()
        label = f"Importing media from {os.path.basename(path.rstrip(os.sep)) or path}"

        def work(job: Job) -> dict:
            pending: list[dict] = []
            last_flush = [time.time()]

            def flush() -> None:
                if pending:
                    self.emit({"type": "frames", "event": "added", "frames": list(pending)})
                    pending.clear()
                    last_flush[0] = time.time()

            def on_frame(index: int, png_path: str, source_path: str) -> None:
                video = media_loader.is_video(source_path)
                ann = FrameAnnotation(
                    frame_index=index,
                    image_path=png_path,
                    source_video=source_path if video else "",
                    source_image="" if video else source_path,
                )
                self._read_size(ann)
                # Under the lock like every other change to the store: an
                # import runs on a job thread while requests are still being
                # served, and a reader walking the store wants it to hold still.
                with self._lock:
                    self.store[index] = ann
                    if video:
                        self.video_stride.setdefault(source_path, stride)
                pending.append(self.frame_meta(ann))
                # Batch UI updates: a 10k-frame video should not mean 10k messages
                if len(pending) >= 24 or (time.time() - last_flush[0]) > 0.4:
                    flush()

            written = media_loader.import_media(
                [path], SESSION_ROOT,
                import_stride=stride,
                frame_offset=offset,
                on_frame=on_frame,
                on_progress=lambda cur, tot: job.set_progress(cur, tot, f"{cur} / {tot} frames"),
                should_abort=lambda: job.is_cancelled,
                on_error=job.log,
            )
            flush()
            self.touch()

            labelled = 0
            if labels_dir and os.path.isdir(labels_dir):
                labelled = self.apply_dataset_labels(labels_dir, dataset_task)
                self.emit({"type": "frames", "event": "reload"})

            self.save_state()
            return {"frames": written, "labelled": labelled, "total": len(self.store)}

        return self.jobs.start("import", label, work)

    def _read_size(self, ann: FrameAnnotation) -> None:
        if ann.width and ann.height:
            return
        try:
            from PIL import Image

            with Image.open(ann.image_path) as img:
                ann.width, ann.height = img.size
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    # Existing-dataset support
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def resolve_dataset_dirs(path: str) -> tuple[str, str | None, str | None]:
        """Return (images_dir, labels_dir|None, data.yaml|None) for a dataset path.

        Handles the three layouts people actually have:
          1. path is the dataset root   → path/images/, path/labels/, path/data.yaml
          2. path IS the images/ dir    → ../labels/, ../data.yaml
          3. path is images/train/ etc. → ../../labels/train/, ../../data.yaml
        """
        def first_existing(*paths: str) -> str | None:
            for p in paths:
                if os.path.exists(p):
                    return p
            return None

        images_sub = os.path.join(path, "images")
        if os.path.isdir(images_sub):
            return (images_sub,
                    first_existing(os.path.join(path, "labels")),
                    first_existing(os.path.join(path, "data.yaml")))

        parent = os.path.dirname(path)
        folder_name = os.path.basename(path)
        parent_labels = os.path.join(parent, "labels")
        if os.path.isdir(parent_labels):
            return path, parent_labels, first_existing(os.path.join(parent, "data.yaml"))

        grandparent = os.path.dirname(parent)
        gp_split = os.path.join(grandparent, "labels", folder_name)
        gp_flat = os.path.join(grandparent, "labels")
        if os.path.isdir(gp_split):
            return path, gp_split, first_existing(os.path.join(grandparent, "data.yaml"))
        if os.path.isdir(gp_flat):
            return path, gp_flat, first_existing(os.path.join(grandparent, "data.yaml"))

        return path, None, None

    def apply_dataset_yaml(self, yaml_path: str) -> dict:
        """Load class names + task from a data.yaml.  Returns a summary dict."""
        import yaml as _yaml

        with open(yaml_path, encoding="utf-8") as fh:
            data = _yaml.safe_load(fh) or {}

        task = data.get("task", "detect")
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        if isinstance(names, list) and names:
            self.set_classes([str(n) for n in names])
        if task in TASKS:
            self.task = task
            self.emit({"type": "task", "task": task})
        self.touch()
        return {"task": task, "classes": list(self.class_names)}

    def apply_dataset_labels(self, labels_dir: str, dataset_task: str = "detect") -> int:
        """Parse YOLO .txt label files and populate boxes in the store."""
        import cv2

        labelled = 0
        is_pose = dataset_task == "pose"
        is_seg = dataset_task == "segment"

        for ann in self.store.values():
            source = ann.source_image
            if not source:
                continue
            stem = os.path.splitext(os.path.basename(source))[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
            if not os.path.isfile(label_path):
                continue

            if not (ann.width and ann.height):
                img = cv2.imread(ann.image_path)
                if img is None:
                    continue
                ann.height, ann.width = img.shape[:2]
            w, h = ann.width, ann.height

            boxes: list[BBox] = []
            with open(label_path, encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(parts[0])
                        floats = [float(p) for p in parts[1:]]
                    except ValueError:
                        continue
                    box = _parse_label_line(cid, floats, w, h, is_pose, is_seg)
                    if box is not None:
                        boxes.append(box)

            if boxes:
                ann.boxes = boxes
                ann.status = "verified"
                labelled += 1

        if labelled:
            self.touch()
        return labelled

    # ──────────────────────────────────────────────────────────────
    # Auto-labelling
    # ──────────────────────────────────────────────────────────────

    def frames_for_range(self, start: int = 0, stride: int = 1, exclude: int | None = None) -> list[tuple[int, str]]:
        stride = max(1, stride)
        return [
            (idx, ann.image_path)
            for idx, ann in sorted(self.store.items())
            if idx >= start and idx % stride == 0 and idx != exclude
        ]

    def start_concept_search(
        self,
        concepts: list[tuple[str, int]],
        frames: list[tuple[int, str]],
        *,
        policy: str = REPLACE,
        threshold: float | None = None,
        label: str = "Auto-labelling",
    ) -> Job:
        threshold = self.threshold if threshold is None else threshold
        # apply_auto_boxes refuses a held-out frame whatever arrives for it, so
        # running the model over one only spends the time to throw the answer
        # away.  Dropping them here also makes the count in the progress line
        # the number of frames that can actually change.
        frames = [(i, p) for i, p in frames
                  if not (i in self.store and self.store[i].held_out)]
        if policy == SKIP:
            frames = [(i, p) for i, p in frames
                      if i in self.store and not self.store[i].boxes]

        def work(job: Job) -> dict:
            model, processor = self.ensure_sam(job)
            job.raise_if_cancelled()
            hits = {"boxes": 0, "frames": 0}

            def on_boxes(index: int, boxes: list[BBox]) -> None:
                if boxes:
                    hits["boxes"] += len(boxes)
                    hits["frames"] += 1
                self.apply_auto_boxes(index, boxes, policy)

            run_concepts_over_frames(
                model, processor, frames, concepts,
                threshold=threshold,
                want_masks=self.wants_masks, detail=self.mask_detail,
                on_boxes=on_boxes,
                on_progress=lambda cur, tot: job.set_progress(
                    cur, tot, f"{cur} / {tot} frames · {hits['boxes']} objects found"
                ),
                should_abort=lambda: job.is_cancelled,
                on_error=job.log,
            )
            self.save_state()
            return {"objects": hits["boxes"], "framesWithObjects": hits["frames"]}

        return self.jobs.start("autolabel", label, work)

    def start_tracking(
        self,
        seed_index: int,
        *,
        max_frames: int | None = None,
        policy: str = REPLACE,
    ) -> Job:
        seed = self.store.get(seed_index)
        if seed is None or not seed.boxes:
            raise ValueError("Label the current frame first — tracking follows the boxes you drew.")

        seed_source = seed.source_video
        frames = [
            (idx, ann.image_path)
            for idx, ann in sorted(self.store.items())
            if ann.source_video == seed_source
        ]
        seeds = [((b.x1, b.y1, b.x2, b.y2), b.class_id) for b in seed.boxes]

        def work(job: Job) -> dict:
            model, processor = self.ensure_tracker(job)
            job.raise_if_cancelled()
            count = {"n": 0}

            def on_boxes(index: int, boxes: list[BBox]) -> None:
                if index == seed_index:
                    return  # never overwrite the frame the user labelled by hand
                count["n"] += len(boxes)
                self.apply_auto_boxes(index, boxes, policy)

            tracked = track_through_frames(
                model, processor, frames, seeds, seed_index,
                max_frames=max_frames,
                want_masks=self.wants_masks, detail=self.mask_detail,
                on_boxes=on_boxes,
                on_progress=lambda cur, tot: job.set_progress(
                    cur, tot, f"{cur} / {tot} frames · {count['n']} objects tracked"
                ),
                should_abort=lambda: job.is_cancelled,
                on_status=job.set_message,
            )
            self.save_state()
            return {"frames": tracked, "objects": count["n"]}

        return self.jobs.start("track", f"Tracking objects from frame {seed_index}", work)

    def run_prompt(
        self,
        frame_index: int,
        pos: list[list[float]],
        neg: list[list[float]],
        *,
        text: str | None = None,
        class_id: int = 0,
        threshold: float | None = None,
        policy: str = MERGE,
    ) -> Job:
        ann = self.store.get(frame_index)
        if ann is None:
            raise ValueError(f"Frame {frame_index} is not loaded.")
        if not pos and not text:
            raise ValueError(
                "Draw at least one green example box around something you want, "
                "or type what to look for."
            )
        threshold = self.threshold if threshold is None else threshold

        def work(job: Job) -> dict:
            from PIL import Image

            model, processor = self.ensure_sam(job)
            job.raise_if_cancelled()
            job.set_message("Looking for matches on this frame…")
            pil = Image.open(ann.image_path).convert("RGB")
            boxes = run_sam3(
                model, processor, pil,
                text=text or None,
                pos_boxes=[tuple(b) for b in pos],
                neg_boxes=[tuple(b) for b in neg],
                class_id=class_id,
                threshold=threshold,
                want_masks=self.wants_masks, detail=self.mask_detail,
            )
            self.apply_auto_boxes(frame_index, boxes, policy)
            self.save_state()
            return {"objects": len(boxes), "frame": frame_index}

        return self.jobs.start("prompt", f"Finding matches on frame {frame_index}", work)

    @staticmethod
    def _frame_key(image_path: str) -> tuple:
        """Identify an encoded frame.

        The modification time rides along with the path so that a re-imported
        frame reusing a filename can never match a stale encoding.
        """
        return (image_path, os.path.getmtime(image_path))

    def prewarm_frame(self, frame_index: int) -> bool:
        """Encode a frame ahead of time so the next snap on it is instant.

        Turning the picture into features is nearly all of what a snap costs,
        and it only depends on the picture — not on where the box lands.  Doing
        it while the user is still deciding where to drag turns the first snap
        on a frame from a wait into no wait at all.

        Runs on its own thread and reports nothing: it is an optimisation, and
        a pre-warm that fails simply means the snap pays the cost as before.
        """
        ann = self.store.get(frame_index)
        if ann is None or not self.wants_masks:
            return False
        try:
            key = self._frame_key(ann.image_path)
        except OSError:
            return False
        # Claim the frame before deciding whether there is work to do.  A
        # pre-warm already in flight for the frame the user just left would
        # otherwise encode it and evict the one they are looking at now.
        self._prewarm_want = key
        if self._snap_cache.holds(key):
            return False

        def work() -> None:
            from PIL import Image

            try:
                # Let the frame settle first.  Someone arrowing through the
                # filmstrip fires one of these per frame, and an encode cannot
                # be interrupted once it starts — so the frames passed through
                # are dropped here rather than each evicting the next.
                time.sleep(PREWARM_SETTLE)
                if self._prewarm_want != key:
                    return
                model, processor = self.ensure_tracker()
                with self._snap_lock:
                    # Checked again: the wait for the lock is itself a delay,
                    # and a real snap may have encoded the frame meanwhile.
                    if self._prewarm_want != key or self._snap_cache.holds(key):
                        return
                    pil = Image.open(ann.image_path).convert("RGB")
                    encode_frame(model, processor, pil, self._snap_cache, key)
            except Exception:
                pass  # nothing is owed: the snap itself will encode the frame

        threading.Thread(target=work, name="prewarm", daemon=True).start()
        return True

    def snap_to_object(
        self,
        frame_index: int,
        box: list[float],
        *,
        class_id: int = 0,
    ) -> Job:
        """Fit an outline to the object inside a drawn box.

        The box only has to be roughly right — the tracker segments what sits
        inside it, so the outline lands on the object rather than on the box.
        """
        ann = self.store.get(frame_index)
        if ann is None:
            raise ValueError(f"Frame {frame_index} is not loaded.")
        if len(box) != 4 or box[2] - box[0] < 2 or box[3] - box[1] < 2:
            raise ValueError("Drag a box around the object you want outlined.")

        def work(job: Job) -> dict:
            from PIL import Image

            model, processor = self.ensure_tracker(job)
            job.raise_if_cancelled()
            job.set_message("Fitting an outline…")

            pil = Image.open(ann.image_path).convert("RGB")
            key = self._frame_key(ann.image_path)
            with self._snap_lock:
                polygon, bounds = segment_box(
                    model, processor, pil, tuple(box), detail=self.mask_detail,
                    cache=self._snap_cache, cache_key=key,
                )
            if not polygon or not bounds:
                return {"objects": 0, "frame": frame_index, "missed": True}

            x1, y1, x2, y2 = bounds
            outlined = BBox(x1=x1, y1=y1, x2=x2, y2=y2, class_id=class_id,
                            source="sam", polygon=polygon)
            self.apply_auto_boxes(frame_index, [outlined], MERGE)
            self.save_state()
            return {"objects": 1, "frame": frame_index, "points": len(polygon) // 2}

        return self.jobs.start("snap", f"Outlining an object on frame {frame_index}", work)

    # ──────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Counts for the sidebar.

        Everything is counted off one snapshot taken under the lock, because an
        import adds frames from a job thread while requests are being served.
        Every walk added here wants to read `frames` for the same reason — a
        fresh `self.store.values()` slipped in below is another chance to be
        halfway through the store when it changes size.
        """
        with self._lock:
            frames = list(self.store.values())
            class_names = list(self.class_names)
        total = len(frames)
        labelled = sum(1 for a in frames if a.boxes)
        boxes = sum(len(a.boxes) for a in frames)
        per_class: dict[str, int] = {}
        for ann in frames:
            for box in ann.boxes:
                name = (class_names[box.class_id]
                        if 0 <= box.class_id < len(class_names)
                        else f"class {box.class_id}")
                per_class[name] = per_class.get(name, 0) + 1
        return {
            "frames": total,
            "heldOut": sum(1 for a in frames if a.held_out),
            "exported": sum(1 for a in frames if a.exported),
            "labelled": labelled,
            "unlabelled": total - labelled,
            "boxes": boxes,
            "perClass": per_class,
            "videos": sorted({os.path.basename(a.source_video)
                              for a in frames if a.source_video}),
        }


def _parse_label_line(
    cid: int, floats: list[float], w: int, h: int, is_pose: bool, is_seg: bool
) -> BBox | None:
    n = len(floats)

    # Pose: cx cy w h  x1 y1 v1  x2 y2 v2 …
    if is_pose and n >= 10 and (n - 4) % 3 == 0:
        cx, cy, bw, bh = floats[:4]
        kpt_raw = floats[4:]
        keypoints = [
            (kpt_raw[i], kpt_raw[i + 1]) if kpt_raw[i + 2] > 0 else None
            for i in range(0, len(kpt_raw), 3)
        ]
        return BBox(x1=(cx - bw / 2) * w, y1=(cy - bh / 2) * h,
                    x2=(cx + bw / 2) * w, y2=(cy + bh / 2) * h,
                    class_id=cid, source="dataset", keypoints=keypoints or None)

    # Segmentation: class_id + polygon points
    if (is_seg or not is_pose) and n >= 6 and n % 2 == 0:
        xs, ys = floats[0::2], floats[1::2]
        return BBox(x1=min(xs) * w, y1=min(ys) * h, x2=max(xs) * w, y2=max(ys) * h,
                    class_id=cid, source="dataset", polygon=floats)

    # Detection: cx cy w h
    if n == 4:
        cx, cy, bw, bh = floats
        return BBox(x1=(cx - bw / 2) * w, y1=(cy - bh / 2) * h,
                    x2=(cx + bw / 2) * w, y2=(cy + bh / 2) * h,
                    class_id=cid, source="dataset")

    return None
