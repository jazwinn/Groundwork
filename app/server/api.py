"""
HTTP + websocket layer.

Thin translation only: every route turns a request into a call on the Session
(app/core/session.py) and returns JSON.  Long operations answer immediately
with a job id; progress arrives over the /ws websocket.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.core.jobs import Job
from app.core.sam3_handler import BBox, SAM_MODELS
from app.core.sam2_trainer import SAM2_MODELS, train_sam2
from app.core import maskops, overlay, scripts
from app.core.session import MERGE, REPLACE, SKIP, TASKS, Session
from app.core.yolo_trainer import (
    AUGMENTATIONS, DETECTION_MODELS, POSE_MODELS, RUNS_DIR, SEGMENTATION_MODELS,
    _PROJECT_ROOT, _is_pose_key, _is_sam2_key, _is_seg_key, metric_label_for,
    train_yolo,
)
from app.core.onnx_exporter import export_onnx
from app.server import fsbrowse, nativedialog
from app.utils.yolo_exporter import export_dataset

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


# ---------------------------------------------------------------------------
# Websocket event bus
# ---------------------------------------------------------------------------

class EventBus:
    """Fan-out of job/session events to every connected browser tab."""

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._queues.discard(queue)

    def emit(self, payload: dict) -> None:
        """Safe to call from any thread."""
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._dispatch, payload)
        except RuntimeError:
            pass

    def _dispatch(self, payload: dict) -> None:
        for queue in list(self._queues):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass


bus = EventBus()
session = Session()
session.set_emitter(bus.emit)

app = FastAPI(title="Groundwork")


@app.on_event("startup")
async def _startup() -> None:
    bus.bind_loop(asyncio.get_running_loop())
    restored = session.load_state()
    session.start_autosave()
    if restored:
        print(f"Restored previous session: {restored} frame(s)")


@app.on_event("shutdown")
async def _shutdown() -> None:
    session.shutdown()


# ---------------------------------------------------------------------------
# Static app shell
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Serve the UI fresh — a cached stylesheet after an update is pure confusion."""
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Push state to one open tab until it goes away.

    The browser never sends anything up this socket, so it would be natural to
    only ever wait on the queue — but then nothing ends this task when the tab
    closes or the server is asked to stop, and a shutdown waits forever on a
    task that cannot finish.  So the wait is on either the next event or the
    connection ending, whichever comes first.
    """
    await ws.accept()
    queue = bus.subscribe()

    async def until_gone() -> None:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                return

    gone = asyncio.create_task(until_gone())
    nxt: asyncio.Task | None = None
    try:
        await ws.send_json({"type": "hello", "state": _state(ws)})
        while True:
            nxt = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({nxt, gone}, return_when=asyncio.FIRST_COMPLETED)
            if nxt not in done:
                break
            await ws.send_json(nxt.result())
    except (WebSocketDisconnect, asyncio.CancelledError):
        # CancelledError is a BaseException, so the clause below never sees it.
        # Letting it out turns an ordinary shutdown into a page of traceback.
        pass
    except Exception:
        pass
    finally:
        for task in (nxt, gone):
            if task is not None and not task.done():
                task.cancel()
        bus.unsubscribe(queue)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _device_info() -> dict:
    info: dict[str, Any] = {"torch": None, "cuda": False, "gpu": None, "vram": None}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _local_client(request: Request | None) -> bool:
    """Native dialogs open on the server's screen, so only offer them to someone
    sitting at that machine."""
    host = getattr(getattr(request, "client", None), "host", "") if request else ""
    return host in ("127.0.0.1", "::1", "localhost", "")


def _state(request: Request | None = None) -> dict:
    return {
        "nativeDialogs": nativedialog.available() and _local_client(request),
        "classes": session.class_names,
        "task": session.task,
        "samModel": session.sam_model_key,
        "samModels": list(SAM_MODELS),
        "threshold": session.threshold,
        "maskDetail": session.mask_detail,
        "maskDetails": list(maskops.DETAIL_LEVELS),
        "augmentations": AUGMENTATIONS,
        "wantsMasks": session.wants_masks,
        "frames": session.frames_meta(),
        "jobs": session.jobs.snapshot(),
        "stats": session.stats(),
        "device": _device_info(),
        "runsDir": RUNS_DIR,
    }


@app.get("/api/state")
async def get_state(request: Request) -> dict:
    return _state(request)


@app.get("/api/models")
async def get_models() -> dict:
    return {
        "detect": DETECTION_MODELS,
        "segment": SEGMENTATION_MODELS + list(SAM2_MODELS),
        "pose": POSE_MODELS,
        "sam": list(SAM_MODELS),
    }


@app.post("/api/settings")
async def set_settings(payload: dict = Body(...)) -> dict:
    if "task" in payload and payload["task"] in TASKS:
        session.task = payload["task"]
    if "samModel" in payload and payload["samModel"] in SAM_MODELS:
        session.sam_model_key = payload["samModel"]
    if "threshold" in payload:
        session.threshold = max(0.05, min(0.95, float(payload["threshold"])))
    if payload.get("maskDetail") in maskops.DETAIL_LEVELS:
        session.mask_detail = payload["maskDetail"]
    session.touch()
    if "task" in payload:
        session.emit({"type": "task", "task": session.task})
    return {"task": session.task, "samModel": session.sam_model_key,
            "threshold": session.threshold, "maskDetail": session.mask_detail,
            "wantsMasks": session.wants_masks}


@app.post("/api/classes")
async def set_classes(payload: dict = Body(...)) -> dict:
    names = payload.get("names") or []
    return {"classes": session.set_classes([str(n) for n in names])}


# ---------------------------------------------------------------------------
# Filesystem browsing (replaces the native folder dialogs)
# ---------------------------------------------------------------------------

@app.get("/api/fs")
async def browse(path: str | None = None, want: str = "dir") -> dict:
    return fsbrowse.listdir(path, want=want)


@app.post("/api/fs/native")
async def native_dialog(request: Request, payload: dict = Body(...)) -> dict:
    """Open the operating system's own file dialog and return what was picked."""
    if not _local_client(request):
        return {"unavailable": "This browser is on another machine, so a dialog "
                               "here would open on the server's screen."}
    if nativedialog.in_use():
        return {"busy": True}

    want = payload.get("want", "dir")
    filters = None
    if want not in ("dir", "media"):
        filters = [(f"{want} files", f"*{want}"), ("All files", "*.*")]

    try:
        path = await run_in_threadpool(
            nativedialog.pick,
            folders=want in ("dir", "media"),
            title=payload.get("title") or "Select",
            start=payload.get("start") or _PROJECT_ROOT,
            filters=filters,
            ok_label=payload.get("okLabel") or "",
        )
    except nativedialog.DialogBusy:
        return {"busy": True}
    except nativedialog.DialogUnavailable as exc:
        return {"unavailable": str(exc)}

    return {"path": path} if path else {"cancelled": True}


@app.post("/api/fs/native/cancel")
async def native_dialog_cancel() -> dict:
    """Take down a dialog the person has given up waiting for."""
    return {"closed": nativedialog.cancel_open()}


@app.get("/api/fs/inspect")
async def inspect_dataset(path: str) -> dict:
    """Tell the UI what it would be importing before it commits."""
    if not os.path.isdir(path):
        raise HTTPException(404, f"Not a folder: {path}")
    images_dir, labels_dir, yaml_path = Session.resolve_dataset_dirs(path)
    result: dict[str, Any] = {
        "imagesDir": images_dir,
        "labelsDir": labels_dir,
        "yaml": yaml_path,
        "isDataset": bool(labels_dir or yaml_path),
    }
    if yaml_path:
        try:
            import yaml as _yaml

            with open(yaml_path, encoding="utf-8") as fh:
                data = _yaml.safe_load(fh) or {}
            names = data.get("names", [])
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
            result["task"] = data.get("task", "detect")
            result["classes"] = [str(n) for n in names]
        except Exception as exc:
            result["yamlError"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# Media import
# ---------------------------------------------------------------------------

@app.post("/api/import")
async def import_media(payload: dict = Body(...)) -> dict:
    path = payload.get("path") or ""
    mode = payload.get("mode", "replace")
    stride = max(1, int(payload.get("stride", 1)))

    # A folder is what the picker gives, but the loader has always been able to
    # read a single video or image, and the command line can say one.
    if not os.path.exists(path):
        raise HTTPException(400, f"No such file or folder: {path}")
    busy = session.jobs.is_busy("import")
    if busy:
        raise HTTPException(409, "An import is already running.")

    images_dir, labels_dir, dataset_task = session.plan_import(path, mode)

    job = session.start_import(
        images_dir, mode=mode, stride=stride,
        labels_dir=labels_dir, dataset_task=dataset_task,
    )
    return {"job": job.to_dict()}


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

@app.get("/api/frames")
async def list_frames() -> dict:
    return {"frames": session.frames_meta(), "stats": session.stats()}


@app.get("/api/frames/{index}/image")
async def frame_image(index: int):
    ann = session.get(index)
    if ann is None or not os.path.isfile(ann.image_path):
        raise HTTPException(404, "Frame not found")
    return FileResponse(ann.image_path, media_type="image/png")


@app.get("/api/frames/{index}/render")
async def frame_render(index: int, labels: bool = True):
    """The frame with its labels drawn on — what the editor shows, as a picture.

    For anything that has to judge a label without being a browser: whether an
    outline follows the object, whether one object has been cut in two.
    """
    ann = session.get(index)
    if ann is None:
        raise HTTPException(404, "Frame not found")
    img = await run_in_threadpool(overlay.render, ann, session.class_names, labels=labels)
    if img is None:
        raise HTTPException(404, "The picture for this frame could not be read.")
    return Response(content=overlay.encode_png(img), media_type="image/png")


@app.get("/api/frames/{index}/thumb")
async def frame_thumb(index: int):
    thumb = session.thumbnail(index)
    if not thumb:
        raise HTTPException(404, "Frame not found")
    return FileResponse(thumb, media_type="image/jpeg")


@app.get("/api/frames/{index}")
async def frame_detail(index: int) -> dict:
    ann = session.get(index)
    if ann is None:
        raise HTTPException(404, "Frame not found")
    return {
        "index": index,
        "status": ann.status,
        "width": ann.width,
        "height": ann.height,
        "source": ann.source_video or ann.source_image,
        "isVideo": bool(ann.source_video),
        "importStride": session.video_stride.get(ann.source_video, 1),
        "boxes": [b.to_dict() for b in ann.boxes],
    }


@app.put("/api/frames/{index}/boxes")
async def put_boxes(index: int, payload: dict = Body(...)) -> dict:
    ann = session.get(index)
    if ann is None:
        raise HTTPException(404, "Frame not found")
    if ann.held_out:
        raise HTTPException(
            409, f"Frame {index} is held out to measure against, so its labels "
                 f"cannot be changed. Put it back in play first.")
    boxes = [BBox.from_dict(b) for b in payload.get("boxes", [])]
    meta = session.set_boxes(index, boxes, status=payload.get("status"))
    return {"frame": meta, "stats": session.stats()}


@app.post("/api/frames/hold")
async def hold_frames(payload: dict = Body(...)) -> dict:
    """Hold frames out to measure against, or put them back in play."""
    indices = payload.get("indices")
    if not isinstance(indices, list) or not indices:
        raise HTTPException(400, "Say which frames, as a list of indices.")
    held = bool(payload.get("held", True))
    changed = session.set_held_out([int(i) for i in indices], held)
    return {"changed": changed, "held": held, "stats": session.stats()}


@app.put("/api/frames/{index}/status")
async def put_status(index: int, payload: dict = Body(...)) -> dict:
    status = payload.get("status", "verified")
    if status not in ("pending", "verified"):
        raise HTTPException(
            400, f"Unknown status: {status}. A frame is 'pending' or 'verified' — "
                 f"whether it has been exported is kept separately and set by exporting.")
    meta = session.set_status(index, status)
    if meta is None:
        raise HTTPException(404, "Frame not found")
    return {"frame": meta, "stats": session.stats()}


@app.delete("/api/frames/{index}")
async def delete_frame(index: int) -> dict:
    removed = session.delete_frames([index])
    if not removed:
        raise HTTPException(404, "Frame not found")
    return {"removed": removed, "stats": session.stats()}


@app.post("/api/frames/delete")
async def delete_frames(payload: dict = Body(...)) -> dict:
    """Drop several frames at once — what the filmstrip selection acts on."""
    indices = [int(i) for i in payload.get("indices", [])]
    if not indices:
        raise HTTPException(400, "No frames given.")
    removed = session.delete_frames(indices)
    return {"removed": removed, "stats": session.stats()}


@app.post("/api/frames/{index}/merge")
async def merge_boxes(index: int, payload: dict = Body(...)) -> dict:
    """Merge the chosen boxes on this frame into one that covers them all."""
    try:
        merged = session.merge_boxes_on_frame(index, [int(i) for i in payload.get("indices", [])])
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"boxes": [b.to_dict() for b in merged], "stats": session.stats()}


@app.post("/api/labels/merge-overlaps")
async def merge_overlaps(payload: dict = Body(...)) -> dict:
    """Fold overlapping duplicates together on one frame or across the session."""
    threshold = max(0.1, min(1.0, float(payload.get("threshold", 0.8))))
    same_class_only = bool(payload.get("sameClassOnly", True))

    frames = None
    if payload.get("scope") == "frame":
        index = payload.get("frame")
        if index is None or session.get(int(index)) is None:
            raise HTTPException(400, "That frame is not loaded.")
        frames = [int(index)]

    result = await run_in_threadpool(
        session.merge_overlapping, frames,
        threshold=threshold, same_class_only=same_class_only,
    )
    result["stats"] = session.stats()
    return result


@app.post("/api/frames/clear-labels")
async def clear_labels(payload: dict = Body(...)) -> dict:
    indices = payload.get("indices")
    targets = indices if indices else list(session.store)
    for index in targets:
        session.set_boxes(index, [], status="pending")
    return {"cleared": len(targets), "stats": session.stats()}


# Everything that writes boxes to many frames at once — the auto-labellers and a
# script run — takes its turn through here.  Two of them reading the same frames
# and then writing them back would quietly lose whichever finished first.
def _reject_if_busy() -> None:
    busy = session.jobs.is_busy("autolabel", "track", "prompt", "snap", "script")
    if busy:
        raise HTTPException(409, f"{busy.label} is still running — stop it first.")


# ---------------------------------------------------------------------------
# User scripts
# ---------------------------------------------------------------------------

@app.get("/api/scripts")
async def list_scripts() -> dict:
    """What is in the scripts folder right now.

    Off the event loop, because reading the list imports the scripts, and a
    script is free to do something slow at the top of the file.
    """
    listed = await run_in_threadpool(session.script_list)
    return {"scripts": listed, "dir": scripts.SCRIPTS_DIR}


@app.post("/api/scripts/run")
async def run_script(payload: dict = Body(...)) -> dict:
    """Run one script on the current frame, or across the whole session."""
    script_id = str(payload.get("scriptId") or "")
    params = payload.get("params") or {}
    try:
        info = scripts.get(script_id)
    except scripts.ScriptError as exc:
        raise HTTPException(400, str(exc))

    if payload.get("scope") == "frame":
        index = payload.get("frame")
        if index is None or session.get(int(index)) is None:
            raise HTTPException(400, "That frame is not loaded.")
        if not info.per_frame:
            raise HTTPException(400, f"{info.name} only defines apply_all, so it runs "
                                     f"across every frame rather than on one.")
        _reject_if_busy()
        try:
            result = await run_in_threadpool(session.run_script, script_id, [int(index)], params)
        except scripts.ScriptError as exc:
            raise HTTPException(400, str(exc))
        result["stats"] = session.stats()
        return result

    # Across everything: slow enough to want a progress bar and a Stop button,
    # and sweeping enough to want a way back first.
    _reject_if_busy()
    if not session.store:
        raise HTTPException(400, "There are no frames to run it on.")
    backup = await run_in_threadpool(session.backup_state, script_id)

    def work(job: Job) -> dict:
        result = session.run_script(script_id, None, params, job=job)
        result["backup"] = backup
        return result

    # A per-frame script can be stopped between frames.  apply_all is one call
    # into the script and cannot be interrupted, so no Stop button is offered
    # rather than one that would not do anything.
    job = session.jobs.start("script", info.name, work, cancellable=not info.across)
    return {"jobId": job.id, "backup": backup}


@app.post("/api/scripts/open")
async def open_scripts_folder(request: Request) -> dict:
    """Show the scripts folder in the file manager, for someone at this machine."""
    if not _local_client(request):
        raise HTTPException(403, "The scripts folder is on the machine running the server.")
    os.makedirs(scripts.SCRIPTS_DIR, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(scripts.SCRIPTS_DIR)          # noqa: S606 — a fixed path
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, scripts.SCRIPTS_DIR])
    except OSError as exc:
        raise HTTPException(500, f"Could not open the folder: {exc}")
    return {"dir": scripts.SCRIPTS_DIR}


# ---------------------------------------------------------------------------
# Auto-labelling
# ---------------------------------------------------------------------------

def _policy(payload: dict) -> str:
    policy = payload.get("policy", REPLACE)
    return policy if policy in (REPLACE, MERGE, SKIP) else REPLACE


@app.post("/api/autolabel/describe")
async def autolabel_describe(payload: dict = Body(...)) -> dict:
    """Find by description: one text concept, searched across a range of frames."""
    _reject_if_busy()
    concept = (payload.get("concept") or "").strip()
    class_id = int(payload.get("classId", 0))
    if not concept:
        concept = (session.class_names[class_id]
                   if 0 <= class_id < len(session.class_names) else "object")

    frames = session.frames_for_range(
        start=int(payload.get("start", 0)), stride=int(payload.get("stride", 1))
    )
    if not frames:
        raise HTTPException(400, "No frames match that range.")

    job = session.start_concept_search(
        [(concept, class_id)], frames,
        policy=_policy(payload),
        threshold=payload.get("threshold"),
        label=f"Searching for “{concept}” in {len(frames)} frames",
    )
    return {"job": job.to_dict(), "frames": len(frames), "concept": concept}


@app.post("/api/autolabel/propagate")
async def autolabel_propagate(payload: dict = Body(...)) -> dict:
    """Copy this frame's labels everywhere: its class names become the concepts."""
    _reject_if_busy()
    seed_index = int(payload.get("seed", -1))
    seed = session.get(seed_index)
    if seed is None or not seed.boxes:
        raise HTTPException(400, "Label this frame first — its classes become the search terms.")

    concepts: list[tuple[str, int]] = []
    for cid in sorted({b.class_id for b in seed.boxes}):
        if 0 <= cid < len(session.class_names) and session.class_names[cid].strip():
            concepts.append((session.class_names[cid].strip(), cid))
    if not concepts:
        raise HTTPException(400, "Give your classes names first (Classes → Edit).")

    frames = session.frames_for_range(
        start=int(payload.get("start", 0)),
        stride=int(payload.get("stride", 1)),
        exclude=seed_index,
    )
    if not frames:
        raise HTTPException(400, "No other frames match that range.")

    names = ", ".join(c for c, _ in concepts)
    job = session.start_concept_search(
        concepts, frames,
        policy=_policy(payload),
        threshold=payload.get("threshold"),
        label=f"Copying [{names}] to {len(frames)} frames",
    )
    return {"job": job.to_dict(), "frames": len(frames),
            "concepts": [c for c, _ in concepts]}


@app.post("/api/autolabel/track")
async def autolabel_track(payload: dict = Body(...)) -> dict:
    """Follow the exact objects boxed on the seed frame through the video."""
    _reject_if_busy()
    seed_index = int(payload.get("seed", -1))
    seed = session.get(seed_index)
    if seed is None:
        raise HTTPException(404, "Frame not found")

    stride = session.video_stride.get(seed.source_video, 1)
    if stride > 1 and not payload.get("confirm"):
        return {
            "needsConfirm": True,
            "reason": (
                f"This video was imported keeping 1 in every {stride} frames. "
                "The tracker follows objects between neighbouring frames, so big gaps "
                "make it lose them. Re-import with stride 1, or use “Copy labels to all "
                "frames” instead — it treats each frame independently."
            ),
        }

    max_frames = int(payload.get("range", 0)) or None
    try:
        job = session.start_tracking(seed_index, max_frames=max_frames, policy=_policy(payload))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"job": job.to_dict()}


@app.post("/api/autolabel/snap")
async def autolabel_snap(payload: dict = Body(...)) -> dict:
    """Fit an outline to the object inside a drawn box."""
    _reject_if_busy()
    try:
        job = session.snap_to_object(
            int(payload.get("frame", -1)),
            [float(v) for v in payload.get("box", [])],
            class_id=int(payload.get("classId", 0)),
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))
    return {"job": job.to_dict()}


@app.post("/api/autolabel/prewarm")
async def autolabel_prewarm(payload: dict = Body(...)) -> dict:
    """Get a frame ready for snapping while the user is still aiming.

    Deliberately not a job: there is nothing to watch and nothing to stop, and
    it must not make the snap that follows it look busy.
    """
    started = session.prewarm_frame(int(payload.get("frame", -1)))
    return {"started": started}


@app.post("/api/autolabel/prompt")
async def autolabel_prompt(payload: dict = Body(...)) -> dict:
    """Find things like these: positive/negative example boxes on one frame."""
    _reject_if_busy()
    try:
        job = session.run_prompt(
            int(payload.get("frame", -1)),
            payload.get("positive") or [],
            payload.get("negative") or [],
            text=(payload.get("text") or "").strip() or None,
            class_id=int(payload.get("classId", 0)),
            threshold=payload.get("threshold"),
            policy=_policy(payload) if payload.get("policy") else MERGE,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"job": job.to_dict()}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@app.get("/api/jobs")
async def list_jobs() -> dict:
    return {"active": session.jobs.snapshot(), "recent": session.jobs.recent()}


@app.get("/api/jobs/{job_id}/log")
async def job_log(job_id: str) -> dict:
    job = session.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {"job": job.to_dict(), "lines": session.jobs.log_lines(job_id)}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    job = session.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {"cancelled": session.jobs.cancel(job_id), "job": job.to_dict()}


# ---------------------------------------------------------------------------
# Export / convert / train
# ---------------------------------------------------------------------------

@app.post("/api/export")
async def export(payload: dict = Body(...)) -> dict:
    out_dir = payload.get("outDir") or ""
    if not out_dir:
        raise HTTPException(400, "Choose an output folder.")
    task = payload.get("task", session.task)
    if task not in TASKS:
        task = "detect"
    os.makedirs(out_dir, exist_ok=True)

    written = export_dataset(
        session.store, session.class_names, out_dir,
        skip_unverified=bool(payload.get("skipUnreviewed")),
        is_seg=task == "segment",
        is_pose=task == "pose",
    )
    session.task = task
    session.touch()
    session.emit({"type": "frames", "event": "reload"})
    if written["warning"]:
        session.toast(written["warning"], "warn")
    return {
        "frames": written["frames"],
        "train": written["train"],
        "val": written["val"],
        "warning": written["warning"],
        "outDir": out_dir,
        "dataYaml": os.path.join(out_dir, "data.yaml"),
        "task": task,
        "stats": session.stats(),
    }


@app.post("/api/convert/seg")
async def convert_seg(payload: dict = Body(...)) -> dict:
    source = payload.get("source") or ""
    if not os.path.isfile(os.path.join(source, "data.yaml")):
        raise HTTPException(400, "Pick a dataset folder that contains data.yaml.")
    if session.jobs.is_busy("convert"):
        raise HTTPException(409, "A conversion is already running.")

    output = payload.get("output") or (os.path.normpath(source) + "seg")

    def work(job: Job) -> dict:
        from app.core.yolo_seg_converter import YoloDatasetConverter

        model, processor = session.cached_sam()
        converter = YoloDatasetConverter(
            source_root=source, output_root=output,
            progress_callback=lambda cur, tot: job.set_progress(cur, tot),
            status_callback=job.set_message,
            model=model, processor=processor,
            should_abort=lambda: job.is_cancelled,
        )
        return converter.convert()

    job = session.jobs.start("convert", f"Converting {os.path.basename(source)} to segmentation", work)
    return {"job": job.to_dict(), "output": output}


@app.post("/api/convert/pose")
async def convert_pose(payload: dict = Body(...)) -> dict:
    source = payload.get("source") or ""
    if not os.path.isfile(os.path.join(source, "data.yaml")):
        raise HTTPException(400, "Pick a dataset folder that contains data.yaml.")
    if session.jobs.is_busy("convert"):
        raise HTTPException(409, "A conversion is already running.")

    output = payload.get("output") or (os.path.normpath(source) + "pose")
    margin = max(0.0, min(0.49, float(payload.get("edgeMargin", 2.0)) / 100.0))

    def work(job: Job) -> dict:
        from app.core.yolo_pose_converter import YoloPoseConverter

        converter = YoloPoseConverter(
            source_root=source, output_root=output, edge_margin=margin,
            progress_callback=lambda cur, tot: job.set_progress(cur, tot),
            status_callback=job.set_message,
            should_abort=lambda: job.is_cancelled,
        )
        return converter.convert()

    job = session.jobs.start("convert", f"Converting {os.path.basename(source)} to pose", work)
    return {"job": job.to_dict(), "output": output}


def _dataset_task(data_yaml: str) -> str:
    try:
        import yaml as _yaml

        with open(data_yaml, encoding="utf-8") as fh:
            return (_yaml.safe_load(fh) or {}).get("task", "detect")
    except Exception:
        return "detect"


@app.post("/api/train")
async def train(payload: dict = Body(...)) -> dict:
    model_key = payload.get("model") or ""
    epochs = max(1, int(payload.get("epochs", 50)))
    dataset_dir = payload.get("datasetDir") or ""
    if session.jobs.is_busy("train"):
        raise HTTPException(409, "Training is already running.")

    # ── SAM 2 fine-tuning: reads images/ + labels/ straight from the folder ──
    if _is_sam2_key(model_key):
        if not os.path.isdir(os.path.join(dataset_dir, "images")):
            raise HTTPException(400, "Pick an exported dataset folder (it needs images/ and labels/).")
        output_dir = payload.get("outputDir") or os.path.join(RUNS_DIR, "sam2_finetune")

        def sam2_work(job: Job) -> dict:
            return train_sam2(
                model_key, dataset_dir=dataset_dir, epochs=epochs, output_dir=output_dir,
                on_epoch=lambda e, t, v: job.set_progress(e, t, f"epoch {e}/{t} · loss {v:.4f}"),
                on_log=job.log,
                should_stop=lambda: job.is_cancelled,
            )

        job = session.jobs.start("train", f"Fine-tuning {model_key} · {epochs} epochs", sam2_work)
        return {"job": job.to_dict(), "metric": "loss"}

    # ── YOLO / FastSAM ──
    data_yaml = os.path.join(dataset_dir, "data.yaml")
    if not os.path.isfile(data_yaml):
        raise HTTPException(400, f"No data.yaml in {dataset_dir} — export your labels first.")

    ds_task = _dataset_task(data_yaml)
    warnings: list[str] = []
    if _is_seg_key(model_key) and ds_task != "segment":
        warnings.append(
            f"{model_key} is a segmentation model but this dataset is “{ds_task}”. "
            "Segmentation training needs polygon labels — run Convert to Segmentation first."
        )
    if _is_pose_key(model_key) and ds_task != "pose":
        warnings.append(
            f"{model_key} is a pose model but this dataset is “{ds_task}”. "
            "Pose training needs keypoint labels — run Convert to Pose first."
        )
    if warnings and not payload.get("confirm"):
        return {"needsConfirm": True, "warnings": warnings}

    imgsz = int(payload.get("imgsz", 640))
    cache = {"off": False, "disk": "disk", "ram": "ram"}.get(payload.get("cache", "off"), False)
    workers = max(0, int(payload.get("workers", 4)))
    augment = payload.get("augment") or {}
    batch = payload.get("batch", -1)
    batch = int(batch) if batch not in (None, "", "auto", -1) else -1
    metric = metric_label_for(model_key)

    def work(job: Job) -> dict:
        job.set_progress(0, epochs, "Preparing…")
        return train_yolo(
            model_key, data_yaml,
            epochs=epochs, imgsz=imgsz, batch=batch, cache=cache, workers=workers,
            augment=augment,
            on_epoch=lambda e, t, v: job.set_progress(e, t, f"epoch {e}/{t} · {metric} {v:.4f}"),
            on_log=job.log,
            should_stop=lambda: job.is_cancelled,
        )

    job = session.jobs.start("train", f"Training {model_key} · {epochs} epochs", work)
    return {"job": job.to_dict(), "metric": metric, "datasetTask": ds_task}


@app.post("/api/onnx")
async def onnx(payload: dict = Body(...)) -> dict:
    pt_path = payload.get("ptPath") or ""
    if not os.path.isfile(pt_path):
        raise HTTPException(400, "Pick a trained .pt checkpoint.")
    if session.jobs.is_busy("onnx"):
        raise HTTPException(409, "An export is already running.")

    half = str(payload.get("precision", "FP32")).upper() == "FP16"
    dynamic = bool(payload.get("dynamic", True))
    imgsz = int(payload.get("imgsz", 640))

    def work(job: Job) -> dict:
        job.set_message("Exporting — this cannot be interrupted once started.")
        return export_onnx(pt_path, half=half, imgsz=imgsz, dynamic=dynamic, on_log=job.log)

    job = session.jobs.start(
        "onnx", f"Exporting {os.path.basename(pt_path)} to ONNX", work, cancellable=False
    )
    return {"job": job.to_dict()}


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@app.post("/api/session/save")
async def save_session() -> dict:
    return {"path": session.save_state()}


@app.post("/api/session/reset")
async def reset_session() -> dict:
    session.jobs.cancel_all()
    session.clear()
    session.save_state()
    return {"ok": True, "stats": session.stats()}


@app.exception_handler(HTTPException)
async def http_error(_request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
