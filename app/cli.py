"""Groundwork from a terminal — for CI, for a scheduled run, and for an agent.

The browser is one way to drive a session and not the only useful one.  Everything
here works on the session directly, prints JSON, and sets an exit code, so the
same commands serve a person at a prompt, a cron line, and something reading the
output back.

    python -m app.cli stats
    python -m app.cli render 24 --out frame24.png
    python -m app.cli hold 3 9 17 --tag gate
    python -m app.cli run merge_split_modules --all --set dry_run=false
    python -m app.cli gate --script merge_split_modules --epochs 20

Two processes writing one session would lose whichever saved first, so anything
that writes stops when the browser server is up.  Reading is allowed either way,
with a note saying the answer may be a few seconds stale.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from app.core import overlay, scripts
from app.core.session import Session
from app.core.yolo_trainer import RUNS_DIR, metric_label_for, score_model, train_yolo
from app.utils.yolo_exporter import export_dataset

EXIT_OK = 0
EXIT_FAIL = 1        # the command ran and the answer was no
EXIT_MISUSE = 2      # it could not run at all


class Failed(Exception):
    """Something the caller did, said plainly.  Not a traceback."""


def out(payload: dict, code: int = EXIT_OK) -> int:
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return code


# ---------------------------------------------------------------------------
# Getting at the session
# ---------------------------------------------------------------------------

def _went_well(result: dict) -> bool:
    """Whether a finished piece of work actually finished.

    The job machinery says "done"; the dataset converters say "success"; both
    say "cancelled" when stopped.  Rather than teach one to speak the other's
    language and change what the browser reads, the two vocabularies are
    accepted here, where the exit code is decided.
    """
    return str(result.get("status", "done")).lower() in ("done", "success", "ok")


def _server_up(args) -> bool:
    """Whether a browser server is holding *this* session open.

    A Groundwork on the port is not enough to go on.  A checkout somewhere else
    serves a different session from its own folder, and refusing to write
    because an unrelated copy happens to be running would be wrong — as would
    writing because the one that is running looked unfamiliar.  Both sides
    report where their project root is, so that is what gets compared.
    """
    import json as _json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
                f"http://{args.host}:{args.port}/api/state", timeout=1.5) as response:
            state = _json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        return False
    if not isinstance(state, dict) or "frames" not in state:
        return False
    theirs = os.path.normcase(os.path.abspath(state.get("runsDir") or ""))
    return theirs == os.path.normcase(os.path.abspath(RUNS_DIR))


def _open(args, *, writing: bool, require_frames: bool = True) -> Session:
    if writing and _server_up(args) and not args.force:
        raise Failed(
            f"Groundwork is open at http://{args.host}:{args.port} and would overwrite "
            f"whatever this writes. Close the browser server first, or pass --force "
            f"if you are certain nothing else is editing this session.")
    session = Session()
    session.load_state()
    # Everything else needs frames to work on; import is how they get here.
    if require_frames and not session.store:
        raise Failed("This session has no frames. Import some media first.")
    return session


def _finish(session: Session) -> None:
    session.save_state()


def _pick(session: Session, indices: list[int] | None, want_all: bool) -> list[int]:
    if want_all:
        return sorted(session.store)
    if not indices:
        raise Failed("Say which frames, or pass --all.")
    missing = [i for i in indices if i not in session.store]
    if missing:
        raise Failed(f"No such frame: {', '.join(str(i) for i in missing)}")
    return sorted(set(indices))


# ---------------------------------------------------------------------------
# Looking
# ---------------------------------------------------------------------------

def cmd_stats(args) -> int:
    session = _open(args, writing=False)
    stats = session.stats()
    stats["task"] = session.task
    stats["classes"] = session.class_names
    stats["live"] = _server_up(args)
    return out(stats)


def cmd_frames(args) -> int:
    session = _open(args, writing=False)
    rows = []
    for index, ann in sorted(session.store.items()):
        if args.held_out and not ann.held_out:
            continue
        if args.status and ann.status != args.status:
            continue
        if args.exported and not ann.exported:
            continue
        rows.append({
            "index": index,
            "status": ann.status,
            "boxes": len(ann.boxes),
            "heldOut": ann.held_out,
            "exported": ann.exported,
            "image": ann.image_path,
        })
    return out({"frames": rows, "count": len(rows)})


def cmd_render(args) -> int:
    session = _open(args, writing=False)
    ann = session.store.get(args.index)
    if ann is None:
        raise Failed(f"No such frame: {args.index}")
    img = overlay.render(ann, session.class_names, labels=not args.no_labels)
    if img is None:
        raise Failed(f"The picture for frame {args.index} could not be read.")
    path = args.out or os.path.join(RUNS_DIR, "render", f"frame_{args.index:06d}.png")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(overlay.encode_png(img))
    return out({"frame": args.index, "path": os.path.abspath(path),
                "boxes": len(ann.boxes), "heldOut": ann.held_out})


# ---------------------------------------------------------------------------
# The held-out half
# ---------------------------------------------------------------------------

def cmd_hold(args) -> int:
    session = _open(args, writing=True)
    indices = _pick(session, args.frames, args.all)
    changed = session.set_held_out(indices, not args.release)
    _finish(session)
    return out({"changed": changed, "held": not args.release,
                "heldOut": sorted(i for i, a in session.store.items() if a.held_out),
                "stats": session.stats()})


# ---------------------------------------------------------------------------
# Changing labels
# ---------------------------------------------------------------------------

def cmd_scripts(args) -> int:
    return out({"scripts": [s.to_dict() for s in scripts.discover()]})


def _settings(pairs: list[str]) -> dict:
    """--set key=value, with the value read as JSON where it looks like it."""
    values: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise Failed(f"--set wants key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            values[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            values[key.strip()] = raw
    return values


def cmd_run(args) -> int:
    session = _open(args, writing=True)
    frames = None if args.all else _pick(session, args.frames, False)
    try:
        result = session.run_script(args.script, frames, _settings(args.set))
    except scripts.ScriptError as exc:
        raise Failed(str(exc)) from None
    _finish(session)
    result["stats"] = session.stats()
    return out(result, EXIT_OK if not result.get("errors") else EXIT_FAIL)


def cmd_backup(args) -> int:
    session = _open(args, writing=True)
    return out({"backup": session.backup_state(args.tag or "cli")})


def cmd_restore(args) -> int:
    session = _open(args, writing=True)
    if not os.path.isfile(args.path):
        raise Failed(f"No such backup: {args.path}")
    before = session.stats()
    restored = session.load_state(args.path)
    session.save_state()
    return out({"restored": restored, "from": args.path,
                "before": before, "after": session.stats()})


# ---------------------------------------------------------------------------
# Tidying up
# ---------------------------------------------------------------------------

def _plural(items, noun: str) -> str:
    """"1 frame", "3 frames" — these strings are what a person reads before
    agreeing to lose work, so they should not read like a stack trace."""
    n = items if isinstance(items, int) else len(items)
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _writable(session, args) -> tuple[list[int], list[int]]:
    """Split the frames asked for into the ones that can change and the rest.

    Everything in this section writes labels, and a held-out frame refuses a
    write wherever it comes from.  Counting them in and reporting a total that
    included frames nothing happened to would overstate the work.
    """
    picked = _pick(session, args.frames, args.all)
    held = [i for i in picked if session.store[i].held_out]
    return [i for i in picked if i not in set(held)], held


def cmd_tidy(args) -> int:
    """Fold stacked duplicate boxes together.

    This is the overlap cleanup, not the split-module merge — it clusters boxes
    that sit on top of each other, which is what two passes of an auto-labeller
    leave behind.  Halves of one module abut rather than overlap, so this never
    fires on them; `run merge_split_modules` is what does.
    """
    live = _server_up(args)
    session = _open(args, writing=not live)
    targets, held = _writable(session, args)
    if not targets:
        raise Failed("Nothing to tidy — every frame named is held out.")

    backup = _snapshot("tidy")
    if live:
        merged = {"changed": 0, "removed": 0}
        calls = ([{"threshold": args.threshold, "sameClassOnly": not args.any_class}]
                 if args.all and not held else
                 [{"threshold": args.threshold, "sameClassOnly": not args.any_class,
                   "scope": "frame", "frame": i} for i in targets])
        for payload in calls:
            answer = _api(args, "/api/labels/merge-overlaps", payload)
            merged["changed"] += answer.get("changed", 0)
            merged["removed"] += answer.get("removed", 0)
            merged["stats"] = answer.get("stats") or merged.get("stats")
        result = merged
    else:
        result = session.merge_overlapping(
            targets, threshold=args.threshold, same_class_only=not args.any_class)
        _finish(session)

    return out({"frames": len(targets), "heldOutSkipped": held,
                "changed": result.get("changed", 0), "removed": result.get("removed", 0),
                "threshold": args.threshold, "ranOn": "server" if live else "here",
                "backup": backup, "stats": result.get("stats") or session.stats()})


def cmd_status(args) -> int:
    """Mark frames as checked, or put them back to unchecked.

    `verified` means a person looked at this frame and stands behind it.  That
    is the only thing the field records and the only reason held-out frames are
    worth measuring against, so setting it from a script says a person checked
    something they did not.  It is here because the CLI should be able to
    record a decision you actually made — not so a run can mark its own work.
    """
    want = "pending" if args.pending else "verified"
    live = _server_up(args)
    session = _open(args, writing=not live)
    targets = _pick(session, args.frames, args.all)
    was = {s: sum(1 for i in targets if session.store[i].status == s)
           for s in ("pending", "verified")}

    if live:
        answer: dict = {}
        for index in targets:
            answer = _api(args, f"/api/frames/{index}/status", {"status": want},
                          method="PUT")
        stats = answer.get("stats") or session.stats()
    else:
        for index in targets:
            session.set_status(index, want)
        _finish(session)
        stats = session.stats()

    result = {"frames": len(targets), "status": want, "wasBefore": was,
              "ranOn": "server" if live else "here", "stats": stats}
    if want == "verified":
        result["note"] = ("These now read as checked by a person. If they were not, "
                          "put them back with --pending.")
    return out(result)


def cmd_clear_labels(args) -> int:
    """Throw away the labels on some frames and keep the pictures."""
    live = _server_up(args)
    session = _open(args, writing=not live)
    targets, held = _writable(session, args)
    losing = sum(len(session.store[i].boxes) for i in targets)
    if not targets:
        raise Failed("Nothing to clear — every frame named is held out.")
    if losing and not args.yes:
        raise Failed(f"That would throw away {losing} labels on {_plural(targets, 'frame')}. "
                     f"Pass --yes if that is what you want.")

    backup = _snapshot("clear-labels")
    if live:
        stats = _api(args, "/api/frames/clear-labels",
                     {"indices": targets}).get("stats") or session.stats()
    else:
        for index in targets:
            session.set_boxes(index, [], status="pending")
        _finish(session)
        stats = session.stats()

    return out({"cleared": len(targets), "labelsLost": losing, "heldOutSkipped": held,
                "ranOn": "server" if live else "here", "backup": backup,
                "stats": stats})


def cmd_delete(args) -> int:
    """Drop frames from the session entirely."""
    live = _server_up(args)
    session = _open(args, writing=not live)
    targets, held = _writable(session, args)
    losing = sum(len(session.store[i].boxes) for i in targets)
    if not targets:
        raise Failed("Nothing to delete — every frame named is held out.")
    if not args.yes:
        raise Failed(f"That would remove {_plural(targets, 'frame')} and the {losing} "
                     f"labels on them, and there is no undo but a restore point. "
                     f"Pass --yes.")

    backup = _snapshot("delete")
    if live:
        answer = _api(args, "/api/frames/delete", {"indices": targets})
        removed, stats = answer.get("removed", 0), answer.get("stats") or session.stats()
    else:
        removed = session.delete_frames(targets)
        _finish(session)
        stats = session.stats()

    return out({"removed": removed, "labelsLost": losing, "heldOutSkipped": held,
                "ranOn": "server" if live else "here", "backup": backup,
                "stats": stats})


# ---------------------------------------------------------------------------
# Starting a session, and watching what one is doing
# ---------------------------------------------------------------------------

def cmd_import(args) -> int:
    """Bring media into the session — a folder, a dataset, or one video.

    The browser defaults this to "replace", which is right when someone has
    just picked a folder and means that folder.  Here it defaults to "add",
    because a line in a script has nobody watching it, and the difference
    between adding to a session and emptying one should never be a default.
    """
    if not os.path.exists(args.path):
        raise Failed(f"No such file or folder: {args.path}")

    live = _server_up(args)
    session = _open(args, writing=not live, require_frames=False)
    had = len(session.store)
    if args.mode in ("replace", "dataset") and had and not args.yes:
        raise Failed(
            f"--mode {args.mode} empties the session first, and this one holds {had} "
            f"frames with {sum(len(a.boxes) for a in session.store.values())} labels on "
            f"them. Pass --yes if that is really what you want, or --mode add to keep "
            f"them.")

    backup = _snapshot(f"import-{args.mode}") if had else None
    path = os.path.abspath(args.path)
    print(f"importing {path} ({args.mode}, stride {args.stride})", file=sys.stderr)

    if live:
        started = _api(args, "/api/import",
                       {"path": path, "mode": args.mode, "stride": args.stride})
        result = _await_there(args, (started.get("job") or {}).get("id", ""), args.verbose)
        session.load_state()
    else:
        images_dir, labels_dir, dataset_task = session.plan_import(path, args.mode)
        job = session.start_import(images_dir, mode=args.mode, stride=args.stride,
                                   labels_dir=labels_dir, dataset_task=dataset_task)
        result = _await_here(job, args.verbose)
        _finish(session)

    stats = session.stats()
    return out({"mode": args.mode, "path": path, "ranOn": "server" if live else "here",
                "status": result.get("status"),
                # What the import read, not the difference in session size:
                # replacing 6 frames with 3 imported 3, and calling that -3
                # would be arithmetic about the wrong thing.
                "imported": result.get("frames", 0),
                "labelsRead": result.get("labelled", 0),
                "framesBefore": had, "framesAfter": stats["frames"],
                "backup": backup, "stats": stats},
               EXIT_OK if _went_well(result) else EXIT_FAIL)


def cmd_classes(args) -> int:
    """Show the class names, or set them.

    Names are positional, so the order is the class id — the labels already on
    disk hold ids, not names, and renaming is only ever a relabelling of the
    same numbers.
    """
    live = _server_up(args)
    if not args.names:
        session = _open(args, writing=False, require_frames=False)
        return out({"classes": session.class_names})

    names = [n.strip() for chunk in args.names for n in chunk.split(",") if n.strip()]
    if not names:
        raise Failed("Give at least one name.")

    session = _open(args, writing=not live, require_frames=False)
    # Shortening the list does not touch a single label, so anything pointing
    # past the end keeps working and exports as a class that no longer exists.
    orphans = sorted({b.class_id for a in session.store.values() for b in a.boxes
                      if b.class_id >= len(names)})
    if live:
        updated = _api(args, "/api/classes", {"names": names}).get("classes", names)
    else:
        updated = session.set_classes(names)
        _finish(session)
    result = {"classes": updated, "ranOn": "server" if live else "here"}
    if orphans:
        result["warning"] = (
            f"Labels on this session still use class {', '.join(str(o) for o in orphans)}, "
            f"which {'is' if len(orphans) == 1 else 'are'} past the end of that list. "
            f"Those labels are still there and will export under a class that no longer "
            f"has a name.")
    return out(result)


def _resolve_sam(wanted: str) -> str:
    from app.core.sam3_handler import SAM_MODELS

    if wanted in SAM_MODELS:
        return wanted
    hits = [k for k in SAM_MODELS if wanted.lower() in k.lower()]
    if len(hits) == 1:
        return hits[0]
    raise Failed(f"No single model matches {wanted!r}. Pick one of: "
                 f"{'; '.join(SAM_MODELS)}")


def cmd_settings(args) -> int:
    """Show the settings that change what the auto-labeller does, or change them."""
    from app.core import maskops
    from app.core.sam3_handler import SAM_MODELS
    from app.core.session import TASKS

    live = _server_up(args)
    changes: dict = {}
    if args.task:
        changes["task"] = args.task
    if args.threshold is not None:
        if not 0.05 <= args.threshold <= 0.95:
            raise Failed("--threshold runs from 0.05 to 0.95.")
        changes["threshold"] = args.threshold
    if args.mask_detail:
        changes["maskDetail"] = args.mask_detail
    if args.sam_model:
        changes["samModel"] = _resolve_sam(args.sam_model)

    session = _open(args, writing=bool(changes) and not live, require_frames=False)
    now = {"task": session.task, "samModel": session.sam_model_key,
           "threshold": session.threshold, "maskDetail": session.mask_detail}
    if changes:
        if live:
            # Believe the answer, not the file.  Setting these only marks the
            # session dirty on the server; the write happens on its next
            # autosave, so re-reading from disk here would report back the
            # values we just changed.
            answered = _api(args, "/api/settings", changes)
            now = {k: answered.get(k, v) for k, v in now.items()}
        else:
            if "task" in changes:
                session.task = changes["task"]
            if "samModel" in changes:
                session.sam_model_key = changes["samModel"]
            if "threshold" in changes:
                session.threshold = changes["threshold"]
            if "maskDetail" in changes:
                session.mask_detail = changes["maskDetail"]
            session.touch()
            _finish(session)
            now = {"task": session.task, "samModel": session.sam_model_key,
                   "threshold": session.threshold, "maskDetail": session.mask_detail}

    return out({
        **now,
        "changed": sorted(changes) or None,
        "ranOn": ("server" if live else "here") if changes else None,
        "choices": {"task": list(TASKS), "samModel": list(SAM_MODELS),
                    "maskDetail": list(maskops.DETAIL_LEVELS)},
    })


def cmd_jobs(args) -> int:
    """Show what the app is working on, and stop it.

    Jobs only exist inside the running app — importing, auto-labelling and
    training all happen on its worker threads.  With nothing running there is
    nothing to show, which is worth saying plainly rather than printing an
    empty list that looks like an idle app.
    """
    if not _server_up(args):
        raise Failed(f"Nothing is running at http://{args.host}:{args.port}. Jobs live "
                     f"inside the app, so with it closed there are none to show.")
    if args.cancel:
        stopped = []
        targets = ([j["id"] for j in (_api(args, "/api/jobs").get("active") or [])]
                   if args.cancel == "all" else [args.cancel])
        for job_id in targets:
            answer = _api(args, f"/api/jobs/{job_id}/cancel", {})
            stopped.append({"id": job_id, "cancelled": answer.get("cancelled", False),
                            "label": (answer.get("job") or {}).get("label", "")})
        return out({"cancelled": stopped},
                   EXIT_OK if any(s["cancelled"] for s in stopped) else EXIT_FAIL)
    if args.log:
        return out(_api(args, f"/api/jobs/{args.log}/log"))

    data = _api(args, "/api/jobs")
    def brief(job: dict) -> dict:
        return {k: job.get(k) for k in
                ("id", "kind", "status", "current", "total", "message", "elapsed")}
    active = [brief(j) for j in (data.get("active") or [])]
    return out({"active": active,
                "recent": [brief(j) for j in (data.get("recent") or [])][:10],
                "busy": bool(active)})


# ---------------------------------------------------------------------------
# Auto-labelling
# ---------------------------------------------------------------------------

def _api(args, path: str, payload: dict | None = None, timeout: float = 30.0,
         method: str | None = None) -> dict:
    """Ask the running server for something.

    GET with no payload, POST with one, and *method* where the route wants
    something else — setting a frame's status is a PUT, and sending it as a
    POST gets a 405 that reads like the server is broken.
    """
    import urllib.error
    import urllib.request

    url = f"http://{args.host}:{args.port}{path}"
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=body, method=method or ("POST" if body is not None else "GET"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = (json.load(exc) or {}).get("detail", "")
        except Exception:
            pass
        raise Failed(detail or f"The server answered {exc.code} to {path}.") from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise Failed(f"Could not reach Groundwork at {url}: {exc}") from None


def _snapshot(tag: str) -> str | None:
    """Copy the saved session aside without loading it.

    Session.backup_state saves before it copies, which is right when this
    process owns the session and wrong when the server does — saving would put
    this process's stale copy over the live one.  Copying the file on disk
    leaves the session alone; the server autosaves every few seconds, so the
    restore point is at most that far behind.
    """
    import shutil
    from app.core.session import BACKUP_DIR, STATE_FILE

    if not os.path.isfile(STATE_FILE):
        return None
    os.makedirs(BACKUP_DIR, exist_ok=True)
    safe = "".join(c for c in tag if c.isalnum() or c in "-_")[:40] or "backup"
    path = os.path.join(BACKUP_DIR, f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}.json")
    shutil.copy2(STATE_FILE, path)
    return path


def _mode(args) -> str:
    if args.track:
        return "track"
    return "describe" if args.concept else "propagate"


def _note(message: str, last: str, verbose: bool) -> str:
    if verbose and message and message != last:
        print(f"  {message}", file=sys.stderr)
        return message
    return last


def _await_here(job, verbose: bool) -> dict:
    """Wait for a job running in this process, and stop it properly on Ctrl-C.

    Leaving on the interrupt would drop a daemon thread mid-frame and lose the
    frames already labelled, because the save happens when the run returns.
    Asking it to stop and waiting for it out costs a frame's worth of time and
    keeps the work — which is what the Stop button in the browser does too.
    """
    from app.core.jobs import CANCELLED, DONE, ERROR

    done = (DONE, ERROR, CANCELLED)
    last = ""
    try:
        while job.status not in done:
            last = _note(job.message, last, verbose)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopping — keeping the frames done so far.", file=sys.stderr)
        job.cancel()
        while job.status not in done:
            time.sleep(0.2)
    if job.status == ERROR:
        raise Failed(job.error or "The auto-labeller stopped with an error.")
    return {"status": job.status, **(job.result or {})}


def _await_there(args, job_id: str, verbose: bool) -> dict:
    """Wait for a job running on the server, and stop that too on Ctrl-C.

    Closing the terminal is not the same as changing your mind, but it looks
    identical from the server's side: without this, Ctrl-C ends the watching
    and leaves it working through the rest of the frames with nothing left
    reporting on it.
    """
    done = ("done", "error", "cancelled")
    last = ""
    stopping = False
    while True:
        try:
            job = _api(args, f"/api/jobs/{job_id}/log").get("job") or {}
            last = _note(job.get("message", ""), last, verbose)
            status = job.get("status")
            if status in done:
                if status == "error":
                    raise Failed(job.get("error")
                                 or "The auto-labeller stopped with an error.")
                return {"status": status, **(job.get("result") or {})}
            time.sleep(1.0)
        except KeyboardInterrupt:
            if stopping:
                raise
            stopping = True
            print("Stopping the job on the server — Ctrl-C again to stop waiting "
                  "and leave it running.", file=sys.stderr)
            _api(args, f"/api/jobs/{job_id}/cancel", {})


def _targets(session, args) -> dict:
    """Which frames a run would write to, and which it would pass over.

    Worth knowing before the model spends ten minutes rather than after.
    `--policy skip` only fills empty frames and held-out frames are refused
    wherever the request comes from, so a sweep over "everything" can come down
    to very little, and saying which frames those are is the difference between
    a run that did nothing and a run that did nothing *for a reason*.
    """
    stride = max(1, args.stride)
    every = [i for i in sorted(session.store)
             if i >= args.start and i % stride == 0 and i != args.seed]
    held = [i for i in every if session.store[i].held_out]
    live = [i for i in every if not session.store[i].held_out]
    filled = [i for i in live if session.store[i].boxes] if args.policy == "skip" else []
    return {"write": [i for i in live if i not in set(filled)],
            "heldOut": held, "alreadyLabelled": filled}


def _concepts_from_seed(session, seed_index: int) -> list[tuple[str, int]]:
    seed = session.store[seed_index]
    if not seed.boxes:
        raise Failed(f"Frame {seed_index} has no labels to copy. Label it first — its "
                     f"classes are what the labeller goes looking for.")
    named = [(session.class_names[c].strip(), c)
             for c in sorted({b.class_id for b in seed.boxes})
             if 0 <= c < len(session.class_names) and session.class_names[c].strip()]
    if not named:
        raise Failed("Give your classes names first — the names are the search terms.")
    return named


def _autolabel_here(session, args, verbose: bool) -> dict:
    if args.track:
        stride = session.video_stride.get(session.store[args.seed].source_video, 1)
        if stride > 1 and not args.confirm:
            raise Failed(
                f"This video was imported keeping 1 frame in every {stride}. The tracker "
                f"follows objects between neighbouring frames, so gaps that size make it "
                f"lose them. Re-import at stride 1, use --seed without --track to treat "
                f"each frame separately, or pass --confirm to go ahead anyway.")
        try:
            job = session.start_tracking(args.seed, policy=args.policy)
        except ValueError as exc:
            raise Failed(str(exc)) from None
        return _await_here(job, verbose)

    frames = session.frames_for_range(args.start, args.stride, exclude=args.seed)
    if args.concept:
        concepts = [(args.concept, args.class_id)]
        label = f"Searching for {args.concept!r} in {len(frames)} frames"
    else:
        concepts = _concepts_from_seed(session, args.seed)
        label = f"Copying [{', '.join(c for c, _ in concepts)}] to {len(frames)} frames"
    job = session.start_concept_search(concepts, frames, policy=args.policy,
                                       threshold=args.threshold, label=label)
    return _await_here(job, verbose)


def _autolabel_there(args, verbose: bool) -> dict:
    payload: dict = {"policy": args.policy, "start": args.start, "stride": args.stride}
    if args.threshold is not None:
        payload["threshold"] = args.threshold
    if args.track:
        payload.update({"seed": args.seed, "confirm": args.confirm})
        started = _api(args, "/api/autolabel/track", payload)
        if started.get("needsConfirm"):
            raise Failed(f"{started.get('reason', '').strip()} Or pass --confirm.")
    elif args.concept:
        payload.update({"concept": args.concept, "classId": args.class_id})
        started = _api(args, "/api/autolabel/describe", payload)
    else:
        payload["seed"] = args.seed
        started = _api(args, "/api/autolabel/propagate", payload)
    job_id = (started.get("job") or {}).get("id", "")
    if not job_id:
        raise Failed("The server accepted the request but started no job.")
    return _await_there(args, job_id, verbose)


def cmd_autolabel(args) -> int:
    """Run the auto-labeller over a range of frames.

    This is the one writing command that does not refuse while the browser is
    open.  The others refuse because two processes writing one session lose
    each other's work — but this is the job whose cost is the model, the server
    already has it in memory, and a second process loading its own copy to
    write the same session would be slower and more dangerous at the same time.
    So when the server is up the work is handed to it, and when it is not, it
    runs here.  Either way exactly one process writes.
    """
    if args.concept and args.seed is not None:
        raise Failed("Pass --concept to search for a description or --seed to copy one "
                     "frame's labels, not both — they are different searches.")
    if args.track and args.concept:
        raise Failed("--track follows the shapes on a frame, not a description.")
    if args.track and args.seed is None:
        raise Failed("--track follows the shapes you have already drawn: say which frame "
                     "with --seed N.")
    if not args.track and not args.concept and args.seed is None:
        raise Failed('Say what to look for: --concept "solar panel" to search by '
                     "description, or --seed N to copy one frame's labels to the rest.")

    live = _server_up(args)
    session = _open(args, writing=not live)
    if args.seed is not None and args.seed not in session.store:
        raise Failed(f"No such frame: {args.seed}")
    if args.concept:
        # Defaulting this to 0 would quietly file every hit under whatever class
        # happens to be first, and the labels would look right until something
        # trained on them.  --seed does not need it: the seed frame's own
        # classes say which is which.
        named = [f"{i} {n}" for i, n in enumerate(session.class_names) if n.strip()]
        if args.class_id is None:
            if len(named) > 1:
                raise Failed("Say which class these belong to with --class-id — this "
                             f"session has {', '.join(named)}.")
            args.class_id = 0
        if not 0 <= args.class_id < len(session.class_names):
            raise Failed(f"No class {args.class_id}. This session has {', '.join(named)}.")
    if args.seed is not None and not args.track:
        _concepts_from_seed(session, args.seed)      # fail before the model loads

    plan = None if args.track else _targets(session, args)
    if plan is not None and not plan["write"]:
        return out({"wrote": 0, "mode": _mode(args), "policy": args.policy, **plan,
                    "reason": ("every frame in range already has labels, and --policy skip "
                               "leaves those alone" if plan["alreadyLabelled"]
                               else "no frames match that range")}, EXIT_FAIL)
    if args.dry_run:
        return out({"dryRun": True, "mode": _mode(args), "policy": args.policy,
                    "wouldRunOn": "server" if live else "here",
                    "seed": args.seed, "concept": args.concept,
                    **(plan or {"note": "tracking follows the whole video from the seed"})})

    # Plain ASCII: this goes straight to a console, and stderr is cp1252 on
    # Windows, which turns anything prettier into question marks.
    print(f"{_mode(args)}, policy {args.policy}, running "
          f"{'on the server' if live else 'here'}"
          + (f", {len(plan['write'])} frames" if plan else ""), file=sys.stderr)

    backup = _snapshot(f"autolabel-{_mode(args)}")
    if live:
        result = _autolabel_there(args, args.verbose)
        session.load_state()          # the server saved when the job finished
    else:
        result = _autolabel_here(session, args, args.verbose)
        _finish(session)

    return out({
        "mode": _mode(args),
        "policy": args.policy,
        "ranOn": "server" if live else "here",
        "status": result.get("status"),
        "objects": result.get("objects", 0),
        "framesWithObjects": result.get("framesWithObjects", result.get("frames", 0)),
        "heldOutSkipped": plan["heldOut"] if plan else [],
        "backup": backup,
        "stats": session.stats(),
    }, EXIT_OK if _went_well(result) else EXIT_FAIL)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def cmd_export(args) -> int:
    session = _open(args, writing=True)
    task = args.task or session.task
    os.makedirs(args.out, exist_ok=True)
    written = export_dataset(
        session.store, session.class_names, args.out,
        skip_unverified=args.skip_unreviewed,
        is_seg=task == "segment", is_pose=task == "pose",
    )
    _finish(session)
    written["dataYaml"] = os.path.join(os.path.abspath(args.out), "data.yaml")
    written["task"] = task
    return out(written, EXIT_FAIL if written["train"] == 0 else EXIT_OK)


def _quiet():
    """Send anything printed to stdout to stderr instead, for the duration.

    Stdout here is the report, and something reading it expects JSON and
    nothing else.  Ultralytics prints its banner, its model summary and a line
    per epoch straight to stdout, which lands in the middle of the JSON and
    leaves it unparseable.  None of that is worth losing, so it is moved rather
    than swallowed.
    """
    from contextlib import redirect_stdout
    return redirect_stdout(sys.stderr)


def _train_and_score(model: str, data_yaml: str, *, epochs: int, imgsz: int,
                     name: str, log: bool) -> dict:
    with _quiet():
        trained = train_yolo(
            model, data_yaml, epochs=epochs, imgsz=imgsz, name=name,
            on_log=(lambda line: print(line, file=sys.stderr)) if log else None,
        )
        if not trained.get("best"):
            raise Failed("Training finished without producing weights.")
        scored = score_model(trained["best"], data_yaml, imgsz=imgsz)
    scored["save_dir"] = trained["save_dir"]
    scored["epochs"] = epochs
    scored["metric"] = metric_label_for(model)
    return scored


def cmd_train(args) -> int:
    data_yaml = os.path.join(args.data, "data.yaml")
    if not os.path.isfile(data_yaml):
        raise Failed(f"No data.yaml in {args.data} — export first.")
    return out(_train_and_score(args.model, data_yaml, epochs=args.epochs,
                                imgsz=args.imgsz, name=args.name, log=args.verbose))


def cmd_convert(args) -> int:
    """Turn an exported dataset into another task's shape.

    Neither of these touches the session — they read one dataset folder and
    write another.  They still go to the server when it is up, because turning
    boxes into outlines runs SAM, and the server already has it loaded.
    """
    source = os.path.abspath(args.source)
    if not os.path.isfile(os.path.join(source, "data.yaml")):
        raise Failed(f"No data.yaml in {source} — point this at an exported dataset.")
    output = os.path.abspath(args.out) if args.out else os.path.normpath(source) + args.to
    if os.path.abspath(output) == source:
        raise Failed("The output folder would be the source folder.")

    live = _server_up(args)
    print(f"converting {source} to {args.to} -> {output}", file=sys.stderr)
    if live:
        payload = {"source": source, "output": output}
        if args.to == "pose":
            payload["edgeMargin"] = args.edge_margin
        started = _api(args, f"/api/convert/{args.to}", payload)
        result = _await_there(args, (started.get("job") or {}).get("id", ""), args.verbose)
    else:
        def note(line) -> None:
            if args.verbose:
                print(f"  {line}", file=sys.stderr)
        with _quiet():
            if args.to == "seg":
                from app.core.yolo_seg_converter import YoloDatasetConverter
                session = _open(args, writing=False, require_frames=False)
                model, processor = session.cached_sam()
                result = YoloDatasetConverter(
                    source_root=source, output_root=output,
                    progress_callback=lambda c, t: note(f"{c} / {t}"),
                    status_callback=note, model=model, processor=processor,
                ).convert()
            else:
                from app.core.yolo_pose_converter import YoloPoseConverter
                result = YoloPoseConverter(
                    source_root=source, output_root=output,
                    edge_margin=max(0.0, min(0.49, args.edge_margin / 100.0)),
                    progress_callback=lambda c, t: note(f"{c} / {t}"),
                    status_callback=note,
                ).convert()
        result = {"status": "done", **(result or {})}

    return out({"to": args.to, "source": source, "output": output,
                "ranOn": "server" if live else "here", **result},
               EXIT_OK if _went_well(result) else EXIT_FAIL)


def cmd_onnx(args) -> int:
    """Export trained weights to ONNX, for running them outside this app."""
    pt_path = os.path.abspath(args.weights)
    if not os.path.isfile(pt_path):
        raise Failed(f"No such checkpoint: {pt_path}")

    live = _server_up(args)
    print(f"exporting {pt_path} to ONNX (this cannot be interrupted)", file=sys.stderr)
    if live:
        started = _api(args, "/api/onnx",
                       {"ptPath": pt_path, "precision": "FP16" if args.fp16 else "FP32",
                        "dynamic": not args.fixed, "imgsz": args.imgsz})
        result = _await_there(args, (started.get("job") or {}).get("id", ""), args.verbose)
    else:
        from app.core.onnx_exporter import export_onnx
        with _quiet():
            written = export_onnx(
                pt_path, half=args.fp16, imgsz=args.imgsz, dynamic=not args.fixed,
                on_log=(lambda line: print(f"  {line}", file=sys.stderr))
                if args.verbose else None)
        result = {"status": "done", **(written or {})}

    return out({"weights": pt_path, "precision": "FP16" if args.fp16 else "FP32",
                "dynamic": not args.fixed, "imgsz": args.imgsz,
                "ranOn": "server" if live else "here", **result},
               EXIT_OK if _went_well(result) else EXIT_FAIL)


def cmd_score(args) -> int:
    data_yaml = os.path.join(args.data, "data.yaml")
    if not os.path.isfile(data_yaml):
        raise Failed(f"No data.yaml in {args.data} — export first.")
    if not os.path.isfile(args.weights):
        raise Failed(f"No such weights: {args.weights}")
    with _quiet():
        scored = score_model(args.weights, data_yaml, imgsz=args.imgsz)
    return out(scored)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def cmd_gate(args) -> int:
    """Change the labels, and keep the change only if the model still scores.

    The order matters.  Score first, then change, then score again: a number
    from before the change is the only thing the number after it can be
    compared with, and measuring after the fact against a remembered figure
    from some other run compares two things that were never alike.
    """
    session = _open(args, writing=True)
    held = sorted(i for i, a in session.store.items() if a.held_out)
    if not held:
        raise Failed(
            "No frames are held out, so there is nothing to measure against and "
            "the gate would be scoring the model on its own training labels. "
            "Hold some frames out first: python -m app.cli hold 3 9 17")
    if not args.script:
        raise Failed("Say what to change, with --script.")

    report: dict = {
        "heldOut": held, "scripts": args.script, "epochs": args.epochs,
        "tolerance": args.tolerance, "kept": False,
    }
    work = os.path.abspath(args.data or os.path.join(RUNS_DIR, "gate", "dataset"))
    task = args.task or session.task

    def export_now() -> str:
        os.makedirs(work, exist_ok=True)
        written = export_dataset(session.store, session.class_names, work,
                                 is_seg=task == "segment", is_pose=task == "pose")
        if written["train"] == 0:
            raise Failed("Nothing to train on — every frame is held out.")
        return os.path.join(work, "data.yaml")

    def note(message: str) -> None:
        print(message, file=sys.stderr)

    # ── before ────────────────────────────────────────────────────────
    if args.baseline is not None:
        baseline = {"score": args.baseline, "from": "given"}
    else:
        note("scoring the labels as they are…")
        baseline = _train_and_score(args.model, export_now(), epochs=args.epochs,
                                    imgsz=args.imgsz, name="gate-before", log=args.verbose)
    report["before"] = baseline

    # ── the change ────────────────────────────────────────────────────
    backup = session.backup_state("gate")
    report["backup"] = backup
    labels_before = session.stats()["boxes"]

    def put_back(why: str) -> None:
        session.load_state(backup)
        session.save_state()
        report["kept"] = False
        report["verdict"] = why

    # Anything from here on leaves the labels changed until they have been
    # weighed up, so a failure in the middle has to put them back.  A crash
    # between changing and deciding would otherwise leave exactly the state the
    # gate exists to prevent: an unmeasured change, kept.
    try:
        changes = []
        for script_id in args.script:
            note(f"running {script_id}…")
            try:
                result = session.run_script(script_id, None, _settings(args.set))
            except scripts.ScriptError as exc:
                raise Failed(str(exc)) from None
            changes.append({"script": script_id, "changed": result.get("changed", 0),
                            "before": result.get("before"), "after": result.get("after"),
                            "errors": result.get("errors", [])})
        session.save_state()
        report["changes"] = changes
        report["touched"] = sum(c["changed"] for c in changes)

        if report["touched"] == 0:
            report["kept"] = True
            report["verdict"] = "nothing changed, so nothing to weigh up"
            return out(report)

        # How much of the labelling went missing.  Worth asking before training
        # again, and not only to save the training: on a small held-out set the
        # score is a blunt enough instrument that deleting most of the labels
        # can pass for a change that did nothing.  Labels disappearing wholesale
        # is not something a score should be asked to adjudicate.
        labels_after = session.stats()["boxes"]
        lost = (labels_before - labels_after) / max(labels_before, 1)
        report["labels"] = {"before": labels_before, "after": labels_after,
                            "lost": round(lost, 4)}
        if lost > args.max_loss:
            put_back(f"put back — {labels_before - labels_after} of {labels_before} labels "
                     f"({lost:.0%}) disappeared, over the {args.max_loss:.0%} allowed. "
                     f"No score was asked for: a change this large is not something "
                     f"a few held-out frames can weigh up.")
            report["stats"] = session.stats()
            return out(report, EXIT_FAIL)

        # ── after ─────────────────────────────────────────────────────
        note("scoring the changed labels…")
        candidate = _train_and_score(args.model, export_now(), epochs=args.epochs,
                                     imgsz=args.imgsz, name="gate-after", log=args.verbose)
    except Failed:
        put_back("put back — the run failed before the change could be weighed up")
        raise
    except BaseException:
        put_back("put back — the run failed before the change could be weighed up")
        raise

    report["after"] = candidate
    delta = candidate["score"] - baseline["score"]
    report["delta"] = round(delta, 5)

    if delta >= -args.tolerance:
        report["kept"] = True
        report["verdict"] = (f"kept — {candidate['score']:.4f} against {baseline['score']:.4f} "
                             f"({delta:+.4f})")
        session.save_state()
    else:
        put_back(f"put back — {candidate['score']:.4f} against {baseline['score']:.4f} "
                 f"({delta:+.4f}), worse than the {args.tolerance} allowed")
    report["stats"] = session.stats()

    # A handful of held-out frames is a coarse instrument.  Saying so next to a
    # small delta is the difference between a number and a number that has been
    # read properly.
    if len(held) < 10 and abs(delta) < 0.02:
        report["caution"] = (
            f"{len(held)} held-out frames and a delta of {delta:+.4f} — that is "
            f"within the noise of a run. Treat this as 'nothing broke', not as "
            f"evidence the change helped.")
    return out(report, EXIT_OK if report["kept"] else EXIT_FAIL)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Groundwork without the browser — for CI, a schedule, or an agent.")
    parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8321, help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true",
                        help="write even though the browser server is running")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="what is in the session").set_defaults(fn=cmd_stats)

    p = sub.add_parser("frames", help="list frames")
    p.add_argument("--held-out", action="store_true", help="only the held-out ones")
    p.add_argument("--status", choices=("pending", "verified"),
                   help="whether a person has checked the frame")
    p.add_argument("--exported", action="store_true",
                   help="only frames already written into a dataset")
    p.set_defaults(fn=cmd_frames)

    p = sub.add_parser("render", help="write a frame with its labels drawn on")
    p.add_argument("index", type=int)
    p.add_argument("--out", help="where to write the png")
    p.add_argument("--no-labels", action="store_true", help="outlines only")
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("hold", help="hold frames out to measure against")
    p.add_argument("frames", nargs="*", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--release", action="store_true", help="put them back in play")
    p.add_argument("--tag", help=argparse.SUPPRESS)
    p.set_defaults(fn=cmd_hold)

    sub.add_parser("scripts", help="list the scripts available").set_defaults(fn=cmd_scripts)

    p = sub.add_parser("run", help="run a script over frames")
    p.add_argument("script")
    p.add_argument("frames", nargs="*", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", default=[])
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("backup", help="take a restore point")
    p.add_argument("--tag")
    p.set_defaults(fn=cmd_backup)

    p = sub.add_parser("restore", help="put a restore point back")
    p.add_argument("path")
    p.set_defaults(fn=cmd_restore)

    p = sub.add_parser("tidy", help="fold stacked duplicate boxes together")
    p.add_argument("frames", nargs="*", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--threshold", type=float, default=0.8,
                   help="how much two boxes must overlap to count as one (default 0.8)")
    p.add_argument("--any-class", dest="any_class", action="store_true",
                   help="merge across classes too, not only within one")
    p.set_defaults(fn=cmd_tidy)

    p = sub.add_parser("status", help="mark frames checked, or back to unchecked")
    p.add_argument("frames", nargs="*", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--pending", action="store_true",
                   help="mark them unchecked instead of checked")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("clear-labels", help="throw away labels, keep the pictures")
    p.add_argument("frames", nargs="*", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--yes", action="store_true", help="confirm losing the labels")
    p.set_defaults(fn=cmd_clear_labels)

    p = sub.add_parser("delete", help="drop frames from the session")
    p.add_argument("frames", nargs="*", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--yes", action="store_true", help="confirm removing them")
    p.set_defaults(fn=cmd_delete)

    p = sub.add_parser("convert", help="turn a dataset into another task's shape")
    p.add_argument("source", help="an exported dataset folder, with its data.yaml")
    p.add_argument("--to", choices=("seg", "pose"), required=True)
    p.add_argument("--out", help="where to write it (default: the source name + seg/pose)")
    p.add_argument("--edge-margin", dest="edge_margin", type=float, default=2.0,
                   help="pose only: percent of the edge to ignore (default 2.0)")
    p.add_argument("-v", "--verbose", action="store_true", help="progress on stderr")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("onnx", help="export trained weights to ONNX")
    p.add_argument("weights", help="a trained .pt checkpoint")
    p.add_argument("--fp16", action="store_true", help="half precision instead of FP32")
    p.add_argument("--fixed", action="store_true", help="fixed input size, not dynamic")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("-v", "--verbose", action="store_true", help="progress on stderr")
    p.set_defaults(fn=cmd_onnx)

    p = sub.add_parser("import", help="bring in video, pictures or a YOLO dataset")
    p.add_argument("path", help="a folder, a dataset root, or one video or image")
    p.add_argument("--mode", choices=("add", "replace", "dataset"), default="add",
                   help="add: keep what is there (default). replace: empty the session "
                        "first. dataset: read labels and classes from a YOLO dataset.")
    p.add_argument("--stride", type=int, default=1,
                   help="keep 1 video frame in every N (default 1, every frame)")
    p.add_argument("--yes", action="store_true",
                   help="confirm emptying a session that already has frames")
    p.add_argument("-v", "--verbose", action="store_true", help="progress on stderr")
    p.set_defaults(fn=cmd_import)

    p = sub.add_parser("classes", help="show or set the class names")
    p.add_argument("names", nargs="*", help="in class-id order; commas work too")
    p.set_defaults(fn=cmd_classes)

    p = sub.add_parser("settings", help="show or change what the auto-labeller does")
    p.add_argument("--task", choices=("detect", "segment", "pose"))
    p.add_argument("--threshold", type=float,
                   help="confidence to accept a detection, 0.05 to 0.95")
    p.add_argument("--sam-model", dest="sam_model",
                   help="which SAM to use; part of the name is enough")
    p.add_argument("--mask-detail", dest="mask_detail",
                   choices=("coarse", "medium", "fine"))
    p.set_defaults(fn=cmd_settings)

    p = sub.add_parser("jobs", help="what the running app is working on")
    p.add_argument("--cancel", metavar="ID", help='a job id, or "all"')
    p.add_argument("--log", metavar="ID", help="the log lines for one job")
    p.set_defaults(fn=cmd_jobs)

    p = sub.add_parser("autolabel", help="let the model label frames for you")
    p.add_argument("--concept", help='what to look for, in words: "solar panel"')
    p.add_argument("--seed", type=int, metavar="N",
                   help="copy frame N's labels everywhere — its class names become "
                        "the search terms")
    p.add_argument("--track", action="store_true",
                   help="follow the exact shapes on --seed through the video instead")
    p.add_argument("--class-id", type=int, dest="class_id",
                   help="which class --concept belongs to; needed when the session "
                        "has more than one")
    p.add_argument("--policy", choices=("skip", "merge", "replace"), default="skip",
                   help="skip: only frames with no labels (default). merge: add to what "
                        "is there. replace: overwrite it.")
    p.add_argument("--threshold", type=float,
                   help="confidence to accept a detection; lower finds more and invents "
                        "more (default: the session's)")
    p.add_argument("--start", type=int, default=0, help="first frame to consider")
    p.add_argument("--stride", type=int, default=1, help="take every Nth frame")
    p.add_argument("--dry-run", action="store_true",
                   help="say which frames would be written and stop")
    p.add_argument("--confirm", action="store_true",
                   help="go ahead with --track on a video imported with gaps")
    p.add_argument("-v", "--verbose", action="store_true", help="progress on stderr")
    p.set_defaults(fn=cmd_autolabel)

    p = sub.add_parser("export", help="write a YOLO dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--task", choices=("detect", "segment", "pose"))
    p.add_argument("--skip-unreviewed", action="store_true")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("train", help="train on an exported dataset and score it")
    p.add_argument("--data", required=True, help="the exported dataset folder")
    p.add_argument("--model", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--name", default="cli")
    p.add_argument("-v", "--verbose", action="store_true", help="per-epoch lines on stderr")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("score", help="score existing weights on the held-out half")
    p.add_argument("--data", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("gate", help="change the labels, keep it only if the score holds")
    p.add_argument("--script", action="append", default=[],
                   help="what to change; repeat to run several in order")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", default=[])
    p.add_argument("--model", default="YOLO11n-seg")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tolerance", type=float, default=0.005,
                   help="how far the score may fall and still be kept (default 0.005)")
    p.add_argument("--max-loss", type=float, default=0.2,
                   help="how much of the labelling may disappear before the change is "
                        "put back unweighed (default 0.2)")
    p.add_argument("--baseline", type=float,
                   help="a score from an earlier identical run, to skip training twice")
    p.add_argument("--data", help="where to write the dataset it trains from")
    p.add_argument("--task", choices=("detect", "segment", "pose"))
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except Failed as exc:
        return out({"error": str(exc)}, EXIT_MISUSE)
    except KeyboardInterrupt:
        return out({"error": "stopped"}, EXIT_MISUSE)


if __name__ == "__main__":
    sys.exit(main())
