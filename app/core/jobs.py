"""
Background job manager — the Qt-free replacement for the old QThread workers.

Every long-running operation (media import, auto-labelling, tracking, dataset
conversion, training, ONNX export) runs as a Job on a worker thread.  A Job
carries progress, a status line, a log, and — crucially — a cancel flag that
the worker polls, so the UI can always stop what it started.

Jobs publish their state changes through a callback (wired to the websocket
event bus in app/server/api.py) so the browser sees live progress.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

# Job lifecycle
PENDING   = "pending"
RUNNING   = "running"
DONE      = "done"
ERROR     = "error"
CANCELLED = "cancelled"

_TERMINAL = {DONE, ERROR, CANCELLED}


class JobCancelled(Exception):
    """Raised inside a worker when the user pressed Stop."""


@dataclass
class Job:
    """A single cancellable background task."""

    id: str
    kind: str                      # "import" | "autolabel" | "track" | "train" | …
    label: str                     # human-readable, shown in the job bar
    cancellable: bool = True
    status: str = PENDING
    current: int = 0
    total: int = 0
    message: str = ""
    error: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _log: deque = field(default_factory=lambda: deque(maxlen=400), repr=False)
    _emit: Callable[["Job", str], None] | None = field(default=None, repr=False)

    # ── worker-side API ───────────────────────────────────────────

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def raise_if_cancelled(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()

    def set_progress(self, current: int, total: int, message: str | None = None) -> None:
        self.current = current
        self.total = total
        if message is not None:
            self.message = message
        self._publish("progress")

    def set_message(self, message: str) -> None:
        self.message = message
        self._publish("progress")

    def log(self, line: str) -> None:
        self._log.append(line)
        self._publish("log", line)

    # ── manager-side API ──────────────────────────────────────────

    def cancel(self) -> bool:
        if not self.cancellable or self.status in _TERMINAL:
            return False
        self._cancel.set()
        self.message = "Stopping…"
        self._publish("progress")
        return True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "cancellable": self.cancellable,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "cancelRequested": self._cancel.is_set(),
            "elapsed": round((self.ended_at or time.time()) - self.started_at, 1),
        }

    def _publish(self, event: str, line: str = "") -> None:
        if self._emit:
            try:
                self._emit(self, event if not line else f"log:{line}")
            except Exception:
                pass


class JobManager:
    """Tracks every background job, broadcasts their state, and cancels on demand."""

    def __init__(self, emit: Callable[[dict], None] | None = None):
        self._jobs: dict[str, Job] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        self._emit = emit

    # ── wiring ────────────────────────────────────────────────────

    def set_emitter(self, emit: Callable[[dict], None]) -> None:
        self._emit = emit

    def _broadcast(self, job: Job, event: str) -> None:
        if not self._emit:
            return
        if event.startswith("log:"):
            payload: dict[str, Any] = {"type": "log", "jobId": job.id, "line": event[4:]}
        else:
            payload = {"type": "job", "event": event, "job": job.to_dict()}
        self._emit(payload)

    # ── queries ───────────────────────────────────────────────────

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def active(self) -> list[Job]:
        return [j for j in self._jobs.values() if j.status in (PENDING, RUNNING)]

    def is_busy(self, *kinds: str) -> Job | None:
        """Return the first running job matching any of *kinds* (any kind if empty)."""
        for job in self._jobs.values():
            if job.status in (PENDING, RUNNING) and (not kinds or job.kind in kinds):
                return job
        return None

    def snapshot(self) -> list[dict]:
        return [j.to_dict() for j in self._jobs.values() if j.status in (PENDING, RUNNING)]

    def recent(self, limit: int = 20) -> list[dict]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

    def log_lines(self, job_id: str) -> list[str]:
        job = self._jobs.get(job_id)
        return list(job._log) if job else []

    # ── running ───────────────────────────────────────────────────

    def start(
        self,
        kind: str,
        label: str,
        target: Callable[[Job], dict | None],
        *,
        cancellable: bool = True,
        on_done: Callable[[Job], None] | None = None,
    ) -> Job:
        """Run *target(job)* on a worker thread.  Returns the Job immediately."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label, cancellable=cancellable)
        job._emit = self._broadcast

        with self._lock:
            self._jobs[job.id] = job
            self._prune()

        def _run() -> None:
            job.status = RUNNING
            self._broadcast(job, "started")
            try:
                result = target(job)
                if job.is_cancelled:
                    job.status = CANCELLED
                    job.message = "Stopped"
                    # A target that returns after being asked to stop has kept
                    # everything it reached — a script writes its frames before
                    # handing the report back.  Throwing that report away left
                    # the labels changed and nothing to say what changed, which
                    # reads as Stop having done nothing at all.
                    job.result = result or {}
                else:
                    job.status = DONE
                    job.result = result or {}
                    if not job.message:
                        job.message = "Complete"
            except JobCancelled:
                job.status = CANCELLED
                job.message = "Stopped"
            except Exception as exc:
                job.status = ERROR
                job.error = f"{exc}"
                job.message = f"{exc}"
                job.log(traceback.format_exc())
            finally:
                job.ended_at = time.time()
                self._broadcast(job, "finished")
                if on_done:
                    try:
                        on_done(job)
                    except Exception:
                        pass

        thread = threading.Thread(target=_run, name=f"job-{kind}-{job.id}", daemon=True)
        self._threads[job.id] = thread
        thread.start()
        return job

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        return bool(job and job.cancel())

    def cancel_all(self) -> None:
        for job in list(self._jobs.values()):
            job.cancel()

    def wait_all(self, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        for thread in list(self._threads.values()):
            thread.join(max(0.0, deadline - time.time()))

    def _prune(self, keep: int = 40) -> None:
        finished = [j for j in self._jobs.values() if j.status in _TERMINAL]
        if len(finished) <= keep:
            return
        finished.sort(key=lambda j: j.ended_at or 0)
        for job in finished[: len(finished) - keep]:
            self._jobs.pop(job.id, None)
            self._threads.pop(job.id, None)
