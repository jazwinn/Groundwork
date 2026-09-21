"""Regression tests for the concurrent-save race behind a 500 on an applied write.

`POST /api/labels/merge-overlaps` intermittently returned HTTP 500 while the
merge itself had already been applied.  The merge was never at fault: the
handler's `session.merge_overlapping` ends in `save_state()`, and every save
went through one shared "session.json.tmp".  The autosave thread and the
request thread would land in there together, one renaming the temp file away
while the other still held it open — PermissionError (WinError 32) on Windows,
silently interleaved JSON on POSIX.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import session as session_module
from app.core.sam3_handler import BBox, FrameAnnotation
from app.core.session import Session


def make_session(tmpdir: str, frames: int = 40) -> Session:
    """A Session that saves into `tmpdir` and never touches a real session."""
    sess = Session()
    sess.store = {
        i: FrameAnnotation(
            frame_index=i,
            image_path=os.path.join(tmpdir, f"{i:06d}.jpg"),
            boxes=[BBox(x1=10 + k, y1=10 + k, x2=130 + k, y2=130 + k, class_id=0)
                   for k in range(6)],
            width=1280, height=720,
        )
        for i in range(frames)
    }
    return sess


class SaveStateTests(unittest.TestCase):
    """save_state is called from the autosave thread and from request threads."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.state = os.path.join(self.tmp.name, "session.json")
        patcher = mock.patch.object(session_module, "STATE_FILE", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        self.session = make_session(self.tmp.name)

    def _hammer(self, workers: int, rounds: int) -> list[BaseException]:
        """Save from several threads at once; hand back whatever they raised."""
        errors: list[BaseException] = []
        start = threading.Barrier(workers)

        def run() -> None:
            start.wait()
            for _ in range(rounds):
                try:
                    self.session.save_state()
                except BaseException as exc:   # recorded, so the test can show it
                    errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return errors

    def test_concurrent_saves_do_not_fail(self) -> None:
        """The reported symptom: overlapping saves blew up on the shared temp file."""
        errors = self._hammer(workers=4, rounds=60)
        self.assertEqual(errors, [], f"concurrent save_state raised: {errors[:3]}")

    def test_concurrent_saves_leave_a_complete_file(self) -> None:
        """A reader must never catch session.json half-written."""
        stop = threading.Event()
        bad: list[str] = []

        def read() -> None:
            while not stop.is_set():
                # Polled rather than read back to back: a reader that never
                # lets go of the handle is an adversary Windows cannot rename
                # past, which is a different problem from the one under test.
                time.sleep(0.002)
                try:
                    with open(self.state, encoding="utf-8") as fh:
                        data = json.load(fh)
                except OSError:
                    # Windows will not open the file mid-replace; try again.
                    continue
                except ValueError as exc:
                    bad.append(f"unparseable session.json: {exc}")
                    return
                if len(data.get("frames", [])) != len(self.session.store):
                    bad.append(f"truncated session.json: "
                               f"{len(data.get('frames', []))} frames")
                    return

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            errors = self._hammer(workers=4, rounds=15)
        finally:
            stop.set()
            reader.join(timeout=5)

        self.assertEqual(errors, [], f"concurrent save_state raised: {errors[:3]}")
        self.assertEqual(bad, [], str(bad[:3]))

    def test_only_one_save_writes_at_a_time(self) -> None:
        """The invariant behind the fix, checked without relying on the platform.

        Windows reports the old behaviour as a PermissionError; POSIX just lets
        the two writers scribble over one another.  Counting how many saves are
        inside the write at once catches it either way.
        """
        occupancy = {"now": 0, "peak": 0}
        guard = threading.Lock()
        real_open = open

        def slow_open(*args, **kwargs):
            with guard:
                occupancy["now"] += 1
                occupancy["peak"] = max(occupancy["peak"], occupancy["now"])
            try:
                time.sleep(0.01)          # widen the window the race needs
                return real_open(*args, **kwargs)
            finally:
                with guard:
                    occupancy["now"] -= 1

        with mock.patch.object(session_module, "open", slow_open, create=True):
            errors = self._hammer(workers=4, rounds=8)

        self.assertEqual(errors, [], f"concurrent save_state raised: {errors[:3]}")
        self.assertEqual(occupancy["peak"], 1,
                         f"{occupancy['peak']} saves were writing at once — "
                         f"they have to take turns")

    def test_an_edit_during_a_save_is_not_marked_saved(self) -> None:
        """The snapshot is what gets written; anything later is still unsaved."""
        self.session.save_state()
        self.assertFalse(self.session._dirty)

        writing = threading.Event()
        real_open = open

        def slow_open(*args, **kwargs):
            writing.set()
            time.sleep(0.2)               # an edit lands while the write is in flight
            return real_open(*args, **kwargs)

        with mock.patch.object(session_module, "open", slow_open, create=True):
            saver = threading.Thread(target=self.session.save_state)
            saver.start()
            writing.wait(timeout=5)
            self.session.set_boxes(0, [BBox(x1=1, y1=1, x2=9, y2=9, class_id=0)])
            saver.join(timeout=10)

        self.assertTrue(self.session._dirty,
                        "an edit made during a save was marked as saved, so it "
                        "would only reach disk if something else changed later")

    def test_a_failed_save_leaves_the_session_dirty(self) -> None:
        """Nothing landed, so autosave has to come back for it."""
        self.session.set_boxes(0, [BBox(x1=1, y1=1, x2=9, y2=9, class_id=0)])
        with mock.patch.object(session_module, "open", create=True,
                               side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                self.session.save_state()
        self.assertTrue(self.session._dirty)

    def test_a_failed_save_leaves_no_temp_file_behind(self) -> None:
        with mock.patch.object(session_module.os, "replace",
                               side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                self.session.save_state()
        leftovers = [f for f in os.listdir(self.tmp.name) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [], f"left a temp file behind: {leftovers}")


class MergeReportsWhatItDidTests(unittest.TestCase):
    """A merge that was applied must never be reported as a failure."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.state = os.path.join(self.tmp.name, "session.json")
        patcher = mock.patch.object(session_module, "STATE_FILE", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        self.session = make_session(self.tmp.name, frames=3)

    def test_merge_survives_a_failing_save(self) -> None:
        before = len(self.session.store[0].boxes)
        with mock.patch.object(self.session, "save_state",
                               side_effect=PermissionError("denied")):
            result = self.session.merge_overlapping([0], threshold=0.8)

        self.assertGreater(result["removed"], 0)
        self.assertEqual(result["frames"], 1)
        self.assertLess(len(self.session.store[0].boxes), before,
                        "the merge should have been applied in memory")
        self.assertTrue(self.session._dirty,
                        "the unsaved merge must still be queued for autosave")

    def test_merge_endpoint_answers_200_when_the_save_fails(self) -> None:
        """The reported bug at the HTTP layer: a 500 on a write that succeeded."""
        from fastapi.testclient import TestClient

        from app.server import api

        with mock.patch.object(api, "session", self.session), \
                mock.patch.object(self.session, "save_state",
                                  side_effect=PermissionError("denied")), \
                TestClient(api.app) as client:
            response = client.post(
                "/api/labels/merge-overlaps",
                json={"scope": "frame", "frame": 0,
                      "threshold": 0.8, "sameClassOnly": True},
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertGreater(body["removed"], 0)
        self.assertIn("stats", body)


FIRST_IMPORTED = 10_000


class GuardedStore(dict):
    """A store that records every walk taken without the session's lock held."""

    def __init__(self, lock, data) -> None:
        super().__init__(data)
        self.lock = lock
        self.violations: list[str] = []

    def _check(self, how: str) -> None:
        if not self.lock._is_owned():
            self.violations.append(how)

    def values(self):
        self._check("values()")
        return super().values()

    def items(self):
        self._check("items()")
        return super().items()

    def __iter__(self):
        self._check("iteration")
        return super().__iter__()


class StatsTests(unittest.TestCase):
    """stats() walks the store while an import may be adding to it."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.session = make_session(self.tmp.name, frames=200)

    def test_stats_survives_frames_arriving_underneath_it(self) -> None:
        stop = threading.Event()
        errors: list[BaseException] = []

        def churn() -> None:
            """Stand in for an import adding frames from its job thread.

            Frames have to accumulate rather than be added and taken straight
            back out: a dict iterator only notices a change that leaves the
            dict a different size than when the walk started.
            """
            index = FIRST_IMPORTED
            while not stop.is_set():
                with self.session._lock:
                    self.session.store[index] = FrameAnnotation(
                        frame_index=index, image_path="x.jpg",
                        boxes=[BBox(x1=0, y1=0, x2=5, y2=5, class_id=0)],
                    )
                index += 1
                if index > FIRST_IMPORTED + 500:
                    with self.session._lock:
                        for key in [k for k in self.session.store
                                    if k >= FIRST_IMPORTED]:
                            del self.session.store[key]
                    index = FIRST_IMPORTED

        churner = threading.Thread(target=churn, daemon=True)
        churner.start()
        try:
            for _ in range(300):
                try:
                    self.session.stats()
                except BaseException as exc:   # RuntimeError: dict changed size
                    errors.append(exc)
                    break
        finally:
            stop.set()
            churner.join(timeout=5)

        self.assertEqual(errors, [], f"stats() raised: {errors[:1]}")

    def test_stats_reads_the_store_only_under_the_lock(self) -> None:
        """Every count comes off one snapshot, however many get added later.

        The timing test above only catches a long unlocked walk.  A short one —
        a `sum(... for a in self.store.values())` added to the returned dict —
        races just as truly and slips past it, so the rule is checked directly.
        """
        guarded = GuardedStore(self.session._lock, self.session.store)
        self.session.store = guarded

        self.session.stats()

        self.assertEqual(guarded.violations, [],
                         f"stats() walked the store without the lock: "
                         f"{guarded.violations}")


if __name__ == "__main__":
    unittest.main()
