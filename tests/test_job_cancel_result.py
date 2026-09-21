"""Regression tests for Stop throwing away the report of what it stopped.

Pressing Stop on a script running across every frame looked like it did
nothing.  It did stop — `_run_each` breaks out between frames — but stopping
does not undo: `Session.run_script` writes every frame the script reached and
saves before it hands the report back.  `JobManager.start` then took the
cancelled branch and set the status without ever assigning `job.result`, so the
labels had changed and the only record of what changed was dropped on the
floor.  The UI had nothing to show and said so, which read as Stop not working.

A target that *raises* JobCancelled is a different case: it never returned a
report, so there is genuinely nothing to keep.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.jobs import JobCancelled, JobManager

TERMINAL = ("done", "error", "cancelled", "failed")


def wait_for(job, timeout: float = 5.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if job.status in TERMINAL:
            return
        time.sleep(0.01)
    raise AssertionError(f"job {job.id} still {job.status} after {timeout}s")


class CancelKeepsTheReport(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = JobManager()

    def test_graceful_stop_keeps_what_the_target_returned(self) -> None:
        """The case behind the bug: stopped between frames, report handed back."""

        def target(job):
            for scanned in range(2000):
                if job.is_cancelled:
                    return {"scanned": scanned, "stopped": True,
                            "changed": scanned, "removed": scanned * 2}
                time.sleep(0.002)
            return {"scanned": 2000, "stopped": False}

        job = self.jobs.start("script", "merge split modules", target)
        time.sleep(0.1)
        self.jobs.cancel(job.id)
        wait_for(job)

        self.assertEqual(job.status, "cancelled")
        self.assertTrue(job.result, "a stopped run must keep the report it returned")
        self.assertTrue(job.result["stopped"])
        # What it wrote is the whole point of keeping the report.
        self.assertEqual(job.result["removed"], job.result["changed"] * 2)
        self.assertIn("result", job.to_dict())
        self.assertTrue(job.to_dict()["result"], "the report has to reach the client")

    def test_raising_target_has_no_report_to_keep(self) -> None:
        def target(job):
            time.sleep(0.02)
            raise JobCancelled()

        job = self.jobs.start("script", "raises", target)
        wait_for(job)

        self.assertEqual(job.status, "cancelled")
        self.assertEqual(job.result, {})

    def test_ordinary_completion_is_unchanged(self) -> None:
        job = self.jobs.start("script", "fine", lambda _job: {"scanned": 7, "changed": 3})
        wait_for(job)

        self.assertEqual(job.status, "done")
        self.assertEqual(job.message, "Complete")
        self.assertEqual(job.result["changed"], 3)

    def test_a_target_returning_nothing_still_cancels_cleanly(self) -> None:
        def target(job):
            while not job.is_cancelled:
                time.sleep(0.002)
            return None

        job = self.jobs.start("script", "quiet", target)
        time.sleep(0.05)
        self.jobs.cancel(job.id)
        wait_for(job)

        self.assertEqual(job.status, "cancelled")
        self.assertEqual(job.result, {})


if __name__ == "__main__":
    unittest.main()
