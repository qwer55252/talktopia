from __future__ import annotations
import contextlib
import asyncio
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from talktopia.models import servers


class SpeechPoolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from talktopia.speech_agent import SpeechServerPool

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        runtime = patch("talktopia.speech_agent.RUNTIME_DIR", Path(temp.name))
        runtime.start()
        self.addCleanup(runtime.stop)
        self.pool = SpeechServerPool(["gpu0", "gpu0-2"])

    async def test_episodes_exclusively_lease_and_reuse_servers(self):
        active, seen = set(), []
        release, started = asyncio.Event(), asyncio.Event()

        async def episode(index):
            async with self.pool.lease(str(index)) as (worker, url):
                self.assertNotIn(worker, active)
                active.add(worker)
                seen.append(worker)
                if len(active) == 2:
                    started.set()
                await release.wait()
                active.remove(worker)

        with (
            patch.object(servers, "ensure_speech_ready", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            tasks = [asyncio.create_task(episode(i)) for i in range(4)]
            try:
                await asyncio.wait_for(started.wait(), 2)
                self.assertEqual(len(seen), 2)
                release.set()
                await asyncio.wait_for(asyncio.gather(*tasks), 2)
            finally:
                release.set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(len(seen), 4)
        self.assertEqual(set(self.pool.available), {"gpu0", "gpu0-2"})

    async def test_all_broken_workers_wake_waiters_instead_of_hanging(self):
        async def episode():
            async with self.pool.lease("test"):
                self.fail("An unavailable worker was leased")

        with patch.object(
            servers, "ensure_speech_ready", side_effect=RuntimeError("restart failed")
        ):
            results = await asyncio.wait_for(
                asyncio.gather(*(episode() for _ in range(5)), return_exceptions=True),
                2,
            )
        self.assertTrue(all(isinstance(result, RuntimeError) for result in results))
        self.assertFalse(self.pool.enabled)

    async def test_separate_pipelines_cannot_lease_the_same_server(self):
        from talktopia.speech_agent import SpeechServerPool

        first, second = SpeechServerPool(["gpu0"]), SpeechServerPool(["gpu0"])
        with (
            patch.object(servers, "ensure_speech_ready", return_value=False) as ready,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            async with first.lease("first"):
                with self.assertRaises(BlockingIOError):
                    async with second.lease("second"):
                        self.fail("Exclusive lease was violated")
                self.assertEqual(ready.call_count, 1)

    async def test_failed_recovery_disables_only_its_worker(self):
        calls = []

        def ready(worker):
            calls.append(worker)
            if worker == "gpu0" and calls.count(worker) == 2:
                raise RuntimeError("restart failed")
            return False

        with (
            patch.object(servers, "ensure_speech_ready", side_effect=ready),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            async with self.pool.lease("failed") as (worker, _):
                self.assertEqual(worker, "gpu0")
            async with self.pool.lease("next") as (worker, _):
                self.assertEqual(worker, "gpu0-2")
        self.assertEqual(self.pool.enabled, {"gpu0-2"})

    async def test_cancellation_keeps_lease_until_in_progress_restart_finishes(self):
        from talktopia.speech_agent import SpeechServerPool

        started, release = threading.Event(), threading.Event()

        def ready(worker):
            started.set()
            if not release.wait(3):
                raise AssertionError("Test did not release restart")
            return True

        async def episode():
            async with self.pool.lease("cancelled"):
                self.fail("A cancelled episode started")

        with patch.object(servers, "ensure_speech_ready", side_effect=ready):
            task = asyncio.create_task(episode())
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                task.cancel()
                await asyncio.sleep(0)
                other = SpeechServerPool(["gpu0"])
                with self.assertRaises(BlockingIOError):
                    async with other.lease("other"):
                        self.fail("Restart lock was released early")
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
