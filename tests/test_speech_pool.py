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


def test_shared_asr_uses_text_only_decoding_without_changing_pcm():
    import wave
    from types import SimpleNamespace

    import numpy as np
    from scipy.signal import resample_poly

    from talktopia.experiment import settings
    from talktopia.speech_agent import SpeechBackend

    # A backend without model loading; exercise the production WAV/VAD path.
    backend = SpeechBackend.__new__(SpeechBackend)
    backend.asr_lock = threading.Lock()
    backend.np = np
    backend.torch = SimpleNamespace(from_numpy=lambda samples: samples)
    backend.resample_poly = resample_poly
    backend.vad = object()
    raw = np.arange(-1800, 1800, dtype="<i2")
    expected = resample_poly(raw.astype(np.float32) / 32768, 2, 3).astype(np.float32)
    spans = [{"start": 100, "end": 700}, {"start": 900, "end": 2200}]

    def vad(samples, model, **kwargs):
        assert model is backend.vad
        np.testing.assert_array_equal(samples, expected)
        assert kwargs == dict(
            sampling_rate=16000,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
            speech_pad_ms=30,
        )
        return spans

    def transcribe(samples, **kwargs):
        # Both sentences and all leading/intermediate/trailing PCM survive VAD.
        np.testing.assert_array_equal(samples, expected)
        assert kwargs == dict(
            language="en",
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
        )
        # Preserve recognized words; never use generated dialogue as a hint.
        return iter([SimpleNamespace(text=" Recognized words. ")]), None

    backend.get_speech_timestamps = vad
    backend.asr = SimpleNamespace(transcribe=transcribe)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(raw.tobytes())
    assert backend.transcribe(buf.getvalue()) == "Recognized words."
    assert (
        settings("round-robin")["asr_decoding"]
        == settings("surface5-full-duplex")["asr_decoding"]
        == {"without_timestamps": True, "vad_mode": "speech_presence_only"}
    )
    backend.get_speech_timestamps = lambda *args, **kwargs: []
    backend.asr.transcribe = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("silence must not be decoded")
    )
    assert backend.transcribe(buf.getvalue()) == ""


def test_speech_health_publishes_and_checks_decoder_policy(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import pytest
    from fastapi.testclient import TestClient

    from talktopia.models.config import SPEECH_PROTOCOL

    backend = SimpleNamespace(voices={})
    with TestClient(servers.create_speech_app(backend=backend, db=tmp_path)) as client:
        health = client.get("/health").json()
    assert health["speech_protocol"] == SPEECH_PROTOCOL == "surface5-http-v6"
    assert health["asr_without_timestamps"] is True
    assert health["asr_vad_mode"] == "speech_presence_only"
    monkeypatch.setattr(servers, "database_path", lambda: tmp_path)
    monkeypatch.setattr(servers, "get_json", lambda *args, **kwargs: health)
    assert servers.speech_health()["asr_without_timestamps"] is True
    health["speech_protocol"] = "surface5-http-v5"
    with pytest.raises(ValueError, match="Incompatible speech service"):
        servers.speech_health()
    health["speech_protocol"] = SPEECH_PROTOCOL
    health["asr_without_timestamps"] = False
    with pytest.raises(ValueError, match="Incompatible speech service"):
        servers.speech_health()
    health.pop("asr_without_timestamps")
    with pytest.raises(ValueError, match="Incompatible speech service"):
        servers.speech_health()
    health["asr_without_timestamps"] = True
    health["asr_vad_mode"] = "trim_and_concatenate"
    with pytest.raises(ValueError, match="Incompatible speech service"):
        servers.speech_health()
    health.pop("asr_vad_mode")
    with pytest.raises(ValueError, match="Incompatible speech service"):
        servers.speech_health()
