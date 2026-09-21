"""Profile voices, WAV validation, and the managed ASR/TTS server backend."""

from __future__ import annotations

import hashlib
import asyncio
import fcntl
import io
import json
import logging
import os
import queue
import threading
import time
import wave
from concurrent.futures import Future
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from talktopia.models.config import (
    ASR_REPO,
    ASR_REVISION,
    TTS_REPO,
    TTS_REVISION,
    TTS_BATCH_SIZE,
    RUNTIME_DIR,
    speech_url,
)

from pydantic import Field

from talktopia.task_space import require_local

# SOTOPIA selects its model classes at import time. Set the backend first.
require_local(os.environ.get("SOTOPIA_STORAGE_BACKEND", "local"))

from sotopia.database import AgentProfile as SotopiaAgentProfile


# WAV validation: API output may use other PCM widths; ASR references require PCM16.


def validate_wav(audio: bytes) -> None:
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getcomptype() != "NONE":
                raise ValueError("TTS must return an uncompressed mono WAV")
            frame_count = wav.getnframes()
            if frame_count <= 0:
                raise ValueError("TTS returned an empty WAV")
            frames = wav.readframes(frame_count)
            if len(frames) != frame_count * wav.getsampwidth():
                raise ValueError("TTS returned a truncated WAV")
    except (wave.Error, EOFError) as exc:
        raise ValueError("TTS returned invalid WAV audio") from exc


def read_pcm_wav(audio: bytes) -> tuple[bytes, int]:
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if (
                wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or wav.getcomptype() != "NONE"
            ):
                raise ValueError("Expected an uncompressed mono PCM16 WAV")
            count = wav.getnframes()
            pcm = wav.readframes(count)
            if count <= 0 or len(pcm) != count * 2 or wav.getframerate() <= 0:
                raise ValueError("Empty or truncated WAV")
            return pcm, wav.getframerate()
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid WAV") from exc


# Reference voices: shared by dataset preparation and the speech server.


def reference_profile(pk: str, source: Path) -> dict[str, str]:
    if not pk or Path(pk).name != pk or pk in {".", ".."}:
        raise ValueError(f"Invalid Agent PK: {pk!r}")
    directory = source / pk
    try:
        audio = (directory / "reference.wav").read_bytes()
        read_pcm_wav(audio)
        text = (directory / "reference.txt").read_text(encoding="utf-8").strip()
        design = json.loads((directory / "design.json").read_text(encoding="utf-8"))
        voice = design["voice"]
        if (
            design["agent_pk"] != pk
            or not text
            or voice["reference_text"].strip() != text
        ):
            raise ValueError("Agent PK or reference text mismatch")
        if hashlib.sha256(audio).hexdigest() != voice["reference_audio_checksum"]:
            raise ValueError("reference.wav checksum mismatch")
        if not isinstance(voice["voice_id"], str) or not voice["voice_id"].strip():
            raise ValueError("Missing voice_id")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Agent {pk}, voice files at {directory}: {exc}") from exc
    return {
        "voice_id": voice["voice_id"],
        "voice_reference_wav": f"voices/{pk}/reference.wav",
        "voice_reference_text": text,
    }


def load_voice_registry(db: Path) -> dict[str, dict[str, Any]]:
    registry = {}
    for path in sorted((db / "AgentProfile").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        pk = row["pk"]
        expected = reference_profile(pk, db / "voices")
        for field_name, value in expected.items():
            if row.get(field_name) != value:
                raise ValueError(
                    f"Agent {pk}: {field_name} does not match copied reference files"
                )
        voice_id = expected["voice_id"]
        if voice_id in registry:
            raise ValueError(f"Duplicate voice_id: {voice_id}")
        registry[voice_id] = {
            **expected,
            "agent_pk": pk,
            "wav_path": db / expected["voice_reference_wav"],
        }
    if not registry:
        raise ValueError(f"No profile voices at {db}; run ./load_profiles.sh first")
    return registry


# Agent profile: preserve the original fields and add the character's voice.


class AgentProfile(SotopiaAgentProfile):
    voice_id: str = Field(min_length=1)
    voice_reference_wav: str = Field(min_length=1)
    voice_reference_text: str = Field(min_length=1)


# SOTOPIA adds query descriptors as class attributes after defining its model.
# Pydantic mistakes those attributes for defaults when creating a subclass.
# Restore the original field definitions, without changing the engine's class.
for _name, _field in SotopiaAgentProfile.model_fields.items():
    AgentProfile.model_fields[_name] = deepcopy(_field)
AgentProfile.model_rebuild(force=True)


class EmptyAudioError(RuntimeError):
    """An empty TTS result is retryable with a different deterministic seed."""


@dataclass
class TTSRequest:
    text: str
    voice_id: str
    seed: int | None = None
    future: Future = field(default_factory=Future)
    queued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None


class SpeechServerPool:
    """Lease one persistent ASR/TTS process for an entire episode attempt."""

    def __init__(self, workers: Sequence[str]):
        self.available = list(workers)
        self.enabled = set(workers)
        self.condition = asyncio.Condition()

    @asynccontextmanager
    async def lease(self, episode_id: str):
        from talktopia.models.servers import ensure_speech_ready, speech_spec

        async def check(endpoint):
            # A restart runs in a thread; keep the exclusive lock until it
            # finishes even if the episode is cancelled in the meantime.
            task = asyncio.create_task(asyncio.to_thread(ensure_speech_ready, endpoint))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise

        async with self.condition:
            await self.condition.wait_for(lambda: self.available or not self.enabled)
            if not self.available:
                raise RuntimeError("No healthy speech workers remain in this run")
            endpoint = self.available.pop(0)
        reusable = False
        locked = False
        lease_file = None
        try:
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            name = speech_spec(endpoint)["name"]
            lease_file = (RUNTIME_DIR / f"{name}.episode.lock").open("a")
            # Also exclude a second pipeline process using the same server.
            fcntl.flock(lease_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            if await check(endpoint):
                print(f"{endpoint}: restarted before {episode_id}", flush=True)
            reusable = True
            print(f"{episode_id}: assigned speech worker {endpoint}", flush=True)
            try:
                yield endpoint, speech_url(endpoint)
            finally:
                # Cancellation should stop work, not start new server processes.
                if asyncio.current_task().cancelling():
                    reusable = False
                else:
                    try:
                        if await check(endpoint):
                            print(
                                f"{endpoint}: restarted after {episode_id}", flush=True
                            )
                    except Exception as exc:
                        reusable = False
                        print(
                            f"{endpoint}: unavailable ({type(exc).__name__}); other workers continue",
                            flush=True,
                        )
        finally:
            if lease_file is not None:
                if locked:
                    fcntl.flock(lease_file, fcntl.LOCK_UN)
                lease_file.close()
            async with self.condition:
                if reusable:
                    self.available.append(endpoint)
                else:
                    self.enabled.discard(endpoint)
                self.condition.notify_all()


class SpeechBackend:
    """One ASR worker and one batched TTS worker on the process's visible GPU."""

    request_timeout = 110

    def __init__(self, db: Path):
        from talktopia.speech_agent import load_voice_registry

        self.voices = load_voice_registry(db)
        self.asr_lock = threading.Lock()
        self.batch_size = TTS_BATCH_SIZE
        if self.batch_size < 1:
            raise ValueError("TALKTOPIA_TTS_BATCH_SIZE must be positive")
        self.requests: queue.Queue = queue.Queue(maxsize=128)
        self.closed = False
        self.fatal_error: str | None = None
        self.tts_active_requests = 0
        self.metrics = dict(
            tts_batches=0, tts_requests=0, tts_seconds=0.0, max_tts_batch=0
        )
        self.prompts: dict[str, Any] = {}
        import numpy as np
        import torch
        from faster_whisper import WhisperModel
        from huggingface_hub import snapshot_download
        from omnivoice import OmniVoice
        from scipy.signal import resample_poly
        from silero_vad import get_speech_timestamps, load_silero_vad

        self.np, self.torch = np, torch
        self.resample_poly = resample_poly
        self.get_speech_timestamps = get_speech_timestamps
        self.vad = load_silero_vad(onnx=True)
        self.asr = WhisperModel(
            snapshot_download(ASR_REPO, revision=ASR_REVISION, local_files_only=True),
            device="cuda",
            device_index=0,
            compute_type="float16",
            local_files_only=True,
        )
        self.tts = OmniVoice.from_pretrained(
            snapshot_download(TTS_REPO, revision=TTS_REVISION, local_files_only=True),
            device_map="cuda:0",
            dtype=torch.float16,
            local_files_only=True,
        )
        if int(self.tts.sampling_rate) != 24000:
            raise ValueError("OmniVoice must return 24000 Hz audio")
        self.thread = threading.Thread(
            target=self._tts_loop, name="tts-batches", daemon=True
        )
        self.thread.start()

    def synthesize(self, text: str, voice_id: str, seed: int | None = None) -> bytes:
        if self.closed or self.fatal_error:
            raise RuntimeError(self.fatal_error or "Speech backend is closed")
        request = TTSRequest(text, voice_id, seed=seed)
        self.requests.put(request, timeout=5)
        try:
            return request.future.result(timeout=self.request_timeout)
        except TimeoutError:
            now = time.monotonic()
            cancelled = request.future.cancel()
            logging.warning(
                "TTS timeout: phase=%s queue_seconds=%.3f generation_seconds=%.3f "
                "chars=%d queue_size=%d cancelled_before_generation=%s",
                "queued" if request.started_at is None else "generating",
                (request.started_at or now) - request.queued_at,
                now - request.started_at if request.started_at is not None else 0,
                len(text),
                self.requests.qsize(),
                cancelled,
            )
            raise
        except BaseException:
            request.future.cancel()
            raise

    def close(self) -> None:
        self.closed = True
        self.thread.join(timeout=5)

    def _tts_loop(self) -> None:
        pending = None
        while not self.closed:
            try:
                first = (
                    pending if pending is not None else self.requests.get(timeout=0.1)
                )
                pending = None
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + 0.02
            while first.seed is None and len(batch) < self.batch_size:
                try:
                    candidate = self.requests.get(
                        timeout=max(0, deadline - time.monotonic())
                    )
                    if candidate.seed is not None:
                        pending = candidate
                        break
                    batch.append(candidate)
                except queue.Empty:
                    break
            active = [
                item for item in batch if item.future.set_running_or_notify_cancel()
            ]
            self.tts_active_requests = len(active)
            started = time.monotonic()
            for item in active:
                item.started_at = started
            if active:
                logging.info(
                    "TTS batch started: requests=%d chars=%s queue_seconds=%s queue_size=%d",
                    len(active),
                    [len(item.text) for item in active],
                    [round(started - item.queued_at, 3) for item in active],
                    self.requests.qsize(),
                )
            try:
                if self.fatal_error:
                    raise RuntimeError(self.fatal_error)
                if active:
                    audios = self._synthesize_batch(active)
                    if len(audios) != len(active):
                        raise RuntimeError(
                            "TTS batch output count does not match requests"
                        )
                    for item, audio in zip(active, audios):
                        if not item.future.cancelled():
                            item.future.set_result(audio)
            except Exception as exc:
                logging.exception("TTS batch failed: requests=%d", len(active))
                if (
                    "out of memory" in str(exc).lower()
                    or "cuda error" in str(exc).lower()
                ):
                    self.fatal_error = f"Speech GPU failed: {type(exc).__name__}"
                for item in active:
                    if not item.future.done():
                        item.future.set_exception(exc)
            finally:
                self.tts_active_requests = 0
                if active:
                    logging.info(
                        "TTS batch finished: requests=%d generation_seconds=%.3f",
                        len(active),
                        time.monotonic() - started,
                    )
                if active and hasattr(self, "metrics"):
                    self.metrics["tts_batches"] += 1
                    self.metrics["tts_requests"] += len(active)
                    self.metrics["tts_seconds"] += time.monotonic() - started
                    self.metrics["max_tts_batch"] = max(
                        self.metrics["max_tts_batch"], len(active)
                    )
                for _ in batch:
                    self.requests.task_done()
        if pending is not None:
            if not pending.future.done():
                pending.future.set_exception(RuntimeError("Speech backend stopped"))
            self.requests.task_done()
        while True:
            try:
                item = self.requests.get_nowait()
            except queue.Empty:
                break
            if not item.future.done():
                item.future.set_exception(RuntimeError("Speech backend stopped"))
            self.requests.task_done()

    def _synthesize_batch(self, requests: list[TTSRequest]) -> list[bytes]:
        for item in requests:
            voice_id = item.voice_id
            voice = self.voices[voice_id]
            if voice_id not in self.prompts:
                self.prompts[voice_id] = self.tts.create_voice_clone_prompt(
                    ref_audio=str(voice["wav_path"]),
                    ref_text=voice["voice_reference_text"],
                )
        if requests[0].seed is not None:
            if len(requests) != 1:
                raise ValueError("Seeded TTS requests must be synthesized individually")
            item = requests[0]
            # Prompt creation can consume RNG; seed immediately before generation.
            self.torch.manual_seed(item.seed)
            audios = self.tts.generate(
                text=item.text,
                language="English",
                voice_clone_prompt=self.prompts[item.voice_id],
            )
        else:
            audios = self.tts.generate(
                text=[item.text for item in requests],
                language="English",
                voice_clone_prompt=[self.prompts[item.voice_id] for item in requests],
            )
        results = []
        for audio in audios:
            samples = self.np.asarray(audio, dtype=self.np.float32).reshape(-1)
            if not len(samples):
                raise EmptyAudioError("OmniVoice returned empty audio")
            if not self.np.isfinite(samples).all():
                raise RuntimeError("OmniVoice returned non-finite audio")
            pcm = (self.np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
            result = io.BytesIO()
            with wave.open(result, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(pcm)
            results.append(result.getvalue())
        return results

    def transcribe(self, audio: bytes) -> str:
        from math import gcd
        from talktopia.speech_agent import read_pcm_wav

        pcm, rate = read_pcm_wav(audio)
        with self.asr_lock:
            samples = (
                self.np.frombuffer(pcm, dtype="<i2").astype(self.np.float32) / 32768
            )
            if rate != 16000:
                divisor = gcd(rate, 16000)
                samples = self.resample_poly(
                    samples, 16000 // divisor, rate // divisor
                ).astype(self.np.float32)
            timestamps = self.get_speech_timestamps(
                self.torch.from_numpy(samples),
                self.vad,
                sampling_rate=16000,
                min_speech_duration_ms=100,
                min_silence_duration_ms=100,
                speech_pad_ms=30,
            )
            if not timestamps:
                return ""
            samples = self.np.concatenate(
                [samples[item["start"] : item["end"]] for item in timestamps]
            )
            segments, _ = self.asr.transcribe(
                samples,
                language="en",
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
