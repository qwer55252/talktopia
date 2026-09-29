"""Windowed ASR and sentence TTS over the Talktopia speech API."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .actions import SpeechChunk
from .audio import AudioChunk, AudioFrame
from .speech_client import SpeechClient, SpeechClientError


class ASRUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    utterance_id: str = Field(min_length=1)
    listener: str = Field(min_length=1)
    text: str
    is_final: bool
    is_stable: bool
    revision_id: int = Field(ge=0)
    sentence_texts: dict[int, str] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeechSynthesisFailure:
    chunk: SpeechChunk
    source_agent: str
    attempts: int
    reason: Literal["empty_audio"]
    error_detail: str

    def __post_init__(self) -> None:
        if not self.source_agent:
            raise ValueError("source_agent must not be blank")
        if self.attempts < 1:
            raise ValueError("attempts must be positive")
        if not self.error_detail.strip():
            raise ValueError("error_detail must not be blank")


@runtime_checkable
class OnlineASR(Protocol):
    async def start_utterance(self, utterance_id: str, listener: str) -> None: ...

    async def push_audio(self, frame: AudioFrame) -> None: ...

    def updates(self) -> AsyncIterator[ASRUpdate]: ...

    async def finish_utterance(self, utterance_id: str) -> ASRUpdate: ...

    async def cancel_utterance(self, utterance_id: str) -> ASRUpdate: ...

    async def close(self) -> None: ...


@runtime_checkable
class IncrementalTTS(Protocol):
    async def synthesize(
        self,
        chunk: SpeechChunk,
        voice_reference: Path,
    ) -> AudioChunk | SpeechSynthesisFailure: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class _ASRSession:
    listener: str
    pcm: bytearray = field(default_factory=bytearray)
    revision_id: int = 0
    previous_hypothesis: tuple[str, ...] = ()
    stable_words: tuple[str, ...] = ()
    last_decode_size: int = 0
    decode_task: asyncio.Task[None] | None = None
    chunk_pcm: dict[int, bytearray] = field(default_factory=dict)
    chunk_tasks: dict[int, asyncio.Task[str]] = field(default_factory=dict)


class WindowedASR:
    """Expose only common-prefix stable partials from delivered PCM."""

    def __init__(
        self,
        worker: SpeechClient,
        *,
        decode_interval_ms: int,
        window_ms: int,
        sample_rate_hz: int = 24_000,
    ) -> None:
        if decode_interval_ms <= 0 or window_ms < decode_interval_ms:
            raise ValueError("ASR interval/window configuration is invalid")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.worker = worker
        self.decode_interval_ms = decode_interval_ms
        self.window_ms = window_ms
        self.sample_rate_hz = sample_rate_hz
        self.buffers: dict[str, bytearray] = {}
        self._sessions: dict[str, _ASRSession] = {}
        self._updates: asyncio.Queue[ASRUpdate | None] = asyncio.Queue()
        self._closed = False
        self._bytes_per_interval = sample_rate_hz * decode_interval_ms // 1000 * 2
        self._bytes_per_window = sample_rate_hz * window_ms // 1000 * 2

    async def start_utterance(self, utterance_id: str, listener: str) -> None:
        self._require_open()
        if not utterance_id or not listener:
            raise ValueError("ASR utterance identity must not be blank")
        if utterance_id in self._sessions:
            raise ValueError(f"ASR utterance already active: {utterance_id}")
        session = _ASRSession(listener=listener)
        self._sessions[utterance_id] = session
        self.buffers[utterance_id] = session.pcm

    async def push_audio(self, frame: AudioFrame) -> None:
        self._require_open()
        session = self._sessions.get(frame.utterance_id)
        if session is None:
            raise ValueError(f"ASR utterance is not active: {frame.utterance_id}")
        # A failed decoder rejects the next frame before any PCM is accepted.
        if session.decode_task is not None and session.decode_task.done():
            session.decode_task.result()
            session.decode_task = None
        session.pcm.extend(frame.pcm_s16le)
        session.chunk_pcm.setdefault(frame.chunk_index, bytearray()).extend(
            frame.pcm_s16le
        )
        if frame.is_chunk_end:
            session.chunk_tasks[frame.chunk_index] = asyncio.create_task(
                self.worker.decode(
                    bytes(session.chunk_pcm[frame.chunk_index]),
                    self.sample_rate_hz,
                ),
                name=f"surface5-asr-sentence-{frame.utterance_id}-{frame.chunk_index}",
            )
        enough_new_audio = (
            len(session.pcm) - session.last_decode_size >= self._bytes_per_interval
        )
        if enough_new_audio and (
            session.decode_task is None or session.decode_task.done()
        ):
            session.last_decode_size = len(session.pcm)
            snapshot = bytes(session.pcm[-self._bytes_per_window :])
            session.decode_task = asyncio.create_task(
                self._decode_partial(frame.utterance_id, session, snapshot),
                name=f"surface5-asr-partial-{frame.utterance_id}",
            )

    async def updates(self) -> AsyncIterator[ASRUpdate]:
        while True:
            update = await self._updates.get()
            if update is None:
                return
            yield update

    async def finish_utterance(self, utterance_id: str) -> ASRUpdate:
        return await self._finalize(utterance_id)

    async def cancel_utterance(self, utterance_id: str) -> ASRUpdate:
        return await self._finalize(utterance_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = [
            session.decode_task
            for session in self._sessions.values()
            if session.decode_task is not None and not session.decode_task.done()
        ]
        tasks.extend(
            task
            for session in self._sessions.values()
            for task in session.chunk_tasks.values()
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._sessions.clear()
        self.buffers.clear()
        await self._updates.put(None)

    async def _decode_partial(
        self,
        utterance_id: str,
        session: _ASRSession,
        snapshot: bytes,
    ) -> None:
        try:
            text = await self.worker.decode(snapshot, self.sample_rate_hz)
            if self._sessions.get(utterance_id) is not session:
                return
            hypothesis = tuple(text.split())
            common = _common_prefix(session.previous_hypothesis, hypothesis)
            session.previous_hypothesis = hypothesis
            if len(common) <= len(session.stable_words):
                return
            if common[: len(session.stable_words)] != session.stable_words:
                return
            session.stable_words = common
            session.revision_id += 1
            await self._updates.put(
                ASRUpdate(
                    utterance_id=utterance_id,
                    listener=session.listener,
                    text=" ".join(common),
                    is_final=False,
                    is_stable=True,
                    revision_id=session.revision_id,
                )
            )
        except asyncio.CancelledError:
            raise

    async def _finalize(self, utterance_id: str) -> ASRUpdate:
        self._require_open()
        session = self._sessions.get(utterance_id)
        if session is None:
            raise ValueError(f"ASR utterance is not active: {utterance_id}")
        if session.decode_task is not None:
            await session.decode_task
        for index, chunk_pcm in session.chunk_pcm.items():
            if index not in session.chunk_tasks:
                session.chunk_tasks[index] = asyncio.create_task(
                    self.worker.decode(bytes(chunk_pcm), self.sample_rate_hz)
                )
        sentence_texts = {
            index: await task for index, task in session.chunk_tasks.items()
        }
        pcm = bytes(session.pcm)
        if len(sentence_texts) == 1:
            text = next(iter(sentence_texts.values()))
        elif not pcm:
            text = ""
        else:
            text = await self.worker.decode(pcm, self.sample_rate_hz)
        session.revision_id += 1
        update = ASRUpdate(
            utterance_id=utterance_id,
            listener=session.listener,
            text=text,
            is_final=True,
            is_stable=True,
            revision_id=session.revision_id,
            sentence_texts=sentence_texts,
        )
        self._sessions.pop(utterance_id, None)
        self.buffers.pop(utterance_id, None)
        return update

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("ASR backend is closed")


class SentenceTTS:
    """Turn one validated sentence chunk into one validated PCM segment."""

    def __init__(
        self,
        worker: SpeechClient,
        *,
        source_agent: str,
        sample_rate_hz: int = 24_000,
    ) -> None:
        if not source_agent:
            raise ValueError("source_agent must not be blank")
        if sample_rate_hz != 24_000:
            raise ValueError("Surface5 OmniVoice output must be 24000 Hz")
        self.worker = worker
        self.source_agent = source_agent
        self.sample_rate_hz = sample_rate_hz
        self._closed = False

    async def synthesize(
        self,
        chunk: SpeechChunk,
        voice_reference: Path,
    ) -> AudioChunk | SpeechSynthesisFailure:
        if self._closed:
            raise RuntimeError("TTS backend is closed")
        last_empty_error: SpeechClientError | None = None
        for attempt in (1, 2):
            try:
                pcm = await self.worker.synthesize(
                    chunk.text,
                    voice_reference,
                    _deterministic_tts_seed(
                        voice_reference=voice_reference,
                        text=chunk.text,
                        attempt=attempt,
                    ),
                )
            except SpeechClientError as error:
                if error.code != "empty_audio":
                    raise
                last_empty_error = error
                continue
            if not pcm or len(pcm) % 2:
                raise RuntimeError("TTS produced invalid PCM16")
            return AudioChunk(
                utterance_id=chunk.utterance_id,
                source_agent=self.source_agent,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                pcm_s16le=pcm,
                sample_rate_hz=self.sample_rate_hz,
                is_final=chunk.is_final,
                synthesis_attempt=attempt,
            )
        assert last_empty_error is not None
        return SpeechSynthesisFailure(
            chunk=chunk,
            source_agent=self.source_agent,
            attempts=2,
            reason="empty_audio",
            error_detail=str(last_empty_error),
        )

    async def close(self) -> None:
        self._closed = True


def _common_prefix(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    length = 0
    for left_word, right_word in zip(left, right):
        if left_word.casefold() != right_word.casefold():
            break
        length += 1
    return right[:length]


def _deterministic_tts_seed(
    *,
    voice_reference: Path,
    text: str,
    attempt: int,
) -> int:
    if attempt < 1:
        raise ValueError("TTS attempt must be positive")
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("TTS text must not be blank")
    voice_profile_id = voice_reference.expanduser().resolve().parent.name
    if not voice_profile_id:
        raise ValueError("voice profile ID must not be blank")
    identity = "\0".join(
        (
            "surface5-omnivoice-seed-v1",
            voice_profile_id,
            normalized,
            str(attempt),
        )
    )
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:4], "big")


__all__ = [
    "ASRUpdate",
    "IncrementalTTS",
    "OnlineASR",
    "SpeechSynthesisFailure",
    "SentenceTTS",
    "WindowedASR",
]
