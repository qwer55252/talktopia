"""PCM routing and deterministic stereo artifact writing."""

from __future__ import annotations

import wave
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioChunk:
    utterance_id: str
    source_agent: str
    chunk_index: int
    text: str
    pcm_s16le: bytes
    sample_rate_hz: int = 24_000
    is_final: bool
    synthesis_attempt: int = 1

    def __post_init__(self) -> None:
        if not self.utterance_id or not self.source_agent:
            raise ValueError("audio chunk identity must not be blank")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.synthesis_attempt < 1:
            raise ValueError("synthesis_attempt must be positive")
        if not self.pcm_s16le or len(self.pcm_s16le) % 2:
            raise ValueError("pcm_s16le must contain complete PCM16 samples")


@dataclass(frozen=True, slots=True)
class AudioFrame:
    utterance_id: str
    source_agent: str
    chunk_index: int
    frame_index: int
    pcm_s16le: bytes
    duration_ms: int
    is_chunk_end: bool
    is_utterance_end: bool

    def __post_init__(self) -> None:
        if not self.utterance_id or not self.source_agent:
            raise ValueError("audio frame identity must not be blank")
        if self.chunk_index < 0 or self.frame_index < 0:
            raise ValueError("audio indexes must be non-negative")
        if not self.pcm_s16le or len(self.pcm_s16le) % 2:
            raise ValueError("pcm_s16le must contain complete PCM16 samples")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")


class AudioRouter:
    """Keep one independent FIFO per source and expose one frame per tick."""

    def __init__(
        self,
        sources: tuple[str, str],
        *,
        sample_rate_hz: int = 24_000,
        frame_ms: int = 40,
    ) -> None:
        if len(set(sources)) != 2 or any(not source for source in sources):
            raise ValueError("AudioRouter requires two distinct sources")
        if sample_rate_hz <= 0 or frame_ms <= 0:
            raise ValueError("audio timing must be positive")
        samples_per_frame = sample_rate_hz * frame_ms
        if samples_per_frame % 1000:
            raise ValueError("sample_rate_hz * frame_ms must form whole samples")
        self._sources = sources
        self._sample_rate_hz = sample_rate_hz
        self._frame_ms = frame_ms
        self._bytes_per_frame = samples_per_frame // 1000 * 2
        self._queues: dict[str, deque[AudioFrame]] = {
            source: deque() for source in sources
        }

    def enqueue(self, chunk: AudioChunk) -> None:
        if chunk.source_agent not in self._queues:
            raise ValueError(f"unknown audio source: {chunk.source_agent}")
        if chunk.sample_rate_hz != self._sample_rate_hz:
            raise ValueError("audio chunk sample rate does not match router")

        pcm = chunk.pcm_s16le
        frame_count = (len(pcm) + self._bytes_per_frame - 1) // self._bytes_per_frame
        queue = self._queues[chunk.source_agent]
        for frame_index in range(frame_count):
            start = frame_index * self._bytes_per_frame
            frame_pcm = pcm[start : start + self._bytes_per_frame]
            sample_count = len(frame_pcm) // 2
            duration_ms = max(1, round(sample_count * 1000 / self._sample_rate_hz))
            is_chunk_end = frame_index == frame_count - 1
            queue.append(
                AudioFrame(
                    utterance_id=chunk.utterance_id,
                    source_agent=chunk.source_agent,
                    chunk_index=chunk.chunk_index,
                    frame_index=frame_index,
                    pcm_s16le=frame_pcm,
                    duration_ms=duration_ms,
                    is_chunk_end=is_chunk_end,
                    is_utterance_end=is_chunk_end and chunk.is_final,
                )
            )

    def pop_tick(self) -> dict[str, AudioFrame | None]:
        return {
            source: queue.popleft() if queue else None
            for source, queue in self._queues.items()
        }

    def cancel(self, utterance_id: str) -> int:
        removed = 0
        for source, queue in self._queues.items():
            kept = deque(frame for frame in queue if frame.utterance_id != utterance_id)
            removed += len(queue) - len(kept)
            self._queues[source] = kept
        return removed

    def has_pending(self) -> bool:
        return any(self._queues.values())


class StereoWavWriter:
    """Write fixed left/right source channels, padding idle channels with silence."""

    def __init__(
        self,
        path: Path,
        sources: tuple[str, str],
        *,
        sample_rate_hz: int = 24_000,
        frame_ms: int = 40,
    ) -> None:
        if len(set(sources)) != 2:
            raise ValueError("StereoWavWriter requires two distinct sources")
        self._sources = sources
        self._bytes_per_tick = sample_rate_hz * frame_ms // 1000 * 2
        output_path = path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._wave = wave.open(str(output_path), "wb")
        self._wave.setnchannels(2)
        self._wave.setsampwidth(2)
        self._wave.setframerate(sample_rate_hz)
        self._closed = False

    def write_tick(self, frames: Mapping[str, AudioFrame | None]) -> None:
        if self._closed:
            raise RuntimeError("stereo WAV writer is closed")
        unknown = set(frames) - set(self._sources)
        if unknown:
            raise ValueError(f"unknown stereo channel sources: {sorted(unknown)}")

        channels: list[bytes] = []
        for source in self._sources:
            frame = frames.get(source)
            pcm = frame.pcm_s16le if frame is not None else b""
            if len(pcm) > self._bytes_per_tick:
                raise ValueError("audio frame exceeds one configured tick")
            channels.append(pcm.ljust(self._bytes_per_tick, b"\x00"))

        interleaved = bytearray(self._bytes_per_tick * 2)
        for offset in range(0, self._bytes_per_tick, 2):
            target = offset * 2
            interleaved[target : target + 2] = channels[0][offset : offset + 2]
            interleaved[target + 2 : target + 4] = channels[1][offset : offset + 2]
        self._wave.writeframesraw(bytes(interleaved))

    def close(self) -> None:
        if not self._closed:
            self._wave.close()
            self._closed = True


__all__ = ["AudioChunk", "AudioFrame", "AudioRouter", "StereoWavWriter"]
