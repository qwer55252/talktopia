"""Surface5 evaluation evidence from measured delivery and sentence ASR."""

from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path

from talktopia.full_duplex.events import (
    ASRUpdateEvent,
    AudioDeliveryEvent,
    EpisodeEnded,
    EpisodeStarted,
)
from talktopia.full_duplex.rendering import _timestamp

TEMPORAL_INSTRUCTION = (
    (Path(__file__).with_name("prompts") / "temporal_v1.txt")
    .read_text()
    .strip()
)


def _require_silence(wav, channel: int, start: int, end: int) -> None:
    wav.setpos(start)
    while start < end:
        size = min(24000, end - start)
        samples = memoryview(wav.readframes(size)).cast("h")[channel::2]
        if any(samples):
            raise ValueError(
                "Recorded audio contains PCM outside measured deliveries"
            )
        start += size


def validate_timing(events, events_path: Path, source) -> None:
    starts = [event for event in events if isinstance(event, EpisodeStarted)]
    if len(starts) != 1:
        raise ValueError("Expected one Surface5 episode start")
    start = starts[0]
    if (
        start.run_config.get("clock") != "monotonic_elapsed_ms"
        or start.run_config.get("audio_capture") != "live_delivered_pcm_v2"
    ):
        raise ValueError(
            "Surface5 temporal evaluation requires measured live timing; frame-clock logs are not eligible"
        )
    if (
        start.environment_id != source.environment
        or list(start.agent_ids) != source.agents
    ):
        raise ValueError("Timing evidence belongs to another episode")
    ends = [event for event in events if isinstance(event, EpisodeEnded)]
    if (
        len(ends) != 1
        or ends[0].status != "completed"
        or events[-1] != ends[0]
    ):
        raise ValueError("Timing evidence must end with one completed episode")
    previous = -1.0
    for event in events:
        if (
            event.episode_id != start.episode_id
            or not math.isfinite(event.timestamp_ms)
            or event.timestamp_ms < previous
        ):
            raise ValueError(
                "Timing events must share one episode and a monotonic clock"
            )
        previous = event.timestamp_ms
    audio = (
        events_path.parent.parent
        / "audio"
        / events_path.stem
        / "conversation.wav"
    )
    delivered_chunks = set()
    channel_ends = {name: 0 for name in start.agent_names}
    with wave.open(str(audio), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (
            2,
            2,
            24000,
        ):
            raise ValueError(
                "Expected the recorded Surface5 stereo PCM stream"
            )
        for event in events:
            if not isinstance(event, AudioDeliveryEvent):
                continue
            key = (event.utterance_id, event.chunk_index)
            if key in delivered_chunks or event.end_ms > event.timestamp_ms:
                raise ValueError("Duplicate or future audio delivery")
            delivered_chunks.add(key)
            spans = event.frame_spans
            if not spans or len(spans) != event.delivered_frames:
                raise ValueError("Missing measured frame delivery evidence")
            if (
                event.start_ms != spans[0]["start_ms"]
                or event.end_ms != spans[-1]["end_ms"]
            ):
                raise ValueError(
                    "Sentence timing differs from delivered frames"
                )
            channel = start.agent_names.index(event.source_agent)
            pcm = bytearray()
            end_sample = channel_ends[event.source_agent]
            for span in spans:
                index, size = span["start_sample"], span["samples"]
                if (
                    not isinstance(index, int)
                    or not isinstance(size, int)
                    or size <= 0
                    or index < end_sample
                    or index + size > wav.getnframes()
                    or not all(
                        isinstance(span[key], (int, float))
                        and math.isfinite(span[key])
                        for key in ("start_ms", "end_ms")
                    )
                    or span["end_ms"] < span["start_ms"]
                    or abs(index * 1000 / 24000 - span["start_ms"])
                    > 1000 / 24000
                ):
                    raise ValueError("Invalid measured frame position")
                _require_silence(wav, channel, end_sample, index)
                wav.setpos(index)
                frame = (
                    memoryview(wav.readframes(size))
                    .cast("h")[channel::2]
                    .tobytes()
                )
                if hashlib.sha256(frame).hexdigest() != span["pcm_sha256"]:
                    raise ValueError(
                        "Timing evidence differs from recorded PCM"
                    )
                pcm.extend(frame)
                end_sample = index + size
            channel_ends[event.source_agent] = end_sample
            if hashlib.sha256(pcm).hexdigest() != event.delivered_pcm_sha256:
                raise ValueError(
                    "Sentence PCM hash differs from delivered audio"
                )
        for channel, name in enumerate(start.agent_names):
            _require_silence(
                wav, channel, channel_ends[name], wav.getnframes()
            )


def timed_messages(source, commits, entries, events):
    """Add sentence spans to ASR evidence, leaving the source EpisodeLog untouched."""
    finals = {
        event.utterance_id: event
        for event in events
        if isinstance(event, ASRUpdateEvent) and event.is_final
    }
    by_commit = {
        (entry.commit_id, entry.speaker): entry
        for entry in entries
        if entry.commit_id
    }
    messages = [source.messages[0]]
    for commit in commits:
        turn = []
        for speaker, action in commit.actions.items():
            if action.action_type == "none":
                continue
            entry = by_commit.get((commit.commit_id, speaker))
            if entry is None:
                rendered = f"[{_timestamp(commit.timestamp_ms)}] {action.to_natural_language()}"
            else:
                final = finals[entry.utterance_id]
                if not entry.sentences or set(final.sentence_texts) != {
                    span.chunk_index for span in entry.sentences
                }:
                    raise ValueError(
                        "Missing ASR evidence for a delivered sentence"
                    )
                rendered = (
                    f"[{_timestamp(entry.start_ms)} - {_timestamp(entry.end_ms)}] "
                    + action.model_copy(
                        update={"argument": entry.received_text}
                    ).to_natural_language()
                )
                rendered += "\nSentence ASR (the same utterance, not additional actions):"
                for span in entry.sentences:
                    rendered += (
                        f"\n  [{_timestamp(span.start_ms)} - {_timestamp(span.end_ms)}] "
                        f"{speaker}: {span.received_text or '[no recognized words]'}"
                    )
            turn.append((speaker, "Environment", rendered))
        if turn:
            messages.append(turn)
    return messages
