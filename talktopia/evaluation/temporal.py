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


def _require_silence(wav, channel: int, start: int, end: int) -> None:
    wav.setpos(start)
    while start < end:
        size = min(24000, end - start)
        samples = memoryview(wav.readframes(size)).cast("h")[channel::2]
        if any(samples):
            raise ValueError("Recorded audio contains PCM outside measured deliveries")
        start += size


def validate_timing(events, events_path: Path, source) -> None:
    starts = [event for event in events if isinstance(event, EpisodeStarted)]
    if len(starts) != 1:
        raise ValueError("Expected one Surface5 episode start")
    start = starts[0]
    capture_policy = start.run_config.get("audio_capture")
    if start.run_config.get(
        "clock"
    ) != "monotonic_elapsed_ms" or capture_policy not in {
        "live_delivered_pcm_v2",
        "live_pcm_sample_clock_v3",
    }:
        raise ValueError(
            "Surface5 temporal evaluation requires measured live timing; frame-clock logs are not eligible"
        )
    if (
        start.environment_id != source.environment
        or list(start.agent_ids) != source.agents
    ):
        raise ValueError("Timing evidence belongs to another episode")
    ends = [event for event in events if isinstance(event, EpisodeEnded)]
    if len(ends) != 1 or ends[0].status != "completed" or events[-1] != ends[0]:
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
    audio = events_path.parent.parent / "audio" / events_path.stem / "conversation.wav"
    delivered_chunks = set()
    channel_ends = {name: 0 for name in start.agent_names}
    clocks: dict[int, tuple[float, dict[int, tuple[int, float, float]]]] = {}
    frame_ms = start.run_config.get("audio_frame_ms")
    if capture_policy == "live_pcm_sample_clock_v3" and (
        not isinstance(frame_ms, (int, float))
        or not math.isfinite(frame_ms)
        or not 0 < frame_ms <= 40
    ):
        raise ValueError("Missing or invalid audio frame interval")
    with wave.open(str(audio), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (
            2,
            2,
            24000,
        ):
            raise ValueError("Expected the recorded Surface5 stereo PCM stream")
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
                raise ValueError("Sentence timing differs from delivered frames")
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
                        isinstance(span[key], (int, float)) and math.isfinite(span[key])
                        for key in ("start_ms", "end_ms")
                    )
                    or span["end_ms"] < span["start_ms"]
                ):
                    raise ValueError("Invalid measured frame position")
                if capture_policy == "live_delivered_pcm_v2":
                    if abs(index / 24 - span["start_ms"]) > 1 / 24:
                        raise ValueError("Invalid measured frame position")
                else:
                    _validate_sample_clock(span, frame_ms, clocks)
                _require_silence(wav, channel, end_sample, index)
                wav.setpos(index)
                frame = memoryview(wav.readframes(size)).cast("h")[channel::2].tobytes()
                if hashlib.sha256(frame).hexdigest() != span["pcm_sha256"]:
                    raise ValueError("Timing evidence differs from recorded PCM")
                pcm.extend(frame)
                end_sample = index + size
            channel_ends[event.source_agent] = end_sample
            if hashlib.sha256(pcm).hexdigest() != event.delivered_pcm_sha256:
                raise ValueError("Sentence PCM hash differs from delivered audio")
        for channel, name in enumerate(start.agent_names):
            _require_silence(wav, channel, channel_ends[name], wav.getnframes())
    previous_end = None
    for anchor, (anchor_ms, ticks) in sorted(clocks.items()):
        cursor = anchor
        if previous_end is not None and anchor < previous_end:
            raise ValueError("Overlapping PCM clock anchors")
        for index, (samples, measured_start_ms, underrun_ms) in sorted(ticks.items()):
            if index != cursor or (index == anchor and measured_start_ms != anchor_ms):
                raise ValueError("Discontinuous PCM sample clock")
            if underrun_ms and (
                previous_end is None
                or abs(underrun_ms - (measured_start_ms - previous_end / 24)) > 1 / 24
            ):
                raise ValueError("Underrun differs from measured delivery gap")
            cursor += samples
            previous_end = cursor


def _validate_sample_clock(span, frame_ms, clocks) -> None:
    """Keep measured callback times separate from the anchored PCM timeline."""
    anchor_ms = span.get("clock_anchor_ms")
    anchor = span.get("clock_anchor_sample")
    underrun = span.get("underrun_ms")
    index, size = span["start_sample"], span["samples"]
    if (
        not isinstance(anchor, int)
        or not isinstance(anchor_ms, (int, float))
        or not math.isfinite(anchor_ms)
        or anchor_ms < 0
        or anchor != round(anchor_ms * 24)
        or index < anchor
        or not isinstance(underrun, (int, float))
        or not math.isfinite(underrun)
        or underrun < 0
        or (underrun and (underrun <= frame_ms or index != anchor))
        # Absolute cursor, not the preceding callback: drift cannot accumulate.
        or not -1 / 24 <= span["start_ms"] - index / 24 <= frame_ms
        or span["end_ms"] + 1 / 24 < (index + size) / 24
        or size > frame_ms * 24
    ):
        raise ValueError("Invalid anchored PCM clock or callback timing")
    known_ms, ticks = clocks.setdefault(anchor, (anchor_ms, {}))
    if known_ms != anchor_ms:
        raise ValueError("Inconsistent PCM clock anchor")
    previous_size, previous_start, previous_underrun = ticks.get(
        index, (0, span["start_ms"], underrun)
    )
    if previous_start != span["start_ms"] or previous_underrun != underrun:
        raise ValueError("Stereo channels have different callback evidence")
    ticks[index] = (max(size, previous_size), previous_start, underrun)


def timed_messages(source, commits, entries, events):
    """Add sentence spans to ASR evidence, leaving the source EpisodeLog untouched."""
    finals = {
        event.utterance_id: event
        for event in events
        if isinstance(event, ASRUpdateEvent) and event.is_final
    }
    by_commit = {
        (entry.commit_id, entry.speaker): entry for entry in entries if entry.commit_id
    }
    messages = [source.messages[0]]
    for commit in commits:
        turn = []
        for speaker, action in commit.actions.items():
            if action.action_type == "none" and commit.metadata.get("actor") != speaker:
                continue
            entry = by_commit.get((commit.commit_id, speaker))
            if entry is None:
                rendered = f"[{_timestamp(commit.timestamp_ms)}] {action.to_natural_language()}"
            else:
                final = finals[entry.utterance_id]
                if not entry.sentences or set(final.sentence_texts) != {
                    span.chunk_index for span in entry.sentences
                }:
                    raise ValueError("Missing ASR evidence for a delivered sentence")
                rendered = (
                    f"[{_timestamp(entry.start_ms)} - {_timestamp(entry.end_ms)}] "
                    + action.model_copy(
                        update={"argument": entry.received_text}
                    ).to_natural_language()
                )
                rendered += (
                    "\nSentence ASR (the same utterance, not additional actions):"
                )
                for span in entry.sentences:
                    rendered += (
                        f"\n  [{_timestamp(span.start_ms)} - {_timestamp(span.end_ms)}] "
                        f"{speaker}: {span.received_text or '[no recognized words]'}"
                    )
            turn.append((speaker, "Environment", rendered))
        if turn:
            messages.append(turn)
    return messages
