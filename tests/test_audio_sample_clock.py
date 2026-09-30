"""Packet scheduling jitter must not become periodic holes in recorded speech."""

import hashlib
import wave
from copy import deepcopy
from types import SimpleNamespace

import pytest

from talktopia.evaluation.temporal import validate_timing
from talktopia.full_duplex.audio import AudioFrame, StereoWavWriter
from talktopia.full_duplex.events import (
    AudioDeliveryEvent,
    EpisodeEnded,
    EpisodeStarted,
    EventWriter,
    read_events,
)


def frame(index, *, source="Alice", samples=960):
    return AudioFrame(
        utterance_id=f"speech-{source}",
        source_agent=source,
        chunk_index=0,
        frame_index=index,
        pcm_s16le=(1200 if source == "Alice" else -1800).to_bytes(
            2, "little", signed=True
        )
        * samples,
        duration_ms=round(samples / 24),
        is_chunk_end=False,
        is_utterance_end=False,
    )


def recorded_stream(tmp_path, timestamps, *, idle_before=()):
    events_path = tmp_path / "events" / "episode.jsonl"
    audio_path = tmp_path / "audio" / "episode" / "conversation.wav"
    writer = StereoWavWriter(audio_path, ("Alice", "Bob"))
    journal = EventWriter(events_path, "episode")
    journal.emit(
        EpisodeStarted,
        0,
        environment_id="env",
        agent_ids=("a", "b"),
        agent_names=("Alice", "Bob"),
        models=("test", "test"),
        initial_observations={},
        run_config={
            "clock": "monotonic_elapsed_ms",
            "audio_capture": "live_pcm_sample_clock_v3",
            "audio_frame_ms": 40,
        },
    )
    spans = {name: [] for name in ("Alice", "Bob")}
    pcm = {name: bytearray() for name in spans}
    for i, timestamp in enumerate(timestamps):
        if i in idle_before:
            writer.silence_until(timestamp)
        # A short final BC frame shares the bus with a full-length speech frame.
        frames = {"Alice": frame(i), "Bob": frame(i, source="Bob", samples=360)}
        capture = writer.write_tick(frames, timestamp_ms=timestamp)
        for name, item in frames.items():
            size = len(item.pcm_s16le) // 2
            spans[name].append(
                {
                    "start_ms": timestamp,
                    "end_ms": max(timestamp, (capture.start_sample + size) / 24) + 0.8,
                    "start_sample": capture.start_sample,
                    "samples": size,
                    "clock_anchor_ms": capture.clock_anchor_ms,
                    "clock_anchor_sample": capture.clock_anchor_sample,
                    "underrun_ms": capture.underrun_ms,
                    "pcm_sha256": hashlib.sha256(item.pcm_s16le).hexdigest(),
                }
            )
            pcm[name].extend(item.pcm_s16le)
    ended = max(s[-1]["end_ms"] for s in spans.values()) + 10
    for name in spans:
        journal.emit(
            AudioDeliveryEvent,
            ended,
            utterance_id=f"speech-{name}",
            source_agent=name,
            chunk_index=0,
            delivered_frames=len(timestamps),
            planned_frames=len(timestamps),
            start_ms=spans[name][0]["start_ms"],
            end_ms=spans[name][-1]["end_ms"],
            frame_spans=tuple(spans[name]),
            delivered_pcm_sha256=hashlib.sha256(pcm[name]).hexdigest(),
        )
    writer.silence_until(ended)
    writer.close()
    journal.emit(
        EpisodeEnded, ended, status="completed", reason="test", duration_ms=ended
    )
    journal.close()
    source = SimpleNamespace(environment="env", agents=["a", "b"])
    events = read_events(events_path)
    validate_timing(events, events_path, source)
    return events, events_path, audio_path, source


def test_thousands_of_jittered_frames_keep_pcm_continuous(tmp_path):
    times = [1000 + i * 40 + (0 if i == 0 else 0.3 + (i % 9) / 10) for i in range(2000)]
    events, _, path, _ = recorded_stream(tmp_path, times)
    delivery = next(e for e in events if isinstance(e, AudioDeliveryEvent))
    spans = delivery.frame_spans
    assert all(
        b["start_sample"] == a["start_sample"] + a["samples"]
        for a, b in zip(spans, spans[1:])
    )
    assert all(s["clock_anchor_ms"] == 1000 and s["underrun_ms"] == 0 for s in spans)
    assert max(abs(s["start_ms"] - s["start_sample"] / 24) for s in spans) < 1.2
    with wave.open(str(path), "rb") as wav:
        wav.setpos(spans[0]["start_sample"])
        samples = memoryview(wav.readframes(2000 * 960)).cast("h")
        assert all(value == 1200 for value in samples[::2])
        assert list(samples[1:720:2]) == [-1800] * 360
        assert not any(samples[721:1920:2])  # Short final frame is not stretched.


@pytest.mark.parametrize("idle", [False, True])
def test_real_starvation_and_large_stalls_remain_silence(tmp_path, idle):
    events, _, path, _ = recorded_stream(
        tmp_path, [1000, 1040.7, 1180.5, 1221.1], idle_before=(2,) if idle else ()
    )
    delivery = next(e for e in events if isinstance(e, AudioDeliveryEvent))
    a, b = delivery.frame_spans[1:3]
    assert b["start_sample"] / 24 == 1180.5
    assert b["clock_anchor_ms"] == 1180.5
    assert b["underrun_ms"] == (0 if idle else 100.5)
    with wave.open(str(path), "rb") as wav:
        end = a["start_sample"] + a["samples"]
        wav.setpos(end)
        assert not any(wav.readframes(b["start_sample"] - end))


@pytest.mark.parametrize("damage", ["drift", "anchor", "underrun", "stereo_underrun"])
def test_clock_validation_rejects_tampered_measurements(tmp_path, damage):
    events, path, _, source = recorded_stream(tmp_path, [1000, 1040.7, 1180.5, 1221.1])
    index = next(i for i, e in enumerate(events) if isinstance(e, AudioDeliveryEvent))
    if damage == "stereo_underrun":
        index += 1
    spans = deepcopy(events[index].frame_spans)
    if damage == "drift":
        spans[1]["start_ms"] += 41
    elif damage == "anchor":
        spans[1]["clock_anchor_ms"] += 0.1
    else:
        spans[2]["underrun_ms"] = 9999
    events[index] = events[index].model_copy(update={"frame_spans": spans})
    with pytest.raises(ValueError):
        validate_timing(events, path, source)


def test_short_last_frame_advances_by_its_actual_samples(tmp_path):
    writer = StereoWavWriter(tmp_path / "short.wav", ("Alice", "Bob"))
    first = writer.write_tick({"Alice": frame(0, samples=360)}, timestamp_ms=200)
    second = writer.write_tick({"Alice": frame(1)}, timestamp_ms=215.8)
    assert second.start_sample == first.start_sample + 360
    assert second.clock_anchor_ms == first.clock_anchor_ms
    writer.close()
