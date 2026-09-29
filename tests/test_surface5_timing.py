"""Exercise measured delivery with deliberately slow model and speech backends."""

import asyncio
import json
import time
import wave
from itertools import pairwise
from types import SimpleNamespace

import pytest
from sotopia.database import EpisodeLog, SotopiaDimensions
from test_duplex_episode import FakeSpeech
from test_duplex_episode import profiles as profiles  # noqa: PLC0414 -- pytest fixture

from talktopia import pipeline, utils
from talktopia.evaluation import evaluator
from talktopia.full_duplex import generation
from talktopia.full_duplex.actions import DuplexActionDecision
from talktopia.full_duplex.agent import CascadedDuplexAgent
from talktopia.full_duplex.audio import (
    AudioChunk,
    AudioFrame,
    AudioRouter,
    StereoWavWriter,
)
from talktopia.full_duplex.config import (
    RuntimeConfig,
    runtime_options,
    runtime_settings,
)
from talktopia.full_duplex.episode import summarize_latencies
from talktopia.full_duplex.events import (
    ASRUpdateEvent,
    AudioDeliveryEvent,
    EventWriter,
    ResponseLatencyEvent,
    read_events,
)
from talktopia.full_duplex.runtime import (
    DuplexRuntime,
    FloorController,
    RuntimeState,
    _ChunkDelivery,
    _UtteranceRuntime,
)
from talktopia.full_duplex.speech_backends import WindowedASR
from talktopia.full_duplex.speech_client import SpeechClient


async def run_timed_episode(
    profiles, tmp_path, monkeypatch, *, late_backchannel=False
):
    decisions = 0
    sent_backchannel = False

    async def generate(**kwargs):
        nonlocal decisions, sent_backchannel
        await asyncio.sleep(0.06)
        values = kwargs["input_values"]
        if "available_actions" in values:
            observation = json.loads(values["observation"])
            available = json.loads(values["available_actions"])
            assert (
                "correction" not in available
                and "interruption" not in available
            )
            if observation["source"] == "asr_partial":
                action = "backchanneling" if not sent_backchannel else "none"
                sent_backchannel = True
            else:
                decisions += 1
                action = "speak" if decisions <= 2 else "leave"
            return json.dumps({"action_type": action})
        return json.dumps(
            {"text": "Generated first sentence. Generated second sentence."}
        )

    async def synthesize(self, text, reference, seed):
        backchannel = text in generation._BACKCHANNELS
        await asyncio.sleep(2.8 if backchannel and late_backchannel else 0.09)
        count = 4800 if backchannel else 28800
        sample = 2000 if backchannel else 1000
        return sample.to_bytes(2, "little", signed=True) * count

    async def decode(self, pcm, rate):
        backchannel = pcm[:2] == (2000).to_bytes(2, "little", signed=True)
        await asyncio.sleep(0.7 if backchannel else 0.12)
        return (
            "Heard an acknowledgment."
            if backchannel
            else "Received words about the proposed meeting time."
        )

    monkeypatch.setattr(generation, "agenerate", generate)
    monkeypatch.setattr(SpeechClient, "synthesize", synthesize)
    monkeypatch.setattr(SpeechClient, "decode", decode)
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    args.tag = "timing-test"
    speech = FakeSpeech()
    async with speech.client() as client:
        resolved, agents = pipeline.build_episode(
            profiles, args, client, client
        )
        result = await pipeline.run_one_episode(
            resolved, agents, args, tmp_path, "episode_0001"
        )
    return result, read_events(tmp_path / result["events"])


@pytest.mark.asyncio
async def test_live_pcm_survives_model_waits_and_slow_final_asr(
    profiles, tmp_path, monkeypatch
):
    started = time.monotonic()
    result, events = await run_timed_episode(profiles, tmp_path, monkeypatch)
    wall_ms = (time.monotonic() - started) * 1000
    deliveries = [
        event for event in events if isinstance(event, AudioDeliveryEvent)
    ]
    assert (
        deliveries[0].start_ms >= 190
    )  # Decision, text generation and TTS really waited.
    assert 0 <= wall_ms - result["duration_ms"] < 500
    assert all(a.timestamp_ms <= b.timestamp_ms for a, b in pairwise(events))
    with wave.open(str(tmp_path / result["conversation_audio"]), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
        assert abs(wav.getnframes() / 24 - result["duration_ms"]) < 60
        first = deliveries[0].frame_spans[0]["start_sample"]
        assert not any(
            pcm[: first * 4]
        )  # Initial model latency is present as silence.
        samples = memoryview(pcm).cast("h")
        assert any(
            left and right for left, right in zip(samples[::2], samples[1::2])
        )
    latencies = [
        event for event in events if isinstance(event, ResponseLatencyEvent)
    ]
    assert {event.kind for event in latencies} == {
        "normal_response",
        "backchannel",
    }
    for event in latencies:
        assert event.latency_ms == event.first_audio_ms - event.origin_ms
        if event.kind == "normal_response":
            final_delivery = max(
                e.end_ms
                for e in deliveries
                if e.utterance_id == event.peer_utterance_id
            )
            assert event.origin_ms == final_delivery
            assert event.latency_ms >= 190
        else:
            decision = next(
                e
                for e in events
                if e.event_type == "decision"
                and e.decision_id == event.decision_id
                and e.status == "selected"
            )
            assert event.origin_ms == decision.request_started_ms
            assert decision.observation.source == "asr_partial"
    backchannel = next(
        event for event in latencies if event.kind == "backchannel"
    )
    bc_end = max(
        e.end_ms
        for e in deliveries
        if e.utterance_id == backchannel.utterance_id
    )
    bc_asr = next(
        e
        for e in events
        if isinstance(e, ASRUpdateEvent)
        and e.is_final
        and e.utterance_id == backchannel.utterance_id
    )
    assert bc_asr.timestamp_ms - bc_end > 400
    assert any(
        bc_end < span["start_ms"] < bc_asr.timestamp_ms
        for delivery in deliveries
        if delivery.utterance_id == backchannel.peer_utterance_id
        for span in delivery.frame_spans
    )
    report = json.loads((tmp_path / result["latency_report"]).read_text())
    assert report["statistics"] == result["latency"]
    for kind in ("normal_response", "backchannel"):
        samples = [
            event.latency_ms for event in latencies if event.kind == kind
        ]
        assert report["statistics"][kind] == {
            "count": len(samples),
            "mean_ms": sum(samples) / len(samples),
        }
    assert result["latency_report"] in utils.result_artifacts(result, tmp_path)

    source_path = tmp_path / result["original"]
    source_bytes = source_path.read_bytes()
    source = EpisodeLog.model_validate_json(source_bytes)
    _, turns = evaluator.duplex_history(source, tmp_path / result["events"])
    history = "\n".join(turns)
    assert "Sentence ASR" in history and "[00:" in history
    assert (
        "Received words" in history
        and "Generated first sentence" not in history
    )
    calls = []

    class Judge:
        def __init__(self, **kwargs):
            pass

        async def __acall__(self, **kwargs):
            calls.append(kwargs)
            return [
                (
                    agent,
                    (
                        (dimension, 0),
                        f"{agent} recorded evidence for {dimension}",
                    ),
                )
                for agent in ("agent_1", "agent_2")
                for dimension in SotopiaDimensions.model_fields
            ]

    monkeypatch.setattr(evaluator, "EpisodeLLMEvaluator", Judge)
    args = pipeline.parse_args(
        [
            "--stage",
            "reevaluate",
            "--interaction-mode",
            "surface5-full-duplex",
            "--episode-json",
            str(source_path),
        ]
    )
    args.reeval_tag = "evaluated"
    assert (
        await evaluator.evaluate_episode(args, tmp_path / "evaluation-run")
        == 0
    )
    assert source_path.read_bytes() == source_bytes
    assert (
        "Timing evidence for this Surface5 interaction" in calls[0]["history"]
    )
    assert calls[0]["temperature"] == 0.0 and calls[0]["num_agents"] == 2
    # Delivered samples are evidence, not an unverified timestamp decoration.
    audio = tmp_path / result["conversation_audio"]
    original_audio = audio.read_bytes()
    args.source_conversation_audio_sha256 = "changed-since-manifest"
    assert (
        await evaluator.evaluate_episode(args, tmp_path / "changed-audio") == 1
    )
    assert len(calls) == 1
    damaged = bytearray(original_audio)
    damaged[44] = 1
    audio.write_bytes(damaged)
    with pytest.raises(ValueError, match="outside measured deliveries"):
        evaluator.duplex_history(source, tmp_path / result["events"])
    damaged = bytearray(original_audio)
    damaged[44 + deliveries[0].frame_spans[0]["start_sample"] * 4] ^= 1
    audio.write_bytes(damaged)
    with pytest.raises(ValueError, match="recorded PCM"):
        evaluator.duplex_history(source, tmp_path / result["events"])


@pytest.mark.asyncio
async def test_late_backchannel_is_not_played_or_counted(
    profiles, tmp_path, monkeypatch
):
    result, events = await run_timed_episode(
        profiles, tmp_path, monkeypatch, late_backchannel=True
    )
    assert result["latency"]["backchannel"] == {"count": 0, "mean_ms": None}
    assert any(
        event.event_type == "speech_lifecycle"
        and event.reason == "peer_audio_already_finished"
        for event in events
    )
    late_ids = {
        event.utterance_id
        for event in events
        if event.event_type == "speech_lifecycle"
        and event.reason == "peer_audio_already_finished"
    }
    assert not any(
        event.utterance_id in late_ids
        for event in events
        if isinstance(event, AudioDeliveryEvent)
    )


def test_action_controls_and_weighted_latency_summary():
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    config = RuntimeConfig(**runtime_options(args))
    assert (
        config.allow_backchannels
        and not config.allow_corrections
        and not config.allow_interruptions
    )
    state = RuntimeState(
        episode_id="test",
        now_ms=0,
        opener_agent="Alice",
        active_utterances={"speech": "Alice"},
    )
    available = FloorController(config=config).available_actions("Bob", state)
    assert "backchanneling" in available and not {
        "correction",
        "interruption",
    }.intersection(available)
    off = pipeline.parse_args(
        [
            "--interaction-mode",
            "surface5-full-duplex",
            "--no-duplex-backchannels",
        ]
    )
    assert not runtime_settings(**runtime_options(off))["allow_backchannels"]
    assert "backchanneling" not in FloorController(
        config=RuntimeConfig(**runtime_options(off))
    ).available_actions("Bob", state)
    with pytest.raises(SystemExit):
        pipeline.parse_args(["--duplex-interruptions"])
    summary = {
        "episodes": [
            {
                "status": "completed",
                "latency": {
                    "normal_response": {"count": 1, "mean_ms": 100},
                    "backchannel": {"count": 0, "mean_ms": None},
                },
            },
            {
                "status": "completed",
                "latency": {
                    "normal_response": {"count": 3, "mean_ms": 20},
                    "backchannel": {"count": 0, "mean_ms": None},
                },
            },
        ]
    }
    summarize_latencies(summary)
    assert summary["latency"]["normal_response"] == {"count": 4, "mean_ms": 40}
    assert summary["latency"]["backchannel"]["mean_ms"] is None


def test_prompt_is_the_versioned_appendix_text():
    text = generation._PROMPT_PATH.read_text()
    assert generation.SIMULATION_PROMPT_VERSION == "simulation_v2.1"
    for section in (
        generation._ROLE_PROMPT,
        generation._DECISION_PROMPT,
        generation._NON_AUDIO_ARGUMENT_PROMPT,
        generation._HIDDEN_SAID_PROMPT,
        generation._CLOSING_PROMPT,
    ):
        assert section in text


def test_temporal_evaluation_requires_recorded_sidecars():
    class Source:
        agent_classes = ["CascadedDuplexAgent"] * 2

    with pytest.raises(ValueError, match="requires its event journal"):
        evaluator.duplex_history(Source(), None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action", ["correction", "interruption", "backchanneling"]
)
async def test_runtime_rejects_disabled_actions_before_execution(action):
    runtime = object.__new__(DuplexRuntime)
    runtime.config = RuntimeConfig(allow_backchannels=False)
    decision = DuplexActionDecision(
        decision_id="disabled",
        action_type=action,
        target_utterance_id="peer-speech"
        if action != "backchanneling"
        else None,
    )
    with pytest.raises(ValueError, match="Action disabled"):
        await runtime.handle_agent_output("Alice", decision)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "timeout", "final_asr"])
async def test_active_audio_failure_keeps_capture_and_stops_tasks(
    profiles, tmp_path, monkeypatch, failure
):
    receiving = asyncio.Event()
    received = []
    original_receive = CascadedDuplexAgent.receive_audio

    async def receive(self, frame):
        await original_receive(self, frame)
        received.append(frame)
        receiving.set()

    async def generate(**kwargs):
        values = kwargs["input_values"]
        if "available_actions" in values:
            return json.dumps({"action_type": "speak"})
        return json.dumps({"text": "Generated sentence."})

    async def synthesize(self, text, reference, seed):
        # Keep the audio short, with a 15 ms final frame.
        return (1000).to_bytes(2, "little", signed=True) * 1320

    async def decode(self, pcm, rate):
        if failure == "final_asr":
            raise RuntimeError("final ASR unavailable")
        await asyncio.Event().wait()

    monkeypatch.setattr(CascadedDuplexAgent, "receive_audio", receive)
    monkeypatch.setattr(generation, "agenerate", generate)
    monkeypatch.setattr(SpeechClient, "synthesize", synthesize)
    monkeypatch.setattr(SpeechClient, "decode", decode)
    if failure == "timeout":
        monkeypatch.setattr(pipeline, "EPISODE_TIMEOUT_S", 0.15)
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    args.tag = "failed-live-audio"
    speech = FakeSpeech()
    async with speech.client() as client:
        resolved, agents = pipeline.build_episode(
            profiles, args, client, client
        )
        task = asyncio.create_task(
            pipeline.run_one_episode(
                resolved, agents, args, tmp_path, "episode_0001"
            )
        )
        await receiving.wait()
        if failure == "cancel":
            task.cancel()
        error = {
            "cancel": asyncio.CancelledError,
            "timeout": TimeoutError,
            "final_asr": RuntimeError,
        }[failure]
        with pytest.raises(error):
            await task
    events = read_events(tmp_path / "simulation/events/episode_0001.jsonl")
    deliveries = [
        event for event in events if isinstance(event, AudioDeliveryEvent)
    ]
    assert sum(event.delivered_frames for event in deliveries) == len(received)
    assert all(
        span["end_ms"] > span["start_ms"]
        for event in deliveries
        for span in event.frame_spans
    )
    with wave.open(
        str(tmp_path / "simulation/audio/episode_0001/conversation.wav"), "rb"
    ) as wav:
        for delivery in deliveries:
            for span in delivery.frame_spans:
                wav.setpos(span["start_sample"])
                assert all(
                    memoryview(wav.readframes(span["samples"])).cast("h")[::2]
                )
    assert events[-1].status == (
        "cancelled" if failure == "cancel" else "failed"
    )
    assert all(not agent.state.session_active for agent in agents)
    assert not any(
        task.get_name().startswith("surface5-")
        for task in asyncio.all_tasks()
        if not task.done()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["receiver", "drain_cancel"])
async def test_failure_preserves_exactly_the_accepted_audio(tmp_path, failure):
    names = ("Alice", "Bob")
    accepted = []
    receiving = asyncio.Event()

    class Receiver:
        async def receive_audio(self, frame):
            if failure == "receiver" and frame.source_agent == "Bob":
                raise RuntimeError("second channel rejected before buffering")
            accepted.append(frame)
            receiving.set()

        def note_audio_played(self, utterance_id):
            pass

    runtime = object.__new__(DuplexRuntime)
    runtime.state = RuntimeState(
        episode_id="receiver-failure", now_ms=0, opener_agent="Alice"
    )
    runtime.event_writer = EventWriter(
        tmp_path / "events.jsonl", "receiver-failure"
    )
    runtime.stereo_writer = StereoWavWriter(
        tmp_path / "conversation.wav", names
    )
    runtime.audio_router = AudioRouter(names)
    runtime.sample_rate_hz = 24000
    runtime._agents = {name: Receiver() for name in names}
    runtime._utterances = {}
    runtime._deliveries = {}
    runtime._decision_observations = {}
    runtime._closed_deliveries = set()
    runtime._audio_end_ms = {}
    runtime.latencies = {"normal_response": [], "backchannel": []}
    for name in names:
        utterance_id = f"speech-{name}"
        decision = DuplexActionDecision(decision_id=name, action_type="speak")
        runtime._utterances[utterance_id] = _UtteranceRuntime(
            speaker=name,
            listener=names[1 - names.index(name)],
            decision=decision,
            action_type="speak",
            closing=False,
            hidden_said_id=name,
        )
        runtime._deliveries[(utterance_id, 0)] = _ChunkDelivery(
            planned_frames=2
        )
        runtime._decision_observations[name] = SimpleNamespace(
            peer_utterance_id=None
        )
        runtime.audio_router.enqueue(
            AudioChunk(
                utterance_id=utterance_id,
                source_agent=name,
                chunk_index=0,
                text=name,
                pcm_s16le=(1000).to_bytes(2, "little") * 1920,
                is_final=True,
            )
        )
    try:
        if failure == "receiver":
            with pytest.raises(RuntimeError, match="second channel rejected"):
                await runtime.tick_audio()
        else:
            audio_task = asyncio.create_task(runtime.tick_audio())
            runtime._audio_task = audio_task
            await receiving.wait()
            drain_task = asyncio.create_task(runtime._stop_audio())
            await asyncio.sleep(0)
            drain_task.cancel()
            # A second cancellation while draining must also preserve the frame.
            await asyncio.sleep(0)
            drain_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await drain_task
            assert audio_task.done() and not audio_task.cancelled()
            assert runtime._audio_task is None
        for name in names:
            runtime._emit_open_delivery_prefixes(f"speech-{name}")
    finally:
        runtime.event_writer.close()
        runtime.stereo_writer.close()
    events = read_events(tmp_path / "events.jsonl")
    deliveries = [
        event for event in events if isinstance(event, AudioDeliveryEvent)
    ]
    assert (
        len(accepted) == len(deliveries) == (1 if failure == "receiver" else 2)
    )
    assert deliveries[0].source_agent == "Alice"
    assert all(
        span["end_ms"] > span["start_ms"]
        for delivery in deliveries
        for span in delivery.frame_spans
    )
    if failure == "receiver":
        assert not any(
            event.event_type == "speech_lifecycle" and event.speaker == "Bob"
            for event in events
        )
    assert runtime.latencies == {"normal_response": [], "backchannel": []}
    with wave.open(str(tmp_path / "conversation.wav"), "rb") as wav:
        pcm = memoryview(wav.readframes(wav.getnframes())).cast("h")
        assert sum(sample != 0 for sample in pcm[::2]) == 960
        assert sum(sample != 0 for sample in pcm[1::2]) == (
            0 if failure == "receiver" else 960
        )


@pytest.mark.asyncio
async def test_partial_asr_failure_rejects_frame_before_buffer_changes():
    async def decode(pcm, rate):
        raise RuntimeError("partial ASR backend failed")

    asr = WindowedASR(
        SimpleNamespace(decode=decode), decode_interval_ms=40, window_ms=80
    )
    frame = AudioFrame(
        utterance_id="speech",
        source_agent="Alice",
        chunk_index=0,
        frame_index=0,
        pcm_s16le=(1000).to_bytes(2, "little") * 960,
        duration_ms=40,
        is_chunk_end=False,
        is_utterance_end=False,
    )
    try:
        await asr.start_utterance("speech", "Bob")
        await asr.push_audio(frame)
        await asyncio.sleep(0)
        received = bytes(asr.buffers["speech"])
        with pytest.raises(RuntimeError, match="partial ASR backend failed"):
            await asr.push_audio(frame)
        assert bytes(asr.buffers["speech"]) == received
        assert bytes(asr._sessions["speech"].chunk_pcm[0]) == received
    finally:
        await asr.close()
