"""Exercise leave and time-limit termination during active speech work."""

import asyncio
import json
import wave

import pytest
from test_duplex_episode import FakeSpeech, joint_result
from test_duplex_episode import profiles as profiles  # noqa: PLC0414 -- fixture

from talktopia import pipeline
from talktopia.full_duplex import generation
from talktopia.full_duplex.actions import DuplexActionDecision
from talktopia.full_duplex.agent import CascadedDuplexAgent
from talktopia.full_duplex.audio import AudioChunk
from talktopia.full_duplex.events import (
    ActionCommitted,
    ASRUpdateEvent,
    AudioDeliveryEvent,
    read_events,
)
from talktopia.full_duplex.runtime import DuplexRuntime
from talktopia.full_duplex.speech_client import SpeechClient


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["audio", "tts", "asr", "overlap", "commit"])
async def test_time_limit_finishes_selected_speech_and_final_asr(
    profiles, tmp_path, monkeypatch, stage
):
    from sotopia.database import EpisodeLog
    from talktopia.evaluation.evaluator import duplex_history
    from talktopia.full_duplex.sotopia_adapter import SotopiaSession
    from talktopia.full_duplex.transcript import TranscriptBuilder

    requests = []
    with_backchannel = stage in {"overlap", "commit"}
    text = "First sentence. Final sentence."
    if with_backchannel:
        text = "First sentence. Second sentence. Final sentence."

    async def generate(**kwargs):
        observation = json.loads(kwargs["input_values"]["observation"])
        requests.append(observation)
        if observation["source"] == "asr_partial":
            return joint_result(kwargs, "backchanneling")
        return joint_result(kwargs, "speak", text)

    async def synthesize(self, sentence, reference, seed):
        if stage == "tts":
            await asyncio.sleep(0.2)
        duration = {"tts": 0.1, "asr": 0.02}.get(stage, 0.2)
        sample = 1000
        if sentence == "yeah":
            duration, sample = (1.2 if stage == "commit" else 0.6), 2000
        return sample.to_bytes(2, "little") * round(24000 * duration)

    async def decode(self, pcm, rate):
        if stage == "asr":
            await asyncio.sleep(0.2)
        return "Yeah." if pcm[:2] == (2000).to_bytes(2, "little") else "Heard spoken words."

    monkeypatch.setattr(generation, "generate_structured_action", generate)
    monkeypatch.setattr(generation, "BACKCHANNEL_TTS_INPUTS", ("yeah",))
    monkeypatch.setattr(SpeechClient, "synthesize", synthesize)
    monkeypatch.setattr(SpeechClient, "decode", decode)
    if stage == "commit":
        original_commit = SotopiaSession.commit

        async def slow_commit(self, commit):
            # Final ASR arrives before the limit, but its commit finishes after it.
            await asyncio.sleep(0.3)
            return await original_commit(self, commit)

        monkeypatch.setattr(SotopiaSession, "commit", slow_commit)
    time_limit = {"overlap": 0.35, "commit": 0.75}.get(stage, 0.12)
    monkeypatch.setattr(pipeline, "EPISODE_TIMEOUT_S", time_limit)
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    args.tag = "finish-last-speech"
    if with_backchannel:
        # Reaching the action budget while draining must not cut the backchannel.
        args.max_turns = 1
    async with FakeSpeech().client() as client:
        resolved, agents = pipeline.build_episode(profiles, args, client, client)
        result = await asyncio.wait_for(
            pipeline.run_one_episode(resolved, agents, args, tmp_path, "episode_0001"),
            3,
        )
    assert result["status"] == "completed" and result["end_reason"] == "time_limit"
    assert result["duration_ms"] > time_limit * 1000 and result["budget_turns"] == 1
    assert result["action_counts"] == (
        {"speak": 1, "backchanneling": 1} if with_backchannel else {"speak": 1}
    )
    assert [r["source"] for r in requests] == (
        ["reset", "asr_partial"] if with_backchannel else ["reset"]
    )
    events = read_events(tmp_path / result["events"])
    assert events[0].run_config["time_limit_policy"] == "finish_selected_speech_v1"
    deliveries = [e for e in events if isinstance(e, AudioDeliveryEvent)]
    assert len(deliveries) == (4 if with_backchannel else 2)
    assert all(e.delivered_frames == e.planned_frames for e in deliveries)
    assert not any(
        e.event_type == "speech_lifecycle" and e.phase == "cancelled" for e in events
    )
    entries = TranscriptBuilder.from_events(events).build()
    assert all(entry.completed and entry.commit_id for entry in entries)
    speech = next(entry for entry in entries if entry.action_type == "speak")
    assert speech.generated_text == text
    assert len(speech.sentences) == (3 if with_backchannel else 2)
    if stage == "commit":
        final = next(
            e for e in events if isinstance(e, ASRUpdateEvent)
            and e.is_final and e.utterance_id == speech.utterance_id
        )
        committed = next(
            e for e in events if isinstance(e, ActionCommitted)
            and e.commit_id == speech.commit_id
        )
        assert final.timestamp_ms < time_limit * 1000 < committed.timestamp_ms
    source = EpisodeLog.model_validate_json((tmp_path / result["original"]).read_text())
    _, history = duplex_history(source, tmp_path / result["events"])
    assert "Heard spoken words." in "\n".join(history)
    readable = (tmp_path / result["readable"]).read_text()
    assert "\n\n".join(history[:-2]) in readable
    assert "[00:" in readable and "Sentence ASR" in readable
    assert events[-1].status == "completed" and events[-1].reason == "time_limit"
    assert all(not agent.state.session_active for agent in agents)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["audio", "llm", "tts", "asr_error", "drain_cancel"])
async def test_first_leave_stops_active_work_without_a_closing_reply(
    profiles, tmp_path, monkeypatch, stage
):
    receiving = asyncio.Event()
    model_waiting = asyncio.Event()
    model_cancelled = asyncio.Event()
    draining = asyncio.Event()
    received = []
    requests = []
    injected = False
    syntheses = 0
    original_receive = CascadedDuplexAgent.receive_audio
    original_output = DuplexRuntime.handle_agent_output
    original_stop_audio = DuplexRuntime._stop_audio

    async def wait_for_cancellation():
        model_waiting.set()
        try:
            await asyncio.Event().wait()
        finally:
            model_cancelled.set()

    async def generate(**kwargs):
        requests.append(kwargs)
        values = kwargs["input_values"]
        observation = json.loads(values["observation"])
        assert observation["source"] != "peer_left"
        if stage == "llm" and observation["source"] == "asr_partial":
            await wait_for_cancellation()
        return joint_result(kwargs, "speak", "First sentence. Second sentence.")

    async def synthesize(self, text, reference, seed):
        nonlocal syntheses
        syntheses += 1
        if stage == "tts" and syntheses == 2:
            await wait_for_cancellation()
        return (1000).to_bytes(2, "little") * 9600

    async def decode(self, pcm, rate):
        if stage == "asr_error":
            raise RuntimeError("ASR failed during leave cleanup")
        return "Received speech before leaving."

    async def receive(self, frame):
        await original_receive(self, frame)
        received.append(frame)
        receiving.set()

    async def handle_output(self, speaker, output):
        nonlocal injected
        await original_output(self, speaker, output)
        if injected or not isinstance(output, AudioChunk):
            return
        injected = True
        await receiving.wait()
        listener = self._peer(speaker)
        if stage == "llm":
            observation = self._streaming_observation(
                listener,
                source="asr_partial",
                canonical=self._snapshot.observations[listener],
                available_actions=["none", "backchanneling"],
                stable_text="Received speech",
                peer_utterance_id=output.utterance_id,
                peer_speaking=True,
            )
            await self._agents[listener].submit_observation(observation)
        if stage in {"llm", "tts"}:
            await model_waiting.wait()
        # Invoke the controller's leave boundary at a deterministic in-flight frame.
        await self.commit_selected_action(
            listener,
            DuplexActionDecision(
                decision_id="test-voluntary-leave",
                action_type="leave",
            ),
        )

    async def stop_audio(self, **kwargs):
        if self.state.leave_phase == "completed" and self._audio_task is not None:
            draining.set()
        await original_stop_audio(self, **kwargs)

    monkeypatch.setattr(generation, "generate_structured_action", generate)
    monkeypatch.setattr(SpeechClient, "synthesize", synthesize)
    monkeypatch.setattr(SpeechClient, "decode", decode)
    monkeypatch.setattr(CascadedDuplexAgent, "receive_audio", receive)
    monkeypatch.setattr(DuplexRuntime, "handle_agent_output", handle_output)
    monkeypatch.setattr(DuplexRuntime, "_stop_audio", stop_audio)
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    args.tag = "first-leave"
    async with FakeSpeech().client() as client:
        resolved, agents = pipeline.build_episode(profiles, args, client, client)
        task = asyncio.create_task(
            pipeline.run_one_episode(resolved, agents, args, tmp_path, "episode_0001")
        )
        if stage == "drain_cancel":
            await asyncio.wait_for(draining.wait(), 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif stage == "asr_error":
            with pytest.raises(RuntimeError, match="ASR failed"):
                await asyncio.wait_for(task, 3)
        else:
            result = await asyncio.wait_for(task, 3)
            assert result["end_reason"] == "agent_left" and result["budget_turns"] == 1
    events = read_events(tmp_path / "simulation/events/episode_0001.jsonl")
    commits = [event for event in events if isinstance(event, ActionCommitted)]
    assert len(commits) == 1
    assert [action.action_type for action in commits[0].actions.values()].count(
        "leave"
    ) == 1
    deliveries = [event for event in events if isinstance(event, AudioDeliveryEvent)]
    assert len(received) == sum(event.delivered_frames for event in deliveries) == 1
    assert all(
        span["end_ms"] > span["start_ms"]
        for event in deliveries
        for span in event.frame_spans
    )
    assert not any(
        event.event_type == "speech_chunk_synthesized"
        and event.sequence > commits[0].sequence
        for event in events
    )
    if stage not in {"asr_error", "drain_cancel"}:
        assert any(
            isinstance(event, ASRUpdateEvent)
            and event.is_final
            and event.text == "Received speech before leaving."
            for event in events
        )
    if stage in {"llm", "tts"}:
        assert model_cancelled.is_set()
    assert len(requests) == (2 if stage == "llm" else 1)
    assert events[-1].status == {
        "asr_error": "failed",
        "drain_cancel": "cancelled",
    }.get(stage, "completed")
    cancellations = [
        event.utterance_id
        for event in events
        if event.event_type == "speech_lifecycle" and event.phase == "cancelled"
    ]
    assert len(cancellations) == len(set(cancellations))
    assert all(
        not agent.state.session_active and not agent.has_pending_generation
        for agent in agents
    )
    assert not any(
        task.get_name().startswith("surface5-")
        for task in asyncio.all_tasks()
        if not task.done()
    )
    with wave.open(
        str(tmp_path / "simulation/audio/episode_0001/conversation.wav"), "rb"
    ) as wav:
        pcm = memoryview(wav.readframes(wav.getnframes())).cast("h")
        assert sum(sample != 0 for sample in pcm[::2]) == 960
        assert not any(pcm[1::2])
