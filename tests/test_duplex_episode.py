from __future__ import annotations

import asyncio
import io
import json
import wave
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

from talktopia import pipeline, utils
from talktopia.full_duplex import episode as episode_module, generation
from talktopia.full_duplex.actions import (
    HiddenSaid,
    DuplexActionDecision,
    DuplexObservation,
    StreamingObservation,
)
from talktopia.full_duplex.config import RuntimeConfig
from talktopia.full_duplex.episode import run_with_timeout
from talktopia.full_duplex.events import ActionCommitted, EpisodeEnded, read_events
from talktopia.full_duplex.generation import AgentSessionContext, DuplexGenerationEngine
from talktopia.full_duplex.speech_backends import WindowedASR
from talktopia.speech_agent import AgentProfile


def wav_bytes(frames=38_400, sample=1000):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(sample.to_bytes(2, "little", signed=True) * frames)
    return output.getvalue()


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    from sotopia.database import EnvironmentProfile, storage_backend

    db = tmp_path / "db"
    monkeypatch.setenv("TALKTOPIA_DB_DIR", str(db))
    monkeypatch.setattr(
        storage_backend, "_storage_backend", storage_backend.LocalJSONBackend(str(db))
    )
    agents = []
    for index, name in enumerate(("Alice", "Bob")):
        pk = f"agent-{index}"
        reference = db / "voices" / pk / "reference.wav"
        reference.parent.mkdir(parents=True)
        reference.write_bytes(wav_bytes(960))
        profile = AgentProfile(
            pk=pk,
            first_name=name,
            last_name="Test",
            age=30,
            occupation="Shopkeeper",
            secret=f"Secret known only to {name}.",
            voice_id=f"voice-{pk}",
            voice_reference_wav=str(reference.relative_to(db)),
            voice_reference_text="A reference sentence.",
        )
        profile.save()
        agents.append(profile)
    env = EnvironmentProfile(
        pk="env-test",
        codename="duplex-test",
        scenario="Agree on a meeting time.",
        agent_goals=["Arrange a meeting before noon.", "Arrange a meeting after nine."],
        relationship=0,
        agent_constraint=None,
    )
    env.save()
    return {
        "env_id": env.pk,
        "agent_ids": [agent.pk for agent in agents],
        "episode_id": "episode_0001",
    }


class FakeSpeech:
    def __init__(self):
        self.seeds = []
        self.asr_audio_lengths = []

    def handle(self, request):
        if request.url.path.endswith("/audio/speech"):
            body = json.loads(request.content)
            assert body["voice"].startswith("voice-agent-")
            self.seeds.append(body["seed"])
            return httpx.Response(
                200, content=wav_bytes(), headers={"Content-Type": "audio/wav"}
            )
        assert request.url.path.endswith("/audio/transcriptions")
        assert b'name="prompt"' not in request.content
        assert b"Generated" not in request.content
        self.asr_audio_lengths.append(len(request.content))
        return httpx.Response(
            200, json={"text": "Received words describing the proposed meeting time."}
        )

    def client(self):
        return AsyncOpenAI(
            base_url="http://speech.test/v1",
            api_key="EMPTY",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle)),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intervention", ["none", "backchanneling", "interruption", "correction"]
)
async def test_http_episode_limits_and_audible_history(
    profiles, tmp_path, monkeypatch, intervention
):
    count = 0
    intervened = False
    histories = []
    original_cancel_asr = WindowedASR.cancel_utterance

    async def slow_cancel_asr(self, utterance_id):
        await asyncio.sleep(0.35)
        return await original_cancel_asr(self, utterance_id)

    monkeypatch.setattr(WindowedASR, "cancel_utterance", slow_cancel_asr)

    async def fake_generation(**kwargs):
        nonlocal count, intervened
        values = kwargs["input_values"]
        if "available_actions" in values:
            obs = json.loads(values["observation"])
            histories.append(values["recent_history"])
            if obs["source"] == "asr_partial":
                if intervention != "none" and not intervened:
                    intervened = True
                    return json.dumps({"action_type": intervention})
                return '{"action_type":"none"}'
            count += 1
            # Exercise both natural termination and a hard cutoff during closing.
            return json.dumps({"action_type": "speak" if count <= 2 else "leave"})
        return '{"text":"Generated speech that is private until heard."}'

    monkeypatch.setattr(generation, "agenerate", fake_generation)
    monkeypatch.setattr(
        episode_module,
        "RuntimeConfig",
        lambda **kw: RuntimeConfig(
            **{
                **kw,
                "allow_corrections": intervention == "correction",
                "allow_interruptions": intervention == "interruption",
                "correction_min_stable_words": 2,
                "interruption_min_stable_words": 2,
            }
        ),
    )
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    args.tag = "test-duplex"
    speech = FakeSpeech()
    async with speech.client() as client:
        resolved, agents = pipeline.build_episode(profiles, args, client, client)
        result = await pipeline.run_one_episode(
            resolved, agents, args, tmp_path, "episode_0001"
        )
    events = read_events(tmp_path / result["events"])
    commits = [event for event in events if isinstance(event, ActionCommitted)]
    assert result["turns"] == len(commits)
    assert result["budget_turns"] <= 12
    assert [event.turn_number for event in commits] == list(range(1, len(commits) + 1))
    assert events[-1].status == "completed"
    assert len([event for event in events if isinstance(event, EpisodeEnded)]) == 1
    assert result["end_reason"] == "explicit_leave_handshake"
    saved = json.loads((tmp_path / result["original"]).read_text())
    assert len(saved["models"]) == 3
    from sotopia.database import EpisodeLog
    from talktopia.evaluation.evaluator import duplex_history

    duplex_history(EpisodeLog.model_validate(saved), tmp_path / result["events"])
    assert [message[1] for message in saved["messages"][0]] == [
        "Alice Test",
        "Bob Test",
    ]
    assert "Generated speech" not in json.dumps(saved["messages"])
    assert "Received words" in json.dumps(saved["messages"])
    assert all(
        not agent.state.session_active and not agent.has_pending_generation
        for agent in agents
    )
    assert all(isinstance(seed, int) for seed in speech.seeds)
    hashes = utils.result_artifacts(result, tmp_path)
    assert result["events"] in hashes
    with wave.open(str(tmp_path / result["conversation_audio"])) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
            2,
            2,
            24000,
        )
        samples = memoryview(wav.readframes(wav.getnframes())).cast("h")
        overlap = any(
            left and right for left, right in zip(samples[::2], samples[1::2])
        )
    if intervention == "backchanneling":
        assert overlap
        assert any(
            action.action_type == intervention
            for event in commits
            for action in event.actions.values()
        )
    if intervention in {"correction", "interruption"}:
        assert any(
            event.event_type == "floor" and event.change == "transferred"
            for event in events
        )
        assert any(
            event.event_type == "speech_lifecycle" and event.phase == "cancelled"
            for event in events
        )
        cancelled = next(
            event
            for event in events
            if event.event_type == "speech_lifecycle" and event.phase == "cancelled"
        )
        final = next(
            event
            for event in events
            if event.event_type == "asr_update"
            and event.utterance_id == cancelled.utterance_id
            and event.is_final
        )
        assert any(
            cancelled.timestamp_ms < span["start_ms"] < final.timestamp_ms
            for event in events
            if event.event_type == "audio_delivered"
            and event.utterance_id != cancelled.utterance_id
            for span in event.frame_spans
        )


@pytest.mark.asyncio
async def test_exactly_twelve_commits_with_continuing_agents(
    profiles, tmp_path, monkeypatch
):
    async def generate(**kwargs):
        values = kwargs["input_values"]
        if "available_actions" in values:
            obs = json.loads(values["observation"])
            return json.dumps(
                {"action_type": "none" if obs["source"] == "asr_partial" else "speak"}
            )
        return '{"text":"We can meet tomorrow morning."}'

    monkeypatch.setattr(generation, "agenerate", generate)
    monkeypatch.setattr(
        episode_module,
        "RuntimeConfig",
        lambda **kw: RuntimeConfig(**kw),
    )
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    args.tag = "test-duplex"
    speech = FakeSpeech()
    async with speech.client() as client:
        resolved, agents = pipeline.build_episode(profiles, args, client, client)
        result = await pipeline.run_one_episode(
            resolved, agents, args, tmp_path, "episode_0001"
        )
    assert result["turns"] == 12
    assert result["end_reason"] == "max_turns"
    assert (
        len(
            [
                event
                for event in read_events(tmp_path / result["events"])
                if isinstance(event, ActionCommitted)
            ]
        )
        == 12
    )


@pytest.mark.asyncio
async def test_word_limit_retries_whole_utterance_and_closing(monkeypatch):
    context = AgentSessionContext(
        episode_id="episode",
        agent_name="Alice",
        peer_name="Bob",
        scenario="Meet tomorrow.",
        self_background="A shopkeeper.",
        private_goal="Agree on a time.",
    )
    observation = StreamingObservation(
        canonical=DuplexObservation(
            last_turn="Hello",
            turn_number=0,
            available_actions=["speak"],
            observation_id="obs",
        ),
        source="reset",
    )
    decision = DuplexActionDecision(decision_id="decision", action_type="speak")
    forty = " ".join(["word"] * 50)
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return json.dumps({"text": forty + (" excess" if len(calls) % 2 else "")})

    monkeypatch.setattr(generation, "agenerate", generate)
    engine = DuplexGenerationEngine("fake", max_attempts=2)
    hidden = await engine.generate_hidden_said(context, observation, decision, "")
    closing = await engine.generate_closing(context, "", "closing-decision")
    assert hidden.text == closing.text == forty
    assert len(calls) == 4
    assert all(call["input_values"]["max_words"] == "40" for call in calls)
    with pytest.raises(ValidationError, match="at most 50"):
        HiddenSaid(
            hidden_said_id="h", decision_id="d", speaker="Alice", text=forty + " excess"
        )


@pytest.mark.asyncio
async def test_timeout_records_state_before_cancellation(tmp_path):
    class WaitingRuntime:
        state = SimpleNamespace(episode_id="waiting")
        cancelled = False

        async def run(self, snapshot):
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True

        def liveness_snapshot(self):
            assert not self.cancelled
            return SimpleNamespace(model_dump=lambda **_: {"work_stage": "decision"})

    runtime = WaitingRuntime()
    path = tmp_path / "timeout.json"
    with pytest.raises(TimeoutError):
        await run_with_timeout(runtime, None, 0.01, path)
    assert runtime.cancelled
    assert json.loads(path.read_text())["runtime"]["work_stage"] == "decision"


def test_defaults_and_invalid_timeout():
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    args.tag = "test-duplex"
    assert args.max_turns == 12
    assert args.episode_timeout_s == 120
    for value in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            pipeline.parse_args(["--episode-timeout-s", value])


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "generation", "cancel"])
async def test_failed_episode_has_one_terminal_event_and_stops_tasks(
    profiles, tmp_path, monkeypatch, failure
):
    waiting = asyncio.Event()

    async def generate(**kwargs):
        waiting.set()
        if failure == "generation":
            raise RuntimeError("generation unavailable")
        await asyncio.Event().wait()

    monkeypatch.setattr(generation, "agenerate", generate)
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    args.tag = "test-duplex"
    if failure == "timeout":
        monkeypatch.setattr(pipeline, "EPISODE_TIMEOUT_S", 0.03)
    speech = FakeSpeech()
    async with speech.client() as client:
        resolved, agents = pipeline.build_episode(profiles, args, client, client)
        task = asyncio.create_task(
            pipeline.run_one_episode(resolved, agents, args, tmp_path, "episode_0001")
        )
        await waiting.wait()
        if failure == "cancel":
            task.cancel()
        error = (
            asyncio.CancelledError
            if failure == "cancel"
            else (TimeoutError if failure == "timeout" else RuntimeError)
        )
        with pytest.raises(error):
            await task
    events = read_events(tmp_path / "simulation/events/episode_0001.jsonl")
    terminal = [event for event in events if isinstance(event, EpisodeEnded)]
    assert len(terminal) == 1
    assert terminal[0].status == ("cancelled" if failure == "cancel" else "failed")
    assert all(
        not agent.state.session_active and not agent.has_pending_generation
        for agent in agents
    )
    if failure == "timeout":
        assert terminal[0].reason == "episode_timeout"
        assert (tmp_path / "simulation/diagnostics/episode_0001.json").is_file()


def test_existing_journal_is_not_overwritten(tmp_path):
    from talktopia.full_duplex.events import EventWriter

    path = tmp_path / "events.jsonl"
    path.write_text("existing result\n")
    with pytest.raises(FileExistsError):
        EventWriter(path, "episode")
    assert path.read_text() == "existing result\n"
