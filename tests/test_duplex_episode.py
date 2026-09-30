from __future__ import annotations

import asyncio
import io
import json
import wave
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI

from talktopia import pipeline, utils
from talktopia.full_duplex import episode as episode_module, generation
from talktopia.full_duplex.config import RuntimeConfig
from talktopia.full_duplex.episode import run_with_timeout
from talktopia.full_duplex.events import ActionCommitted, EpisodeEnded, read_events
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
        self.tts_inputs = []

    def handle(self, request):
        if request.url.path.endswith("/audio/speech"):
            body = json.loads(request.content)
            self.tts_inputs.append(body["input"])
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


def joint_result(kwargs, action, text="Generated speech that is private until heard."):
    argument = "" if action in {"none", "leave", "backchanneling"} else text
    return kwargs["output_parser"].parse(
        json.dumps({"action_type": action, "argument": argument, "to": []}),
        context=kwargs["context"],
    )


@pytest.mark.asyncio
async def test_accepted_speech_preserves_generated_synthesized_and_live_pcm(
    profiles, tmp_path, monkeypatch
):
    from sotopia.database import EpisodeLog
    from talktopia.evaluation.evaluator import duplex_history
    from talktopia.full_duplex.events import DecisionEvent
    from talktopia.full_duplex.transcript import TranscriptBuilder

    decisions = 0
    original = "I need the other grand."
    raw_response = (
        '<think>Private raw diagnostic.</think>'
        '{"argument":"I need the other grand."}'
    )

    async def generate(**kwargs):
        nonlocal decisions
        assert "Private raw diagnostic" not in kwargs["input_values"]["history"]
        if json.loads(kwargs["input_values"]["observation"])["source"] == "asr_partial":
            return joint_result(kwargs, "none")
        decisions += 1
        kwargs["responses"].append(raw_response)
        return joint_result(kwargs, "speak" if decisions == 1 else "leave", original)

    monkeypatch.setattr(generation, "generate_structured_action", generate)
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    args.tag = "accepted-speech"
    speech = FakeSpeech()
    async with speech.client() as client:
        resolved, agents = pipeline.build_episode(profiles, args, client, client)
        result = await asyncio.wait_for(
            pipeline.run_one_episode(resolved, agents, args, tmp_path, "episode_0001"),
            10,
        )
    assert result["status"] == "completed"
    assert speech.tts_inputs == [original]
    events = read_events(tmp_path / result["events"])
    assert all(
        event.observation.source != "asr_partial"
        for event in events
        if isinstance(event, DecisionEvent)
    )
    spoken = [
        e
        for e in TranscriptBuilder.from_events(events).build()
        if e.action_type == "speak"
    ]
    assert len(spoken) == 1
    assert spoken[0].generated_text == original
    assert spoken[0].synthesized_text == speech.tts_inputs[0]
    assert (
        spoken[0].received_text
        == "Received words describing the proposed meeting time."
    )
    source = EpisodeLog.model_validate_json((tmp_path / result["original"]).read_text())
    _, timed_history = duplex_history(source, tmp_path / result["events"])
    assert "Private raw diagnostic" not in "\n".join(timed_history)
    selected = next(
        e for e in events if isinstance(e, DecisionEvent) and e.raw_responses
    )
    assert selected.raw_responses == (raw_response,)
    legacy = selected.model_dump()
    legacy.pop("raw_responses")
    assert DecisionEvent.model_validate(legacy).raw_responses == ()
    assert "Private raw diagnostic" not in (tmp_path / result["readable"]).read_text()
    assert "nods" not in (tmp_path / result["readable"]).read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intervention,cancel_delay",
    [
        ("none", 0.35),
        ("backchanneling", 0.35),
    ],
)
async def test_http_episode_limits_and_audible_history(
    profiles, tmp_path, monkeypatch, intervention, cancel_delay
):
    count = 0
    intervened = False
    histories = []
    original_cancel_asr = WindowedASR.cancel_utterance

    async def slow_cancel_asr(self, utterance_id):
        await asyncio.sleep(cancel_delay)
        return await original_cancel_asr(self, utterance_id)

    monkeypatch.setattr(WindowedASR, "cancel_utterance", slow_cancel_asr)

    async def fake_generation(**kwargs):
        nonlocal count, intervened
        values = kwargs["input_values"]
        obs = json.loads(values["observation"])
        histories.append(values["history"])
        if obs["source"] == "asr_partial":
            assert "none" in kwargs["context"]["available_action_types"]
            assert obs["has_next_sentence"] is True
            assert obs["sentence_index"] == 0
            if intervention != "none" and not intervened:
                intervened = True
                return joint_result(kwargs, intervention)
            return joint_result(kwargs, "none")
        if obs["source"] == "asr_final":
            assert "none" not in kwargs["context"]["available_action_types"]
            assert obs["has_next_sentence"] is False
        count += 1
        return joint_result(
            kwargs,
            "speak" if count <= 2 else "leave",
            "Generated first sentence. Generated second sentence.",
        )

    monkeypatch.setattr(generation, "generate_structured_action", fake_generation)
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
    assert result["end_reason"] == "agent_left"
    assert (
        sum(
            action.action_type == "leave"
            for event in commits
            for action in event.actions.values()
        )
        == 1
    )
    assert any(action.action_type == "leave" for action in commits[-1].actions.values())
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
    assert all("Generated speech" not in history for history in histories)
    assert "Received words" in histories[-1]
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


@pytest.mark.asyncio
@pytest.mark.parametrize("last_action", ["speak", "leave"])
async def test_exactly_twelve_commits_with_continuing_agents(
    profiles, tmp_path, monkeypatch, last_action
):
    decisions = 0

    async def generate(**kwargs):
        nonlocal decisions
        obs = json.loads(kwargs["input_values"]["observation"])
        if obs["source"] == "asr_partial":
            return joint_result(kwargs, "none")
        decisions += 1
        return joint_result(
            kwargs,
            last_action if decisions == 12 else "speak",
            "We can meet tomorrow morning.",
        )

    from talktopia.full_duplex.speech_client import SpeechClient

    async def short_speech(self, text, reference, seed):
        return (1000).to_bytes(2, "little") * 4800

    monkeypatch.setattr(SpeechClient, "synthesize", short_speech)
    monkeypatch.setattr(generation, "generate_structured_action", generate)
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
    assert result["budget_turns"] == 12
    assert result["end_reason"] == (
        "agent_left" if last_action == "leave" else "max_turns"
    )
    assert decisions == 12
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
@pytest.mark.parametrize("failure", ["timeout", "cancel"])
async def test_failed_episode_has_one_terminal_event_and_stops_tasks(
    profiles, tmp_path, monkeypatch, failure
):
    waiting = asyncio.Event()

    async def generate(**kwargs):
        waiting.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(generation, "generate_structured_action", generate)
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
        error = asyncio.CancelledError if failure == "cancel" else TimeoutError
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
