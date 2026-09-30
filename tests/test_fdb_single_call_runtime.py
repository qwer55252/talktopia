"""Single-call Surface5 behavior through the pipeline and pre-TTS runtime gate."""

import asyncio
import json
import wave

import pytest
from sotopia.database import EpisodeLog
from test_duplex_episode import FakeSpeech
from test_duplex_episode import profiles as profiles  # noqa: PLC0414 -- fixture

from talktopia import pipeline
from talktopia.evaluation import evaluator
from talktopia.full_duplex import generation
from talktopia.full_duplex.actions import StreamingObservation
from talktopia.full_duplex.audio import AudioRouter, StereoWavWriter
from talktopia.full_duplex.config import RuntimeConfig
from talktopia.full_duplex.events import (
    ActionCommitted,
    AudioDeliveryEvent,
    DecisionEvent,
    EventWriter,
    read_events,
)
from talktopia.full_duplex.generation import AgentSessionContext
from talktopia.full_duplex.rendering import render_episode_for_humans
from talktopia.full_duplex.runtime import DuplexRuntime
from talktopia.full_duplex.sotopia_adapter import SotopiaSession
from talktopia.full_duplex.speech_client import SpeechClient


def parse_action(kwargs, action_type, argument=""):
    return kwargs["output_parser"].parse(
        json.dumps({"action_type": action_type, "argument": argument, "to": []}),
        context=kwargs["context"],
    )


def test_readable_history_preserves_pass_and_speech_containing_did_nothing(profiles):
    source = EpisodeLog(
        environment=profiles["env_id"],
        agents=profiles["agent_ids"],
        tag="pass-rendering",
        models=["env", "left", "right"],
        messages=[
            [
                ("Environment", "Alice Test", "Arrange a meeting."),
                ("Environment", "Bob Test", "Arrange a meeting."),
            ],
            [("Alice Test", "Environment", "[00:01.000]  did nothing")],
            [
                (
                    "Bob Test",
                    "Environment",
                    '[00:02.000 - 00:02.500] said: "I did nothing yesterday."',
                )
            ],
        ],
        reasoning="Original reasoning",
        rewards=[0.0, 0.0],
    )
    _, original = source.render_for_humans()
    rendered_profiles, turns = render_episode_for_humans(source)
    assert [profile.pk for profile in rendered_profiles] == profiles["agent_ids"]
    assert turns[0] == original[0] and turns[-2:] == original[-2:]
    assert turns[1:-2] == [
        "Alice Test: [00:01.000]  did nothing",
        'Bob Test [00:02.000 - 00:02.500] said: "I did nothing yesterday."',
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action_type", "generation_failure"),
    [("leave", False), ("none", False), ("none", True)],
)
async def test_silent_termination_and_all_pass_evaluation_exclusion(
    profiles, tmp_path, monkeypatch, action_type, generation_failure
):
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        if generation_failure:
            raise ValueError(
                "The original response and its SOTOPIA repair were invalid"
            )
        return parse_action(kwargs, action_type)

    async def no_speech(*args, **kwargs):
        raise AssertionError("A leave or pass must not synthesize speech")

    monkeypatch.setattr(generation, "generate_structured_action", generate)
    monkeypatch.setattr(SpeechClient, "synthesize", no_speech)
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    args.tag = "single-call-silent"
    async with FakeSpeech().client() as client:
        resolved, agents = pipeline.build_episode(profiles, args, client, client)
        result = await asyncio.wait_for(
            pipeline.run_one_episode(resolved, agents, args, tmp_path, "episode_0001"),
            3,
        )
    expected_calls = 1 if action_type == "leave" else 3
    assert len(calls) == expected_calls
    assert result["end_reason"] == ("agent_left" if action_type == "leave" else "stale")
    assert result["action_counts"] == {action_type: expected_calls}
    assert result["latency"] == {
        "normal_response": {"count": 0, "mean_ms": None},
        "backchannel": {"count": 0, "mean_ms": None},
    }
    events = read_events(tmp_path / result["events"])
    commits = [event for event in events if isinstance(event, ActionCommitted)]
    assert len(commits) == expected_calls
    assert all(
        event.metadata["generation_fallback"] is generation_failure for event in commits
    )
    assert not any(isinstance(event, AudioDeliveryEvent) for event in events)
    assert [event.metadata["actor"] for event in commits] == (
        ["Alice Test"]
        if action_type == "leave"
        else ["Alice Test", "Bob Test", "Alice Test"]
    )
    with wave.open(str(tmp_path / result["conversation_audio"]), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
            2,
            2,
            24000,
        )
        assert not any(wav.readframes(wav.getnframes()))
    assert all(not agent.state.session_active for agent in agents)

    if action_type == "none":
        original = tmp_path / result["original"]
        assert "did nothing" in (tmp_path / result["readable"]).read_text()
        source = json.loads(original.read_text())
        assert len(source["messages"]) == 4
        assert all(
            len(turn) == 1 and turn[0][2].strip() == "did nothing"
            for turn in source["messages"][1:]
        )
        eval_args = pipeline.parse_args(
            [
                "--stage",
                "reevaluate",
                "--interaction-mode",
                "surface5-full-duplex",
                "--episode-json",
                str(original),
            ]
        )
        eval_args.reeval_tag = "silent-evaluation"
        output = tmp_path / "evaluation-only"
        assert await evaluator.evaluate_episode(eval_args, output) == 0
        summary = json.loads(
            (output / "04_sotopia_eval_reevaluate_existing.json").read_text()
        )
        assert summary["status"] == "excluded" and summary["reason"] == "no_interaction"
        assert summary["attempts"] == 0
        history = (output / summary["history"]).read_text()
        assert "did nothing" in history and "[00:" in history


@pytest.mark.asyncio
async def test_full_history_keeps_initial_context_and_own_asr_beyond_eight_commits(
    profiles, tmp_path, monkeypatch
):
    requests = []

    async def generate(**kwargs):
        requests.append(kwargs)
        index = len(requests)
        if index == 1:
            return parse_action(kwargs, "speak", "Private generated sentence.")
        if index <= 10:
            return parse_action(kwargs, "action", f"Checks calendar item {index - 1}.")
        return parse_action(kwargs, "leave")

    async def synthesize(self, text, reference, seed):
        return (1000).to_bytes(2, "little") * 4800

    async def decode(self, pcm, rate):
        return "ASR delivered statement about the meeting."

    monkeypatch.setattr(generation, "generate_structured_action", generate)
    monkeypatch.setattr(SpeechClient, "synthesize", synthesize)
    monkeypatch.setattr(SpeechClient, "decode", decode)
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    args.tag = "single-call-full-history"
    async with FakeSpeech().client() as client:
        resolved, agents = pipeline.build_episode(profiles, args, client, client)
        result = await asyncio.wait_for(
            pipeline.run_one_episode(resolved, agents, args, tmp_path, "episode_0001"),
            4,
        )
    assert result["end_reason"] == "agent_left" and result["turns"] == 11
    assert len(requests) == 11
    # Alice's next action already observes her own delivered ASR, not generated text.
    assert requests[2]["input_values"]["agent"] == "Alice Test"
    assert (
        "ASR delivered statement about the meeting."
        in requests[2]["input_values"]["history"]
    )
    final = requests[-1]["input_values"]
    events = read_events(tmp_path / result["events"])
    initial = events[0].initial_observations[final["agent"]].to_natural_language()
    assert final["history"].startswith(initial)
    assert "ASR delivered statement about the meeting." in final["history"]
    assert "Private generated sentence." not in final["history"]
    for index in range(1, 10):
        assert f"Checks calendar item {index}." in final["history"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "action_type", "reason"),
    [
        ("superseded", "speak", "newer observation"),
        ("finished_backchannel", "backchanneling", "peer_audio_already_finished"),
        ("disabled_interruption", "interruption", "no longer available"),
        ("missing_target", "interruption", "target_unavailable_before_synthesis"),
    ],
)
async def test_runtime_rejects_obsolete_or_disabled_proposal_before_tts(
    profiles, tmp_path, monkeypatch, case, action_type, reason
):
    async def generate(**kwargs):
        return parse_action(
            kwargs,
            action_type,
            "Another time." if action_type in {"speak", "interruption"} else "",
        )

    async def no_speech(*args, **kwargs):
        raise AssertionError("Rejected proposal reached TTS")

    monkeypatch.setattr(generation, "generate_structured_action", generate)
    monkeypatch.setattr(SpeechClient, "synthesize", no_speech)
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    async with FakeSpeech().client() as client:
        resolved, agents = pipeline.build_episode(profiles, args, client, client)
        names = tuple(agent.agent_name for agent in agents)
        session = SotopiaSession(resolved)
        snapshot = session.open(agents)
        writer = EventWriter(tmp_path / "gate.jsonl", "episode-gate")
        stereo = StereoWavWriter(tmp_path / "gate.wav", names)
        runtime = DuplexRuntime(
            resolved=resolved,
            agents=tuple(agents),
            session=session,
            event_writer=writer,
            audio_router=AudioRouter(names),
            stereo_writer=stereo,
            config=RuntimeConfig(allow_interruptions=case == "missing_target"),
            model_names=tuple(agent.model_name for agent in agents),
        )
        observation = StreamingObservation(
            canonical=snapshot.observations[names[0]].model_copy(
                update={"available_actions": ["none", action_type]}
            ),
            source="reset" if action_type == "speak" else "asr_partial",
            peer_utterance_id="peer-current" if action_type != "speak" else None,
            target_utterance_id="peer-current"
            if action_type == "interruption"
            else None,
            peer_speaking=action_type != "speak",
        )
        runtime.state.latest_observations[names[0]] = observation
        runtime.state.latest_observations[names[1]] = StreamingObservation(
            canonical=snapshot.observations[names[1]], source="reset"
        )
        if action_type != "speak":
            runtime.state.active_utterances["peer-current"] = names[1]
        context = AgentSessionContext(
            episode_id="episode-gate",
            agent_name=names[0],
            peer_name=names[1],
            scenario="Agree on a meeting.",
            self_background="Shopkeeper.",
            private_goal="Meet before noon.",
        )
        try:
            generated = await agents[0].generation.generate_action(
                context, observation, "Initial conversation context."
            )
            assert not generated.fallback
            agents[0]._decision_observations[generated.decision.decision_id] = (
                observation
            )
            if case == "superseded":
                runtime.state.latest_observations[names[0]] = observation.model_copy(
                    update={
                        "canonical": observation.canonical.model_copy(
                            update={"observation_id": "new-observation"}
                        )
                    }
                )
            elif case == "finished_backchannel":
                runtime._audio_end_ms["peer-current"] = runtime.now_ms
            await runtime.handle_agent_output(names[0], generated)
            await asyncio.sleep(0)
            events = read_events(writer.path)
            assert len(events) == 1 and isinstance(events[0], DecisionEvent)
            assert events[0].status == "stale"
            assert any(reason in error for error in events[0].validation_errors)
            assert not agents[0].has_pending_generation
            assert not agents[0].has_pending_output
        finally:
            session.close()
            stereo.close()
            writer.close()
