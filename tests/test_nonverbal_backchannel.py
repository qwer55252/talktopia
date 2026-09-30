"""Backchannel audio remains visible with and without recognized words."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from sotopia.database import EpisodeLog
from test_duplex_episode import FakeSpeech, joint_result, wav_bytes
from test_duplex_episode import profiles as profiles  # noqa: PLC0414 -- pytest fixture

from talktopia import pipeline
from talktopia.evaluation import evaluator
from talktopia.full_duplex import generation
from talktopia.full_duplex.actions import DuplexAction
from talktopia.full_duplex.config import runtime_settings
from talktopia.full_duplex.events import (
    ActionCommitted,
    ASRUpdateEvent,
    AudioDeliveryEvent,
    HiddenSaidCreated,
    SpeechSynthesisFailed,
    read_events,
)
from talktopia.full_duplex.rendering import render_agent_history
from talktopia.full_duplex.runtime import DuplexRuntime
from talktopia.full_duplex.speech_backends import ASRUpdate
from talktopia.full_duplex.speech_client import SpeechClient
from talktopia.full_duplex.transcript import TranscriptBuilder, TranscriptEntry


class BackchannelSpeech(FakeSpeech):
    def __init__(self, text, *, fail=False):
        super().__init__()
        self.text = text
        self.inputs = []
        self.backchannel_seeds = []
        self.fail = fail

    def handle(self, request):
        if request.url.path.endswith("/audio/speech"):
            body = json.loads(request.content)
            self.inputs.append(body["input"])
            if body["input"] == self.text:
                self.backchannel_seeds.append(body["seed"])
                if self.fail:
                    return httpx.Response(
                        400,
                        json={
                            "detail": {"code": "empty_audio", "message": "No samples"}
                        },
                    )
                return httpx.Response(
                    200,
                    content=wav_bytes(4800, sample=2000),
                    headers={"Content-Type": "audio/wav"},
                )
        return super().handle(request)


async def run_backchannel_episode(
    profiles, tmp_path, monkeypatch, *, text="[confirmation-en]", received="", fail=False
):
    regular_decisions = 0
    attempted_backchannel = False
    prompts = []

    async def generate(**kwargs):
        nonlocal regular_decisions, attempted_backchannel
        prompts.append(kwargs["input_values"]["history"])
        observation = json.loads(kwargs["input_values"]["observation"])
        if observation["source"] == "asr_partial":
            if (
                not attempted_backchannel
                and "backchanneling" in kwargs["context"]["available_action_types"]
            ):
                attempted_backchannel = True
                action = "backchanneling"
            else:
                action = "none"
        else:
            regular_decisions += 1
            action = "speak" if regular_decisions <= 2 else "leave"
        return joint_result(
            kwargs, action, "We should agree on the proposed meeting time."
        )

    async def decode(self, pcm, rate):
        if pcm[:2] == (2000).to_bytes(2, "little", signed=True):
            return received
        return "Received words about the proposed meeting time."

    monkeypatch.setattr(generation, "generate_structured_action", generate)
    # Exercise each history path deterministically; selection has its own test.
    monkeypatch.setattr(generation, "BACKCHANNEL_TTS_INPUTS", (text,))
    monkeypatch.setattr(SpeechClient, "decode", decode)
    args = pipeline.parse_args([
        "--interaction-mode", "surface5-full-duplex", "--seed", "17"
    ])
    args.tag = "nonverbal-test"
    speech = BackchannelSpeech(text, fail=fail)
    async with speech.client() as client:
        resolved, agents = pipeline.build_episode(profiles, args, client, client)
        assert all(agent.generation.seed == 17 for agent in agents)
        result = await asyncio.wait_for(
            pipeline.run_one_episode(resolved, agents, args, tmp_path, "episode_0001"),
            15,
        )
    return result, read_events(tmp_path / result["events"]), speech, prompts


@pytest.mark.asyncio
@pytest.mark.parametrize("text,received", [("[confirmation-en]", ""), ("Mm-hmm", "Mm-hmm.")])
async def test_backchannel_survives_into_agent_and_evaluation_history(
    profiles, tmp_path, monkeypatch, text, received
):
    result, events, speech, prompts = await run_backchannel_episode(
        profiles, tmp_path, monkeypatch, text=text, received=received
    )
    assert result["status"] == "completed"
    assert events[0].run_config["backchannel_tts"] == runtime_settings()["backchannel_tts"]
    assert speech.inputs.count(text) == 1
    commits = [event for event in events if isinstance(event, ActionCommitted)]
    committed = [
        event
        for event in commits
        if any(
            action.action_type == "backchanneling" for action in event.actions.values()
        )
    ]
    assert len(committed) == 1
    commit = committed[0]
    assert bool(commit.metadata.get("observed_nonverbal_backchannel")) == (not received)
    action = next(
        action
        for action in commit.actions.values()
        if action.action_type == "backchanneling"
    )
    assert action.argument == received
    entries = TranscriptBuilder.from_events(events).build()
    entry = next(entry for entry in entries if entry.action_type == "backchanneling")
    assert entry.generated_text == entry.synthesized_text == text
    assert entry.received_text == received and entry.sentences[0].received_text == received
    assert entry.commit_id == commit.commit_id and entry.completed
    assert entry.start_ms < entry.end_ms
    assert any(
        isinstance(event, AudioDeliveryEvent)
        and event.utterance_id == entry.utterance_id
        for event in events
    )
    finals = [
        event
        for event in events
        if isinstance(event, ASRUpdateEvent)
        and event.utterance_id == entry.utterance_id
        and event.is_final
    ]
    assert (
        len(finals) == 1
        and finals[0].text == received
        and finals[0].sentence_texts == {0: received}
    )
    assert result["action_counts"]["backchanneling"] == 1
    assert result["latency"]["backchannel"]["count"] == 1
    label = (
        f'backchanneled: "{received}"'
        if received else "made a nonverbal backchannel [no recognized words]"
    )
    assert any(label in history for history in prompts)
    if text.startswith("["):
        assert all(text not in history for history in prompts)
    for viewer in (entry.speaker, entry.listener):
        history = render_agent_history(entries, viewer, len(entries))
        assert label in history
        if text.startswith("["):
            assert text not in history
    source = EpisodeLog.model_validate_json((tmp_path / result["original"]).read_text())
    _, turns = evaluator.duplex_history(source, tmp_path / result["events"])
    history = "\n".join(turns)
    assert label in history and "Sentence ASR" in history and "[00:" in history
    if text.startswith("["):
        assert text not in history
    if not received:
        assert "backchanneled:" not in history


@pytest.mark.asyncio
async def test_empty_tts_does_not_invent_a_nonverbal_backchannel(
    profiles, tmp_path, monkeypatch
):
    result, events, speech, _ = await run_backchannel_episode(
        profiles, tmp_path, monkeypatch, fail=True
    )
    assert result["status"] == "completed"
    assert speech.inputs.count("[confirmation-en]") == 2
    assert len(set(speech.backchannel_seeds)) == 2
    failures = [event for event in events if isinstance(event, SpeechSynthesisFailed)]
    assert len(failures) == 1 and failures[0].attempts == 2
    assert failures[0].reason == "empty_audio"
    assert failures[0].chunk.text == "[confirmation-en]"
    assert any(
        isinstance(event, HiddenSaidCreated)
        and event.hidden_said.text == "[confirmation-en]"
        for event in events
    )
    assert not any(
        isinstance(event, AudioDeliveryEvent)
        and event.utterance_id == failures[0].chunk.utterance_id
        for event in events
    )
    assert not any(
        isinstance(event, ActionCommitted)
        and any(
            action.action_type == "backchanneling" for action in event.actions.values()
        )
        for event in events
    )
    assert result["latency"]["backchannel"] == {"count": 0, "mean_ms": None}
    assert not result["action_counts"].get("backchanneling", 0)


def test_empty_asr_permission_is_limited_to_backchannels():
    assert (
        "nonverbal backchannel"
        in DuplexAction(action_type="backchanneling", argument="").to_natural_language()
    )
    for action in ("speak", "hesitation", "action", "non-verbal communication"):
        with pytest.raises(ValueError, match="non-empty argument"):
            DuplexAction(action_type=action, argument="")


def test_both_agent_views_use_recognized_backchannel_words_instead_of_tts_tag():
    entry = TranscriptEntry(
        episode_id="test",
        utterance_id="ack",
        speaker="Alice",
        listener="Bob",
        start_ms=100,
        end_ms=750,
        action_type="backchanneling",
        generated_text="[confirmation-en]",
        synthesized_text="[confirmation-en]",
        received_text="Mm.",
        origin="agent",
        completed=True,
        interrupted=False,
        cancellation_reason=None,
        decision_id="decision",
        hidden_said_id="hidden",
        commit_id="commit",
    )
    for viewer in (entry.speaker, entry.listener):
        history = render_agent_history([entry], viewer, 1)
        assert 'backchanneled: "Mm."' in history
        assert "[confirmation-en]" not in history


@pytest.mark.asyncio
async def test_empty_asr_cannot_commit_an_undelivered_backchannel():
    runtime = object.__new__(DuplexRuntime)
    runtime._utterances = {
        "unheard": SimpleNamespace(action_type="backchanneling", listener="Bob")
    }
    runtime._deliveries = {}
    update = ASRUpdate(
        utterance_id="unheard",
        listener="Bob",
        text="",
        is_final=True,
        is_stable=True,
        revision_id=1,
    )
    with pytest.raises(RuntimeError, match="without delivered audio"):
        await runtime.commit_asr_final(update)
