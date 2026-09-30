"""Surface5 generates once, validates locally, and never asks an LLM to repair."""

from __future__ import annotations

import asyncio
import importlib
import json
from dataclasses import replace
from types import SimpleNamespace

import gin
import pytest
from jsonschema import Draft202012Validator

from talktopia.full_duplex.actions import DuplexObservation, StreamingObservation
from talktopia.full_duplex.generation import (
    AgentSessionContext,
    DuplexGenerationEngine,
    GeneratedAction,
    GenerationFailure,
    _action_model,
)
from talktopia.models.config import BACKCHANNEL_TTS_INPUTS

MODEL = "custom/local-agent@http://127.0.0.1:18083/v1"
REPAIR_MODEL = "custom/local-repair@http://127.0.0.1:18084/v1"


def schema_branches(schema):
    return schema.get("anyOf", [schema])


def make_context():
    return AgentSessionContext(
        episode_id="test",
        agent_name="Alice One",
        peer_name="Bob Two",
        scenario="They need to settle a plan.",
        self_background="Alice is practical.",
        private_goal="Reach a workable plan.",
    )


def make_observation(actions, **overrides):
    return StreamingObservation(
        canonical=DuplexObservation(
            last_turn="Conversation starts.",
            turn_number=0,
            available_actions=actions,
            observation_id="observation-1",
        ),
        source=overrides.pop("source", "reset"),
        **overrides,
    )


def mock_completions(monkeypatch, responses):
    backend = importlib.import_module("sotopia.generation_utils.generate")
    responses = iter(responses)
    calls = []

    async def complete(**kwargs):
        calls.append(kwargs)
        value = next(responses)
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, dict):
            value = json.dumps(value)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=value))]
        )

    monkeypatch.setattr(backend, "acompletion", complete)
    return calls


@pytest.fixture(autouse=True)
def configured_generation():
    previous = gin.config_str()
    gin.bind_parameter(
        "sotopia.generation_utils.generate.agenerate_action.temperature", 1.0
    )
    gin.bind_parameter(
        "sotopia.generation_utils.generate.agenerate.bad_output_process_model",
        REPAIR_MODEL,
    )
    try:
        yield
    finally:
        gin.clear_config()
        gin.parse_config(previous)


@pytest.mark.asyncio
async def test_backchannel_selection_is_seeded_and_independent_of_dialogue_meaning(
    monkeypatch,
):
    count = 32
    calls = mock_completions(
        monkeypatch,
        [{"action_type": "backchanneling", "argument": "", "to": []}] * (count * 4),
    )
    session = make_context()

    async def select(seed, context, history):
        engine = DuplexGenerationEngine(MODEL, seed=seed)
        selected = []
        for _ in range(count):
            generated = await engine.generate_action(
                context, make_observation(["backchanneling"]), history
            )
            hidden = engine.make_backchannel(context, generated.decision)
            selected.append(hidden.text)
            assert [
                chunk.text for chunk in engine.split_into_sentence_chunks(hidden)
            ] == [hidden.text]
        return selected

    first = await select(17, session, "They agree with the proposal.")
    replay = await select(17, session, "They agree with the proposal.")
    different_context = await select(
        17,
        replace(session, private_goal="Reject the proposal."),
        "They disagree and ask a question.",
    )
    different_seed = await select(18, session, "They agree with the proposal.")
    assert first == replay == different_context
    assert different_seed != first
    assert set(first) == set(BACKCHANNEL_TTS_INPUTS)
    assert len(calls) == count * 4  # Choosing the sound never calls the LLM.


@pytest.mark.asyncio
async def test_cancellation_propagates_from_generation(monkeypatch):
    calls = mock_completions(monkeypatch, [asyncio.CancelledError()])
    engine = DuplexGenerationEngine(MODEL)
    with pytest.raises(asyncio.CancelledError):
        await engine.generate_action(make_context(), make_observation(["speak"]), "")
    assert len(calls) == 1 and not engine._decision_audits


@pytest.mark.asyncio
async def test_concurrent_masks_do_not_change_each_others_schema(monkeypatch):
    from talktopia.full_duplex.generation import _JointAction

    original = _JointAction.model_json_schema()
    backend = importlib.import_module("sotopia.generation_utils.generate")
    calls = []

    async def complete(**kwargs):
        calls.append(kwargs)
        await asyncio.sleep(0)  # Both requests are alive before either completes.
        schema = kwargs["response_format"]["json_schema"]["schema"]
        allowed = [
            action
            for branch in schema_branches(schema)
            for action in branch["properties"]["action_type"]["enum"]
        ]
        action = "speak" if "speak" in allowed else "backchanneling"
        content = json.dumps(
            {
                "action_type": action,
                "argument": "Hello." if action == "speak" else "",
                "to": [],
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    monkeypatch.setattr(backend, "acompletion", complete)
    results = await asyncio.gather(
        *[
            DuplexGenerationEngine(MODEL).generate_action(
                make_context(), make_observation(actions), ""
            )
            for actions in [["none", "speak", "leave"], ["none", "backchanneling"]]
        ]
    )
    assert [r.decision.action_type for r in results] == ["speak", "backchanneling"]
    assert all(isinstance(r, GeneratedAction) for r in results)
    assert len(calls) == 2
    assert _JointAction.model_json_schema() == original
    first_schema = calls[0]["response_format"]["json_schema"]["schema"]
    assert "properties" not in first_schema
    speech_branch = next(
        branch
        for branch in first_schema["anyOf"]
        if "speak" in branch["properties"]["action_type"]["enum"]
    )
    assert "const" not in speech_branch["properties"]["argument"]
    assert speech_branch["properties"]["argument"]["minLength"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,argument",
    [
        ("speak", "I can offer a solution."),
        ("leave", "Goodbye."),
        ("none", "Still listening."),
        ("backchanneling", "Yeah."),
    ],
)
async def test_one_call_routes_to_peer_and_ignores_control_arguments(
    monkeypatch, action, argument
):
    calls = mock_completions(
        monkeypatch,
        [
            {
                "action_type": action,
                "argument": argument,
                "to": ["Unknown", "Alice One", "Unknown"],
                "unused": 123,
            }
        ],
    )
    engine = DuplexGenerationEngine(MODEL)
    generated = await engine.generate_action(
        make_context(), make_observation([action]), "full history"
    )
    assert isinstance(generated, GeneratedAction)
    assert generated.decision.action_type == action and generated.decision.to == []
    assert generated.argument == (argument if action == "speak" else "")
    assert len(calls) == 1 and calls[0]["temperature"] == 1.0
    assert "full history" in calls[0]["messages"][0]["content"]
    assert engine.decision_audit(generated.decision.decision_id) == (1, ())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "The table’s like a blueprint for *my* portfolio.",
        "I can contribute **five hundred** dollars.",
        "*takes a slow breath* We agree (for now).",
        "I need $3K. Спасибо. evaluator social goal.",
        "The table's legs are uneven.",
        " ".join(["word"] * 50),
    ],
)
async def test_speech_words_reach_tts_without_content_filters(monkeypatch, text):
    calls = mock_completions(
        monkeypatch, [{"action_type": "speak", "argument": text, "to": []}]
    )
    engine = DuplexGenerationEngine(MODEL)
    generated = await engine.generate_action(
        make_context(), make_observation(["speak"]), ""
    )
    assert isinstance(generated, GeneratedAction)
    hidden = engine.make_hidden_said(make_context(), generated)
    assert generated.argument == hidden.text == text
    assert " ".join(
        c.text for c in engine.split_into_sentence_chunks(hidden)
    ) == " ".join(text.replace("*", "").split())
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        {"action_type": "speak", "argument": "  ", "to": []},
        {"action_type": "speak", "argument": " ".join(["word"] * 51), "to": []},
        {"action_type": "speak", "argument": 123, "to": []},
        {"action_type": "speak", "argument": "Hello."},
        {"action_type": "action", "argument": "opens a door", "to": []},
        {"action_type": "speak", "argument": "Hello.", "to": "Bob"},
    ],
)
async def test_unusable_response_returns_explicit_error_not_none(monkeypatch, response):
    calls = mock_completions(monkeypatch, [response])
    engine = DuplexGenerationEngine(MODEL)
    result = await engine.generate_action(
        make_context(), make_observation(["speak", "leave"]), ""
    )
    assert isinstance(result, GenerationFailure) and result.error
    assert len(calls) == 1 and len(result.raw_responses) == 1
    assert engine.decision_audit(result.decision_id) == (1, (result.error,))
    assert not hasattr(result, "decision")


@pytest.mark.asyncio
async def test_api_error_is_explicit_and_carries_no_invented_response(monkeypatch):
    calls = mock_completions(monkeypatch, [RuntimeError("server unavailable")])
    result = await DuplexGenerationEngine(MODEL).generate_action(
        make_context(), make_observation(["speak"]), ""
    )
    assert isinstance(result, GenerationFailure)
    assert (
        "server unavailable" in result.error
        and result.raw_responses == ()
        and len(calls) == 1
    )


@pytest.mark.parametrize("actions", [["speak", "leave"], ["none", "backchanneling"]])
def test_schema_keeps_types_and_action_mask_without_rejecting_extra_fields(actions):
    schema = _action_model(actions).model_json_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for action in actions:
        valid = {"action_type": action, "argument": "Hello.", "to": [], "unused": 1}
        assert validator.is_valid(valid)
        assert not validator.is_valid({**valid, "argument": 5})
        assert not validator.is_valid(
            {k: v for k, v in valid.items() if k != "argument"}
        )
    assert not validator.is_valid(
        {"action_type": "interruption", "argument": "Hello", "to": []}
    )


@pytest.mark.parametrize(
    "action",
    ["hesitation", "correction", "interruption", "action", "non-verbal communication"],
)
def test_legacy_actions_cannot_enter_the_live_generation_path(action):
    with pytest.raises(ValueError, match="Unsupported live action mask"):
        _action_model([action])
