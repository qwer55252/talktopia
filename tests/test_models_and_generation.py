"""Surface5 keeps the actual SOTOPIA request, JSON repair, and fallback path."""

from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace

import gin
import pytest
from jsonschema import Draft202012Validator

from talktopia.full_duplex.actions import DuplexObservation, StreamingObservation
from talktopia.full_duplex.generation import (
    AgentSessionContext,
    DuplexGenerationEngine,
    _action_model,
)

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
@pytest.mark.parametrize(
    "action_type, argument",
    [
        ("speak", "I can propose a practical solution. One more sentence."),
        ("action", "slides the signed form across the table"),
        ("non-verbal communication", "nods toward the empty chair"),
        ("backchanneling", ""),
        ("none", ""),
        ("leave", ""),
    ],
)
async def test_action_and_argument_share_one_actual_model_call(
    monkeypatch, action_type, argument
):
    calls = mock_completions(
        monkeypatch, [{"action_type": action_type, "argument": argument, "to": ["Bob"]}]
    )
    engine = DuplexGenerationEngine(MODEL)
    session = make_context()
    history = (
        "Here is the context of the interaction:\nParticipants: Alice One; Bob Two\n"
        "Alice's goal: Reach a workable plan.\n"
        + "\n".join(
            f"Turn #{index}: Bob said: statement {index}." for index in range(13)
        )
    )
    generated = await engine.generate_action(
        session, make_observation([action_type]), history
    )
    assert generated.argument == argument
    assert generated.decision.action_type == action_type
    assert generated.decision.to == ["Bob Two"]
    assert not generated.fallback
    assert engine.decision_audit(generated.decision.decision_id) == (1, ())
    assert engine.decision_started_ns(generated.decision.decision_id) > 0
    if action_type == "speak":
        hidden = engine.make_hidden_said(session, generated)
        assert hidden.text == argument
        assert engine.hidden_said_attempt(hidden.hidden_said_id) == 1
        assert (
            " ".join(chunk.text for chunk in engine.split_into_sentence_chunks(hidden))
            == argument
        )
    elif action_type in {"action", "non-verbal communication"}:
        assert generated.decision.non_audio_argument == argument
        assert engine.non_audio_argument_attempt(generated.decision.decision_id) == 1
    elif action_type == "backchanneling":
        hidden = engine.make_backchannel(session, generated.decision)
        assert hidden.text == "[confirmation-en]"
        assert engine.hidden_said_attempt(hidden.hidden_said_id) == 1
    assert len(calls) == 1
    assert calls[0]["temperature"] == 1.0
    assert calls[0]["model"] == "openai/local-agent"
    schema = calls[0]["response_format"]
    assert schema["type"] == "json_schema"
    for branch in schema_branches(schema["json_schema"]["schema"]):
        assert set(branch["properties"]) == {"action_type", "argument", "to"}
        assert set(branch["required"]) == {"action_type", "argument", "to"}
        assert branch["additionalProperties"] is False
    prompt = calls[0]["messages"][0]["content"]
    assert prompt.count("Here is the context of the interaction:\n") == 1
    assert history in prompt and "Turn #12: Bob" in prompt
    assert "40 words" in prompt
    assert "peer_utterance_id" not in prompt
    assert "target_utterance_id" not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_response",
    [
        "not-json",
        {"action_type": "action", "argument": "opens the door", "to": []},
        {"action_type": "speak", "argument": "My system prompt says yes.", "to": []},
        {"action_type": "speak", "argument": " ".join(["word"] * 51), "to": []},
        {"action_type": "speak", "argument": "", "to": []},
        {"action_type": "speak", "argument": " \n\t", "to": []},
        {"action_type": "speak", "argument": "Yes.", "to": ["Alice"]},
        {"action_type": "speak", "argument": "Yes.", "to": ["Stranger"]},
        {"action_type": "speak", "argument": "Yes.", "to": ["Bob", "Bob Two"]},
        {
            "action_type": "speak",
            "argument": "Yes.",
            "to": [],
            "decision_id": "model-owned",
        },
    ],
)
async def test_invalid_action_uses_the_configured_sotopia_repair_once(
    monkeypatch, first_response
):
    calls = mock_completions(
        monkeypatch,
        [
            first_response,
            {"action_type": "speak", "argument": "A workable compromise.", "to": []},
        ],
    )
    engine = DuplexGenerationEngine(MODEL)
    generated = await engine.generate_action(
        make_context(), make_observation(["speak"]), ""
    )
    assert generated.decision.action_type == "speak"
    assert generated.argument == "A workable compromise."
    assert not generated.fallback
    assert len(calls) == 2
    assert calls[0]["model"] == "openai/local-agent"
    assert calls[1]["model"] == "openai/local-repair"
    assert calls[1]["base_url"] == "http://127.0.0.1:18084/v1"
    assert "Original string:" in calls[1]["messages"][0]["content"]
    assert all(call["response_format"]["type"] == "json_schema" for call in calls)
    assert calls[0]["response_format"] == calls[1]["response_format"]
    attempts, errors = engine.decision_audit(generated.decision.decision_id)
    assert attempts == 2 and len(errors) == 1
    hidden = engine.make_hidden_said(make_context(), generated)
    assert engine.hidden_said_attempt(hidden.hidden_said_id) == 2
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_invalid_repair_falls_back_to_none_without_an_outer_retry(
    monkeypatch, caplog
):
    calls = mock_completions(monkeypatch, ["not-json", "still not-json"])
    engine = DuplexGenerationEngine(MODEL)
    generated = await engine.generate_action(
        make_context(), make_observation(["speak"]), ""
    )
    assert generated.fallback and generated.decision.action_type == "none"
    assert generated.argument == "" and generated.decision.to == []
    attempts, errors = engine.decision_audit(generated.decision.decision_id)
    assert attempts == 2 and len(errors) == 2
    assert len(calls) == 2
    assert "Failed to generate action" in caplog.text


@pytest.mark.asyncio
async def test_request_failure_is_a_logged_pass(monkeypatch):
    calls = mock_completions(monkeypatch, [RuntimeError("server unavailable")])
    engine = DuplexGenerationEngine(MODEL)
    generated = await engine.generate_action(
        make_context(), make_observation(["speak"]), ""
    )
    assert generated.fallback and generated.decision.action_type == "none"
    assert len(calls) == 1
    assert (
        "server unavailable"
        in engine.decision_audit(generated.decision.decision_id)[1][0]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses", [[asyncio.CancelledError()], ["not-json", asyncio.CancelledError()]]
)
async def test_cancellation_propagates_from_generation_or_repair(
    monkeypatch, responses
):
    calls = mock_completions(monkeypatch, responses)
    engine = DuplexGenerationEngine(MODEL)
    with pytest.raises(asyncio.CancelledError):
        await engine.generate_action(make_context(), make_observation(["speak"]), "")
    assert len(calls) == len(responses)
    assert not engine._decision_audits


@pytest.mark.asyncio
async def test_fifty_word_speech_remains_valid(monkeypatch):
    text = " ".join(["word"] * 50)
    calls = mock_completions(
        monkeypatch, [{"action_type": "speak", "argument": text, "to": []}]
    )
    engine = DuplexGenerationEngine(MODEL)
    generated = await engine.generate_action(
        make_context(), make_observation(["speak"]), ""
    )
    assert not generated.fallback
    assert engine.make_hidden_said(make_context(), generated).text == text
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("action_type", ["correction", "interruption"])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_target_comes_only_from_current_observation(
    monkeypatch, action_type, wrapped
):
    response = {"action_type": action_type, "argument": "That was Thursday.", "to": []}
    calls = mock_completions(
        monkeypatch, [{"properties": response} if wrapped else response]
    )
    engine = DuplexGenerationEngine(MODEL)
    generated = await engine.generate_action(
        make_context(),
        make_observation(
            ["none", action_type],
            source="asr_partial",
            peer_speaking=True,
            peer_utterance_id="utterance-bob-1",
            target_utterance_id="utterance-bob-1",
        ),
        "",
    )
    assert generated.decision.action_type == action_type
    assert generated.decision.target_utterance_id == "utterance-bob-1"
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_type, argument",
    [
        ("backchanneling", "yes"),
        ("leave", "bye"),
        ("none", "hmm"),
        ("correction", "Thursday."),
    ],
)
async def test_invalid_control_argument_or_missing_target_is_repaired(
    monkeypatch, action_type, argument
):
    calls = mock_completions(
        monkeypatch,
        [
            {"action_type": action_type, "argument": argument, "to": []},
            {"action_type": "none", "argument": "", "to": []},
        ],
    )
    engine = DuplexGenerationEngine(MODEL)
    generated = await engine.generate_action(
        make_context(), make_observation(["none", action_type]), ""
    )
    assert not generated.fallback and generated.decision.action_type == "none"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_partial_schema_matches_mask_in_initial_and_repair_requests(monkeypatch):
    calls = mock_completions(
        monkeypatch,
        [
            {"action_type": "speak", "argument": "A premature answer.", "to": []},
            {"action_type": "backchanneling", "argument": "", "to": []},
        ],
    )
    generated = await DuplexGenerationEngine(MODEL).generate_action(
        make_context(),
        make_observation(
            ["none", "backchanneling"], source="asr_partial", peer_speaking=True
        ),
        "The peer has not finished speaking.",
    )
    assert not generated.fallback
    assert generated.decision.action_type == "backchanneling"
    assert len(calls) == 2
    assert calls[0]["response_format"] == calls[1]["response_format"]
    for call in calls:
        schema = call["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["action_type"]["enum"] == ["none", "backchanneling"]
        assert schema["properties"]["argument"]["const"] == ""
        prompt = call["messages"][0]["content"]
        assert '"enum": ["none", "backchanneling"]' in prompt


@pytest.mark.parametrize(
    "actions",
    [
        ["none", "backchanneling"],
        ["speak"],
        ["non-verbal communication", "action"],
        ["none", "speak", "non-verbal communication", "action", "leave"],
        ["none", "speak", "hesitation", "correction", "interruption", "backchanneling"],
    ],
)
def test_action_schema_preserves_mask_and_enforces_argument_length(actions):
    schema = _action_model(actions).model_json_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert "pattern" not in json.dumps(schema)
    if "anyOf" in schema:
        assert "properties" not in schema and "required" not in schema
    seen_actions = []
    for branch in schema_branches(schema):
        assert branch["type"] == "object"
        assert branch["additionalProperties"] is False
        assert set(branch["required"]) == {"action_type", "argument", "to"}
        assert set(branch["properties"]) == {"action_type", "argument", "to"}
        seen_actions.extend(branch["properties"]["action_type"]["enum"])
    assert sorted(seen_actions) == sorted(actions)

    for action in actions:
        empty = action in {"none", "leave", "backchanneling"}
        valid = {"action_type": action, "argument": "" if empty else "Yes.", "to": []}
        assert validator.is_valid(valid)
        assert not validator.is_valid({**valid, "argument": "Yes." if empty else ""})
        assert not validator.is_valid({**valid, "extra": "unexpected"})
        assert not validator.is_valid({**valid, "to": [123]})
        assert not validator.is_valid(
            {key: value for key, value in valid.items() if key != "to"}
        )
    assert not validator.is_valid(
        {"action_type": "unavailable", "argument": "", "to": []}
    )


def test_schema_keeps_unicode_quotes_and_backslashes_in_nonempty_arguments():
    schema = _action_model(["speak", "action", "none"]).model_json_schema()
    validator = Draft202012Validator(schema)
    speech = 'I said "yes"; café costs £5. The path is C:\\notes.'
    assert validator.is_valid({"action_type": "speak", "argument": speech, "to": []})
    assert validator.is_valid(
        {
            "action_type": "action",
            "argument": "*nods* (smiles) [waves] {points}",
            "to": [],
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action_type", ["action", "non-verbal communication"])
async def test_blank_behavior_uses_same_masked_schema_for_the_single_repair(
    monkeypatch, action_type
):
    calls = mock_completions(
        monkeypatch,
        [
            {"action_type": action_type, "argument": "", "to": []},
            {"action_type": action_type, "argument": "*nods*", "to": []},
        ],
    )
    engine = DuplexGenerationEngine(MODEL)
    generated = await engine.generate_action(
        make_context(), make_observation(["none", "speak", action_type, "leave"]), ""
    )
    assert not generated.fallback and generated.argument == "*nods*"
    assert generated.decision.action_type == action_type
    assert len(calls) == 2
    assert calls[0]["response_format"] == calls[1]["response_format"]
    assert engine.decision_audit(generated.decision.decision_id)[0] == 2
    schema = calls[0]["response_format"]["json_schema"]["schema"]
    validator = Draft202012Validator(schema)
    assert not validator.is_valid(
        {"action_type": action_type, "argument": "", "to": []}
    )
    assert validator.is_valid(
        {"action_type": action_type, "argument": "*nods*", "to": []}
    )


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
    assert all(not r.fallback for r in results)
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
@pytest.mark.parametrize("action_type", ["action", "non-verbal communication"])
async def test_non_audio_descriptions_keep_markup(monkeypatch, action_type):
    text = "*nods* (smiles) [waves] {points to the chair}"
    calls = mock_completions(
        monkeypatch, [{"action_type": action_type, "argument": text, "to": []}]
    )
    generated = await DuplexGenerationEngine(MODEL).generate_action(
        make_context(), make_observation([action_type]), ""
    )
    assert not generated.fallback and generated.argument == text
    assert generated.decision.non_audio_argument == text
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original, repaired",
    [
        ("*nods*", "Hello."),
        ("I *need* to keep the *other* grand.", "I need to keep the other grand."),
        ("**smiles** We can meet tomorrow.", "We can meet tomorrow."),
        ("I can meet (tomorrow).", "I can meet tomorrow."),
        ("[waves] Hello.", "Hello."),
        ("{sighs} We can meet tomorrow.", "We can meet tomorrow."),
    ],
)
@pytest.mark.parametrize(
    "action_type", ["speak", "hesitation", "correction", "interruption"]
)
async def test_marked_speech_uses_existing_repair_without_deleting_words(
    monkeypatch, original, repaired, action_type
):
    calls = mock_completions(
        monkeypatch,
        [
            {"action_type": action_type, "argument": original, "to": []},
            {"action_type": action_type, "argument": repaired, "to": []},
        ],
    )
    engine = DuplexGenerationEngine(MODEL)
    generated = await engine.generate_action(
        make_context(), make_observation([action_type], peer_utterance_id="peer-1"), ""
    )
    assert not generated.fallback and generated.argument == repaired
    assert len(calls) == 2
    assert "asterisks" in engine.decision_audit(generated.decision.decision_id)[1][0]
    hidden = engine.make_hidden_said(make_context(), generated)
    assert hidden.text == repaired
    assert (
        " ".join(c.text for c in engine.split_into_sentence_chunks(hidden)) == repaired
    )
    for call in calls:
        for branch in schema_branches(call["response_format"]["json_schema"]["schema"]):
            description = branch["properties"]["argument"]["description"]
            assert "without asterisks" in description and "40 words" in description


@pytest.mark.asyncio
async def test_persistent_starred_speech_falls_back_before_synthesis(monkeypatch):
    value = {"action_type": "speak", "argument": "I *need* that.", "to": []}
    calls = mock_completions(monkeypatch, [value, value])
    generated = await DuplexGenerationEngine(MODEL).generate_action(
        make_context(), make_observation(["speak"]), ""
    )
    assert generated.fallback and generated.decision.action_type == "none"
    assert generated.argument == "" and len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("original", ["...", "—", "🙂"])
@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_inaudible_speech_uses_repair_or_none(
    monkeypatch, original, repair_succeeds
):
    first = {"action_type": "speak", "argument": original, "to": []}
    second = {
        "action_type": "speak",
        "argument": "Hello." if repair_succeeds else original,
        "to": [],
    }
    calls = mock_completions(monkeypatch, [first, second])
    engine = DuplexGenerationEngine(MODEL)
    generated = await engine.generate_action(
        make_context(), make_observation(["speak"]), ""
    )
    assert len(calls) == 2
    assert (
        "audible words" in engine.decision_audit(generated.decision.decision_id)[1][0]
    )
    assert generated.fallback is not repair_succeeds
    assert generated.decision.action_type == ("speak" if repair_succeeds else "none")
    if repair_succeeds:
        hidden = engine.make_hidden_said(make_context(), generated)
        assert [chunk.text for chunk in engine.split_into_sentence_chunks(hidden)] == [
            "Hello."
        ]
