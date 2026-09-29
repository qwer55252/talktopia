from __future__ import annotations
import json
import importlib
from typing import Any
from types import SimpleNamespace
import pytest
from sotopia.generation_utils import PydanticOutputParser
import talktopia.full_duplex.generation as generation_module
from talktopia.full_duplex.actions import DuplexObservation, StreamingObservation
from talktopia.full_duplex.generation import (
    AgentSessionContext,
    DuplexGenerationEngine,
    _parse_generated_text,
)


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ('"bare speech"', "bare speech"),
        ('```json\n{"text":"fenced speech"}\n```', "fenced speech"),
        (
            '<think>I should answer briefly.</think>{"text":"spoken answer"}',
            "spoken answer",
        ),
    ],
)
def test_generated_text_normalizes_local_model_json(
    raw_text: str,
    expected: str,
) -> None:
    assert _parse_generated_text(raw_text).text == expected


def test_generated_text_does_not_reinterpret_malformed_json_as_speech() -> None:
    with pytest.raises(json.JSONDecodeError):
        _parse_generated_text('{"text":"missing terminator"')


@pytest.mark.asyncio
async def test_generation_keeps_decision_and_hidden_text_as_separate_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_agenerate(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if len(calls) == 1:
            return json.dumps({"action_type": "speak"})
        # Ollama can collapse a one-field JSON schema to a bare JSON string.
        return '"I can propose a practical solution."'

    monkeypatch.setattr(generation_module, "agenerate", fake_agenerate)
    engine = DuplexGenerationEngine("test-model", max_attempts=2)
    context = AgentSessionContext(
        episode_id="episode-1",
        agent_name="Alice One",
        peer_name="Bob Two",
        scenario="They need to settle a plan.",
        self_background="Alice is practical.",
        private_goal="Reach a workable plan.",
    )
    observation = StreamingObservation(
        canonical=DuplexObservation(
            last_turn="Conversation starts.",
            turn_number=0,
            available_actions=["speak"],
            action_instruction="",
            observation_id="observation-1",
        ),
        source="reset",
    )
    decision = await engine.decide_action(context, observation, "")
    hidden = await engine.generate_hidden_said(
        context,
        observation,
        decision,
        "",
    )
    chunks = engine.split_into_sentence_chunks(hidden)

    assert decision.action_type == "speak"
    assert decision.to == ["Bob Two"]
    assert decision.target_utterance_id is None
    assert decision.non_audio_argument == ""
    assert hidden.decision_id == decision.decision_id
    assert " ".join(chunk.text for chunk in chunks) == hidden.text
    assert calls[0]["structured_output"] is True
    assert isinstance(calls[0]["output_parser"], PydanticOutputParser)
    decision_schema = json.loads(calls[0]["input_values"]["format_instructions"])
    assert decision_schema["title"] == "_ActionChoice"
    assert decision_schema["properties"]["action_type"]["enum"] == ["speak"]
    assert set(decision_schema["properties"]) == {"action_type"}
    assert "peer_utterance_id" not in calls[0]["input_values"]["observation"]
    assert (
        "Choose one action from the available list, using its exact spelling."
        in calls[0]["template"]
    )
    assert 'normally choose "none"' in calls[0]["template"]
    assert calls[1]["structured_output"] is False
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_decision_discards_premature_model_payload_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    responses = iter(
        (
            json.dumps(
                {
                    "action_type": "speak",
                    "text": "This premature speech must not be committed.",
                    "content": "Nor may this alternate field be committed.",
                    "dialogue_act_label": "offer",
                }
            ),
            json.dumps({"text": "This is the separately generated speech."}),
        )
    )

    async def fake_agenerate(**kwargs: Any) -> str:
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(generation_module, "agenerate", fake_agenerate)
    engine = DuplexGenerationEngine("test-model", max_attempts=2)
    context = AgentSessionContext(
        episode_id="episode-premature-payload",
        agent_name="Alice One",
        peer_name="Bob Two",
        scenario="They need to settle a plan.",
        self_background="Alice is practical.",
        private_goal="Reach a workable plan.",
    )
    observation = StreamingObservation(
        canonical=DuplexObservation(
            last_turn="Conversation starts.",
            turn_number=0,
            available_actions=["speak"],
            action_instruction="",
            observation_id="observation-premature-payload",
        ),
        source="reset",
    )

    decision = await engine.decide_action(context, observation, "")
    hidden = await engine.generate_hidden_said(
        context,
        observation,
        decision,
        "",
    )

    assert decision.action_type == "speak"
    assert hidden.text == "This is the separately generated speech."
    assert "premature" not in hidden.text.casefold()
    assert engine.decision_audit(decision.decision_id) == (1, ())
    decision_schema = json.loads(calls[0]["input_values"]["format_instructions"])
    assert decision_schema["additionalProperties"] is False
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_generation_exposes_successful_retry_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        (
            "not-json",
            json.dumps({"action_type": "speak"}),
            "not-json",
            json.dumps({"text": "I can suggest a workable compromise."}),
        )
    )

    async def fake_agenerate(**_kwargs: Any) -> str:
        return next(responses)

    monkeypatch.setattr(generation_module, "agenerate", fake_agenerate)
    engine = DuplexGenerationEngine("test-model", max_attempts=2)
    context = AgentSessionContext(
        episode_id="episode-retry",
        agent_name="Alice One",
        peer_name="Bob Two",
        scenario="They need to settle a plan.",
        self_background="Alice is practical.",
        private_goal="Reach a workable plan.",
    )
    observation = StreamingObservation(
        canonical=DuplexObservation(
            last_turn="Conversation starts.",
            turn_number=0,
            available_actions=["speak"],
            action_instruction="",
            observation_id="observation-retry",
        ),
        source="reset",
    )

    decision = await engine.decide_action(context, observation, "")
    hidden = await engine.generate_hidden_said(
        context,
        observation,
        decision,
        "",
    )

    decision_attempt, decision_errors = engine.decision_audit(decision.decision_id)
    assert decision_attempt == 2
    assert len(decision_errors) == 1
    assert engine.hidden_said_attempt(hidden.hidden_said_id) == 2


@pytest.mark.asyncio
async def test_invalid_json_retries_same_local_endpoint_without_credential_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = importlib.import_module("sotopia.generation_utils.generate")
    responses = iter(("not-json", '```json\n{"action_type":"leave"}\n```'))
    completion_calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> Any:
        completion_calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=next(responses)),
                )
            ]
        )

    async def forbidden_fallback(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("Sotopia format_bad_output must not be called")

    monkeypatch.setattr(backend, "acompletion", fake_acompletion)
    monkeypatch.setattr(backend, "format_bad_output", forbidden_fallback)
    monkeypatch.setenv("CUSTOM_API_KEY", "EMPTY")
    engine = DuplexGenerationEngine(
        "custom/local-surface5-agent@http://127.0.0.1:18083/v1",
        max_attempts=2,
    )
    context = AgentSessionContext(
        episode_id="episode-local-retry",
        agent_name="Alice One",
        peer_name="Bob Two",
        scenario="They have reached a workable agreement.",
        self_background="Alice is practical.",
        private_goal="Conclude the agreement.",
    )
    observation = StreamingObservation(
        canonical=DuplexObservation(
            last_turn="Bob accepted the agreement.",
            turn_number=3,
            available_actions=["speak", "leave"],
            action_instruction="",
            observation_id="observation-local-retry",
        ),
        source="asr_final",
        stable_text="I agree to the plan.",
    )

    decision = await engine.decide_action(context, observation, "")

    assert decision.action_type == "leave"
    assert len(completion_calls) == 2
    assert {call["model"] for call in completion_calls} == {
        "openai/local-surface5-agent"
    }
    assert {call["base_url"] for call in completion_calls} == {
        "http://127.0.0.1:18083/v1"
    }
    assert {call["api_key"] for call in completion_calls} == {"EMPTY"}
    assert all("response_format" in call for call in completion_calls)


@pytest.mark.asyncio
async def test_generation_derives_target_and_recipient_from_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_agenerate(**_kwargs: Any) -> str:
        return json.dumps({"action_type": "correction"})

    monkeypatch.setattr(generation_module, "agenerate", fake_agenerate)
    engine = DuplexGenerationEngine("test-model", max_attempts=2)
    context = AgentSessionContext(
        episode_id="episode-target",
        agent_name="Alice One",
        peer_name="Bob Two",
        scenario="They need to settle a plan.",
        self_background="Alice is practical.",
        private_goal="Reach a workable plan.",
    )
    observation = StreamingObservation(
        canonical=DuplexObservation(
            last_turn="Bob is speaking.",
            turn_number=1,
            available_actions=["none", "correction"],
            action_instruction="",
            observation_id="observation-target",
        ),
        source="asr_partial",
        stable_text="That happened on Thursday",
        peer_utterance_id="utterance-bob-1",
        peer_speaking=True,
        target_utterance_id="utterance-bob-1",
    )

    decision = await engine.decide_action(context, observation, "")

    assert decision.action_type == "correction"
    assert decision.to == ["Bob Two"]
    assert decision.target_utterance_id == "utterance-bob-1"
    assert decision.non_audio_argument == ""


@pytest.mark.asyncio
async def test_non_audio_argument_uses_separate_structured_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    responses = iter(
        (
            json.dumps({"action_type": "action"}),
            json.dumps({"text": "mentions the system prompt"}),
            json.dumps({"text": "slides the signed form across the table"}),
        )
    )

    async def fake_agenerate(**kwargs: Any) -> str:
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(generation_module, "agenerate", fake_agenerate)
    engine = DuplexGenerationEngine("test-model", max_attempts=2)
    context = AgentSessionContext(
        episode_id="episode-action",
        agent_name="Alice One",
        peer_name="Bob Two",
        scenario="They need to settle a plan.",
        self_background="Alice is practical.",
        private_goal="Reach a workable plan.",
    )
    observation = StreamingObservation(
        canonical=DuplexObservation(
            last_turn="Conversation starts.",
            turn_number=0,
            available_actions=["action"],
            action_instruction="",
            observation_id="observation-action",
        ),
        source="reset",
    )

    decision = await engine.decide_action(context, observation, "")

    assert decision.action_type == "action"
    assert decision.non_audio_argument == "slides the signed form across the table"
    assert decision.to == ["Bob Two"]
    assert engine.non_audio_argument_attempt(decision.decision_id) == 2
    assert [call["structured_output"] for call in calls] == [True, True, True]
    assert all(
        isinstance(call["output_parser"], PydanticOutputParser) for call in calls
    )
    action_schema = json.loads(calls[0]["input_values"]["format_instructions"])
    assert set(action_schema["properties"]) == {"action_type"}
    non_audio_schema = json.loads(calls[1]["input_values"]["format_instructions"])
    assert set(non_audio_schema["properties"]) == {"text"}


@pytest.mark.asyncio
async def test_non_audio_argument_accepts_unquoted_local_model_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        (
            json.dumps({"action_type": "action"}),
            "slides the signed form across the table",
        )
    )

    async def fake_agenerate(**_kwargs: Any) -> str:
        return next(responses)

    monkeypatch.setattr(generation_module, "agenerate", fake_agenerate)
    engine = DuplexGenerationEngine("test-model", max_attempts=2)
    context = AgentSessionContext(
        episode_id="episode-unquoted-action",
        agent_name="Alice One",
        peer_name="Bob Two",
        scenario="They need to settle a plan.",
        self_background="Alice is practical.",
        private_goal="Reach a workable plan.",
    )
    observation = StreamingObservation(
        canonical=DuplexObservation(
            last_turn="Conversation starts.",
            turn_number=0,
            available_actions=["action"],
            action_instruction="",
            observation_id="observation-unquoted-action",
        ),
        source="reset",
    )

    decision = await engine.decide_action(context, observation, "")

    assert decision.non_audio_argument == "slides the signed form across the table"
    assert engine.non_audio_argument_attempt(decision.decision_id) == 1
