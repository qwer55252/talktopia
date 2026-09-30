"""Preserve the initial SOTOPIA payload without a second LLM correction call."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import gin
import pytest
from pydantic import BaseModel, Field, PrivateAttr, ValidationInfo, model_validator
from sotopia.generation_utils import PydanticOutputParser
from sotopia.generation_utils import generate as sotopia_generation

from talktopia.full_duplex.requests import generate_structured_action

AGENT_MODEL = "custom/agent@http://127.0.0.1:18083/v1"
REPAIR_MODEL = "custom/repair@http://127.0.0.1:18084/v1"
REPAIR_BINDING = "sotopia.generation_utils.generate.agenerate.bad_output_process_model"
TEMPLATE = """
    Speak as {agent}.
    {history}
    Return {format_instructions}
    """
CONTEXT = {"expected": "Correct answer."}


class Reply(BaseModel):
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def match_context(self, info: ValidationInfo):
        if info.context and self.text != info.context["expected"]:
            raise ValueError("text does not match context")
        return self


class RecordingParser(PydanticOutputParser[Reply]):
    _contexts: list = PrivateAttr(default_factory=list)

    def parse(self, result, context=None):
        self._contexts.append(context)
        return super().parse(result, context=context)


@pytest.fixture(autouse=True)
def isolate_gin_configuration():
    previous = gin.config_str()
    gin.clear_config()
    try:
        yield
    finally:
        gin.clear_config()
        gin.parse_config(previous)


def mock_completions(monkeypatch, responses):
    responses = iter(responses)
    calls = []

    async def complete(**kwargs):
        calls.append(deepcopy(kwargs))
        response = next(responses)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )

    monkeypatch.setattr(sotopia_generation, "acompletion", complete)
    return calls


def request_kwargs(parser=None):
    return {
        "model_name": AGENT_MODEL,
        "template": TEMPLATE,
        "input_values": {"agent": "Alice", "history": 'Bob said: "{hello}".'},
        "output_parser": parser or RecordingParser(pydantic_object=Reply),
        "temperature": 1.0,
        "context": CONTEXT,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "temperature, model, explicit_instructions",
    [
        pytest.param(1.0, AGENT_MODEL, False, id="custom-model"),
        pytest.param(0.7, AGENT_MODEL, False, id="nondefault-temperature"),
        pytest.param(None, AGENT_MODEL, False, id="omitted-temperature"),
        pytest.param(1.0, "gpt-4o", False, id="standard-model"),
        pytest.param(1.0, AGENT_MODEL, True, id="caller-instructions"),
    ],
)
async def test_first_payload_matches_actual_agenerate(
    monkeypatch, temperature, model, explicit_instructions
):
    monkeypatch.setenv("CUSTOM_API_KEY", "test-key")
    calls = mock_completions(
        monkeypatch, ['{"text":"Correct answer."}', '{"text":"Correct answer."}']
    )
    kwargs = request_kwargs()
    kwargs.update(model_name=model, temperature=temperature)
    if explicit_instructions:
        kwargs["input_values"]["format_instructions"] = "Caller supplied instructions."
    initial_inputs = deepcopy(kwargs["input_values"])
    expected = await sotopia_generation.agenerate(
        **deepcopy(kwargs), structured_output=True
    )
    recorded = []
    actual = await generate_structured_action(**kwargs, responses=recorded)
    assert expected == actual == Reply(text="Correct answer.")
    assert len(calls) == 2
    assert calls[1].pop("num_retries") == 0
    assert calls[1].pop("max_retries") == 0
    assert calls[0] == calls[1]
    assert recorded == ['{"text":"Correct answer."}']
    assert kwargs["input_values"] == initial_inputs
    assert kwargs["output_parser"]._contexts == [CONTEXT]


@pytest.mark.asyncio
async def test_null_response_is_recorded_without_repair(monkeypatch):
    calls = mock_completions(monkeypatch, [None])
    parser = RecordingParser(pydantic_object=Reply)
    recorded = []
    with pytest.raises(ValueError, match="Response content is None"):
        await generate_structured_action(**request_kwargs(parser), responses=recorded)
    assert len(calls) == 1
    assert recorded == [None]


@pytest.mark.asyncio
async def test_relaxed_json_and_thinking_fence_cleanup_skip_repair(monkeypatch):
    calls = mock_completions(
        monkeypatch, ['<think>hidden</think>```json\n{text: "Correct answer.",}\n```']
    )
    result = await generate_structured_action(**request_kwargs())
    assert result.text == "Correct answer."
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_concurrent_response_collectors_do_not_mix(monkeypatch):
    async def complete(**kwargs):
        await asyncio.sleep(0)
        response = '{"text":"' + kwargs["messages"][0]["content"] + '"}'
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )

    monkeypatch.setattr(sotopia_generation, "acompletion", complete)
    recorded = [[], []]
    await asyncio.gather(
        *[
            generate_structured_action(
                model_name=AGENT_MODEL,
                template=name,
                input_values={},
                output_parser=RecordingParser(pydantic_object=Reply),
                temperature=1,
                responses=recorded[index],
            )
            for index, name in enumerate(["Alice", "Bob"])
        ]
    )
    assert recorded == [['{"text":"Alice"}'], ['{"text":"Bob"}']]


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ['{"text":"Wrong content."}', "not-json"])
async def test_invalid_output_never_calls_the_configured_repair_model(
    monkeypatch, response
):
    gin.bind_parameter(REPAIR_BINDING, REPAIR_MODEL)
    calls = mock_completions(monkeypatch, [response])
    recorded = []
    with pytest.raises((ValueError, AssertionError)):
        await generate_structured_action(**request_kwargs(), responses=recorded)
    assert len(calls) == 1
    assert calls[0]["model"] == "openai/agent"
    assert recorded == [response]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("HTTP failed"), asyncio.CancelledError()]
)
async def test_request_errors_propagate_without_another_call(monkeypatch, error):
    calls = mock_completions(monkeypatch, [error])
    recorded = []
    with pytest.raises(type(error)):
        await generate_structured_action(**request_kwargs(), responses=recorded)
    assert len(calls) == 1 and not recorded


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [True, False])
async def test_only_transient_transport_failure_is_retried_once(monkeypatch, recover):
    from litellm.exceptions import APIConnectionError

    error = APIConnectionError(
        "connection lost", llm_provider="openai", model=AGENT_MODEL
    )
    calls = mock_completions(
        monkeypatch, [error, '{"text":"Correct answer."}' if recover else error]
    )
    responses = []
    if recover:
        result = await generate_structured_action(
            **request_kwargs(), responses=responses
        )
        assert result.text == "Correct answer." and len(responses) == 1
    else:
        with pytest.raises(APIConnectionError):
            await generate_structured_action(**request_kwargs(), responses=responses)
        assert responses == []
    assert len(calls) == 2 and calls[0] == calls[1]
