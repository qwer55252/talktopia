"""The first request matches SOTOPIA; only the single repair prompt differs."""

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
    _failed: bool = PrivateAttr(default=False)

    def get_format_instructions(self):
        if self._failed:
            return "Correct the reported content error; preserve the spoken words."
        return super().get_format_instructions()

    def parse(self, result, context=None):
        self._contexts.append(context)
        try:
            return super().parse(result, context=context)
        except Exception:
            self._failed = True
            raise


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
@pytest.mark.parametrize("temperature", [1.0, 0.7, None])
@pytest.mark.parametrize("model", [AGENT_MODEL, "gpt-4o"])
@pytest.mark.parametrize("explicit_instructions", [False, True])
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
    actual = await generate_structured_action(**kwargs)
    assert expected == actual == Reply(text="Correct answer.")
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert kwargs["input_values"] == initial_inputs
    assert kwargs["output_parser"]._contexts == [CONTEXT]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "binding, expected_model, expected_endpoint",
    [
        (REPAIR_MODEL, "openai/repair", "http://127.0.0.1:18084/v1"),
        ("custom/repair", "openai/repair", "http://127.0.0.1:18083/v1"),
        ("other-provider-model", "other-provider-model", "http://127.0.0.1:18083/v1"),
        (
            None,
            sotopia_generation.DEFAULT_BAD_OUTPUT_PROCESS_MODEL,
            "http://127.0.0.1:18083/v1",
        ),
    ],
)
async def test_one_content_repair_keeps_schema_context_and_model_configuration(
    monkeypatch, binding, expected_model, expected_endpoint
):
    monkeypatch.setenv("CUSTOM_API_KEY", "test-key")
    gin.bind_parameter(REPAIR_BINDING, binding)
    calls = mock_completions(
        monkeypatch,
        [
            '{"text":"Wrong content."}',
            '<think>private</think>```json\n{"text":"Correct answer."}\n```',
        ],
    )
    parser = RecordingParser(pydantic_object=Reply)
    result = await generate_structured_action(**request_kwargs(parser))
    assert result.text == "Correct answer."
    assert len(calls) == 2
    assert parser._contexts == [CONTEXT, CONTEXT]
    first, repair = calls
    assert repair["model"] == expected_model
    assert repair["base_url"] == expected_endpoint
    assert repair["api_key"] == "test-key"
    assert repair["response_format"] == first["response_format"]
    assert repair["drop_params"] is True
    assert "temperature" not in repair
    content = repair["messages"][0]["content"]
    assert content.lstrip().startswith("The action below failed content validation.")
    assert 'Original string: {"text":"Wrong content."}' in content
    assert parser.get_format_instructions() in content
    assert "Given the string that can not be parsed" not in content


@pytest.mark.asyncio
async def test_scoped_gin_repair_model_matches_actual_agenerate(monkeypatch):
    gin.bind_parameter(REPAIR_BINDING, "custom/global@http://127.0.0.1:18081/v1")
    gin.bind_parameter("trial/" + REPAIR_BINDING, REPAIR_MODEL)
    responses = ['{"text":"Wrong."}', '{"text":"Correct answer."}'] * 2
    calls = mock_completions(monkeypatch, responses)
    with gin.config_scope("trial/nested"):
        await sotopia_generation.agenerate(**request_kwargs(), structured_output=True)
        await generate_structured_action(**request_kwargs())
    assert len(calls) == 4
    assert calls[0] == calls[2]
    original_repair = {
        key: value for key, value in calls[1].items() if key != "messages"
    }
    content_repair = {
        key: value for key, value in calls[3].items() if key != "messages"
    }
    assert original_repair == content_repair
    assert content_repair["model"] == "openai/repair"


@pytest.mark.asyncio
async def test_unbound_repair_model_uses_engine_default_without_initial_endpoint(
    monkeypatch,
):
    calls = mock_completions(
        monkeypatch, ['{"text":"Wrong."}', '{"text":"Correct answer."}']
    )
    kwargs = request_kwargs()
    kwargs["model_name"] = "gpt-4o"
    await generate_structured_action(**kwargs)
    assert calls[1]["model"] == sotopia_generation.DEFAULT_BAD_OUTPUT_PROCESS_MODEL
    assert calls[1]["base_url"] is None
    assert calls[1]["api_key"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses, expected_count",
    [
        ([RuntimeError("initial HTTP failed")], 1),
        (['{"text":"Wrong."}', RuntimeError("repair HTTP failed")], 2),
    ],
)
async def test_http_failure_does_not_trigger_extra_repair(
    monkeypatch, responses, expected_count
):
    calls = mock_completions(monkeypatch, responses)
    with pytest.raises(RuntimeError, match="HTTP failed"):
        await generate_structured_action(**request_kwargs())
    assert len(calls) == expected_count


@pytest.mark.asyncio
async def test_second_parse_failure_has_no_third_call(monkeypatch):
    calls = mock_completions(
        monkeypatch, ['{"text":"Wrong."}', '{"text":"Still wrong."}']
    )
    with pytest.raises(ValueError, match="text does not match context"):
        await generate_structured_action(**request_kwargs())
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("during_repair", [False, True])
async def test_cancellation_propagates(monkeypatch, during_repair):
    responses = ['{"text":"Wrong."}'] if during_repair else []
    calls = mock_completions(monkeypatch, responses + [asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await generate_structured_action(**request_kwargs())
    assert len(calls) == 1 + during_repair


@pytest.mark.asyncio
async def test_initial_none_content_matches_engine_one_call_failure(monkeypatch):
    calls = mock_completions(monkeypatch, [None, None])
    original_parser = RecordingParser(pydantic_object=Reply)
    parser = RecordingParser(pydantic_object=Reply)
    with pytest.raises(ValueError):
        await sotopia_generation.agenerate(
            **request_kwargs(original_parser), structured_output=True
        )
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Response content is None"):
        await generate_structured_action(**request_kwargs(parser))
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert parser._contexts == original_parser._contexts == [CONTEXT]
    assert parser._failed and original_parser._failed


@pytest.mark.asyncio
async def test_repaired_none_content_is_an_error_without_retry(monkeypatch):
    calls = mock_completions(monkeypatch, ['{"text":"Wrong."}', None])
    with pytest.raises(ValueError, match="Response content is None"):
        await generate_structured_action(**request_kwargs())
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_relaxed_json_and_thinking_fence_cleanup_skip_repair(monkeypatch):
    calls = mock_completions(
        monkeypatch, ['<think>hidden</think>```json\n{text: "Correct answer.",}\n```']
    )
    result = await generate_structured_action(**request_kwargs())
    assert result.text == "Correct answer."
    assert len(calls) == 1
