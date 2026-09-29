"""Keep the published prompts aligned with their source text and actual requests."""

from pathlib import Path

import pytest
from sotopia.database import SotopiaDimensions
from sotopia.generation_utils import PydanticOutputParser

from talktopia.evaluation import evaluator
from talktopia.full_duplex import generation
from talktopia.full_duplex.config import runtime_settings


def test_simulation_preserves_sotopia_body_and_round_robin_guidance():
    directory = generation._PROMPT_PATH.parent
    original = (directory / "sotopia_action_v1.txt").read_text()
    body, output_instruction = original.split("\n\nPlease only generate", 1)
    prompt = generation._ACTION_PROMPT
    assert prompt.startswith(body)
    assert prompt.endswith("Please only generate" + output_instruction)
    assert {path.name for path in directory.glob("simulation_*.txt")} == {
        "simulation_FDB_v1.txt",
        "simulation_FDB_v2.txt",
        "simulation_FDB_v3.txt",
        "simulation_FDB_v4.txt",
    }
    assert (
        runtime_settings()["simulation_prompt"] == generation.SIMULATION_PROMPT_VERSION
    )
    assert runtime_settings()["termination_policy"] == "first_leave"
    assert "[DECISION]" not in prompt and "[HIDDEN_SAID]" not in prompt
    assert (
        'For a "speak" action, keep "argument" within 40 words; this is a maximum, not a target.'
        in prompt
    )
    assert 'for "to", use [] for public actions' in prompt
    assert (
        'An "asr_partial" observation contains only the words recognized so far'
        in prompt
    )
    assert 'set "argument" to "" for "backchanneling"' in prompt
    assert "only the words to be spoken aloud" in prompt
    assert 'For "action" or "non-verbal communication"' in prompt
    assert prompt.count("{history}") == 1
    for phrase in (
        "make a concrete opening",
        "Do not prolong",
        "remains unresolved",
        "settled point",
    ):
        assert phrase not in prompt


def test_fdb_evaluation_only_adds_timing_to_the_original_template():
    directory = Path(evaluator.__file__).with_name("prompts")
    original = (directory / "evaluation_v1.txt").read_text()
    prefix, timing_and_tail = evaluator.FDB_EVALUATION_PROMPT.split(
        "\nSpeech timing in the recorded interaction:\n", 1
    )
    timing, tail = timing_and_tail.split("\n{retry_feedback}", 1)
    assert prefix + "{retry_feedback}" + tail == original
    assert "existing scoring dimension" in timing
    assert "whole-utterance ASR" in timing
    assert not (directory / "temporal_v1.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, structured",
    [
        ("custom/structured/test@http://local", True),
        ("schema-supported", True),
        ("schema-unsupported", False),
    ],
)
async def test_fdb_request_preserves_generation_and_score_contract(
    monkeypatch, model, structured
):
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return evaluator.TwoAgentEvaluation.model_validate(
            {
                "evaluations": {
                    agent: {
                        dimension: {
                            "score": 0,
                            "reasoning": f"{agent} evidence for {dimension}",
                        }
                        for dimension in SotopiaDimensions.model_fields
                    }
                    for agent in ("agent_2", "agent_1")
                }
            }
        )

    monkeypatch.setattr(evaluator, "agenerate", generate)
    monkeypatch.setattr(
        evaluator,
        "supports_response_schema",
        lambda *, model: model == "schema-supported",
    )
    values = evaluator.fdb_prompt_values(
        '[00:01.000 - 00:02.000] Alice said: "Nine?"',
        ["Alice", "Bob"],
        "Retry with complete scores.",
    )
    responses = await evaluator.evaluate_fdb(model, values)
    evaluator.validate_evaluation_responses(responses)
    assert len(responses) == 14 and responses[0][0] == "agent_1"
    call = calls[0]
    assert call["template"] == evaluator.FDB_EVALUATION_PROMPT
    assert isinstance(call["output_parser"], PydanticOutputParser)
    assert call["output_parser"].pydantic_object is evaluator.TwoAgentEvaluation
    assert call["structured_output"] is structured
    assert call["temperature"] == 0.0
    assert call["bad_output_process_model"] == model
    rendered = call["template"].format(**call["input_values"])
    for text in (
        "Based on previous interactions",
        "Speech timing in the recorded interaction",
        "Agent mapping:",
        "Retry with complete scores.",
        "Please follow the format:",
    ):
        assert rendered.count(text) == 1
    assert '"agent_1", "agent_2"' in rendered
    assert "{history}" not in rendered and "{format_instructions}" not in rendered
