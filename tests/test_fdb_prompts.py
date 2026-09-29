"""Keep the published prompts aligned with their source text and actual requests."""

from pathlib import Path

import pytest
from sotopia.database import SotopiaDimensions
from sotopia.generation_utils import PydanticOutputParser

from talktopia.evaluation import evaluator
from talktopia.full_duplex import generation
from talktopia.full_duplex.actions import DuplexObservation, StreamingObservation
from talktopia.full_duplex.config import runtime_settings
from talktopia.full_duplex.generation import AgentSessionContext


def test_simulation_common_body_is_the_original_sotopia_body():
    directory = generation._PROMPT_PATH.parent
    original = (directory / "sotopia_action_v1.txt").read_text()
    body, output_instruction = original.split("\n\nPlease only generate", 1)
    assert generation._COMMON_PROMPT == body
    assert "action type and the argument" in output_instruction
    assert {path.name for path in directory.glob("simulation_*.txt")} == {
        "simulation_FDB_v1.txt",
        "simulation_FDB_v2.txt",
        "simulation_FDB_v3.txt",
    }
    assert (
        runtime_settings()["simulation_prompt"] == generation.SIMULATION_PROMPT_VERSION
    )
    assert runtime_settings()["termination_policy"] == "first_leave"
    assert "CLOSING" not in generation._PROMPTS

    context = AgentSessionContext(
        episode_id="test",
        agent_name="Alice",
        peer_name="Bob",
        scenario="Meet tomorrow.",
        self_background="A shopkeeper.",
        private_goal="Agree on a time.",
    )
    observation = StreamingObservation(
        canonical=DuplexObservation(
            last_turn="Nine?",
            turn_number=3,
            available_actions=["speak", "leave"],
            observation_id="obs",
        ),
        source="asr_final",
    )
    common = generation.DuplexGenerationEngine._common_prompt(
        context, observation, "Bob said: Nine?"
    )
    assert "Here is the context of the interaction:" in common
    assert "Your social goal: Agree on a time." in common
    assert "Bob said: Nine?" in common and "Turn #3" in common
    assert 'You can "leave"' in common and '"speak", "leave"' in common
    for template in (
        generation._DECISION_PROMPT,
        generation._NON_AUDIO_ARGUMENT_PROMPT,
        generation._HIDDEN_SAID_PROMPT,
    ):
        assert template.count("{common_prompt}") == 1
        assert "make a concrete opening" not in template
        assert "Do not prolong" not in template
        assert "remains unresolved" not in template
        assert "settled point" not in template


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
