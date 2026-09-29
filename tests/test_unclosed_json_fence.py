"""Transport normalization uses the existing SOTOPIA parser without a fork."""

import importlib
import json

import pytest

from talktopia.full_duplex.generation import Surface5ActionOutputParser, _JointAction


def parse(raw):
    backend = importlib.import_module("sotopia.generation_utils.generate")
    return Surface5ActionOutputParser(pydantic_object=_JointAction).parse(
        backend._strip_thinking_tags(raw),
        context={
            "available_action_types": ["speak"],
            "agent_names": ["Alice", "Bob"],
            "sender": "Alice",
        },
    )


@pytest.mark.parametrize("prefix", ["```json\n", "```\n", "```JSON\r\n"])
@pytest.mark.parametrize("suffix", ["", "\n", "\n```"])
def test_joint_action_with_optional_closing_fence(prefix, suffix):
    raw = (
        prefix
        + '{"action_type":"speak","argument":"Let us try that.","to":[]}'
        + suffix
    )
    assert parse(raw).argument == "Let us try that."


def test_thinking_tags_and_unclosed_fence():
    raw = '<think>Choose an action.</think>\n```json\n{"action_type":"speak","argument":"Yes.","to":[]}'
    assert parse(raw).argument == "Yes."


def test_common_json_repair_accepts_missing_brace_and_trailing_explanation():
    assert parse('{"action_type":"speak","argument":"Yes.","to":[]').argument == "Yes."
    assert (
        parse('{"action_type":"speak","argument":"Yes.","to":[]}\nExplanation').argument
        == "Yes."
    )


@pytest.mark.parametrize(
    "raw", ["plain speech", "```json", "```json\n", "```json\n```", '"bare speech"']
)
def test_text_without_a_joint_action_cannot_bypass_contract(raw):
    with pytest.raises((ValueError, AssertionError)):
        parse(raw)


def test_quotes_braces_and_backticks_inside_argument_are_preserved():
    argument = 'Use {one} and say "yes"; ``` is a marker.'
    assert (
        parse(
            json.dumps({"action_type": "speak", "argument": argument, "to": []})
        ).argument
        == argument
    )


def test_schema_properties_wrapper_still_checks_action_mask():
    # The shared parser's legacy wrapper branch omits context internally.
    raw = json.dumps(
        {
            "properties": {
                "action_type": "action",
                "argument": "opens the door",
                "to": [],
            }
        }
    )
    with pytest.raises(ValueError, match="unavailable action"):
        parse(raw)
