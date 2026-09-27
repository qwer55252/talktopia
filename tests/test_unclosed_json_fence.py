import json

import pytest

from talktopia.full_duplex.generation import (
    _parse_action_choice,
    _parse_generated_text,
    _parse_non_audio_argument,
)


@pytest.mark.parametrize("prefix", ["```json\n", "```\n", "```JSON\r\n"])
@pytest.mark.parametrize("suffix", ["", "\n", "\n```"])
def test_complete_json_with_optional_closing_fence(prefix, suffix):
    raw = prefix + '{"text":"Let us try that."}' + suffix
    assert _parse_generated_text(raw).text == "Let us try that."
    assert _parse_non_audio_argument(raw).text == "Let us try that."


def test_decision_after_thinking_with_unclosed_fence():
    raw = '<think>Choose an action.</think>\n```json\n{"action_type":"speak"}'
    assert _parse_action_choice(raw).action_type == "speak"


@pytest.mark.parametrize(
    "body",
    [
        '{"text":"cut off"',
        '{"text":"cut off',
        '{"text":"valid"}\nextra text',
    ],
)
def test_missing_fence_does_not_repair_invalid_json(body):
    for parse in (_parse_generated_text, _parse_non_audio_argument):
        with pytest.raises(json.JSONDecodeError):
            parse("```json\n" + body)


@pytest.mark.parametrize("raw", ["```json", "```json\n", "```json\n```"])
def test_empty_fenced_response_is_rejected(raw):
    with pytest.raises(ValueError):
        _parse_generated_text(raw)


def test_backticks_inside_json_string_are_preserved():
    assert (
        _parse_generated_text('```json\n{"text":"Use ``` as a marker."}').text
        == "Use ``` as a marker."
    )
