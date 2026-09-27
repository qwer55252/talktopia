import pytest

from talktopia.models import ollama_proxy as proxy


def test_answer_only_request_preserves_schema_and_sampling(monkeypatch):
    calls = []
    schema = {"type": "object", "properties": {"action": {"type": "string"}}}

    def forward(url, payload, timeout):
        calls.append((url, payload, timeout))
        return 200, {
            "response": '{"action":"wait"}',
            "eval_count": 7,
            "prompt_eval_count": 20,
            "eval_duration": 100000000,
        }

    monkeypatch.setattr(proxy, "forward_json", forward)
    status, response = proxy.forward_deepseek_answer(
        "http://localhost:11434",
        {
            "model": "deepseek-r1:8b",
            "messages": [{"role": "user", "content": "Choose an action."}],
            "temperature": 0.4,
            "max_tokens": 128,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": schema},
            },
        },
        150,
    )
    url, request, timeout = calls[0]
    assert url.endswith("/api/generate") and timeout == 150
    assert request["raw"] is True and request["think"] is False
    assert request["prompt"].endswith("<｜Assistant｜><think>\n</think>\n")
    assert request["format"] == schema
    assert request["options"] == {"temperature": 0.4, "num_predict": 128}
    assert status == 200
    assert response["choices"][0]["message"]["content"] == '{"action":"wait"}'
    assert response["usage"]["total_tokens"] == 27
    assert response["talktopia_model_proxy"]["reasoning_chars"] == 0


@pytest.mark.parametrize(
    "native",
    [
        {"response": "answer", "thinking": "unexpected reasoning"},
        {"response": "<think>unexpected reasoning</think>answer"},
    ],
)
def test_reasoning_is_rejected_not_silently_removed(monkeypatch, native):
    monkeypatch.setattr(proxy, "forward_json", lambda *args: (200, native))
    status, response = proxy.forward_deepseek_answer(
        "http://localhost",
        {
            "model": "deepseek-r1:8b",
            "messages": [{"role": "user", "content": "hello"}],
        },
        150,
    )
    assert status == 502
    assert "reasoning" in response["error"]


def test_explicit_reasoning_request_is_preserved():
    payload = {"reasoning_effort": "high"}
    proxy.apply_model_defaults(payload, "deepseek-r1:8b")
    assert payload["reasoning_effort"] == "high"


def test_default_deepseek_request_disables_reasoning():
    payload = {}
    proxy.apply_model_defaults(payload, "deepseek-r1:8b")
    assert payload["reasoning_effort"] == "none"


def test_message_history_keeps_role_boundaries():
    prompt = proxy.deepseek_answer_prompt(
        [
            {"role": "system", "content": "system instruction"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "next question"},
        ]
    )
    assert (
        "first answer<｜end▁of▁sentence｜><｜User｜>next question<｜Assistant｜>"
        in prompt
    )
    assert prompt.count("<think>") == 1


def test_non_text_input_is_rejected():
    with pytest.raises(ValueError, match="text messages"):
        proxy.deepseek_answer_prompt([{"role": "user", "content": [{"image": "x"}]}])
