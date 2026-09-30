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


def qwen_request(**overrides):
    return {
        "model": "qwen3.5:9b",
        "messages": [{"role": "user", "content": "Choose one action."}],
        "reasoning_effort": "none",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "action",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"status": {"const": "SCHEMA_LOCKED"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            },
        },
        **overrides,
    }


def test_qwen_prompt_matches_native_render_and_preserves_message_order():
    messages = [
        {"role": "system", "content": "Keep the schema."},
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": '{"status":"first"}'},
        {"role": "user", "content": "Next question.\nPreserve my newline."},
    ]
    assert proxy.qwen_answer_prompt(messages) == (
        "<|im_start|>system\nKeep the schema.<|im_end|>\n"
        "<|im_start|>user\nFirst question.<|im_end|>\n"
        '<|im_start|>assistant\n{"status":"first"}<|im_end|>\n'
        "<|im_start|>user\nNext question.\nPreserve my newline.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


@pytest.mark.parametrize("sampling", [{}, {"temperature": None, "top_p": None}])
def test_qwen_raw_preserves_openai_defaults_instead_of_model_defaults(
    monkeypatch, sampling
):
    calls = []

    def forward(url, payload, timeout):
        calls.append((url, payload, timeout))
        return 200, {
            "response": '{"status":"SCHEMA_LOCKED"}',
            "done_reason": "length",
            "prompt_eval_count": 31,
            "eval_count": 9,
            "eval_duration": 250_000_000,
        }

    monkeypatch.setattr(proxy, "forward_json", forward)
    payload = qwen_request(**sampling)
    status, response = proxy.forward_qwen_schema_answer("http://upstream", payload, 123)
    assert len(calls) == 1
    url, raw, timeout = calls[0]
    assert url == "http://upstream/api/generate" and timeout == 123
    assert raw["options"] == {"temperature": 1.0, "top_p": 1.0}
    assert raw["format"] == payload["response_format"]["json_schema"]["schema"]
    assert raw["raw"] is True and raw["think"] is False and raw["stream"] is False
    assert raw["prompt"] == proxy.qwen_answer_prompt(payload["messages"])
    assert status == 200
    assert response["choices"][0]["message"]["content"] == '{"status":"SCHEMA_LOCKED"}'
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["usage"] == {
        "prompt_tokens": 31,
        "completion_tokens": 9,
        "total_tokens": 40,
    }
    assert response["talktopia_model_proxy"]["generation_seconds"] == 0.25


@pytest.mark.parametrize(
    "stop, expected",
    [
        ("END", ["END"]),
        (["END", "STOP"], ["END", "STOP"]),
        (["END", 3], ["END"]),
        (None, None),
    ],
)
def test_qwen_sampling_and_stop_match_ollama_openai_mapping(
    monkeypatch, stop, expected
):
    calls = []

    def forward(url, payload, timeout):
        calls.append(payload)
        return 200, {"response": "{}", "done_reason": "stop"}

    monkeypatch.setattr(proxy, "forward_json", forward)
    payload = qwen_request(
        temperature=0.0,
        top_p=0.8,
        seed=0,
        frequency_penalty=0.25,
        presence_penalty=-0.1,
        max_tokens=777,
        stop=stop,
        response_format={"type": "json_object"},
    )
    status, response = proxy.forward_qwen_schema_answer("http://upstream", payload, 123)
    options = {
        "temperature": 0.0,
        "top_p": 0.8,
        "seed": 0,
        "frequency_penalty": 0.25,
        "presence_penalty": -0.1,
        "num_predict": 777,
    }
    if expected is not None:
        options["stop"] = expected
    assert calls[0]["options"] == options
    assert calls[0]["format"] == "json"
    assert status == 200 and response["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "deepseek-r1:8b"},
        {"model": "qwen3.5:122b-a10b"},
        {"reasoning_effort": "high"},
        {"reasoning": {"effort": "high"}},
        {"reasoning": {}},
        {"stream": True},
        {"tools": [{"type": "function"}]},
        {"tool_choice": "auto"},
        {"logprobs": True},
        {"top_logprobs": 2},
        {"_debug_render_only": True},
        {"response_format": {"type": "text"}},
        {"response_format": None},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": "test"}],
                }
            ]
        },
        {
            "messages": [
                {"role": "developer", "content": "Instruction."},
                {"role": "user", "content": "Hello."},
            ]
        },
        {"messages": [{"role": "assistant", "content": "unfinished"}]},
        {"messages": [{"role": "user", "content": "Hello.", "name": "Alice"}]},
        {
            "messages": [
                {"role": "assistant", "content": "hello", "tool_calls": []},
                {"role": "user", "content": "next"},
            ]
        },
        {"messages": []},
    ],
)
def test_qwen_raw_bypass_is_limited_to_supported_requests(overrides):
    assert not proxy.use_qwen_schema_raw(qwen_request(**overrides))


def test_nested_reasoning_override_matches_ollama_precedence():
    assert proxy.use_qwen_schema_raw(
        qwen_request(reasoning_effort="high", reasoning={"effort": "none"})
    )
    assert not proxy.use_qwen_schema_raw(
        qwen_request(reasoning_effort="none", reasoning={"effort": "low"})
    )


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "json_schema"},
        {"type": "json_schema", "json_schema": {"schema": "not a schema"}},
    ],
)
def test_missing_or_invalid_qwen_schema_never_calls_unconstrained_model(
    monkeypatch, response_format
):
    def no_forward(*args):
        raise AssertionError("Malformed schema must not cause an inference request")

    monkeypatch.setattr(proxy, "forward_json", no_forward)
    with pytest.raises(ValueError, match="schema object"):
        proxy.forward_qwen_schema_answer(
            "http://upstream", qwen_request(response_format=response_format), 123
        )


@pytest.mark.parametrize(
    "native",
    [
        {"response": "{}", "thinking": "unexpected reasoning"},
        {"response": '<think>reasoning</think>{"status":"ok"}'},
    ],
)
def test_qwen_reasoning_is_rejected_without_an_extra_call(monkeypatch, native):
    calls = []

    def forward(*args):
        calls.append(args)
        return 200, native

    monkeypatch.setattr(proxy, "forward_json", forward)
    status, response = proxy.forward_qwen_schema_answer(
        "http://upstream", qwen_request(), 123
    )
    assert status == 502 and "Qwen emitted reasoning" in response["error"]
    assert len(calls) == 1


def test_qwen_native_error_status_is_preserved(monkeypatch):
    monkeypatch.setattr(
        proxy, "forward_json", lambda *args: (503, {"error": "unavailable"})
    )
    assert proxy.forward_qwen_schema_answer("http://upstream", qwen_request(), 123) == (
        503,
        {"error": "unavailable"},
    )


@pytest.mark.parametrize("reason", ["stop", "length", "load", "", None])
def test_qwen_native_finish_reason_is_preserved(monkeypatch, reason):
    monkeypatch.setattr(
        proxy,
        "forward_json",
        lambda *args: (200, {"response": "{}", "done_reason": reason}),
    )
    status, response = proxy.forward_qwen_schema_answer(
        "http://upstream", qwen_request(), 123
    )
    assert status == 200
    assert response["choices"][0]["finish_reason"] == (reason or None)


@pytest.fixture
def proxy_http_route(monkeypatch):
    """Exercise the real proxy handler and urllib forwarding to a local fake Ollama."""
    import json
    import threading
    from contextlib import ExitStack
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    calls = []

    class NativeHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, payload))
            if self.path == "/api/generate":
                result = {
                    "response": '{"status":"SCHEMA_LOCKED"}',
                    "done_reason": "stop",
                    "prompt_eval_count": 23,
                    "eval_count": 7,
                }
            else:
                result = {"choices": [{"message": {"content": "native chat path"}}]}
            proxy.json_response(self, 200, result)

    with ExitStack() as cleanup:
        native = ThreadingHTTPServer(("127.0.0.1", 0), NativeHandler)
        native_thread = threading.Thread(
            target=native.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        native_thread.start()
        cleanup.callback(native_thread.join)
        cleanup.callback(native.server_close)
        cleanup.callback(native.shutdown)
        monkeypatch.setitem(
            proxy.OLLAMA_ENDPOINTS,
            "gpu0",
            {"url": f"http://127.0.0.1:{native.server_port}"},
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), proxy.OllamaProxyHandler)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        cleanup.callback(thread.join)
        cleanup.callback(server.server_close)
        cleanup.callback(server.shutdown)
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", calls


@pytest.mark.parametrize(
    "overrides, expected_path",
    [
        ({}, "/api/generate"),
        ({"reasoning_effort": "high"}, "/v1/chat/completions"),
        ({"reasoning": {"effort": "high"}}, "/v1/chat/completions"),
        (
            {"reasoning_effort": "high", "reasoning": {"effort": "none"}},
            "/api/generate",
        ),
        ({"response_format": {"type": "json_object"}}, "/api/generate"),
        ({"response_format": None}, "/v1/chat/completions"),
        ({"tools": [{"type": "function"}]}, "/v1/chat/completions"),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": "test"}],
                    }
                ]
            },
            "/v1/chat/completions",
        ),
        ({"stream": True}, "/v1/chat/completions"),
        ({"model": "structured-talktopia-agent-deepseek-r1-8b"}, "/api/generate"),
        ({"model": "structured-talktopia-evaluator-glm"}, "/v1/chat/completions"),
    ],
)
def test_http_handler_uses_one_matching_upstream_route(
    proxy_http_route, overrides, expected_path
):
    import httpx

    url, calls = proxy_http_route
    payload = qwen_request(model="structured-talktopia-agent-qwen35-9b")
    payload.pop("reasoning_effort")  # Exercise application of defaults in the handler.
    payload.update(overrides)
    response = httpx.post(url, json=payload, timeout=3)
    assert response.status_code == 200
    assert len(calls) == 1 and calls[0][0] == expected_path
    result = response.json()
    assert result["talktopia_model_proxy"]["requested_model"] == payload["model"]
    if expected_path == "/api/generate":
        assert result["usage"] == {
            "prompt_tokens": 23,
            "completion_tokens": 7,
            "total_tokens": 30,
        }
        assert result["choices"][0]["finish_reason"] == "stop"
        assert calls[0][1]["think"] is False
    else:
        assert result["choices"][0]["message"]["content"] == "native chat path"
        for key in overrides:
            if key != "model":
                assert calls[0][1][key] == overrides[key]


def test_malformed_schema_http_request_returns_400_without_inference(proxy_http_route):
    import httpx

    url, calls = proxy_http_route
    response = httpx.post(
        url, json=qwen_request(response_format={"type": "json_schema"}), timeout=3
    )
    assert response.status_code == 400 and "schema object" in response.json()["error"]
    assert not calls
