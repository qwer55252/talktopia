from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import MODEL_ALIASES, OLLAMA_ENDPOINTS, PROXY_HOST, PROXY_PORT

NON_REASONING_DEFAULT_MODELS = {"qwen3.5:9b", "deepseek-r1:8b"}
EVALUATOR_DEFAULT_MAX_TOKENS = 8192
EVALUATOR_MODELS = {
    spec["model"]
    for spec in MODEL_ALIASES.values()
    if spec["role"] == "evaluator" and spec.get("backend", "ollama") == "ollama"
}


def normalize_model(model_name: str) -> tuple[str, str, str]:
    """Return (ollama_model, endpoint_name, requested_model)."""
    requested_model = model_name
    if model_name.startswith("openai/"):
        model_name = model_name.removeprefix("openai/")
    if model_name.startswith("structured-"):
        model_name = model_name.removeprefix("structured-")

    alias = MODEL_ALIASES.get(model_name)
    if alias:
        if alias.get("backend", "ollama") != "ollama":
            raise ValueError(f"{model_name} is not served by the Ollama proxy")
        return alias["model"], alias["endpoint"], requested_model
    return model_name, "gpu0", requested_model


def apply_model_defaults(payload: dict[str, Any], ollama_model: str) -> None:
    """Keep reasoning models from exhausting structured-output token budgets."""
    if ollama_model in NON_REASONING_DEFAULT_MODELS:
        payload.setdefault("reasoning_effort", "none")
    if ollama_model in EVALUATOR_MODELS:
        payload.setdefault("max_tokens", EVALUATOR_DEFAULT_MAX_TOKENS)


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def forward_json(
    url: str, payload: dict[str, Any], timeout: int
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
            return response.status, data
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"error": raw}


def deepseek_answer_prompt(messages: list[dict[str, Any]]) -> str:
    """Close the R1 reasoning prefix before generating the answer tokens."""
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("DeepSeek answer-only requests must end with a user message")
    system = []
    conversation = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("DeepSeek answer-only requests require text messages")
        match message.get("role"):
            case "system":
                system.append(content)
            case "user":
                conversation.append(f"<｜User｜>{content}<｜Assistant｜>")
            case "assistant":
                conversation.append(content + "<｜end▁of▁sentence｜>")
            case _:
                raise ValueError("Unsupported role in DeepSeek answer-only request")
    return (
        "<｜begin▁of▁sentence｜>" + "\n\n".join(system)
        + "".join(conversation) + "<think>\n</think>\n"
    )


def forward_deepseek_answer(
    endpoint: str, payload: dict[str, Any], timeout: int
) -> tuple[int, dict[str, Any]]:
    # R1-0528 ignores think=false through its chat template. Raw generation
    # preserves the model weights and explicitly prefills an empty think block.
    if payload.get("stream"):
        raise ValueError("DeepSeek answer-only streaming is not supported")
    request = {
        "model": payload["model"],
        "prompt": deepseek_answer_prompt(payload.get("messages", [])),
        "raw": True,
        "think": False,
        "stream": False,
        "options": {
            key: payload[key]
            for key in ("temperature", "top_p", "seed") if key in payload
        },
    }
    if payload.get("max_tokens") is not None:
        request["options"]["num_predict"] = payload["max_tokens"]
    response_format = payload.get("response_format") or {}
    if response_format.get("type") == "json_schema":
        request["format"] = response_format["json_schema"]["schema"]
    elif response_format.get("type") == "json_object":
        request["format"] = "json"
    status, response = forward_json(f"{endpoint}/api/generate", request, timeout)
    if status != 200:
        return status, response
    content = response.get("response", "")
    if response.get("thinking") or "<think>" in content or "</think>" in content:
        return 502, {"error": "DeepSeek emitted reasoning despite the empty-think prefix"}
    return 200, {
        "id": f"chatcmpl-{time.time_ns()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "length" if response.get("done_reason") == "length" else "stop"}],
        "usage": {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
        },
        "talktopia_model_proxy": {
            "reasoning_mode": "empty_think_prefill",
            "reasoning_chars": 0,
            "generation_seconds": response.get("eval_duration", 0) / 1e9,
        },
    }


class OllamaProxyHandler(BaseHTTPRequestHandler):
    server_version = "talktopia-ollama-proxy/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def do_GET(self) -> None:
        if self.path in {"/health", "/v1/health"}:
            json_response(self, 200, {"status": "ok", "time": time.time()})
            return
        if self.path == "/v1/models":
            json_response(
                self,
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": f"structured-{name}",
                            "object": "model",
                            "owned_by": "talktopia",
                            "root": spec["model"],
                            "endpoint": spec["endpoint"],
                            "role": spec["role"],
                        }
                        for name, spec in sorted(MODEL_ALIASES.items())
                        if spec.get("backend", "ollama") == "ollama"
                    ],
                },
            )
            return
        json_response(self, 404, {"error": f"unknown path: {self.path}"})

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            json_response(self, 404, {"error": f"unknown path: {self.path}"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            json_response(self, 400, {"error": f"invalid JSON: {exc}"})
            return

        try:
            ollama_model, endpoint_name, requested_model = normalize_model(
                str(payload.get("model", ""))
            )
        except ValueError as exc:
            json_response(self, 400, {"error": str(exc)})
            return
        endpoint = OLLAMA_ENDPOINTS[endpoint_name]
        payload["model"] = ollama_model
        apply_model_defaults(payload, ollama_model)

        try:
            if ollama_model == "deepseek-r1:8b" and payload.get("reasoning_effort") == "none":
                status, response = forward_deepseek_answer(endpoint["url"], payload, timeout=600)
            else:
                status, response = forward_json(
                    f"{endpoint['url']}/v1/chat/completions", payload, timeout=600
                )
        except ValueError as exc:
            json_response(self, 400, {"error": str(exc)})
            return
        if isinstance(response, dict):
            response.setdefault("talktopia_model_proxy", {})
            if isinstance(response["talktopia_model_proxy"], dict):
                response["talktopia_model_proxy"].update(
                    {
                        "requested_model": requested_model,
                        "ollama_model": ollama_model,
                        "endpoint": endpoint_name,
                    }
                )
                if response["talktopia_model_proxy"].get("reasoning_mode"):
                    self.log_message("answer_only %s", json.dumps({
                        **response["talktopia_model_proxy"],
                        "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                    }))
        json_response(self, status, response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible proxy for local Ollama models."
    )
    parser.add_argument("--host", default=PROXY_HOST)
    parser.add_argument("--port", type=int, default=PROXY_PORT)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), OllamaProxyHandler)
    print(
        f"talktopia local model proxy listening on http://{args.host}:{args.port}/v1",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
