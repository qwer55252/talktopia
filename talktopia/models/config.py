from __future__ import annotations

import os
from pathlib import Path


HOME = Path.home()
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = HOME / ".sotopia" / "talktopia_models"

PROXY_HOST = os.environ.get("TALKTOPIA_MODEL_PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("TALKTOPIA_MODEL_PROXY_PORT", "18084"))
PROXY_BASE_URL = f"http://{PROXY_HOST}:{PROXY_PORT}/v1"
SPEECH_HOST = os.environ.get("TALKTOPIA_SPEECH_HOST", "127.0.0.1")
SPEECH_PORT = int(os.environ.get("TALKTOPIA_SPEECH_PORT", "18086"))
SPEECH_BASE_URL = f"http://{SPEECH_HOST}:{SPEECH_PORT}/v1"
SPEECH_GPU = os.environ.get("TALKTOPIA_SPEECH_GPU", "1")
ASR_REPO = "Systran/faster-whisper-small.en"
ASR_REVISION = "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
TTS_REPO = "k2-fsa/OmniVoice"
TTS_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
OLLAMA_NUM_PARALLEL = int(os.environ.get("OLLAMA_NUM_PARALLEL", "2"))
OLLAMA_CONTEXT_LENGTH = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", "32768"))
TTS_BATCH_SIZE = int(os.environ.get("TALKTOPIA_TTS_BATCH_SIZE", "1"))
SPEECH_WORKERS_PER_GPU = int(os.environ.get("TALKTOPIA_SPEECH_WORKERS_PER_GPU", "2"))
if SPEECH_WORKERS_PER_GPU not in (1, 2, 4):
    raise ValueError("TALKTOPIA_SPEECH_WORKERS_PER_GPU must be 1, 2 or 4")
SPEECH_ENDPOINTS = {
    f"gpu{gpu}": {
        "gpu": gpu,
        "host": SPEECH_HOST,
        "port": int(os.environ.get(f"TALKTOPIA_SPEECH_GPU{gpu}_PORT", port)),
    }
    for gpu, port in (("0", "18087"), ("1", "18086"))
}
# Keep the existing configurable single speech endpoint working.
if f"gpu{SPEECH_GPU}" in SPEECH_ENDPOINTS:
    SPEECH_ENDPOINTS[f"gpu{SPEECH_GPU}"].update(port=SPEECH_PORT)

for _endpoint, _spec in list(SPEECH_ENDPOINTS.items()):
    for _slot in range(2, SPEECH_WORKERS_PER_GPU + 1):
        SPEECH_ENDPOINTS[f"{_endpoint}-{_slot}"] = {
            **_spec,
            "port": int(
                os.environ.get(
                    f"TALKTOPIA_SPEECH_GPU{_spec['gpu']}_WORKER{_slot}_PORT",
                    _spec["port"] + 2 * (_slot - 1),
                )
            ),
        }
if len({(spec["host"], spec["port"]) for spec in SPEECH_ENDPOINTS.values()}) != len(
    SPEECH_ENDPOINTS
):
    raise ValueError("Speech worker ports must be distinct")


def speech_worker_keys(endpoint: str | None = None) -> list[str]:
    return [
        key
        for key in SPEECH_ENDPOINTS
        if endpoint is None or key == endpoint or key.startswith(endpoint + "-")
    ]


def speech_url(endpoint: str) -> str:
    spec = SPEECH_ENDPOINTS[endpoint]
    return f"http://{spec['host']}:{spec['port']}/v1"


OLLAMA_BIN = Path(
    os.environ.get("TALKTOPIA_OLLAMA_BIN", str(HOME / ".local/bin/ollama"))
)
OLLAMA_ENDPOINTS = {
    "gpu0": {
        "gpu": "0",
        "host": os.environ.get("TALKTOPIA_GPU0_HOST", "127.0.0.1:11438"),
        "url": os.environ.get("TALKTOPIA_GPU0_URL", "http://127.0.0.1:11438"),
    },
    "gpu1": {
        "gpu": "1",
        "host": os.environ.get("TALKTOPIA_GPU1_HOST", "127.0.0.1:11439"),
        "url": os.environ.get("TALKTOPIA_GPU1_URL", "http://127.0.0.1:11439"),
    },
}

VLLM_ENDPOINTS = {
    "mistral-small4-nvfp4": {
        "url": os.environ.get(
            "TALKTOPIA_MISTRAL_SMALL4_VLLM_URL", "http://127.0.0.1:18085/v1"
        ),
        "served_model": ("structured-talktopia-evaluator-mistral-small4-119b-nvfp4"),
    }
}

# Supported local models. Optional aliases can be pulled independently; startup
# only requires the models referenced by DEFAULT_PIPELINE_ALIASES.
MODEL_ALIASES = {
    "talktopia-agent-qwen35-9b": {
        "model": "qwen3.5:9b",
        "endpoint": "gpu0",
        "role": "agent",
    },
    "talktopia-agent-ministral3-8b": {
        "model": "ministral-3:8b",
        "endpoint": "gpu1",
        "role": "agent",
    },
    "talktopia-agent-deepseek-r1-8b": {
        "model": "deepseek-r1:8b",
        "endpoint": "gpu0",
        "role": "agent",
    },
    "talktopia-agent-llama31-8b": {
        "model": "llama3.1:8b",
        "endpoint": "gpu0",
        "role": "agent",
    },
    "talktopia-agent-gemma3-4b": {
        "model": "gemma3:4b",
        "endpoint": "gpu1",
        "role": "agent",
    },
    "talktopia-agent-fast": {
        "model": "qwen2.5:7b",
        "endpoint": "gpu0",
        "role": "agent",
    },
    "talktopia-agent-strong": {
        "model": "qwen3:30b-a3b",
        "endpoint": "gpu1",
        "role": "agent",
    },
    "talktopia-evaluator": {
        "model": "qwen2.5:32b",
        "endpoint": "gpu0",
        "role": "evaluator",
    },
    "talktopia-evaluator-glm": {
        "model": "glm-4.7-flash:latest",
        "endpoint": "gpu0",
        "role": "evaluator",
    },
    "talktopia-evaluator-qwen35-122b-a10b": {
        "model": "qwen3.5:122b-a10b",
        "endpoint": "gpu0",
        "role": "evaluator",
        "backend": "ollama",
    },
    "talktopia-evaluator-mistral-small4-119b-nvfp4": {
        "model": "mistralai/Mistral-Small-4-119B-2603-NVFP4",
        "endpoint": "mistral-small4-nvfp4",
        "role": "evaluator",
        "backend": "vllm",
    },
    "talktopia-agent-mistral": {
        "model": "mistral:7b",
        "endpoint": "gpu1",
        "role": "agent",
    },
}

DEFAULT_PIPELINE_ALIASES = {
    "env": "talktopia-evaluator-glm",
    "agent1": "talktopia-agent-qwen35-9b",
    "agent2": "talktopia-agent-ministral3-8b",
    "evaluator": "talktopia-evaluator-glm",
}

# Explicit GPU aliases let separate jobs use the same model without sharing a GPU.
for _alias, _spec in list(MODEL_ALIASES.items()):
    if _spec.get("backend", "ollama") == "ollama":
        for _endpoint in OLLAMA_ENDPOINTS:
            MODEL_ALIASES[f"{_alias}-{_endpoint}"] = {**_spec, "endpoint": _endpoint}


def local_alias(model: str) -> str:
    alias = model.split("@", 1)[0].removeprefix("custom/").removeprefix("structured-")
    if (
        alias not in MODEL_ALIASES
        or MODEL_ALIASES[alias].get("backend", "ollama") != "ollama"
    ):
        raise ValueError(f"Expected a configured local Ollama alias: {model}")
    return alias


def model_on_gpu(model: str, endpoint: str) -> str:
    alias = local_alias(model)
    for suffix in OLLAMA_ENDPOINTS:
        alias = alias.removesuffix(f"-{suffix}")
    return custom_model_name(f"{alias}-{endpoint}")


def custom_model_name(alias: str) -> str:
    spec = MODEL_ALIASES[alias]
    if spec.get("backend", "ollama") == "vllm":
        endpoint = VLLM_ENDPOINTS[spec["endpoint"]]
        return f"custom/{endpoint['served_model']}@{endpoint['url']}"
    # The custom/structured prefix makes SOTOPIA request JSON schema output.
    # The local proxy strips "structured-" before routing to the real Ollama tag.
    return f"custom/structured-{alias}@{PROXY_BASE_URL}"


def default_pipeline_models() -> dict[str, str]:
    return {
        key: custom_model_name(alias) for key, alias in DEFAULT_PIPELINE_ALIASES.items()
    }


def default_ollama_models() -> set[str]:
    return {
        MODEL_ALIASES[alias]["model"] for alias in DEFAULT_PIPELINE_ALIASES.values()
    }
