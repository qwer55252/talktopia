from __future__ import annotations

import os
from pathlib import Path


HOME = Path.home()
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = HOME / ".sotopia" / "talktopia_models"

PROXY_HOST = os.environ.get("TALKTOPIA_MODEL_PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("TALKTOPIA_MODEL_PROXY_PORT", "18084"))
PROXY_BASE_URL = f"http://{PROXY_HOST}:{PROXY_PORT}/v1"
OLLAMA_NUM_PARALLEL = int(os.environ.get("OLLAMA_NUM_PARALLEL", "2"))
OLLAMA_CONTEXT_LENGTH = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", "32768"))

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
