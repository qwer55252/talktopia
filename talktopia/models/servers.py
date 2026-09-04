from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from .config import (
    MODEL_ALIASES,
    OLLAMA_BIN,
    OLLAMA_CONTEXT_LENGTH,
    OLLAMA_ENDPOINTS,
    OLLAMA_NUM_PARALLEL,
    PROXY_BASE_URL,
    PROXY_HOST,
    PROXY_PORT,
    REPO_ROOT,
    RUNTIME_DIR,
    VLLM_ENDPOINTS,
    default_ollama_models,
    default_pipeline_models,
)


def pid_path(name: str) -> Path:
    return RUNTIME_DIR / f"{name}.pid"


def log_path(name: str) -> Path:
    return RUNTIME_DIR / f"{name}.log"


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_pidfile(path: Path) -> None:
    pid = read_pid(path)
    if not is_alive(pid):
        path.unlink(missing_ok=True)
        return
    assert pid is not None
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 10
    while time.time() < deadline and is_alive(pid):
        time.sleep(0.2)
    if is_alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            os.kill(pid, signal.SIGKILL)
    path.unlink(missing_ok=True)


def get_json(url: str, timeout: float = 3) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def wait_json(url: str, timeout: int = 120) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: Any = None
    while time.time() < deadline:
        try:
            return get_json(url)
        except Exception as exc:
            last = exc
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {url}: {last}")


def proxy_is_compatible() -> bool:
    try:
        data = get_json(f"{PROXY_BASE_URL}/models")
    except Exception:
        return False
    model_ids = {item.get("id") for item in data.get("data", [])}
    expected = {
        f"structured-{alias}"
        for alias, spec in MODEL_ALIASES.items()
        if spec.get("backend", "ollama") == "ollama"
    }
    return expected.issubset(model_ids)


def start_ollama(name: str, restart: bool) -> None:
    endpoint = OLLAMA_ENDPOINTS[name]
    if restart:
        stop_pidfile(pid_path(name))

    try:
        wait_json(f"{endpoint['url']}/api/tags", timeout=2)
        print(f"{name}: Ollama already responds at {endpoint['url']}")
        return
    except Exception:
        pass

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": endpoint["gpu"],
        "OLLAMA_HOST": endpoint["host"],
        "OLLAMA_KEEP_ALIVE": os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
        "OLLAMA_CONTEXT_LENGTH": str(OLLAMA_CONTEXT_LENGTH),
        "OLLAMA_NUM_PARALLEL": str(OLLAMA_NUM_PARALLEL),
    }
    with log_path(name).open("w") as output:
        process = subprocess.Popen(
            [str(OLLAMA_BIN), "serve"],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    pid_path(name).write_text(str(process.pid))
    wait_json(f"{endpoint['url']}/api/tags", timeout=120)
    print(f"{name}: started Ollama pid={process.pid} url={endpoint['url']}")


def start_proxy(restart: bool) -> None:
    if restart:
        stop_pidfile(pid_path("proxy"))

    try:
        wait_json(f"http://{PROXY_HOST}:{PROXY_PORT}/health", timeout=2)
    except Exception:
        pass
    else:
        if not proxy_is_compatible():
            raise SystemExit(
                f"Incompatible model proxy already occupies {PROXY_HOST}:{PROXY_PORT}. "
                "Stop it before starting Talktopia."
            )
        print(f"proxy: already responds at {PROXY_BASE_URL}")
        return

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with log_path("proxy").open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "talktopia.models.ollama_proxy",
                "--host",
                PROXY_HOST,
                "--port",
                str(PROXY_PORT),
            ],
            cwd=str(REPO_ROOT),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path("proxy").write_text(str(process.pid))
    wait_json(f"http://{PROXY_HOST}:{PROXY_PORT}/health", timeout=30)
    print(f"proxy: started pid={process.pid} url={PROXY_BASE_URL}")


def available_model_names() -> set[str]:
    names: set[str] = set()
    for endpoint in OLLAMA_ENDPOINTS.values():
        try:
            data = get_json(f"{endpoint['url']}/api/tags")
        except Exception:
            continue
        for item in data.get("models", []):
            if "name" in item:
                names.add(item["name"])
            if "model" in item:
                names.add(item["model"])
    return names


def assert_required_models() -> None:
    available = available_model_names()
    missing = sorted(default_ollama_models() - available)
    if missing:
        raise SystemExit(
            "Missing default local Ollama models: "
            + ", ".join(missing)
            + ". Install them with `ollama pull <model>` or change "
            "DEFAULT_PIPELINE_ALIASES in talktopia/models/config.py."
        )


def start(restart: bool = False) -> None:
    for name in OLLAMA_ENDPOINTS:
        start_ollama(name, restart=restart)
    assert_required_models()
    start_proxy(restart=restart)
    print(
        json.dumps(
            {"proxy": PROXY_BASE_URL, "pipeline_models": default_pipeline_models()},
            indent=2,
        )
    )


def stop() -> None:
    stop_pidfile(pid_path("proxy"))
    for name in OLLAMA_ENDPOINTS:
        stop_pidfile(pid_path(name))
    print("talktopia local model API stopped")


def status() -> None:
    def responds(url: str) -> bool:
        try:
            get_json(url, timeout=1)
            return True
        except Exception:
            return False

    payload = {
        "proxy": {
            "url": PROXY_BASE_URL,
            "pid": read_pid(pid_path("proxy")),
            "alive": is_alive(read_pid(pid_path("proxy"))),
            "responds": responds(f"http://{PROXY_HOST}:{PROXY_PORT}/health"),
            "compatible": proxy_is_compatible(),
        },
        "ollama": {},
        "vllm": {},
        "aliases": MODEL_ALIASES,
        "available_models": sorted(available_model_names()),
    }
    for name, endpoint in OLLAMA_ENDPOINTS.items():
        payload["ollama"][name] = {
            "url": endpoint["url"],
            "gpu": endpoint["gpu"],
            "pid": read_pid(pid_path(name)),
            "alive": is_alive(read_pid(pid_path(name))),
            "responds": responds(f"{endpoint['url']}/api/tags"),
        }
    for name, endpoint in VLLM_ENDPOINTS.items():
        payload["vllm"][name] = {
            "url": endpoint["url"],
            "served_model": endpoint["served_model"],
            "managed": False,
            "responds": responds(f"{endpoint['url']}/models"),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage talktopia local model API.")
    parser.add_argument("command", choices=["start", "stop", "restart", "status"])
    args = parser.parse_args()

    if args.command == "start":
        start(restart=False)
    elif args.command == "restart":
        start(restart=True)
    elif args.command == "stop":
        stop()
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
