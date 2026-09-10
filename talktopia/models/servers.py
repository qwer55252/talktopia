import argparse
import json
import logging
import os
import signal
import socket
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
    SPEECH_BASE_URL,
    SPEECH_GPU,
    SPEECH_HOST,
    SPEECH_PORT,
    SPEECH_ENDPOINTS,
    TTS_BATCH_SIZE,
    speech_url,
    ASR_REPO,
    ASR_REVISION,
    TTS_REPO,
    TTS_REVISION,
    default_ollama_models,
    default_pipeline_models,
    local_alias,
)

from talktopia.task_space import database_path


def speech_spec(endpoint: str | None = None) -> dict[str, Any]:
    endpoint = endpoint or f"gpu{SPEECH_GPU}"
    spec = SPEECH_ENDPOINTS[endpoint]
    name = "speech" if endpoint == f"gpu{SPEECH_GPU}" else f"speech-{endpoint}"
    return {**spec, "name": name, "worker": endpoint}


def speech_health(
    endpoint: str | None = None, *, require_ready: bool = True
) -> dict[str, Any]:
    spec = speech_spec(endpoint)
    data = get_json(f"http://{spec['host']}:{spec['port']}/health", timeout=2)
    if (
        data.get("service") != "talktopia-speech"
        or data.get("database") != str(database_path())
        or data.get("asr_revision") != ASR_REVISION
        or data.get("tts_revision") != TTS_REVISION
        or data.get("gpu") != spec["gpu"]
        or data.get("tts_batch_size") != TTS_BATCH_SIZE
        or data.get("worker") != spec["worker"]
    ):
        raise ValueError(
            "Incompatible speech service or database at configured speech port"
        )
    if require_ready and not data.get("ready"):
        raise RuntimeError(f"{spec['worker']}: speech backend is not ready")
    return data


def start_speech(restart: bool = False, endpoint: str | None = None) -> None:
    spec = speech_spec(endpoint)
    host, port, name = spec["host"], spec["port"], spec["name"]
    url = f"http://{host}:{port}/v1"
    if restart:
        stop_pidfile(pid_path(name))
    try:
        get_json(f"http://{host}:{port}/health", timeout=2)
    except Exception:
        try:
            with socket.create_connection((host, port), timeout=1):
                pass
        except OSError:
            pass
        else:
            raise ValueError(f"Another service occupies speech port {host}:{port}")
    else:
        speech_health(endpoint)
        print(f"{name}: already responds at {url}")
        return
    from talktopia.speech_agent import load_voice_registry

    load_voice_registry(database_path())
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with log_path(name).open("a") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "talktopia.models.servers", "serve-speech"],
            cwd=str(REPO_ROOT),
            stdout=output,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": spec["gpu"],
                "TALKTOPIA_SPEECH_GPU": spec["gpu"],
                "TALKTOPIA_SPEECH_HOST": host,
                "TALKTOPIA_SPEECH_PORT": str(port),
                "TALKTOPIA_SPEECH_WORKER": spec["worker"],
            },
            start_new_session=True,
        )
    pid_path(name).write_text(str(process.pid))
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Speech server exited ({process.returncode}); see {log_path(name)}"
                )
            try:
                speech_health(endpoint)
                print(f"{name}: started pid={process.pid} url={url}")
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"Speech startup exceeded 300 seconds; see {log_path(name)}")
    except BaseException:
        if read_pid(pid_path(name)) == process.pid:
            stop_pidfile(pid_path(name))
        raise


def ensure_speech_ready(endpoint: str) -> bool:
    """Called only while the pipeline holds this worker's episode lock."""
    try:
        health = speech_health(endpoint, require_ready=False)
        # A just-finished HTTP call may precede the batch's final bookkeeping.
        if health.get("ready") and not health.get("idle", False):
            time.sleep(0.1)
            health = speech_health(endpoint, require_ready=False)
        if health.get("ready") and health.get("idle", False):
            return False
    except ValueError:
        # Never replace a different application or database on this port.
        raise
    except (OSError, RuntimeError):
        pass
    start_speech(restart=True, endpoint=endpoint)
    return True


def prepare_speech_models() -> None:
    from huggingface_hub import snapshot_download

    for repo, revision in ((ASR_REPO, ASR_REVISION), (TTS_REPO, TTS_REVISION)):
        # Cached files are reused; missing files from interrupted downloads resume.
        print(snapshot_download(repo, revision=revision))


def create_speech_app(backend: Any = None, db: Path | None = None) -> Any:
    # Management commands do not need to import the application or load models.
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response
    from starlette.concurrency import run_in_threadpool
    from talktopia.speech_agent import read_pcm_wav, SpeechBackend

    db = db or database_path()

    @asynccontextmanager
    async def lifespan(app):
        app.state.backend = backend or await run_in_threadpool(SpeechBackend, db)
        try:
            yield
        finally:
            if hasattr(app.state.backend, "close"):
                await run_in_threadpool(app.state.backend.close)

    app = FastAPI(title="Talktopia speech API", lifespan=lifespan)
    app.state.inference_requests = 0

    def worker():
        value = getattr(app.state, "backend", None)
        if value is None:
            raise HTTPException(503, "Speech models are not ready")
        return value

    @app.get("/health")
    async def health():
        value = worker()
        queued = value.requests.qsize() if hasattr(value, "requests") else 0
        active = getattr(value, "tts_active_requests", 0)
        return dict(
            service="talktopia-speech",
            ready=not getattr(value, "fatal_error", None)
            and not getattr(value, "closed", False),
            idle=app.state.inference_requests == 0 and active == 0 and queued == 0,
            active_requests=app.state.inference_requests,
            tts_active_requests=active,
            worker=os.environ.get("TALKTOPIA_SPEECH_WORKER", f"gpu{SPEECH_GPU}"),
            error=getattr(value, "fatal_error", None),
            tts_batch_size=getattr(value, "batch_size", TTS_BATCH_SIZE),
            tts_metrics=dict(getattr(value, "metrics", {})),
            tts_queue_size=queued,
            database=str(db),
            gpu=SPEECH_GPU,
            voices=len(value.voices),
            asr_model=ASR_REPO,
            asr_revision=ASR_REVISION,
            tts_model=TTS_REPO,
            tts_revision=TTS_REVISION,
        )

    @app.get("/v1/models")
    async def models():
        worker()
        return {
            "object": "list",
            "data": [
                dict(id=name, object="model", created=0, owned_by="talktopia")
                for name in ("whisper-1", "tts-1")
            ],
        }

    async def inference(function, *args):
        app.state.inference_requests += 1
        try:
            return await run_in_threadpool(function, *args)
        except TimeoutError as exc:
            logging.exception("Speech inference timed out")
            raise HTTPException(
                504,
                "Speech inference timed out; see speech.log",
                headers={"x-should-retry": "false"},
            ) from exc
        except Exception as exc:
            if "out of memory" in str(exc).lower() or "cuda error" in str(exc).lower():
                worker().fatal_error = f"Speech GPU failed: {type(exc).__name__}"
            logging.exception("Speech inference failed")
            raise HTTPException(500, "Speech inference failed; see speech.log") from exc
        finally:
            app.state.inference_requests -= 1

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        value = worker()
        try:
            data = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "Expected a JSON object") from exc
        if not isinstance(data, dict):
            raise HTTPException(400, "Expected a JSON object")
        if (
            data.get("model") != "tts-1"
            or data.get("response_format", "wav") != "wav"
            or data.get("speed", 1) != 1
        ):
            raise HTTPException(
                400, "Supported: model=tts-1, response_format=wav, speed=1"
            )
        voice, text = data.get("voice"), data.get("input")
        if not isinstance(voice, str) or voice not in value.voices:
            raise HTTPException(400, "Unknown profile voice_id")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(400, "input must contain text")
        audio = await inference(value.synthesize, text.strip(), voice)
        return Response(content=audio, media_type="audio/wav")

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request):
        value = worker()
        async with request.form() as form:
            if (
                form.get("model") != "whisper-1"
                or form.get("language", "en") != "en"
                or form.get("response_format", "json") != "json"
                or "prompt" in form
            ):
                raise HTTPException(
                    400,
                    "Supported: model=whisper-1, language=en, response_format=json, no prompt",
                )
            upload = form.get("file")
            if not hasattr(upload, "read"):
                raise HTTPException(400, "file must be a WAV upload")
            audio = await upload.read(20 * 1024 * 1024 + 1)
        if len(audio) > 20 * 1024 * 1024:
            raise HTTPException(413, "WAV upload exceeds 20 MiB")
        try:
            read_pcm_wav(audio)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"text": await inference(value.transcribe, audio)}

    return app


def serve_speech() -> None:
    # Set CUDA library lookup before importing torch/ctranslate2 or loading models.
    import site

    libraries = [
        str(path)
        for root in site.getsitepackages()
        for path in Path(root).glob("nvidia/*/lib")
    ]
    if os.environ.get("TALKTOPIA_SPEECH_CUDA_READY") != "1":
        env = {
            **os.environ,
            "TALKTOPIA_SPEECH_CUDA_READY": "1",
            "CUDA_VISIBLE_DEVICES": SPEECH_GPU,
            "LD_LIBRARY_PATH": ":".join(
                libraries + [os.environ.get("LD_LIBRARY_PATH", "")]
            ),
        }
        os.execve(
            sys.executable,
            [sys.executable, "-m", "talktopia.models.servers", "serve-speech"],
            env,
        )
    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    uvicorn.run(create_speech_app(), host=SPEECH_HOST, port=SPEECH_PORT)


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
    if pid is None or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state not in {"Z", "X"}
    except OSError:
        return False


def stop_pidfile(path: Path) -> None:
    pid = read_pid(path)
    if not is_alive(pid):
        path.unlink(missing_ok=True)
        return
    assert pid is not None
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        path.unlink(missing_ok=True)
        return
    if path.stem.startswith("speech"):
        owned = b"talktopia.models.servers" in command and b"serve-speech" in command
    elif path.stem == "proxy":
        owned = b"talktopia.models.ollama_proxy" in command
    else:
        owned = (
            bool(command)
            and Path(os.fsdecode(command[0])).name == "ollama"
            and b"serve" in command
        )
    if not owned:
        raise ValueError(
            f"Refusing to stop PID {pid}: command does not match managed service {path.stem}"
        )
    if path.stem.startswith("speech"):
        specs = [speech_spec(key) for key in SPEECH_ENDPOINTS]
        spec = next((spec for spec in specs if spec["name"] == path.stem), None)
        try:
            environment = dict(
                entry.split(b"=", 1)
                for entry in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
                if b"=" in entry
            )
        except FileNotFoundError:
            path.unlink(missing_ok=True)
            return
        expected = (
            {}
            if spec is None
            else {
                "TALKTOPIA_SPEECH_GPU": str(spec["gpu"]),
                "TALKTOPIA_SPEECH_HOST": spec["host"],
                "TALKTOPIA_SPEECH_PORT": str(spec["port"]),
            }
        )
        if not expected or any(
            environment.get(key.encode()) != value.encode()
            for key, value in expected.items()
        ):
            raise ValueError(
                f"Refusing to stop PID {pid}: speech worker identity differs for {path.stem}"
            )
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


def release_worker_models(args, *, keep_models: list[str]) -> None:
    """Unload only this worker's selected models which the next phase does not use."""
    if not getattr(args, "worker_gpu", None):
        return

    all_models = {
        MODEL_ALIASES[local_alias(value)]["model"]
        for value in (
            args.agent1_model,
            args.agent2_model,
            args.evaluator_model,
            args.bad_output_process_model,
        )
    }
    keep = {MODEL_ALIASES[local_alias(value)]["model"] for value in keep_models}
    url = OLLAMA_ENDPOINTS[args.worker_gpu]["url"]
    loaded = {item["name"] for item in get_json(f"{url}/api/ps").get("models", [])}
    for name in sorted((loaded & all_models) - keep):
        request = urllib.request.Request(
            f"{url}/api/generate",
            data=json.dumps(dict(model=name, keep_alive=0)).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()


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
    except Exception:
        pass
    else:
        validate_ollama_settings(name)
        print(f"{name}: Ollama already responds at {endpoint['url']}")
        return

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


def validate_ollama_settings(name: str) -> None:
    """Never label an already-running server with settings it did not start with."""
    pid = read_pid(pid_path(name))
    try:
        data = Path(f"/proc/{pid}/environ").read_bytes()
        environment = dict(
            entry.split(b"=", 1) for entry in data.split(b"\0") if b"=" in entry
        )
    except OSError as exc:
        raise ValueError(
            f"{name}: responding Ollama has no readable managed PID; cannot verify its settings"
        ) from exc
    expected = {
        "OLLAMA_NUM_PARALLEL": str(OLLAMA_NUM_PARALLEL),
        "OLLAMA_CONTEXT_LENGTH": str(OLLAMA_CONTEXT_LENGTH),
        "CUDA_VISIBLE_DEVICES": OLLAMA_ENDPOINTS[name]["gpu"],
        "OLLAMA_HOST": OLLAMA_ENDPOINTS[name]["host"],
    }
    mismatched = [
        key
        for key, value in expected.items()
        if environment.get(key.encode()) != value.encode()
    ]
    if mismatched:
        raise ValueError(
            f"{name}: running settings differ for {', '.join(mismatched)}; restart the managed servers with the intended settings"
        )


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
    if restart:
        stop()
    names = [
        *OLLAMA_ENDPOINTS,
        "proxy",
        *[speech_spec(key)["name"] for key in SPEECH_ENDPOINTS],
    ]
    previous = {name: read_pid(pid_path(name)) for name in names}
    try:
        for name in OLLAMA_ENDPOINTS:
            start_ollama(name, restart=False)
        assert_required_models()
        start_proxy(restart=False)
        for endpoint in SPEECH_ENDPOINTS:
            start_speech(restart=False, endpoint=endpoint)
    except BaseException:
        for name in reversed(names):
            if read_pid(pid_path(name)) != previous[name]:
                stop_pidfile(pid_path(name))
        raise
    print(
        json.dumps(
            {
                "proxy": PROXY_BASE_URL,
                "speech": SPEECH_BASE_URL,
                "pipeline_models": default_pipeline_models(),
            },
            indent=2,
        )
    )


def stop() -> None:
    for endpoint in SPEECH_ENDPOINTS:
        stop_pidfile(pid_path(speech_spec(endpoint)["name"]))
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
        "speech_workers": {},
        "speech": {
            "url": SPEECH_BASE_URL,
            "gpu": SPEECH_GPU,
            "pid": read_pid(pid_path("speech")),
            "alive": is_alive(read_pid(pid_path("speech"))),
            "responds": responds(f"http://{SPEECH_HOST}:{SPEECH_PORT}/health"),
        },
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
    try:
        payload["speech"].update(compatible=True, health=speech_health())
    except Exception:
        payload["speech"]["compatible"] = False
    for name, endpoint in OLLAMA_ENDPOINTS.items():
        payload["ollama"][name] = {
            "url": endpoint["url"],
            "gpu": endpoint["gpu"],
            "pid": read_pid(pid_path(name)),
            "alive": is_alive(read_pid(pid_path(name))),
            "responds": responds(f"{endpoint['url']}/api/tags"),
        }
    for name in SPEECH_ENDPOINTS:
        spec = speech_spec(name)
        row = dict(
            url=speech_url(name), gpu=spec["gpu"], pid=read_pid(pid_path(spec["name"]))
        )
        try:
            row.update(compatible=True, health=speech_health(name))
        except Exception as exc:
            row.update(compatible=False, error=str(exc))
        payload["speech_workers"][name] = row
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
    parser.add_argument(
        "command",
        choices=[
            "start",
            "stop",
            "restart",
            "status",
            "serve-speech",
            "prepare-speech-models",
        ],
    )
    args = parser.parse_args()

    if args.command == "start":
        start(restart=False)
    elif args.command == "restart":
        start(restart=True)
    elif args.command == "stop":
        stop()
    elif args.command == "status":
        status()
    elif args.command == "serve-speech":
        serve_speech()
    elif args.command == "prepare-speech-models":
        prepare_speech_models()


if __name__ == "__main__":
    main()
