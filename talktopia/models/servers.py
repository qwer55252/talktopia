import argparse
import io
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import threading
import urllib.request
import wave
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
    ASR_REPO,
    ASR_REVISION,
    TTS_REPO,
    TTS_REVISION,
    default_ollama_models,
    default_pipeline_models,
)

from talktopia.task_space import database_path


def speech_health() -> dict[str, Any]:
    data = get_json(f"http://{SPEECH_HOST}:{SPEECH_PORT}/health", timeout=2)
    if (
        data.get("service") != "talktopia-speech"
        or not data.get("ready")
        or data.get("database") != str(database_path())
        or data.get("asr_revision") != ASR_REVISION
        or data.get("tts_revision") != TTS_REVISION
        or data.get("gpu") != SPEECH_GPU
    ):
        raise ValueError(
            "Incompatible speech service or database at configured speech port"
        )
    return data


def start_speech(restart: bool = False) -> None:
    if restart:
        stop_pidfile(pid_path("speech"))
    try:
        get_json(f"http://{SPEECH_HOST}:{SPEECH_PORT}/health", timeout=2)
    except Exception:
        try:
            with socket.create_connection((SPEECH_HOST, SPEECH_PORT), timeout=1):
                pass
        except OSError:
            pass
        else:
            raise ValueError(
                f"Another service occupies speech port {SPEECH_HOST}:{SPEECH_PORT}"
            )
    else:
        speech_health()
        print(f"speech: already responds at {SPEECH_BASE_URL}")
        return
    from talktopia.speech_agent import load_voice_registry

    load_voice_registry(database_path())
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with log_path("speech").open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "talktopia.models.servers", "serve-speech"],
            cwd=str(REPO_ROOT),
            stdout=output,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": SPEECH_GPU},
            start_new_session=True,
        )
    pid_path("speech").write_text(str(process.pid))
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Speech server exited ({process.returncode}); see {log_path('speech')}"
            )
        try:
            speech_health()
            print(f"speech: started pid={process.pid} url={SPEECH_BASE_URL}")
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"Speech startup exceeded 300 seconds; see {log_path('speech')}")


def prepare_speech_models() -> None:
    from huggingface_hub import snapshot_download

    for repo, revision in ((ASR_REPO, ASR_REVISION), (TTS_REPO, TTS_REVISION)):
        # Cached files are reused; missing files from interrupted downloads resume.
        print(snapshot_download(repo, revision=revision))


class SpeechBackend:
    """One GPU worker, shared inference lock, lazy per-profile voice prompts."""

    def __init__(self, db: Path):
        from talktopia.speech_agent import load_voice_registry

        self.voices = load_voice_registry(db)
        self.lock = threading.Lock()
        self.prompts: dict[str, Any] = {}
        import numpy as np
        import torch
        from faster_whisper import WhisperModel
        from huggingface_hub import snapshot_download
        from omnivoice import OmniVoice
        from scipy.signal import resample_poly
        from silero_vad import get_speech_timestamps, load_silero_vad

        self.np, self.torch = np, torch
        self.resample_poly = resample_poly
        self.get_speech_timestamps = get_speech_timestamps
        self.vad = load_silero_vad(onnx=True)
        self.asr = WhisperModel(
            snapshot_download(ASR_REPO, revision=ASR_REVISION, local_files_only=True),
            device="cuda",
            device_index=0,
            compute_type="float16",
            local_files_only=True,
        )
        self.tts = OmniVoice.from_pretrained(
            snapshot_download(TTS_REPO, revision=TTS_REVISION, local_files_only=True),
            device_map="cuda:0",
            dtype=torch.float16,
            local_files_only=True,
        )
        if int(self.tts.sampling_rate) != 24000:
            raise ValueError("OmniVoice must return 24000 Hz audio")

    def synthesize(self, text: str, voice_id: str) -> bytes:
        voice = self.voices[voice_id]
        with self.lock:
            if voice_id not in self.prompts:
                self.prompts[voice_id] = self.tts.create_voice_clone_prompt(
                    ref_audio=str(voice["wav_path"]),
                    ref_text=voice["voice_reference_text"],
                )
            audio = self.tts.generate(
                text=text,
                language="English",
                voice_clone_prompt=self.prompts[voice_id],
            )[0]
            samples = self.np.asarray(audio, dtype=self.np.float32).reshape(-1)
            if not len(samples) or not self.np.isfinite(samples).all():
                raise RuntimeError("OmniVoice returned empty or non-finite audio")
            pcm = (self.np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        result = io.BytesIO()
        with wave.open(result, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm)
        return result.getvalue()

    def transcribe(self, audio: bytes) -> str:
        from math import gcd
        from talktopia.speech_agent import read_pcm_wav

        pcm, rate = read_pcm_wav(audio)
        with self.lock:
            samples = (
                self.np.frombuffer(pcm, dtype="<i2").astype(self.np.float32) / 32768
            )
            if rate != 16000:
                divisor = gcd(rate, 16000)
                samples = self.resample_poly(
                    samples, 16000 // divisor, rate // divisor
                ).astype(self.np.float32)
            timestamps = self.get_speech_timestamps(
                self.torch.from_numpy(samples),
                self.vad,
                sampling_rate=16000,
                min_speech_duration_ms=100,
                min_silence_duration_ms=100,
                speech_pad_ms=30,
            )
            if not timestamps:
                return ""
            samples = self.np.concatenate(
                [samples[item["start"] : item["end"]] for item in timestamps]
            )
            segments, _ = self.asr.transcribe(
                samples,
                language="en",
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()


def create_speech_app(backend: Any = None, db: Path | None = None) -> Any:
    # Management commands do not need to import the application or load models.
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response
    from starlette.concurrency import run_in_threadpool
    from talktopia.speech_agent import read_pcm_wav

    db = db or database_path()

    @asynccontextmanager
    async def lifespan(app):
        app.state.backend = backend or await run_in_threadpool(SpeechBackend, db)
        yield

    app = FastAPI(title="Talktopia speech API", lifespan=lifespan)

    def worker():
        value = getattr(app.state, "backend", None)
        if value is None:
            raise HTTPException(503, "Speech models are not ready")
        return value

    @app.get("/health")
    async def health():
        value = worker()
        return dict(
            service="talktopia-speech",
            ready=True,
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
        try:
            return await run_in_threadpool(function, *args)
        except Exception as exc:
            logging.exception("Speech inference failed")
            raise HTTPException(500, "Speech inference failed; see speech.log") from exc

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
    if restart:
        stop()
    names = [*OLLAMA_ENDPOINTS, "proxy", "speech"]
    previous = {name: read_pid(pid_path(name)) for name in names}
    try:
        for name in OLLAMA_ENDPOINTS:
            start_ollama(name, restart=False)
        assert_required_models()
        start_proxy(restart=False)
        start_speech(restart=False)
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
    stop_pidfile(pid_path("speech"))
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
