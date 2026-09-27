"""HTTP interface for the local, read-only results viewer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt

from .catalog import evaluation_rows, read_config, simulation_rows
from .evaluations import DIMENSIONS, evaluation_id, read_evaluations
from .files import ResultError, artifact_path, available, optional_json
from .playback import transcript
from .profiles import load_profiles


STATIC = Path(__file__).parent / "static"
MARKDOWN = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable("table")
MARKDOWN.disable("image")


def render_markdown(text: str) -> str:
    tokens = MARKDOWN.parse(text)
    # markdown-it emits inline styles for table alignment. Use CSS classes so
    # reports work with the same strict style policy as the rest of the page.
    alignments = {"text-align:left": "align-left", "text-align:center": "align-center",
                  "text-align:right": "align-right"}
    for token in tokens:
        if token.type in ("th_open", "td_open") and token.attrGet("style"):
            style = token.attrs.pop("style")
            token.attrSet("class", alignments.get(style, "align-left"))
    return MARKDOWN.renderer.render(tokens, MARKDOWN.options, {})


def create_app(config_path: Path) -> FastAPI:
    config_path = config_path.expanduser().resolve()
    app = FastAPI(title="Talktopia Results", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(ResultError)
    async def invalid_result(request, error):
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.middleware("http")
    async def response_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self'; media-src 'self'; connect-src 'self'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
        )
        # Configs and in-progress results are deliberately re-read on refresh.
        if request.url.path.startswith("/api/") and not request.url.path.endswith("/audio"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def find_run(run_id):
        for run in read_config(config_path):
            if run.id == run_id:
                return run
        raise HTTPException(404, "등록된 실행이 없습니다.")

    def find_episode(run_id, episode_id):
        run = find_run(run_id)
        rows, warnings = simulation_rows(run)
        if episode_id not in rows:
            raise HTTPException(404, "에피소드가 없습니다.")
        return run, rows[episode_id], warnings

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/runs")
    def runs():
        return [{"id": run.id, "label": run.label} for run in read_config(config_path)]

    @app.get("/api/runs/{run_id}/episodes")
    def episodes(run_id: str):
        run = find_run(run_id)
        rows, warnings = simulation_rows(run)
        return {
            "episodes": [
                {
                    "id": row["episode_id"],
                    "codename": row.get("codename", ""),
                    "agent_names": row.get("agent_names", []),
                    "status": row["status"],
                }
                for row in rows.values()
            ],
            "warnings": warnings,
        }

    @app.get("/api/runs/{run_id}/episodes/{episode_id}")
    def episode(run_id: str, episode_id: str):
        run, row, warnings = find_episode(run_id, episode_id)
        source, source_hash = {}, None
        try:
            raw = artifact_path(run.simulation_dir, row.get("original"), ".json").read_bytes()
            source = json.loads(raw)
            if not isinstance(source, dict):
                raise ValueError()
            source_hash = hashlib.sha256(raw).hexdigest()
        except (ResultError, OSError, ValueError) as error:
            source = {}
            if row["status"] == "completed":
                warnings.append("시뮬레이션 원본을 읽을 수 없습니다. 사용 가능한 기록만 표시합니다.")
        config = optional_json(run.simulation_dir / "run_config.json", dict, warnings)
        agents = source.get("agents") or row.get("agent_ids") or []
        environment = source.get("environment") or row.get("env_id") or ""
        profiles = load_profiles(config, agents, environment, run.profiles_dir)
        perspectives = []
        for message in (source.get("messages") or [[]])[0][:2]:
            if isinstance(message, list) and len(message) == 3 and message[0] == "Environment":
                perspectives.append({"name": message[1], "text": message[2]})
        names = [entry["name"] for entry in perspectives]
        if len(names) != 2:
            names = row.get("agent_names") or ["Agent 1", "Agent 2"]
        playback = transcript(run.simulation_dir, row, source)
        return {
            "id": episode_id,
            "run_id": run_id,
            "codename": row.get("codename", ""),
            "status": row["status"],
            "error": row.get("error") if row["status"] == "failed" else None,
            "agent_ids": agents,
            "agent_names": names,
            "models": (source.get("models") or [])[1:3],
            "profiles": profiles,
            "perspectives": perspectives,
            "playback": playback,
            "evaluations": read_evaluations(run, row, source, source_hash),
            "dimensions": DIMENSIONS,
            "report_available": available(run.simulation_dir, row.get("readable"), ".md"),
            "warnings": warnings,
        }

    @app.api_route("/api/runs/{run_id}/episodes/{episode_id}/audio", methods=["GET", "HEAD"])
    def audio(run_id: str, episode_id: str):
        run, row, _ = find_episode(run_id, episode_id)
        path = artifact_path(run.simulation_dir, row.get("conversation_audio"), ".wav")
        if not path.is_file():
            raise HTTPException(404, "음성 파일이 없습니다.")
        return FileResponse(path, media_type="audio/wav", content_disposition_type="inline")

    @app.get("/api/runs/{run_id}/episodes/{episode_id}/reports/{report_id}")
    def report(run_id: str, episode_id: str, report_id: str):
        run, row, _ = find_episode(run_id, episode_id)
        directory = run.simulation_dir
        if report_id != "simulation":
            directory = next(
                (path for path in run.evaluation_dirs if evaluation_id(path) == report_id), None
            )
            if directory is None:
                raise HTTPException(404, "등록된 평가가 없습니다.")
            rows, _ = evaluation_rows(directory)
            row = rows.get(episode_id, {})
        path = artifact_path(directory, row.get("readable"), ".md")
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, ValueError) as error:
            raise HTTPException(404, "보고서를 읽을 수 없습니다.") from error
        return {"markdown": text, "html": render_markdown(text)}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
