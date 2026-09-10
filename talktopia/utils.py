"""Shared run files, input checks, and bounded episode workers."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from talktopia.models.config import (
    REPO_ROOT,
    SPEECH_WORKERS_PER_GPU,
    SPEECH_ENDPOINTS,
    TTS_BATCH_SIZE,
)

EPISODE_IDENTITY = ("episode_id", "env_id", "agent_ids", "combo_id")


def safe_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for name in (
        "CUSTOM_API_KEY",
        "OPENAI_API_KEY",
        "TALKTOPIA_ASR_API_KEY",
        "TALKTOPIA_TTS_API_KEY",
    ):
        secret = os.environ.get(name, "")
        if secret and secret != "EMPTY":
            message = message.replace(secret, "[redacted]")
    return message


def write_json(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@contextmanager
def lock_run(run_dir: Path):
    with (run_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another process is already using {run_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def input_fingerprints(db_path: Path) -> dict[str, str]:
    from talktopia.task_space import CORE_MODELS

    paths = [path for name in CORE_MODELS for path in (db_path / name).glob("*.json")]
    paths.extend(path for path in (db_path / "voices").rglob("*") if path.is_file())
    paths.extend((REPO_ROOT / "talktopia").rglob("*.py"))
    paths.extend((REPO_ROOT / "patches").glob("*.patch"))
    paths.extend(
        REPO_ROOT / name
        for name in ("engine.lock", "requirements.lock", "run_pipeline.sh")
    )
    engine_files = (
        subprocess.check_output(
            ["git", "-C", str(REPO_ROOT / "engine"), "ls-files", "-z"]
        )
        .decode()
        .split("\0")
    )
    paths.extend(REPO_ROOT / "engine" / name for name in engine_files if name)
    return {
        str(path.resolve()): file_hash(path)
        for path in sorted(set(paths))
        if path.is_file()
    }


def create_run_directory(out_dir: Path, tag: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{tag}_{timestamp}_{uuid4().hex[:8]}"
    run_dir = out_dir.expanduser().resolve() / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(
            f"Output already exists: {run_dir}; nothing was overwritten."
        ) from exc
    return run_dir


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Sample manifest must contain a nonempty list")
    seen = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or not isinstance(record.get("env_id"), str):
            raise ValueError(f"Manifest record {index} must have an env_id string")
        agent_ids = record.get("agent_ids")
        if (
            not record["env_id"]
            or not isinstance(agent_ids, list)
            or len(agent_ids) != 2
            or not all(isinstance(pk, str) and pk for pk in agent_ids)
            or agent_ids[0] == agent_ids[1]
        ):
            raise ValueError(
                f"Manifest record {index} needs an env_id and two distinct agent_ids"
            )
        episode_id = record.get("episode_id", f"episode_{index:04d}")
        if (
            not isinstance(episode_id, str)
            or not re.fullmatch(r"episode_[0-9]+", episode_id)
            or episode_id in seen
        ):
            raise ValueError(
                "Manifest episode IDs must be unique episode_<number> names"
            )
        seen.add(episode_id)
    return records


def result_artifacts(result: dict[str, Any], run_dir: Path) -> dict[str, str]:
    """Commit hashes only after all of an episode's output files are readable."""
    from sotopia.database import EpisodeLog

    episode = EpisodeLog.model_validate_json(
        (run_dir / result["original"]).read_bytes()
    )
    if episode.environment != result["env_id"] or episode.agents != result["agent_ids"]:
        raise ValueError("Episode result does not match its environment/agents")
    paths = {
        result[key]
        for key in ("original", "readable", "speech", "history", "conversation_audio")
        if result.get(key)
    }
    if result.get("speech"):
        for line in (run_dir / result["speech"]).read_text().splitlines():
            row = json.loads(line)
            if row.get("wav_path"):
                paths.add(row["wav_path"])
    for name in paths:
        if not (run_dir / name).resolve().is_relative_to(run_dir.resolve()):
            raise ValueError("Artifact path escapes its run directory")
    return {name: file_hash(run_dir / name) for name in sorted(paths)}


def completed_result(
    row: dict[str, Any],
    run_dir: Path,
    *,
    validate_artifacts=result_artifacts,
    identity_keys=EPISODE_IDENTITY,
) -> dict[str, Any] | None:
    """Recover a committed episode even if its batch checkpoint was interrupted."""
    if not row.get("attempt_history"):
        return None
    result_path = run_dir / row["attempt_history"][-1]["result_path"]
    try:
        result = json.loads(result_path.read_text())
        if (
            result["status"] != "completed"
            or result.get("episode_id") != row["episode_id"]
        ):
            return None
        if any(result.get(key) != row.get(key) for key in identity_keys):
            return None
        if (
            not result.get("artifact_hashes")
            or validate_artifacts(result, run_dir) != result["artifact_hashes"]
        ):
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


async def run_episode_batch(
    records,
    args,
    run_dir,
    stage,
    runner,
    *,
    summary_path,
    initial_summary,
    concurrency,
    max_attempts=1,
    validate_artifacts=result_artifacts,
    identity_keys=EPISODE_IDENTITY,
    on_checkpoint=None,
    error_fields=None,
    attempt_fields=(),
) -> int:
    """Run bounded workers; callers supply stage-specific checks and metadata."""
    if args.resume_run and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if [
            tuple(row.get(key) for key in identity_keys) for row in summary["episodes"]
        ] != [tuple(row.get(key) for key in identity_keys) for row in records]:
            raise ValueError(
                "Saved progress does not match the frozen episode manifest"
            )
    else:
        summary = {
            **initial_summary,
            "run_id": run_dir.name,
            "status": "running",
            "episodes": [
                {**record, "status": "pending", "attempt_history": []}
                for record in records
            ],
        }
    rows = summary["episodes"]
    invocation_started = time.monotonic()
    previous_wall = summary.get("wall_seconds", 0)

    def checkpoint(status=None):
        summary.update(
            wall_seconds=previous_wall + time.monotonic() - invocation_started,
            total=len(rows),
            completed=sum(row["status"] == "completed" for row in rows),
            failed=sum(row["status"] == "failed" for row in rows),
            pending=sum(row["status"] in {"pending", "running"} for row in rows),
            elapsed_seconds=sum(
                item.get("duration_seconds", 0)
                for row in rows
                for item in row["attempt_history"]
            ),
        )
        summary["status"] = status or (
            "failed"
            if summary["failed"]
            else "partial" if summary["pending"] else "completed"
        )
        if on_checkpoint is not None:
            on_checkpoint(summary)
        write_json(summary_path, summary)

    for row in rows:
        recovered = completed_result(
            row,
            run_dir,
            validate_artifacts=validate_artifacts,
            identity_keys=identity_keys,
        )
        if recovered:
            row.update(recovered)
            row["attempt_history"][-1]["status"] = "completed"
        else:
            row["status"] = "pending"
    pending = [row for row in rows if row["status"] != "completed"]
    if args.episode_limit:
        pending = pending[: args.episode_limit]
    if args.dry_run:
        if not summary_path.exists():
            checkpoint("dry_run")
        print(
            f"Prepared {len(records)} episodes; {len(pending)} pending episodes selected. No inference or DB writes."
        )
        return 0
    checkpoint("running")

    async def execute(row):
        for _ in range(max_attempts):
            attempt = len(row["attempt_history"]) + 1
            artifact_dir = (
                run_dir
                if attempt == 1
                else run_dir / "attempts" / stage / row["episode_id"] / str(attempt)
            )
            result_path = (
                (run_dir / stage / "status" / f"{row['episode_id']}.json")
                if attempt == 1
                else artifact_dir / "result.json"
            )
            entry = {
                "attempt": attempt,
                "status": "running",
                "result_path": str(result_path.relative_to(run_dir)),
                "started_at": datetime.now(UTC).isoformat(),
            }
            row["attempt_history"].append(entry)
            row["status"] = "running"
            checkpoint("running")
            started = time.monotonic()
            for field in attempt_fields:
                row.pop(field, None)
            try:
                result = await runner(row, artifact_dir, result_path)
                result.update(
                    episode_id=row["episode_id"], combo_id=row.get("combo_id")
                )
                if result["status"] == "completed":
                    if any(result.get(key) != row.get(key) for key in identity_keys):
                        raise ValueError(
                            "Episode result does not match the frozen manifest"
                        )
                    result["artifact_hashes"] = validate_artifacts(result, run_dir)
            except Exception as exc:
                result = {
                    "episode_id": row["episode_id"],
                    "combo_id": row.get("combo_id"),
                    "env_id": row["env_id"],
                    "agent_ids": row["agent_ids"],
                    "status": "failed",
                    "error": safe_error(exc),
                    **(error_fields or {}),
                }
            except BaseException:
                entry.update(
                    status="interrupted", duration_seconds=time.monotonic() - started
                )
                raise
            entry.update(
                status=result["status"], duration_seconds=time.monotonic() - started
            )
            for field in attempt_fields:
                if row.get(field):
                    result[field] = entry[field] = row[field]
            if result.get("error"):
                entry["error"] = result["error"]
                print(result["error"], file=sys.stderr, flush=True)
            write_json(result_path, result)
            history = row["attempt_history"]
            row.update(result, attempt_history=history)
            checkpoint("running")
            print(
                f"{stage} {row['episode_id']}: {row['status']} ({summary['completed']}/{len(rows)} completed)",
                flush=True,
            )
            if row["status"] == "completed":
                break

    # Keep a bounded number of workers busy instead of waiting for a whole batch.
    iterator = iter(pending)

    async def consume():
        for row in iterator:
            await execute(row)

    tasks = []
    try:
        tasks = [
            asyncio.create_task(consume())
            for _ in range(min(concurrency, len(pending)))
        ]
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        checkpoint("interrupted")
        raise
    checkpoint()
    return 1 if summary["failed"] else 0


def restore_run(
    args: argparse.Namespace,
    *,
    path_fields: tuple[str, ...],
    overrides: dict,
) -> argparse.Namespace:
    run_dir = args.resume_run.expanduser().resolve()
    saved = json.loads((run_dir / "run_config.json").read_text())
    if "input_fingerprints" not in saved:
        raise ValueError("This run predates resumable runs; start a new run")
    values = {key: value for key, value in saved.items() if key in vars(args)}
    for name in path_fields:
        if values.get(name) is not None:
            values[name] = Path(values[name])
    values.update(
        resume_run=run_dir,
        episode_limit=args.episode_limit,
        dry_run=args.dry_run,
        **overrides,
    )
    return argparse.Namespace(**{**vars(args), **values})


def speech_runtime() -> dict:
    return dict(
        speech_workers_per_gpu=SPEECH_WORKERS_PER_GPU,
        speech_endpoints=SPEECH_ENDPOINTS,
        tts_batch_size=TTS_BATCH_SIZE,
    )


def validate_run_inputs(run_dir: Path, db_path: Path, manifest_path: Path) -> dict:
    config = json.loads((run_dir / "run_config.json").read_text())
    if config["input_fingerprints"] != input_fingerprints(db_path):
        raise ValueError(
            "Code, profiles or reference voices changed since the run was created; resume refused"
        )
    if config.get("speech_runtime") != speech_runtime():
        raise ValueError(
            "Speech server settings changed since this run was created; resume refused"
        )
    if file_hash(manifest_path) != config["manifest_sha256"]:
        raise ValueError("Frozen manifest changed; resume refused")
    return config


def save_run_config(args, run_dir, db_path, manifest_path, **extra) -> None:
    config = {
        key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_json(
        run_dir / "run_config.json",
        {
            **config,
            "run_id": run_dir.name,
            "input_fingerprints": input_fingerprints(db_path),
            "speech_runtime": speech_runtime(),
            "manifest_sha256": (
                file_hash(manifest_path) if manifest_path.exists() else None
            ),
            **extra,
        },
    )


def configure_generation(args) -> None:
    """Configure SOTOPIA only after the caller has selected the database backend."""
    import gin
    from sotopia.generation_utils import generate

    generate.DEFAULT_BAD_OUTPUT_PROCESS_MODEL = args.bad_output_process_model
    gin.parse_config_file(
        str(REPO_ROOT / "engine/sotopia_conf/generation_utils_conf/generate.gin"),
        skip_unknown=True,
    )
