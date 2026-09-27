"""Configuration and manifest loading; no imports from the experiment runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .files import ResultError, optional_json, read_json


@dataclass(frozen=True)
class Run:
    id: str
    label: str
    simulation_dir: Path
    evaluation_dirs: tuple[Path, ...]
    profiles_dir: Path | None = None


def read_config(config_path: Path) -> list[Run]:
    config = read_json(config_path)
    entries = config.get("runs")
    if not isinstance(entries, list):
        raise ResultError("설정 파일에 runs 목록이 필요합니다.")

    def directory(value) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ResultError("디렉터리 경로는 비어 있지 않은 문자열이어야 합니다.")
        path = Path(value).expanduser()
        return (config_path.parent / path).resolve()

    runs = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ResultError("runs의 각 항목은 객체여야 합니다.")
        run_id = entry.get("id", "")
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ResultError("실행 id에는 영문, 숫자, 밑줄, 하이픈만 사용할 수 있습니다.")
        if run_id in seen:
            raise ResultError(f"중복된 실행 id: {run_id}")
        seen.add(run_id)
        evaluations = entry.get("evaluation_dirs", [])
        if not isinstance(evaluations, list):
            raise ResultError(f"{run_id}: evaluation_dirs는 경로 목록이어야 합니다.")
        paths = tuple(directory(value) for value in evaluations)
        if len(set(paths)) != len(paths):
            raise ResultError(f"{run_id}: 같은 평가 디렉터리가 두 번 등록됐습니다.")
        runs.append(
            Run(
                run_id,
                str(entry.get("label") or run_id),
                directory(entry.get("simulation_dir")),
                paths,
                directory(entry["profiles_dir"]) if entry.get("profiles_dir") else None,
            )
        )
    return runs


def merge_rows(manifest: list, summary: dict) -> dict[str, dict]:
    """A later summary overrides manifest fields, including retry artifact paths."""
    rows = {}
    updates = summary.get("episodes", [])
    if not isinstance(updates, list):
        raise ResultError("진행 요약의 episodes는 목록이어야 합니다.")
    for source in (manifest, updates):
        seen = set()
        for row in source:
            if not isinstance(row, dict):
                raise ResultError("에피소드 기록은 객체여야 합니다.")
            episode_id = row.get("episode_id", "")
            if not isinstance(episode_id, str) or not re.fullmatch(r"episode_\d+", episode_id):
                raise ResultError("에피소드 ID 형식이 잘못되었습니다.")
            if episode_id in seen:
                raise ResultError(f"중복된 에피소드 ID: {episode_id}")
            seen.add(episode_id)
            rows[episode_id] = {**rows.get(episode_id, {}), **row}
    return dict(sorted(rows.items()))


def simulation_rows(run: Run) -> tuple[dict[str, dict], list[str]]:
    warnings = []
    manifest = optional_json(run.simulation_dir / "02_sampled_characters.json", list, warnings)
    summary = optional_json(run.simulation_dir / "03_simulation.json", dict, warnings)
    rows = merge_rows(manifest, summary)
    for episode_id, row in rows.items():
        row.setdefault("status", "pending")
        # Only use canonical names when no explicit path was saved.
        for key, folder, extension in (
            ("original", "original", "json"),
            ("readable", "readable", "md"),
            ("speech", "speech", "jsonl"),
        ):
            row.setdefault(key, f"simulation/{folder}/{episode_id}.{extension}")
    return rows, warnings


def evaluation_rows(directory: Path) -> tuple[dict[str, dict], list[str]]:
    warnings = []
    manifest = optional_json(directory / "evaluation_manifest.json", list, warnings)
    summary = optional_json(
        directory / "04_sotopia_eval_reevaluate_existing.json", dict, warnings
    )
    rows = merge_rows(manifest, summary)
    excluded = summary.get("excluded_episodes", [])
    if not isinstance(excluded, list):
        raise ResultError("excluded_episodes는 목록이어야 합니다.")
    for episode_id, row in merge_rows(excluded, {}).items():
        if episode_id in rows:
            raise ResultError(f"{episode_id}: 평가 기록과 제외 기록이 중복됩니다.")
        rows[episode_id] = {**row, "status": "excluded"}
    return rows, warnings
