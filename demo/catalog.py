"""Configuration and manifest loading; no imports from the experiment runtime."""

from __future__ import annotations

import re
from collections import Counter
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
    experiment_tag: str = ""
    pair_id: str = ""


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
        for field in ("experiment_tag", "pair_id"):
            if field in entry and (not isinstance(entry[field], str) or not entry[field].strip()):
                raise ResultError(f"{run_id}: {field}는 비어 있지 않은 문자열이어야 합니다.")
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
                entry.get("experiment_tag", "").strip(),
                entry.get("pair_id", "").strip(),
            )
        )
    return runs


def short_model(value) -> str:
    if not isinstance(value, str):
        return ""
    name = value.split("@", 1)[0].strip()
    name = re.sub(r"^custom/structured-", "", name)
    name = re.sub(r"^talktopia-(?:agent|evaluator)-", "", name)
    return re.sub(r"-gpu\d+$", "", name)


def run_catalog(runs: list[Run]) -> list[dict]:
    """Group only registered runs; never discover or load sibling simulations."""
    groups = {}
    for run in runs:
        config = optional_json(run.simulation_dir / "run_config.json", dict, [])
        pair_dir = run.simulation_dir.parent
        matrix_pair = pair_dir.parent.name == "pairs" and re.fullmatch(r"pair_\d+", pair_dir.name)
        matrix = optional_json(pair_dir.parent.parent / "run_config.json", dict, []) if matrix_pair else {}
        tag = run.experiment_tag or matrix.get("tag") or config.get("tag")
        if not isinstance(tag, str) or not tag.strip():
            tag = run.label
        tag = tag.strip()
        pair_id = run.pair_id or (pair_dir.name if matrix_pair else "")
        pair_number = re.fullmatch(r"pair[_ -](\d+)", pair_id)
        if pair_number:
            pair_id = f"pair_{int(pair_number[1]):02d}"
        pair_name = f"pair {int(pair_number[1]):02d}" if pair_number else pair_id or run.label
        models = [short_model(config.get(key)) for key in ("agent1_model", "agent2_model")]
        pair_label = f"{pair_name} ({', '.join(model or '모델 정보 없음' for model in models)})"
        groups.setdefault(tag, []).append({
            "id": run.id, "label": run.label, "experiment_tag": tag,
            "pair_id": pair_id, "agent_models": models, "pair_label": pair_label,
        })

    def pair_order(item):
        number = re.fullmatch(r"pair[_ -](\d+)", item["pair_id"])
        return (int(number[1]) if number else float("inf"), item["pair_id"], item["id"])

    catalog = []
    for items in groups.values():
        counts = Counter(item["pair_id"] or item["label"] for item in items)
        for item in sorted(items, key=pair_order):
            if counts[item["pair_id"] or item["label"]] > 1:
                item["pair_label"] += f" · {item['id']}"
            catalog.append(item)
    return catalog


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
