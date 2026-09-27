"""Join evaluations to a simulation only when the saved evidence matches."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from .catalog import evaluation_rows
from .files import ResultError, artifact_json, artifact_path, available, sha256


DIMENSIONS = (
    "believability", "relationship", "knowledge", "secret", "social_rules",
    "financial_and_material_benefits", "goal", "overall_score",
)


def evaluation_id(directory: Path) -> str:
    return hashlib.sha256(str(directory).encode()).hexdigest()[:12]


def association(row: dict, source: dict, source_hash: str | None, root: Path, sim_row: dict):
    if not source_hash or not row.get("source_sha256"):
        return "unverified", "평가 원본을 확인할 수 없습니다."
    if row["source_sha256"] != source_hash:
        return "mismatch", "평가 당시의 시뮬레이션과 현재 원본이 다릅니다."
    if row.get("env_id") != source.get("environment") or row.get("agent_ids") != source.get("agents"):
        return "mismatch", "평가의 환경 또는 에이전트 순서가 다릅니다."
    if row.get("source_events_sha256"):
        try:
            events = artifact_path(root, sim_row.get("events"), ".jsonl")
            if sha256(events) != row["source_events_sha256"]:
                return "mismatch", "평가 당시의 Surface5 이벤트와 현재 기록이 다릅니다."
        except (ResultError, OSError):
            return "unverified", "평가에 사용한 Surface5 이벤트를 확인할 수 없습니다."
    return "matched", ""


def saved_scores(evaluated: dict, source: dict) -> list[dict]:
    if (
        evaluated.get("agents") != source.get("agents")
        or evaluated.get("environment") != source.get("environment")
    ):
        raise ResultError("평가 결과의 에이전트 또는 환경이 원본과 다릅니다.")
    rewards = evaluated.get("rewards")
    if not isinstance(rewards, list) or len(rewards) != 2:
        raise ResultError("두 에이전트의 점수를 읽을 수 없습니다.")
    result = []
    for reward in rewards:
        if not isinstance(reward, list) or len(reward) != 2 or not isinstance(reward[1], dict):
            raise ResultError("평가 점수 형식이 잘못되었습니다.")
        scores = reward[1]
        if any(
            not isinstance(scores.get(name), (int, float))
            or isinstance(scores.get(name), bool)
            or not math.isfinite(scores[name])
            for name in DIMENSIONS
        ):
            raise ResultError("일부 평가 점수가 없거나 숫자가 아닙니다.")
        result.append({name: scores[name] for name in DIMENSIONS})
    return result


def read_evaluations(run, sim_row, source, source_hash):
    results = []
    for directory in run.evaluation_dirs:
        result = {
            "id": evaluation_id(directory),
            "run_name": directory.name,
            "model": directory.name,
            "status": "unavailable",
            "association": "unverified",
            "message": "",
            "scores": None,
            "report_available": False,
            "warnings": [],
        }
        try:
            rows, warnings = evaluation_rows(directory)
            result["warnings"] = warnings
            row = rows.get(sim_row["episode_id"])
            if row is None:
                result["message"] = "이 에피소드의 평가 기록이 없습니다."
            else:
                result.update(
                    model=row.get("evaluator_model") or directory.name,
                    status=row.get("status", "pending"),
                    report_available=available(directory, row.get("readable"), ".md"),
                )
                match, message = association(row, source, source_hash, run.simulation_dir, sim_row)
                result.update(association=match, message=message)
                if row.get("status") == "completed" and match == "matched":
                    evaluated = artifact_json(directory, row, "original", result["warnings"])
                    if evaluated:
                        result["scores"] = saved_scores(evaluated, source)
                elif row.get("status") in ("failed", "excluded"):
                    result["warnings"].append(str(row.get("error") or row.get("reason") or ""))
        except ResultError as error:
            result["warnings"].append(str(error))
        results.append(result)
    return results
