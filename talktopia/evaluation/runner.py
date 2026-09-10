"""Prepare, run, resume, and connect evaluations of saved simulations."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from talktopia.models.config import model_on_gpu
from talktopia.utils import (
    EPISODE_IDENTITY,
    completed_result,
    configure_generation,
    create_run_directory,
    file_hash,
    lock_run,
    read_manifest,
    restore_run,
    result_artifacts,
    run_episode_batch,
    safe_error,
    save_run_config,
    validate_run_inputs,
    write_json,
)
from .cli import PATH_OPTIONS
from .reporting import write_evaluation_report

EVALUATION_SUMMARY = "04_sotopia_eval_reevaluate_existing.json"
EVALUATION_IDENTITY = (*EPISODE_IDENTITY, "source_sha256")


def evaluation_artifacts(result: dict, run_dir: Path) -> dict[str, str]:
    from sotopia.database import EpisodeLog

    hashes = result_artifacts(result, run_dir)
    source_path = Path(result["source_episode"])
    if file_hash(source_path) != result["source_sha256"]:
        raise ValueError("Source episode changed during evaluation")
    source = EpisodeLog.model_validate_json(source_path.read_bytes())
    episode = EpisodeLog.model_validate_json(
        (run_dir / result["original"]).read_bytes()
    )
    if source.messages != episode.messages:
        raise ValueError("Evaluation changed the source conversation")
    return hashes


def evaluation_manifest(
    simulation_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from sotopia.database import EpisodeLog
    from .evaluator import has_agent_interaction

    summary = json.loads((simulation_dir / "03_simulation.json").read_text())
    sources = read_manifest(simulation_dir / "02_sampled_characters.json")
    expected = {
        row.get("episode_id", f"episode_{index:04d}"): row
        for index, row in enumerate(sources, start=1)
    }
    rows = summary["episodes"]
    if (
        len(rows) != len(expected)
        or len({row["episode_id"] for row in rows}) != len(rows)
        or {row["episode_id"] for row in rows} != set(expected)
    ):
        raise ValueError("Simulation summary does not match its manifest")
    if summary["status"] not in {"completed", "failed"} or any(
        row["status"] not in {"completed", "failed"} for row in rows
    ):
        raise ValueError(
            "All simulation episodes must be attempted before batch evaluation"
        )
    records = []
    excluded = []
    for row in rows:
        original = expected.get(row["episode_id"])
        if (
            original is None
            or any(row[key] != original[key] for key in ("env_id", "agent_ids"))
            or row.get("combo_id") != original.get("combo_id")
        ):
            raise ValueError("Simulation summary does not match its manifest")
        exclusion = {
            "episode_id": row["episode_id"],
            "env_id": row["env_id"],
            "agent_ids": row["agent_ids"],
            "combo_id": original.get("combo_id"),
        }
        if row["status"] == "failed":
            excluded.append(
                {**exclusion, "reason": "simulation_failed", "error": row.get("error")}
            )
            continue
        try:
            if row.get("attempt_history"):
                if completed_result(row, simulation_dir) is None:
                    raise ValueError("Missing, invalid or changed simulation artifacts")
            else:
                # Older runs have no per-attempt hashes, but must still have valid files.
                result_artifacts(row, simulation_dir)
            path = (simulation_dir / row["original"]).resolve()
            source_hash = file_hash(path)
            episode = EpisodeLog.model_validate_json(path.read_bytes())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            excluded.append(
                {**exclusion, "reason": "invalid_artifacts", "error": safe_error(exc)}
            )
            continue
        if not has_agent_interaction(episode):
            excluded.append(
                {
                    **exclusion,
                    "reason": "no_interaction",
                    "error": "No agent utterances or actions were recorded.",
                }
            )
            continue
        records.append(
            {
                "episode_id": row["episode_id"],
                "combo_id": original.get("combo_id"),
                "env_id": row["env_id"],
                "agent_ids": row["agent_ids"],
                "source_episode": str(path),
                "source_sha256": source_hash,
                "source_conversation_audio": (
                    str((simulation_dir / row["conversation_audio"]).resolve())
                    if row.get("conversation_audio")
                    else None
                ),
            }
        )
    return records, {
        "source_total": len(rows),
        "source_completed": sum(row["status"] == "completed" for row in rows),
        "source_failed": sum(row["status"] == "failed" for row in rows),
        "source_excluded": len(excluded),
        "excluded_episodes": excluded,
        "source_no_speech_episodes": sum(
            row["source_conversation_audio"] is None for row in records
        ),
    }


def prepare_run(args, run_dir: Path, db_path: Path):
    manifest_path = run_dir / "evaluation_manifest.json"
    if args.resume_run:
        if not args.simulation_dir:
            raise ValueError("Resume requires a simulation or batch evaluation run")
        validate_run_inputs(run_dir, db_path, manifest_path)
        return json.loads(manifest_path.read_text())
    records = None
    extra = {}
    if args.simulation_dir:
        args.simulation_dir = args.simulation_dir.expanduser().resolve()
        records, source_summary = evaluation_manifest(args.simulation_dir)
        write_json(manifest_path, records)
        extra["source_summary"] = source_summary
    save_run_config(args, run_dir, db_path, manifest_path, **extra)
    return records


def checkpoint_evaluation(summary: dict) -> None:
    if not summary["total"] and summary["status"] == "completed":
        summary["status"] = "no_evaluable_episodes"
    summary["evaluation_status"] = summary["status"]


async def run_evaluation_batch(records, args, run_dir) -> int:
    from .evaluator import evaluate_episode

    async def run(record, artifact_dir, result_path):
        if file_hash(Path(record["source_episode"])) != record["source_sha256"]:
            raise ValueError(
                "Source episode changed since the evaluation manifest was frozen"
            )
        episode_args = argparse.Namespace(
            **{**vars(args), "episode_json": Path(record["source_episode"])}
        )
        await evaluate_episode(
            episode_args,
            run_dir,
            episode_id=record["episode_id"],
            artifact_dir=artifact_dir,
            summary_path=result_path,
        )
        result = json.loads(result_path.read_text())
        result["source_conversation_audio"] = record.get("source_conversation_audio")
        return result

    try:
        source_summary = {}
        config_path = run_dir / "run_config.json"
        if config_path.exists():
            source_summary = json.loads(config_path.read_text()).get(
                "source_summary", {}
            )
        status = await run_episode_batch(
            records,
            args,
            run_dir,
            "evaluation",
            run,
            summary_path=run_dir / EVALUATION_SUMMARY,
            initial_summary={
                "tag": args.reeval_tag,
                "evaluation_status": "running",
                "canonical_text": "asr_transcript",
                **source_summary,
                "evaluation_total": len(records),
            },
            concurrency=args.eval_batch_size or args.batch_size,
            validate_artifacts=evaluation_artifacts,
            identity_keys=EVALUATION_IDENTITY,
            on_checkpoint=checkpoint_evaluation,
            error_fields={
                "conversation_audio": None,
                "evaluation_status": "not_performed",
            },
        )
        if not records and not args.dry_run:
            print(
                "No evaluable episodes; no evaluation requests were made.", flush=True
            )
            return 1
        return status
    finally:
        if (
            run_dir / "04_sotopia_eval_reevaluate_existing.json"
        ).exists() and not args.dry_run:
            write_evaluation_report(run_dir)


async def run_evaluation(args, run_dir: Path, db_path: Path) -> int:
    records = prepare_run(args, run_dir, db_path)
    if not args.dry_run:
        configure_generation(args)
        await asyncio.to_thread(release_evaluation_models, args)
    if args.simulation_dir:
        return await run_evaluation_batch(records, args, run_dir)
    from .evaluator import evaluate_episode

    return await evaluate_episode(args, run_dir)


def release_evaluation_models(args) -> None:
    from talktopia.models.servers import release_worker_models

    release_worker_models(
        args, keep_models=[args.evaluator_model, args.bad_output_process_model]
    )


async def run_after_simulation(args, run_dir, db_path) -> int:
    """Evaluate finished results even when individual simulations have failed."""
    summary_path = run_dir / "03_simulation.json"
    summary = json.loads(summary_path.read_text())
    linked_run = summary.get("evaluation_run")
    if linked_run:
        evaluation_dir = Path(linked_run)
        config = json.loads((evaluation_dir / "run_config.json").read_text())
        if (
            config.get("stage") != "reevaluate"
            or Path(config["simulation_dir"]).resolve() != run_dir.resolve()
        ):
            raise ValueError("Linked evaluation does not belong to this simulation run")
        evaluation_args = restore_run(
            argparse.Namespace(
                **{
                    **vars(args),
                    "resume_run": evaluation_dir,
                    "evaluate_after_simulation": False,
                }
            ),
            path_fields=("out_dir", "sample_manifest", *PATH_OPTIONS),
            overrides={"evaluate_after_simulation": False},
        )
    else:
        evaluation_args = argparse.Namespace(
            **{
                **vars(args),
                "stage": "reevaluate",
                "simulation_dir": run_dir,
                "episode_json": None,
                "sample_manifest": None,
                "environment_list_pk": "",
                "env_id": [],
                "use_stored_combos": False,
                "resume_run": None,
                "episode_limit": 0,
                "evaluate_after_simulation": False,
            }
        )
        evaluation_dir = create_run_directory(args.out_dir, args.reeval_tag)

    print(f"Automatic evaluation; output: {evaluation_dir}", flush=True)
    with lock_run(evaluation_dir):
        if not linked_run:
            # Prepare everything before recording the link, and link before inference.
            prepare_run(evaluation_args, evaluation_dir, db_path)
            summary.update(
                evaluation_run=str(evaluation_dir), evaluation_status="pending"
            )
            write_json(summary_path, summary)
            evaluation_args.resume_run = evaluation_dir
        try:
            status = await run_evaluation(evaluation_args, evaluation_dir, db_path)
        except BaseException as exc:
            if not args.dry_run:
                summary.update(
                    evaluation_status=(
                        "failed" if isinstance(exc, Exception) else "interrupted"
                    ),
                    evaluation_error=safe_error(exc),
                )
                write_json(summary_path, summary)
            raise
        evaluation = json.loads(
            (evaluation_dir / "04_sotopia_eval_reevaluate_existing.json").read_text()
        )
        if not args.dry_run:
            summary.update(evaluation_status=evaluation["status"])
            summary.pop("evaluation_error", None)
            write_json(summary_path, summary)
    if args.dry_run:
        return 0
    return int(
        bool(status or summary["failed"] or evaluation.get("source_excluded", 0))
    )


async def resume_linked_evaluation(args, run_dir: Path, db_path: Path) -> int | None:
    """Follow an existing evaluation, or detach it for an explicit simulation retry."""
    if not args.resume_run:
        return None
    summary_path = run_dir / "03_simulation.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if not summary.get("evaluation_run"):
        return None
    if args.evaluate_after_simulation:
        return await run_after_simulation(args, run_dir, db_path)
    if not args.dry_run:
        summary.setdefault("previous_evaluation_runs", []).append(
            summary.pop("evaluation_run")
        )
        summary["evaluation_status"] = "not_performed"
        summary.pop("evaluation_error", None)
        write_json(summary_path, summary)
    return None


async def finish_simulation(
    args, run_dir: Path, db_path: Path, simulation_status: int
) -> int:
    if args.evaluate_after_simulation:
        summary = json.loads((run_dir / "03_simulation.json").read_text())
        if summary["pending"]:
            print(
                "Automatic evaluation deferred: some simulations have not been attempted.",
                flush=True,
            )
        else:
            return await run_after_simulation(args, run_dir, db_path)
    return simulation_status


def validate_linked_evaluation(simulation: dict) -> None:
    run_dir = Path(simulation["evaluation_run"])
    summary = json.loads((run_dir / EVALUATION_SUMMARY).read_text())
    if summary["pending"]:
        raise ValueError("Finished pair has pending episodes")
    for row in summary["episodes"]:
        if (
            row["status"] == "completed"
            and completed_result(
                row,
                run_dir,
                validate_artifacts=evaluation_artifacts,
                identity_keys=EVALUATION_IDENTITY,
            )
            is None
        ):
            raise ValueError(
                f"Saved completed artifact changed: {run_dir} / {row['episode_id']}"
            )


def pair_needs_retry(args, simulation: dict) -> bool:
    if not args.evaluate_after_simulation:
        return bool(simulation["failed"])
    if not simulation.get("evaluation_run"):
        return True
    summary = json.loads(
        (Path(simulation["evaluation_run"]) / EVALUATION_SUMMARY).read_text()
    )
    return bool(summary["failed"])


def check_pair_finished(args, pair_id: str, simulation: dict) -> None:
    if not args.evaluate_after_simulation:
        return
    if simulation.get("evaluation_status") not in {
        "completed",
        "failed",
        "no_evaluable_episodes",
    }:
        raise RuntimeError(f"{pair_id}: evaluation did not finish")
    summary = json.loads(
        (Path(simulation["evaluation_run"]) / EVALUATION_SUMMARY).read_text()
    )
    if summary["pending"] or summary["status"] == "interrupted":
        raise RuntimeError(f"{pair_id}: interrupted evaluation")


def pair_settings(args, pair_id: str, endpoint: str) -> dict:
    return {
        "reeval_tag": f"{pair_id}_eval",
        "evaluator_model": model_on_gpu(args.evaluator_model, endpoint),
    }
