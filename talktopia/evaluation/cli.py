"""Evaluation command-line options; importing this module needs no model SDK."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from talktopia.models.config import local_alias

PATH_OPTIONS = ("simulation_dir", "episode_json")


def add_arguments(parser: argparse.ArgumentParser, models: dict[str, str]) -> None:
    parser.add_argument(
        "--evaluate-after-simulation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Automatically evaluate after all simulations have been attempted (default: enabled).",
    )
    parser.add_argument("--evaluator-model", default=models["evaluator"])
    evaluation_input = parser.add_mutually_exclusive_group()
    evaluation_input.add_argument(
        "--episode-json", type=Path, help="One saved EpisodeLog JSON to evaluate."
    )
    evaluation_input.add_argument(
        "--simulation-dir",
        type=Path,
        help="Evaluate valid results from a finished simulation run, even if some episodes failed.",
    )
    parser.add_argument("--reeval-tag", default="talktopia_pipeline_reeval")
    parser.add_argument("--reeval-max-retries", type=int, default=2)
    parser.add_argument(
        "--reeval-episode-id",
        action="append",
        default=[],
        help="Evaluate only this episode from --simulation-dir; repeat to select more.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        help="Concurrent evaluations per GPU; defaults to --batch-size.",
    )


def validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.evaluate_after_simulation is None:
        args.evaluate_after_simulation = True
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.reeval_tag):
        parser.error(
            "--reeval-tag must use letters, digits, dots, underscores or hyphens"
        )
    if args.agent1_models:
        try:
            local_alias(args.evaluator_model)
        except ValueError as exc:
            parser.error(str(exc))
    if args.reeval_max_retries < 0:
        parser.error("--reeval-max-retries must be nonnegative")
    if args.reeval_episode_id and (
        args.stage != "reevaluate" or not args.simulation_dir
    ):
        parser.error("--reeval-episode-id requires --stage reevaluate --simulation-dir")
    if (
        args.stage == "reevaluate"
        or (args.stage in {"all", "simulate"} and args.evaluate_after_simulation)
    ) and not args.evaluator_model.strip():
        parser.error("--evaluator-model must not be blank")
    if args.stage == "reevaluate":
        if args.episode_json is None and args.simulation_dir is None:
            parser.error(
                "--episode-json or --simulation-dir is required for --stage reevaluate"
            )
        if (
            args.sample_manifest
            or args.env_id
            or args.environment_list_pk
            or args.use_stored_combos
        ):
            parser.error(
                "--stage reevaluate uses --episode-json, not sampling selections"
            )
    elif args.episode_json is not None or args.simulation_dir is not None:
        parser.error("Evaluation inputs require --stage reevaluate")


def resume_overrides(args: argparse.Namespace) -> dict:
    run_dir = args.resume_run.expanduser().resolve()
    saved = json.loads((run_dir / "run_config.json").read_text())
    return {
        "evaluate_after_simulation": (
            args.evaluate_after_simulation
            if args.evaluate_after_simulation is not None
            else saved.get("evaluate_after_simulation", True)
        )
    }


def automatic_switch(args) -> str:
    return (
        "--evaluate-after-simulation"
        if args.evaluate_after_simulation
        else "--no-evaluate-after-simulation"
    )
