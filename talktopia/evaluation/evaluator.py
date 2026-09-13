"""Evaluate one saved conversation with SOTOPIA's existing evaluator."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

from talktopia.utils import safe_error, write_json
from .reporting import write_episode_report

from pydantic import Field, field_validator
from sotopia.database import EpisodeLog, SotopiaDimensions
from sotopia.envs.evaluators import (
    EpisodeLLMEvaluator,
    EvaluationForAgents,
    unweighted_aggregate_evaluate,
)


EVALUATION_EVIDENCE_INSTRUCTION = (
    "Evaluate only the utterances and actions explicitly recorded in the interaction; "
    "background information and goals are context, not evidence that any action "
    "occurred or any goal was achieved.\n"
    "For each agent and each dimension, write 1-3 sentences explaining the score "
    "using that participant's specific recorded behavior. Identify the participant "
    "and the relevant utterance, action, or observed outcome. If there is no evidence "
    "of a change or violation, explain what was not observed in this interaction.\n"
    "Do not copy field descriptions, scoring rubrics, or example values into reasoning. "
    "Never use placeholders such as 'Detailed reasoning for the evaluation', '...', "
    "'string', or 'Placeholder reasoning'. Do not repeat one generic explanation "
    "across all dimensions. Zero is a valid score only with an actual explanation.\n"
    "Check each participant's role and goal separately: a buyer's maximum price is "
    "not a seller's minimum price. Distinguish proposals from accepted agreements "
    "and completed actions. Do not overlook role confusion or contradictory numbers "
    "merely because the language is fluent."
)


def normalize_reason(reason: str) -> str:
    return " ".join(re.findall(r"\w+", reason.casefold()))


def reasoning_problem(reason: str) -> str | None:
    """Reject copied instructions and placeholders, not legitimate zero scores."""
    normalized = normalize_reason(reason)
    if not normalized:
        return "empty or punctuation-only reasoning"
    if "detailed reasoning for the evaluation" in normalized or normalized in {
        "string",
        "placeholder reasoning",
        "reasoning process goes here",
    }:
        return "placeholder reasoning"
    if any(
        normalized == normalize_reason(field.description or "")
        for field in SotopiaDimensions.model_fields.values()
    ):
        return "reasoning copies a scoring rubric"
    return None


def validate_evaluation_responses(responses) -> None:
    expected = {
        (agent, dimension)
        for agent in ("agent_1", "agent_2")
        for dimension in SotopiaDimensions.model_fields
    }
    if (
        len(responses) != len(expected)
        or {(agent, dimension) for agent, ((dimension, _), _) in responses} != expected
    ):
        raise ValueError("Evaluator did not return all 14 dimension scores and reasons")
    reasons = {"agent_1": [], "agent_2": []}
    for agent, ((dimension, _), reason) in responses:
        problem = reasoning_problem(reason)
        if problem:
            raise ValueError(f"{agent}.{dimension}: {problem}")
        reasons[agent].append(normalize_reason(reason))
    for agent, values in reasons.items():
        if len(set(values)) == 1:
            raise ValueError(
                f"{agent}: identical reasoning across all seven dimensions"
            )


def has_agent_interaction(episode: EpisodeLog) -> bool:
    """Ignore initial context and none actions, but retain nonverbal actions."""
    for turn in episode.messages:
        for sender, receiver, message in turn:
            if sender == "Environment" or receiver != "Environment":
                continue
            action = message.strip()
            if action.startswith("[private to "):
                action = action.partition("] ")[2].strip()
            if action and action != "did nothing":
                return True
    return False


def specify_agent_keys(schema: dict[str, Any]) -> None:
    dimension_schema = schema["additionalProperties"]
    schema.update(
        properties={key: dimension_schema for key in ("agent_1", "agent_2")},
        required=["agent_1", "agent_2"],
        additionalProperties=False,
    )


class TwoAgentEvaluation(EvaluationForAgents[SotopiaDimensions]):
    evaluations: dict[str, SotopiaDimensions] = Field(
        json_schema_extra=specify_agent_keys
    )

    @field_validator("evaluations")
    @classmethod
    def validate_agents(
        cls, values: dict[str, SotopiaDimensions]
    ) -> dict[str, SotopiaDimensions]:
        if set(values) != {"agent_1", "agent_2"}:
            raise ValueError("Expected exactly agent_1 and agent_2 evaluations")
        for evaluation in values.values():
            if any(
                not item["reasoning"].strip()
                for item in evaluation.model_dump().values()
            ):
                raise ValueError("Every evaluation dimension needs a nonempty reason")
        # The engine iterates dict values; fix their order before it assigns labels.
        return {key: values[key] for key in ("agent_1", "agent_2")}


async def evaluate_episode(
    args: argparse.Namespace,
    run_dir: Path,
    *,
    episode_id: str = "episode_0001",
    artifact_dir: Path | None = None,
    summary_path: Path | None = None,
) -> int:
    source_path = args.episode_json.expanduser().resolve()
    artifact_dir = artifact_dir or run_dir
    summary_path = summary_path or run_dir / "04_sotopia_eval_reevaluate_existing.json"
    summary: dict[str, Any] = {
        "run_id": run_dir.name,
        "episode_id": episode_id,
        "source_episode": str(source_path),
        "source_sha256": None,
        "reeval_tag": args.reeval_tag,
        "evaluator_model": args.evaluator_model,
        "temperature": 0.0,
        "max_retries": args.reeval_max_retries,
        "push_to_db": args.push_to_db,
        "status": "running",
        "attempts": 0,
        "attempt_errors": [],
        "response_attempts": [],
        "history": None,
        "original": None,
        "readable": None,
        "episode_pk": None,
    }
    write_json(summary_path, summary)
    try:
        source_bytes = source_path.read_bytes()
        summary["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        source = EpisodeLog.model_validate_json(source_bytes)
        if len(source.agents) != 2 or len(set(source.agents)) != 2:
            raise ValueError("Evaluation requires two distinct agents")
        if not source.models or len(source.models) != 3:
            raise ValueError(
                "Episode models must contain the environment and two agent models"
            )
        if not source.messages or len(source.messages[0]) < 2:
            raise ValueError("Episode is missing the two initial agent perspectives")
        profiles, turns = source.render_for_humans()
        names = [
            f"{profile.first_name} {profile.last_name}".strip() for profile in profiles
        ]
        if len(set(names)) != 2 or any(
            sender != "Environment" or receiver != names[index]
            for index, (sender, receiver, _) in enumerate(source.messages[0][:2])
        ):
            raise ValueError(
                "Initial perspectives must match the episode's agent order"
            )
        if args.push_to_db and args.reeval_tag == source.tag:
            raise ValueError(
                "--reeval-tag must differ from the source tag when saving to DB"
            )
        history = (
            "\n".join(turns[:-2])
            + "\n\n"
            + EVALUATION_EVIDENCE_INSTRUCTION
            + f"\nAgent mapping: agent_1 is {names[0]}; agent_2 is {names[1]}."
        )
        history_path = artifact_dir / "evaluation/history" / f"{episode_id}.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(history, encoding="utf-8")
        summary.update(
            source_tag=source.tag,
            source_episode_pk=source.pk or None,
            env_id=source.environment,
            agent_ids=source.agents,
            agent_names=names,
            history=str(history_path.relative_to(run_dir)),
        )
        if not has_agent_interaction(source):
            summary.update(status="excluded", reason="no_interaction")
            print("Evaluation excluded: no agent utterances or actions.", flush=True)
            return 0
        if args.dry_run:
            summary["status"] = "dry_run"
            print(
                "Prepared one episode for evaluation; no inference or DB writes.",
                flush=True,
            )
            return 0

        evaluator = EpisodeLLMEvaluator(
            model_name=args.evaluator_model, response_format_class=TwoAgentEvaluation
        )
        for attempt in range(1, args.reeval_max_retries + 2):
            summary["attempts"] = attempt
            write_json(summary_path, summary)
            print(
                f"SOTOPIA evaluation: attempt {attempt}/{args.reeval_max_retries + 1}",
                flush=True,
            )
            request_history = history
            if summary["attempt_errors"]:
                request_history += (
                    f"\n\nEvaluation attempt {attempt}. The previous evaluation was rejected: "
                    + summary["attempt_errors"][-1][:1000]
                    + "\nEvaluate the complete interaction above again. Return all 14 "
                    "scores with specific evidence, not a reformatted example."
                )
            attempt_path = (
                artifact_dir
                / "evaluation/responses"
                / episode_id
                / f"attempt_{attempt}.json"
            )
            attempt_record = {
                "history": request_history,
                "responses": None,
                "error": None,
            }
            try:
                responses = await evaluator.__acall__(
                    turn_number=-1,
                    messages=None,
                    history=request_history,
                    num_agents=2,
                    temperature=0.0,
                )
                # Check content AFTER generation so these failures retry with the
                # conversation, rather than using the engine's JSON-only repair.
                attempt_record["responses"] = responses
                validate_evaluation_responses(responses)
                response = unweighted_aggregate_evaluate(responses)
                break
            except Exception as exc:
                error = safe_error(exc)
                attempt_record["error"] = error
                summary["attempt_errors"].append(error)
                print(error, file=sys.stderr, flush=True)
            finally:
                write_json(attempt_path, attempt_record)
                summary["response_attempts"].append(
                    str(attempt_path.relative_to(run_dir))
                )
        else:
            raise RuntimeError(
                f"SOTOPIA evaluation failed after {summary['attempts']} attempts"
            )

        evaluated = EpisodeLog(
            **{
                **source.model_dump(),
                "pk": "",
                "tag": args.reeval_tag,
                "models": [args.evaluator_model, *source.models[1:]],
                "rewards": [response.p1_rate, response.p2_rate],
                "reasoning": response.comments,
                # The engine does not expose the completed prompt; save history separately.
                "rewards_prompt": "",
            }
        )
        original_path = artifact_dir / "evaluation/original" / f"{episode_id}.json"
        readable_path = artifact_dir / "evaluation/readable" / f"{episode_id}.md"
        write_json(original_path, evaluated.model_dump(mode="json"))
        write_episode_report(
            readable_path, evaluated, names, responses, args.evaluator_model
        )
        if args.push_to_db:
            evaluated.save()
            write_json(original_path, evaluated.model_dump(mode="json"))
        summary.update(
            status="completed",
            original=str(original_path.relative_to(run_dir)),
            readable=str(readable_path.relative_to(run_dir)),
            episode_pk=evaluated.pk or None,
        )
        print("SOTOPIA evaluation: completed (2 agents, 7 dimensions each)", flush=True)
        return 0
    except Exception as exc:
        summary.update(status="failed", error=safe_error(exc))
        print(summary["error"], file=sys.stderr, flush=True)
        return 1
    except BaseException:
        summary["status"] = "interrupted"
        raise
    finally:
        write_json(summary_path, summary)
