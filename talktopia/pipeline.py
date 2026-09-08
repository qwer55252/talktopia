"""Sample scenarios, run speech conversations, and evaluate saved episodes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
from urllib.parse import urlsplit
from uuid import uuid4

from talktopia.models.config import REPO_ROOT, SPEECH_BASE_URL, default_pipeline_models

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from sotopia.envs import ParallelSotopiaEnv

    from talktopia.speech_agent import CascadedSpeechAgent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    models = default_pipeline_models()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["all", "sample", "simulate", "reevaluate"], default="all"
    )
    parser.add_argument(
        "--storage-backend",
        choices=["local", "redis"],
        default=os.environ.get("SOTOPIA_STORAGE_BACKEND", "local"),
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--tag",
        default="talktopia_pipeline",
        help="Run label and directory prefix; each run adds a UTC timestamp and unique suffix.",
    )
    for role in ("env", "agent1", "agent2"):
        parser.add_argument(f"--{role}-model", default=models[role])
    parser.add_argument("--evaluator-model", default=models["evaluator"])
    parser.add_argument(
        "--episode-json", type=Path, help="One saved EpisodeLog JSON to evaluate."
    )
    parser.add_argument("--reeval-tag", default="talktopia_pipeline_reeval")
    parser.add_argument("--reeval-max-retries", type=int, default=2)
    parser.add_argument(
        "--bad-output-process-model",
        default=models["agent2"],
        help="Local model used by SOTOPIA to repair malformed model output.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--environment-list-pk", default="")
    selection.add_argument("--env-id", action="append", default=[])
    selection.add_argument(
        "--sample-manifest",
        type=Path,
        help="Reuse all env_id/agent_ids pairs; ignores sample counts and seed.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="0 selects all environments."
    )
    parser.add_argument("--pairs-per-env", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--push-to-db", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare samples or evaluation inputs only; no inference or DB writes.",
    )
    speech_options = {
        "asr-base-url": SPEECH_BASE_URL,
        "asr-model": "whisper-1",
        "asr-language": "en",
        "tts-base-url": SPEECH_BASE_URL,
        "tts-model": "tts-1",
    }
    for name, default in speech_options.items():
        env_name = "TALKTOPIA_" + name.replace("-", "_").upper()
        parser.add_argument(f"--{name}", default=os.environ.get(env_name, default))
    args = parser.parse_args(argv)
    if args.storage_backend != "local":
        parser.error(
            "Talktopia requires --storage-backend local; Redis is not supported"
        )
    for name in ("pairs_per_env", "batch_size", "max_turns"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_envs < 0:
        parser.error("--num-envs must be nonnegative (0 means all)")
    for name in ("tag", "reeval_tag"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", getattr(args, name)):
            parser.error(
                f"--{name.replace('_', '-')} must use letters, digits, dots, underscores or hyphens"
            )
    if args.reeval_max_retries < 0:
        parser.error("--reeval-max-retries must be nonnegative")
    if args.stage == "reevaluate":
        if args.episode_json is None:
            parser.error("--episode-json is required for --stage reevaluate")
        if args.sample_manifest or args.env_id or args.environment_list_pk:
            parser.error(
                "--stage reevaluate uses --episode-json, not sampling selections"
            )
        if not args.evaluator_model.strip():
            parser.error("--evaluator-model must not be blank")
    elif args.episode_json is not None:
        parser.error("--episode-json requires --stage reevaluate")
    if args.stage in {"all", "simulate"} and not args.dry_run:
        for kind in ("asr", "tts"):
            url = urlsplit(getattr(args, f"{kind}_base_url"))
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or not url.path.rstrip("/").endswith("/v1")
                or url.username is not None
                or url.password is not None
                or url.query
                or url.fragment
            ):
                parser.error(
                    f"--{kind}-base-url / TALKTOPIA_{kind.upper()}_BASE_URL must be "
                    "an HTTP(S) base URL ending in /v1, without credentials or query parameters"
                )
        for name in speech_options:
            if not getattr(args, name.replace("-", "_")).strip():
                parser.error(f"--{name} must not be blank")
    return args


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")


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


def model_names(args: argparse.Namespace) -> dict[str, str]:
    return {
        role: getattr(args, f"{role}_model") for role in ("env", "agent1", "agent2")
    }


def env_params(args: argparse.Namespace) -> dict[str, Any]:
    from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator

    return {
        "model_name": args.env_model,
        "action_order": "round-robin",
        "evaluators": [
            RuleBasedTerminatedEvaluator(
                max_turn_number=args.max_turns, max_stale_turn=2
            )
        ],
        "terminal_evaluators": [],
    }


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Sample manifest must contain a nonempty list")
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
    return records


def stage_1_sample_env_profiles(
    args: argparse.Namespace, manifest: list[dict[str, Any]] | None = None
) -> list[Any]:
    from sotopia.database import EnvironmentProfile
    from sotopia.database.persistent_profile import EnvironmentList

    if manifest is not None:
        selected = list(dict.fromkeys(record["env_id"] for record in manifest))
    else:
        if args.env_id:
            candidates = sorted(set(args.env_id))
        elif args.environment_list_pk:
            candidates = sorted(
                set(EnvironmentList.get(args.environment_list_pk).environments)
            )
        else:
            candidates = sorted(EnvironmentProfile.all_pks())
        if not candidates:
            raise ValueError(
                "No environments found. Prepare profiles with ./load_profiles.sh first."
            )
        selected = (
            candidates
            if args.num_envs == 0
            else random.Random(args.seed).sample(
                candidates, min(args.num_envs, len(candidates))
            )
        )
    profiles = [EnvironmentProfile.get(pk) for pk in selected]
    for profile in profiles:
        if len(profile.agent_goals) != 2:
            raise ValueError(
                f"Environment {profile.pk} must have exactly two agent goals"
            )
    return profiles


def stage_2_sample_characters(
    env_profiles: Sequence[Any],
    args: argparse.Namespace,
    manifest: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    from sotopia.agents import LLMAgent
    from talktopia.speech_agent import AgentProfile
    from sotopia.samplers import ConstraintBasedSampler

    records = []

    def add_record(env: Any, profiles: Sequence[Any]) -> None:
        names = [
            f"{profile.first_name} {profile.last_name}".strip() for profile in profiles
        ]
        if len(set(names)) != 2:
            raise ValueError("The two agent profiles must have distinct names")
        records.append(
            {
                "env_id": env.pk,
                "codename": env.codename,
                "agent_ids": [profile.pk for profile in profiles],
                "agent_names": names,
                "models": model_names(args),
            }
        )

    if manifest is not None:
        environments = {profile.pk: profile for profile in env_profiles}
        for record in manifest:
            profiles = [AgentProfile.get(pk) for pk in record["agent_ids"]]
            add_record(environments[record["env_id"]], profiles)
        return records

    # SOTOPIA's sampler uses the module RNG; limit its seed to synchronous sampling.
    rng_state = random.getstate()
    random.seed(args.seed)
    try:
        for profile in env_profiles:
            sampler = ConstraintBasedSampler(env_candidates=[profile.pk])
            for env, agents in sampler.sample(
                agent_classes=[LLMAgent, LLMAgent],
                replacement=False,
                size=args.pairs_per_env,
                env_params=env_params(args),
                agents_params=[
                    {"model_name": args.agent1_model},
                    {"model_name": args.agent2_model},
                ],
            ):
                add_record(env.profile, [agent.profile for agent in agents])
    finally:
        random.setstate(rng_state)
    return records


def build_episode(
    record: dict[str, Any],
    args: argparse.Namespace,
    asr_client: AsyncOpenAI,
    tts_client: AsyncOpenAI,
) -> tuple[ParallelSotopiaEnv, list[CascadedSpeechAgent]]:
    from sotopia.database import EnvironmentProfile
    from sotopia.envs import ParallelSotopiaEnv
    from talktopia.speech_agent import AgentProfile, CascadedSpeechAgent

    env = ParallelSotopiaEnv(
        env_profile=EnvironmentProfile.get(record["env_id"]), **env_params(args)
    )
    agents = [
        CascadedSpeechAgent(
            agent_profile=AgentProfile.get(pk),
            model_name=getattr(args, f"agent{index}_model"),
            asr_client=asr_client,
            tts_client=tts_client,
            asr_model=args.asr_model,
            tts_model=args.tts_model,
            asr_language=args.asr_language,
        )
        for index, pk in enumerate(record["agent_ids"], start=1)
    ]
    return env, agents


def episode_markdown(episode: Any) -> str:
    _, turns = episode.render_for_humans()
    return (
        "# Speech conversation\n\nEvaluation: not performed.\n\n"
        + "\n\n".join(turns[:-2])
        + "\n"
    )


async def run_one_episode(
    env: ParallelSotopiaEnv,
    agent_list: Sequence[CascadedSpeechAgent],
    args: argparse.Namespace,
    run_dir: Path,
    episode_id: str,
) -> dict[str, Any]:
    from sotopia.agents import Agents
    from sotopia.database import EpisodeLog
    from sotopia.messages import AgentAction
    from sotopia.messages.message_classes import ScriptBackground
    from talktopia.speech_agent import combine_conversation_audio, prepare_tts_text

    agents = Agents({agent.agent_name: agent for agent in agent_list})
    observations = env.reset(agents=agents, omniscient=False)
    agents.reset()
    # LLMAgent uses only the names here for recipient validation, not private goals.
    names_only = ScriptBackground(
        scenario="", agent_names=list(agents), agent_backgrounds=[], agent_goals=[]
    )
    for index, agent in enumerate(agent_list):
        agent.goal = env.profile.agent_goals[index]
        agent.script_background = names_only

    messages = [
        [
            ("Environment", name, obs.to_natural_language())
            for name, obs in observations.items()
        ]
    ]
    speech_path = run_dir / "simulation" / "speech" / f"{episode_id}.jsonl"
    speech_path.parent.mkdir(parents=True, exist_ok=True)
    utterances: list[tuple[Path, int]] = []
    with speech_path.open("x", encoding="utf-8") as speech_output:
        while True:
            actions = {}
            # Inactive agents receive the last turn; LLMAgent returns none without inference.
            for index, agent in enumerate(agent_list):
                action = await agent.aact(observations[agent.agent_name])
                action = AgentAction.model_validate(
                    action.model_dump(),
                    context={"agent_names": list(agents), "sender": agent.agent_name},
                )
                if (
                    action.action_type
                    not in observations[agent.agent_name].available_actions
                ):
                    raise ValueError(
                        f"{agent.agent_name} returned an unavailable action"
                    )
                if action.action_type == "speak":
                    listener = agent_list[1 - index]
                    tts_text = prepare_tts_text(action.argument)
                    speech_record = {
                        "turn": env.turn_number + 1,
                        "speaker": agent.agent_name,
                        "listener": listener.agent_name,
                        "to": action.to,
                        "llm_text": action.argument,
                        "tts_text": tts_text,
                        "tts_model": agent.tts_model,
                        "voice": agent.voice,
                        "wav_path": None,
                        "asr_model": listener.asr_model,
                        "asr_text": None,
                    }
                    if not tts_text:
                        speech_record.update(
                            status="tts_skipped", reason="no_spoken_text"
                        )
                        action = AgentAction(action_type="none", argument="", to=[])
                    else:
                        filename = f"{env.turn_number + 1:04d}_agent{index + 1}.wav"
                        wav_path = (
                            run_dir / "simulation" / "audio" / episode_id / filename
                        )
                        audio = await agent.synthesize(tts_text)
                        wav_path.parent.mkdir(parents=True, exist_ok=True)
                        wav_path.write_bytes(audio)
                        speech_record["wav_path"] = str(wav_path.relative_to(run_dir))
                        try:
                            transcript = await listener.transcribe(audio, filename)
                        except Exception:
                            speech_record["status"] = "asr_failed"
                            speech_output.write(
                                json.dumps(speech_record, ensure_ascii=False) + "\n"
                            )
                            speech_output.flush()
                            raise
                        speech_record.update(asr_text=transcript, status="completed")
                        utterances.append((wav_path, index))
                        action = action.model_copy(update={"argument": transcript})
                    speech_output.write(
                        json.dumps(speech_record, ensure_ascii=False) + "\n"
                    )
                    speech_output.flush()
                actions[agent.agent_name] = action
                messages[-1].append(
                    (agent.agent_name, "Environment", action.to_natural_language())
                )

            observations, _, terminated, _, info = await env.astep(actions)
            messages.append(
                [
                    ("Environment", name, obs.to_natural_language())
                    for name, obs in observations.items()
                ]
            )
            if all(terminated.values()):
                break

    conversation_audio = combine_conversation_audio(
        utterances,
        run_dir / "simulation" / "audio" / episode_id / "conversation.wav",
    )
    episode = EpisodeLog(
        environment=env.profile.pk,
        agents=[agent.profile.pk for agent in agent_list],
        tag=args.tag,
        models=[env.model_name] + [agent.model_name for agent in agent_list],
        agent_classes=[type(agent).__name__ for agent in agent_list],
        messages=messages,
        reasoning="Not evaluated. " + info[agent_list[0].agent_name]["comments"],
        rewards=[0.0, 0.0],
    )
    original_path = run_dir / "simulation" / "original" / f"{episode_id}.json"
    readable_path = run_dir / "simulation" / "readable" / f"{episode_id}.md"
    write_json(original_path, episode.model_dump(mode="json"))
    readable_path.parent.mkdir(parents=True, exist_ok=True)
    readable_path.write_text(episode_markdown(episode), encoding="utf-8")
    if args.push_to_db:
        episode.save()
        write_json(original_path, episode.model_dump(mode="json"))
    return {
        "episode_id": episode_id,
        "status": "completed",
        "env_id": env.profile.pk,
        "agent_ids": episode.agents,
        "turns": env.turn_number,
        "episode_pk": episode.pk or None,
        "evaluation_status": "not_performed",
        "original": str(original_path.relative_to(run_dir)),
        "readable": str(readable_path.relative_to(run_dir)),
        "speech": str(speech_path.relative_to(run_dir)),
        "conversation_audio": (
            str(conversation_audio.relative_to(run_dir)) if conversation_audio else None
        ),
    }


async def stage_3_simulate(
    records: Sequence[dict[str, Any]], args: argparse.Namespace, run_dir: Path
) -> int:
    from openai import AsyncOpenAI

    summary: dict[str, Any] = {
        "tag": args.tag,
        "run_id": run_dir.name,
        "status": "running",
        "evaluation_status": "not_performed",
        "canonical_text": "asr_transcript",
        "episodes": [],
    }
    summary_path = run_dir / "03_simulation.json"
    write_json(summary_path, summary)
    asr_key = (
        os.environ.get("TALKTOPIA_ASR_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )
    tts_key = (
        os.environ.get("TALKTOPIA_TTS_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )
    async with (
        AsyncOpenAI(
            base_url=args.asr_base_url, api_key=asr_key, timeout=120, max_retries=2
        ) as asr_client,
        AsyncOpenAI(
            base_url=args.tts_base_url, api_key=tts_key, timeout=120, max_retries=2
        ) as tts_client,
    ):

        async def run(index: int, record: dict[str, Any]) -> dict[str, Any]:
            episode_id = f"episode_{index:04d}"
            try:
                env, agents = build_episode(record, args, asr_client, tts_client)
                result = await run_one_episode(env, agents, args, run_dir, episode_id)
            except Exception as exc:
                result = {
                    "episode_id": episode_id,
                    "status": "failed",
                    "env_id": record["env_id"],
                    "agent_ids": record["agent_ids"],
                    "error": safe_error(exc),
                    "conversation_audio": None,
                    "evaluation_status": "not_performed",
                }
            print(f"{episode_id}: {result['status']}", flush=True)
            if result["status"] == "failed":
                print(result["error"], file=sys.stderr, flush=True)
            return result

        try:
            for start in range(0, len(records), args.batch_size):
                results = await asyncio.gather(
                    *[
                        run(index, record)
                        for index, record in enumerate(
                            records[start : start + args.batch_size], start=start + 1
                        )
                    ]
                )
                summary["episodes"].extend(results)
                write_json(summary_path, summary)
        except BaseException:
            summary["status"] = "interrupted"
            write_json(summary_path, summary)
            raise
    failed = sum(result["status"] == "failed" for result in summary["episodes"])
    summary.update(
        status="failed" if failed else "completed",
        failed=failed,
        completed=len(records) - failed,
    )
    write_json(summary_path, summary)
    return 1 if failed else 0


async def stage_4_reevaluate_existing(args: argparse.Namespace, run_dir: Path) -> int:
    from pydantic import Field, field_validator
    from sotopia.database import EpisodeLog, SotopiaDimensions
    from sotopia.envs.evaluators import (
        EpisodeLLMEvaluator,
        EvaluationForAgents,
        unweighted_aggregate_evaluate,
    )

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
                    raise ValueError(
                        "Every evaluation dimension needs a nonempty reason"
                    )
            # The engine iterates dict values; fix their order before it assigns labels.
            return {key: values[key] for key in ("agent_1", "agent_2")}

    source_path = args.episode_json.expanduser().resolve()
    summary_path = run_dir / "04_sotopia_eval_reevaluate_existing.json"
    summary: dict[str, Any] = {
        "run_id": run_dir.name,
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
        history = "\n".join(turns[:-2])
        history_path = run_dir / "evaluation/history/episode_0001.txt"
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
        expected = {
            (agent, dimension)
            for agent in ("agent_1", "agent_2")
            for dimension in SotopiaDimensions.model_fields
        }
        for attempt in range(1, args.reeval_max_retries + 2):
            summary["attempts"] = attempt
            write_json(summary_path, summary)
            print(
                f"SOTOPIA evaluation: attempt {attempt}/{args.reeval_max_retries + 1}",
                flush=True,
            )
            try:
                responses = await evaluator.__acall__(
                    turn_number=-1,
                    messages=None,
                    history=history,
                    num_agents=2,
                    temperature=0.0,
                )
                if (
                    len(responses) != len(expected)
                    or {(agent, dimension) for agent, ((dimension, _), _) in responses}
                    != expected
                ):
                    raise ValueError(
                        "Evaluator did not return all 14 dimension scores and reasons"
                    )
                response = unweighted_aggregate_evaluate(responses)
                break
            except Exception as exc:
                error = safe_error(exc)
                summary["attempt_errors"].append(error)
                print(error, file=sys.stderr, flush=True)
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
        original_path = run_dir / "evaluation/original/episode_0001.json"
        readable_path = run_dir / "evaluation/readable/episode_0001.md"
        lines = [
            "# SOTOPIA episode evaluation",
            "",
            f"Evaluator: {args.evaluator_model}",
            "",
            f"| Dimension | {names[0]} (agent1) | {names[1]} (agent2) |",
            "|---|---:|---:|",
        ]
        for dimension in (*SotopiaDimensions.model_fields, "overall_score"):
            lines.append(
                f"| {dimension} | {evaluated.rewards[0][1][dimension]:g} | {evaluated.rewards[1][1][dimension]:g} |"
            )
        lines.extend(
            [
                "",
                "Overall score is the unweighted mean of the seven original scores; it is not normalized.",
                "",
            ]
        )
        for index, name in enumerate(names, start=1):
            lines.extend([f"## {name} (agent{index})", ""])
            for agent, ((dimension, score), reason) in responses:
                if agent == f"agent_{index}":
                    lines.extend([f"### {dimension}: {score}", "", reason.strip(), ""])
        write_json(original_path, evaluated.model_dump(mode="json"))
        readable_path.parent.mkdir(parents=True, exist_ok=True)
        readable_path.write_text("\n".join(lines), encoding="utf-8")
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


def run_pipeline(args: argparse.Namespace) -> int:
    # SOTOPIA chooses its database classes at import time.
    os.environ["SOTOPIA_STORAGE_BACKEND"] = args.storage_backend
    from talktopia.task_space import configure_database

    db_path = configure_database(args.storage_backend)
    print(f"Talktopia DB: {db_path}")
    os.environ.setdefault("CUSTOM_API_KEY", "EMPTY")
    if args.stage != "reevaluate":
        manifest = read_manifest(args.sample_manifest) if args.sample_manifest else None
        profiles = stage_1_sample_env_profiles(args, manifest)
        records = stage_2_sample_characters(profiles, args, manifest)
    run_dir = create_run_directory(
        args.out_dir, args.reeval_tag if args.stage == "reevaluate" else args.tag
    )
    print(f"Run started; output: {run_dir}", flush=True)
    try:
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        write_json(
            run_dir / "run_config.json",
            {**config, "run_id": run_dir.name},
        )
        if not args.dry_run and args.stage != "sample":
            import gin
            from sotopia.generation_utils import generate

            generate.DEFAULT_BAD_OUTPUT_PROCESS_MODEL = args.bad_output_process_model
            gin.parse_config_file(
                str(
                    REPO_ROOT / "engine/sotopia_conf/generation_utils_conf/generate.gin"
                ),
                skip_unknown=True,
            )
        if args.stage == "reevaluate":
            return asyncio.run(stage_4_reevaluate_existing(args, run_dir))

        write_json(
            run_dir / "01_scenarios_and_social_goals.json",
            [
                {
                    "env_id": profile.pk,
                    **profile.model_dump(mode="json", exclude={"pk"}),
                }
                for profile in profiles
            ],
        )
        write_json(run_dir / "02_sampled_characters.json", records)
        if args.stage == "sample" or args.dry_run:
            print(f"Sampled {len(profiles)} environments and {len(records)} pairs.")
            print("No inference requests or DB writes performed.")
            return 0

        return asyncio.run(stage_3_simulate(records, args, run_dir))
    finally:
        print(f"Run finished; output: {run_dir}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Talktopia: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
