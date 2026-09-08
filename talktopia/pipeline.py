"""Sample SOTOPIA scenarios and run turn-based speech conversations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
from urllib.parse import urlsplit

from talktopia.models.config import REPO_ROOT, SPEECH_BASE_URL, default_pipeline_models

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from sotopia.envs import ParallelSotopiaEnv

    from talktopia.speech_agent import CascadedSpeechAgent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    models = default_pipeline_models()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "sample", "simulate"], default="all")
    parser.add_argument(
        "--storage-backend",
        choices=["local", "redis"],
        default=os.environ.get("SOTOPIA_STORAGE_BACKEND", "local"),
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--tag", default="talktopia_pipeline")
    for role in ("env", "agent1", "agent2"):
        parser.add_argument(f"--{role}-model", default=models[role])
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
        help="Write samples only; no inference or DB writes.",
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
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        parser.error("--tag must use letters, digits, dots, underscores or hyphens")
    if args.stage != "sample" and not args.dry_run:
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")


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
                    filename = f"{env.turn_number + 1:04d}_agent{index + 1}.wav"
                    wav_path = run_dir / "simulation" / "audio" / episode_id / filename
                    audio = await agent.synthesize(action.argument)
                    wav_path.parent.mkdir(parents=True, exist_ok=True)
                    wav_path.write_bytes(audio)
                    speech_record = {
                        "turn": env.turn_number + 1,
                        "speaker": agent.agent_name,
                        "listener": listener.agent_name,
                        "to": action.to,
                        "llm_text": action.argument,
                        "tts_model": agent.tts_model,
                        "voice": agent.voice,
                        "wav_path": str(wav_path.relative_to(run_dir)),
                        "asr_model": listener.asr_model,
                        "asr_text": None,
                    }
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
                    speech_output.write(
                        json.dumps(speech_record, ensure_ascii=False) + "\n"
                    )
                    speech_output.flush()
                    action = action.model_copy(update={"argument": transcript})
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
    }


async def stage_3_simulate(
    records: Sequence[dict[str, Any]], args: argparse.Namespace, run_dir: Path
) -> int:
    from openai import AsyncOpenAI

    summary: dict[str, Any] = {
        "tag": args.tag,
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
                error = f"{type(exc).__name__}: {exc}"
                for secret in (asr_key, tts_key, os.environ.get("CUSTOM_API_KEY", "")):
                    if secret and secret != "EMPTY":
                        error = error.replace(secret, "[redacted]")
                result = {
                    "episode_id": episode_id,
                    "status": "failed",
                    "env_id": record["env_id"],
                    "agent_ids": record["agent_ids"],
                    "error": error,
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


def run_pipeline(args: argparse.Namespace) -> int:
    # SOTOPIA chooses its database classes at import time.
    os.environ["SOTOPIA_STORAGE_BACKEND"] = args.storage_backend
    from talktopia.task_space import configure_database

    db_path = configure_database(args.storage_backend)
    print(f"Talktopia DB: {db_path}")
    os.environ.setdefault("CUSTOM_API_KEY", "EMPTY")
    run_dir = args.out_dir.expanduser().resolve() / args.tag
    if run_dir.exists():
        raise ValueError(f"Output already exists: {run_dir}. Choose a new --tag.")
    manifest = read_manifest(args.sample_manifest) if args.sample_manifest else None
    profiles = stage_1_sample_env_profiles(args, manifest)
    records = stage_2_sample_characters(profiles, args, manifest)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "01_scenarios_and_social_goals.json",
        [
            {"env_id": profile.pk, **profile.model_dump(mode="json", exclude={"pk"})}
            for profile in profiles
        ],
    )
    write_json(run_dir / "02_sampled_characters.json", records)
    write_json(
        run_dir / "run_config.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    if args.stage == "sample" or args.dry_run:
        print(
            f"Sampled {len(profiles)} environments and {len(records)} pairs in {run_dir}."
        )
        print("No inference requests or DB writes performed.")
        return 0

    import gin
    from sotopia.generation_utils import generate

    generate.DEFAULT_BAD_OUTPUT_PROCESS_MODEL = args.bad_output_process_model
    gin.parse_config_file(
        str(
            REPO_ROOT
            / "engine"
            / "sotopia_conf"
            / "generation_utils_conf"
            / "generate.gin"
        ),
        skip_unknown=True,
    )
    return asyncio.run(stage_3_simulate(records, args, run_dir))


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
