"""Sample scenarios, run speech conversations, and evaluate saved episodes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
from urllib.parse import urlsplit

from talktopia import evaluation
from talktopia import utils
from talktopia.experiment import (
    INTERACTION_MODES,
    MAX_TURNS,
    EPISODE_TIMEOUT_S,
    UNCOUNTED_ACTIONS,
    canonical_dataset,
    validate_dataset,
    validate_manifest,
)
from talktopia.utils import (
    completed_result,
    configure_generation,
    create_run_directory,
    file_hash,
    input_fingerprints,
    lock_run,
    read_manifest,
    result_artifacts,
    safe_error,
    save_run_config,
    validate_run_inputs,
    write_json,
)
from talktopia.models.config import (
    REPO_ROOT,
    SPEECH_BASE_URL,
    default_pipeline_models,
    MODEL_ALIASES,
    OLLAMA_ENDPOINTS,
    SPEECH_ENDPOINTS,
    OLLAMA_NUM_PARALLEL,
    OLLAMA_CONTEXT_LENGTH,
    TTS_BATCH_SIZE,
    SPEECH_WORKERS_PER_GPU,
    speech_worker_keys,
    local_alias,
    model_on_gpu,
    speech_url,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from sotopia.envs import ParallelSotopiaEnv
    from talktopia.speech_agent import CascadedSpeechAgent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    models = default_pipeline_models()
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
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
        "--interaction-mode",
        choices=INTERACTION_MODES,
        help="Conversation mode; new simulations default to round-robin.",
    )
    for feature, default in (
        ("backchannels", True),
        ("corrections", False),
        ("interruptions", False),
    ):
        parser.add_argument(
            f"--duplex-{feature}",
            action=argparse.BooleanOptionalAction,
            default=default,
            help=f"Allow {feature} in Surface5 (default: {default}).",
        )
    parser.set_defaults(
        max_turns=MAX_TURNS,
        episode_timeout_s=EPISODE_TIMEOUT_S,
        tag=None,
        reeval_tag=None,
        database_path=None,
    )
    for role in ("env", "agent1", "agent2"):
        parser.add_argument(f"--{role}-model", default=models[role])
    parser.add_argument(
        "--agent1-models",
        nargs="+",
        help="Local aliases for matrix rows; requires --agent2-models.",
    )
    parser.add_argument(
        "--agent2-models",
        nargs="+",
        help="Local aliases for matrix columns; same-model pairs are included.",
    )
    parser.add_argument(
        "--worker-gpu", choices=list(OLLAMA_ENDPOINTS), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        help="Resume saved settings and unfinished episodes only.",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=0,
        help="Maximum unfinished episodes per model pair this invocation; 0 means all 450.",
    )
    parser.add_argument(
        "--bad-output-process-model",
        default=models["agent2"],
        help="Local model used by SOTOPIA to repair malformed model output.",
    )
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        help="Reuse the complete canonical 450-combo manifest.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
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
    evaluation.add_arguments(parser, models)
    args = parser.parse_args(argv)
    if args.episode_limit < 0:
        parser.error("--episode-limit must be nonnegative")
    if args.resume_run:
        allowed = {
            "--resume-run",
            "--episode-limit",
            "--dry-run",
            "--evaluate-after-simulation",
            "--no-evaluate-after-simulation",
        }
        if any(
            token.split("=", 1)[0] not in allowed
            for token in argv
            if token.startswith("-")
        ):
            parser.error(
                "--resume-run only accepts --episode-limit, --dry-run and the automatic evaluation switch; experiment settings are loaded from the saved run"
            )
        return args
    if bool(args.agent1_models) != bool(args.agent2_models):
        parser.error("--agent1-models and --agent2-models must be supplied together")
    if args.agent1_models:
        if args.stage == "reevaluate" or args.worker_gpu or args.push_to_db:
            parser.error(
                "Matrix mode requires sampling/simulation, no --worker-gpu and no --push-to-db"
            )
        if any(
            token.split("=", 1)[0] in {"--agent1-model", "--agent2-model"}
            for token in argv
        ):
            parser.error(
                "Model lists cannot be combined with singular agent model options"
            )
        try:
            for role in ("agent1", "agent2"):
                aliases = [
                    local_alias(name) for name in getattr(args, f"{role}_models")
                ]
                tags = [MODEL_ALIASES[name]["model"] for name in aliases]
                if len(set(tags)) != len(tags) or any(
                    MODEL_ALIASES[name]["role"] != "agent" for name in aliases
                ):
                    raise ValueError(
                        "Each agent list must contain distinct agent models"
                    )
                setattr(args, f"{role}_models", aliases)
            local_alias(args.env_model)
            local_alias(args.bad_output_process_model)
        except ValueError as exc:
            parser.error(str(exc))
    if args.storage_backend != "local":
        parser.error(
            "Talktopia requires --storage-backend local; Redis is not supported"
        )
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.stage != "reevaluate" and args.interaction_mode is None:
        args.interaction_mode = "round-robin"
    if args.interaction_mode == "round-robin" and (
        not args.duplex_backchannels
        or args.duplex_corrections
        or args.duplex_interruptions
    ):
        parser.error(
            "Duplex action controls require --interaction-mode surface5-full-duplex"
        )
    evaluation.validate_arguments(parser, args)
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


def env_params(args: argparse.Namespace) -> dict[str, Any]:
    from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator
    from sotopia.messages import AgentAction

    class BudgetTerminatedEvaluator(RuleBasedTerminatedEvaluator):
        def __call__(self, turn_number, messages, **kwargs):
            # Keep engine step numbers for alternation and history. Only actual
            # actions consume the comparison budget; stale/leave rules stay intact.
            budget_turns = sum(
                isinstance(message, AgentAction)
                and message.action_type not in UNCOUNTED_ACTIONS
                for speaker, message in messages
                if speaker != "Environment"
            )
            return super().__call__(budget_turns, messages, **kwargs)

    return {
        "model_name": args.env_model,
        "action_order": "round-robin",
        "evaluators": [
            BudgetTerminatedEvaluator(max_turn_number=args.max_turns, max_stale_turn=2)
        ],
        "terminal_evaluators": [],
    }


def stage_1_sample_env_profiles(args, manifest=None) -> list[Any]:
    from sotopia.database import EnvironmentProfile

    source, _ = canonical_dataset()
    if manifest is not None:
        validate_manifest(manifest)
    profiles = [
        EnvironmentProfile.get(row["pk"])
        for row in sorted(source["EnvironmentProfile"], key=lambda row: row["pk"])
    ]
    if any(len(profile.agent_goals) != 2 for profile in profiles):
        raise ValueError("Every environment must have exactly two agent goals")
    return profiles


def stage_2_sample_characters(env_profiles, args, manifest=None) -> list[dict]:
    from talktopia.speech_agent import AgentProfile

    source, _ = canonical_dataset()
    environments = {profile.pk: profile for profile in env_profiles}
    records = []
    for index, combo in enumerate(
        sorted(
            source["EnvAgentComboStorage"], key=lambda row: (row["env_id"], row["pk"])
        ),
        1,
    ):
        profiles = [AgentProfile.get(pk) for pk in combo["agent_ids"]]
        names = [
            f"{profile.first_name} {profile.last_name}".strip() for profile in profiles
        ]
        if len(set(names)) != 2:
            raise ValueError("The two agent profiles must have distinct names")
        records.append(
            {
                "episode_id": f"episode_{index:04d}",
                "combo_id": combo["pk"],
                "env_id": combo["env_id"],
                "codename": environments[combo["env_id"]].codename,
                "agent_ids": combo["agent_ids"],
                "agent_names": names,
            }
        )
    validate_manifest(records)
    if manifest is not None:
        validate_manifest(manifest)
        # Names and scenario labels must also come from the canonical DB.
        if manifest != records:
            raise ValueError("Imported manifest differs from canonical records")
    return records


async def run_simulation_batch(
    records, args, run_dir, runner, *, concurrency_limit=None
) -> int:
    summary_path = run_dir / "03_simulation.json"
    if args.resume_run and summary_path.exists():
        if (
            json.loads(summary_path.read_text()).get("interaction_mode")
            != args.interaction_mode
        ):
            raise ValueError(
                "Saved simulation progress has a different interaction mode"
            )
    concurrency = args.batch_size
    if concurrency_limit is not None:
        concurrency = min(concurrency, concurrency_limit)
    from talktopia.full_duplex.episode import summarize_latencies

    return await utils.run_episode_batch(
        records,
        args,
        run_dir,
        "simulation",
        runner,
        summary_path=summary_path,
        initial_summary={
            "tag": args.tag,
            "evaluation_status": "not_performed",
            "canonical_text": "asr_transcript",
            "interaction_mode": args.interaction_mode,
            "coverage": validate_manifest(records),
        },
        concurrency=concurrency,
        max_attempts=3,
        validate_artifacts=result_artifacts,
        error_fields={
            "conversation_audio": None,
            "evaluation_status": "not_performed",
            "interaction_mode": args.interaction_mode,
        },
        attempt_fields=("speech_worker",),
        on_checkpoint=summarize_latencies
        if args.interaction_mode == "surface5-full-duplex"
        else None,
    )


def prepare_simulation_run(args, run_dir, db_path, records=None):
    manifest_path = run_dir / "02_sampled_characters.json"
    if args.resume_run:
        if args.stage == "sample":
            raise ValueError("Resume requires a simulation or batch evaluation run")
        validate_run_inputs(run_dir, db_path, manifest_path)
        records = read_manifest(manifest_path)
        validate_manifest(records)
        return records
    validate_manifest(records)
    write_json(manifest_path, records)
    save_run_config(args, run_dir, db_path, manifest_path)
    return records


def matrix_report(run_dir: Path, state: dict) -> None:
    pairs = []
    for pair in state["pairs"]:
        row = {
            key: pair.get(key)
            for key in (
                "pair_id",
                "agent1_model",
                "agent2_model",
                "gpu",
                "status",
                "run_dir",
                "error",
            )
        }
        row.update(
            simulation_total=state["episodes_per_pair"],
            simulation_completed=0,
            simulation_failed=0,
        )
        if pair.get("run_dir"):
            sim_path = Path(pair["run_dir"]) / "03_simulation.json"
            if sim_path.exists():
                sim = json.loads(sim_path.read_text())
                row.update(
                    simulation_completed=sim["completed"],
                    simulation_failed=sim["failed"],
                    simulation_wall_seconds=sim.get("wall_seconds", 0),
                )
        pairs.append(row)
    evaluation.write_matrix_report(run_dir, state, pairs)


def build_round_robin_episode(
    record: dict[str, Any],
    args: argparse.Namespace,
    asr_client: AsyncOpenAI,
    tts_client: AsyncOpenAI,
    tts_semaphore: asyncio.Semaphore | None = None,
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
            tts_semaphore=tts_semaphore,
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


async def run_round_robin_episode(
    env: ParallelSotopiaEnv,
    agent_list: Sequence[CascadedSpeechAgent],
    args: argparse.Namespace,
    run_dir: Path,
    episode_id: str,
    *,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    from sotopia.agents import Agents
    from sotopia.database import EpisodeLog
    from sotopia.messages import AgentAction
    from sotopia.messages.message_classes import ScriptBackground
    from talktopia.speech_agent import combine_conversation_audio, prepare_tts_text

    started = time.monotonic()
    budget_turns = 0
    action_counts = {}
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
    artifact_dir = artifact_dir or run_dir
    speech_path = artifact_dir / "simulation" / "speech" / f"{episode_id}.jsonl"
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
                            artifact_dir
                            / "simulation"
                            / "audio"
                            / episode_id
                            / filename
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
                if action.action_type not in UNCOUNTED_ACTIONS:
                    budget_turns += 1
                if observations[agent.agent_name].available_actions != ["none"]:
                    action_counts[action.action_type] = (
                        action_counts.get(action.action_type, 0) + 1
                    )
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
        artifact_dir / "simulation" / "audio" / episode_id / "conversation.wav",
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
    original_path = artifact_dir / "simulation" / "original" / f"{episode_id}.json"
    readable_path = artifact_dir / "simulation" / "readable" / f"{episode_id}.md"
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
        "budget_turns": budget_turns,
        "action_counts": action_counts,
        "interaction_mode": args.interaction_mode,
        "opener": "agent1",
        "wall_seconds": time.monotonic() - started,
        "end_reason": "max_turns"
        if budget_turns >= MAX_TURNS
        else "environment_terminated",
        "episode_pk": episode.pk or None,
        "evaluation_status": "not_performed",
        "original": str(original_path.relative_to(run_dir)),
        "readable": str(readable_path.relative_to(run_dir)),
        "speech": str(speech_path.relative_to(run_dir)),
        "conversation_audio": (
            str(conversation_audio.relative_to(run_dir)) if conversation_audio else None
        ),
    }


def build_duplex_episode(
    record: dict[str, Any], args, asr_client, tts_client, tts_semaphore=None
):
    from sotopia.database import EnvironmentProfile
    from talktopia.full_duplex.agent import CascadedDuplexAgent
    from talktopia.full_duplex.config import (
        ASR_DECODE_INTERVAL_MS,
        ASR_WINDOW_MS,
    )
    from talktopia.full_duplex.episode import RuntimeConfig, profile_name
    from talktopia.full_duplex.generation import DuplexGenerationEngine
    from talktopia.full_duplex.speech_backends import SentenceTTS, WindowedASR
    from talktopia.full_duplex.speech_client import SpeechClient
    from talktopia.full_duplex.sotopia_adapter import ResolvedEpisode
    from talktopia.speech_agent import AgentProfile
    from talktopia.task_space import database_path

    profiles = tuple(AgentProfile.get(pk) for pk in record["agent_ids"])
    resolved = ResolvedEpisode(
        combo_pk=record.get("combo_id") or record["env_id"],
        env_profile=EnvironmentProfile.get(record["env_id"]),
        agent_profiles=profiles,
    )
    agents = []
    db = database_path()
    for index, profile in enumerate(profiles, start=1):
        reference = (db / profile.voice_reference_wav).resolve()
        if not reference.is_relative_to(db):
            raise ValueError("Voice reference must be inside the Talktopia DB")
        speech = SpeechClient(
            asr_client,
            tts_client,
            voice_id=profile.voice_id,
            voice_reference=reference,
            asr_model=args.asr_model,
            tts_model=args.tts_model,
            asr_language=args.asr_language,
            tts_semaphore=tts_semaphore,
        )
        agents.append(
            CascadedDuplexAgent(
                profile=profile,
                generation=DuplexGenerationEngine(
                    getattr(args, f"agent{index}_model"),
                    max_attempts=RuntimeConfig().generation_max_attempts,
                ),
                asr=WindowedASR(
                    speech,
                    decode_interval_ms=ASR_DECODE_INTERVAL_MS,
                    window_ms=ASR_WINDOW_MS,
                ),
                tts=SentenceTTS(speech, source_agent=profile_name(profile)),
                voice_reference=reference,
                history_entries=RuntimeConfig().history_entries,
            )
        )
    return resolved, agents


async def run_duplex_episode(
    resolved,
    agent_list,
    args,
    run_dir: Path,
    episode_id: str,
    *,
    artifact_dir: Path | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    import time

    from sotopia.database import EpisodeLog
    from talktopia.full_duplex.audio import AudioRouter, StereoWavWriter
    from talktopia.full_duplex.config import (
        FRAME_MS,
        INTERACTION_MODE,
        SAMPLE_RATE_HZ,
        runtime_options,
    )
    from talktopia.full_duplex.episode import (
        RuntimeConfig,
        profile_background,
        write_speech_log,
    )
    from talktopia.full_duplex.events import (
        ActionCommitted,
        EpisodeEnded,
        EventWriter,
        read_events,
    )
    from talktopia.full_duplex.generation import AgentSessionContext
    from talktopia.full_duplex.rendering import render_sotopia_messages
    from talktopia.full_duplex.runtime import DuplexRuntime
    from talktopia.full_duplex.sotopia_adapter import SotopiaSession
    from talktopia.full_duplex.transcript import TranscriptBuilder

    artifact_dir = artifact_dir or run_dir
    base = artifact_dir / "simulation"
    events_path = base / "events" / f"{episode_id}.jsonl"
    speech_path = base / "speech" / f"{episode_id}.jsonl"
    wav_path = base / "audio" / episode_id / "conversation.wav"
    original_path = base / "original" / f"{episode_id}.json"
    readable_path = base / "readable" / f"{episode_id}.md"
    timeout_path = base / "diagnostics" / f"{episode_id}.json"
    latency_path = base / "latency" / f"{episode_id}.json"
    names = tuple(agent.agent_name for agent in agent_list)
    agents = tuple(agent_list)
    session = SotopiaSession(resolved, seed=args.seed)
    writer = EventWriter(events_path, episode_id)
    stereo = StereoWavWriter(wav_path, names)
    runtime = DuplexRuntime(
        resolved=resolved,
        agents=agents,
        session=session,
        event_writer=writer,
        audio_router=AudioRouter(names),
        stereo_writer=stereo,
        config=RuntimeConfig(**runtime_options(args)),
        model_names=tuple(agent.model_name for agent in agents),
        seed=args.seed,
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_ms=FRAME_MS,
    )
    opened = False
    started = time.monotonic()
    try:
        for index, agent in enumerate(agents):
            await agent.start_session(
                AgentSessionContext(
                    episode_id=episode_id,
                    agent_name=names[index],
                    peer_name=names[1 - index],
                    scenario=resolved.env_profile.scenario,
                    self_background=profile_background(agent.profile),
                    private_goal=resolved.env_profile.agent_goals[index],
                )
            )
        snapshot = session.open(agents)
        opened = True
        ended = await runtime.run(snapshot)
        write_json(
            latency_path,
            {
                "clock": "monotonic_elapsed_ms",
                "normal_response": "peer last audio delivery end to response first audio delivery start",
                "backchannel": "partial ASR first LLM request to backchannel first audio delivery start",
                "statistics": runtime.latency_summary(),
                "samples_ms": runtime.latencies,
            },
        )
        events = read_events(events_path)
        transcript = TranscriptBuilder.from_events(events)
        entries = transcript.build()
        write_speech_log(speech_path, entries, agents, args)
        commits = [event for event in events if isinstance(event, ActionCommitted)]
        episode = EpisodeLog(
            environment=resolved.env_profile.pk,
            agents=[agent.profile.pk for agent in agents],
            tag=args.tag,
            models=[args.env_model, *(agent.model_name for agent in agents)],
            agent_classes=[type(agent).__name__ for agent in agents],
            messages=render_sotopia_messages(snapshot.observations, commits, entries),
            reasoning=f"Not evaluated. Ended: {ended.reason}.",
            rewards=[0.0, 0.0],
        )
        if args.push_to_db:
            episode.save()
        write_json(original_path, episode.model_dump(mode="json"))
        _, turns = episode.render_for_humans()
        readable_path.parent.mkdir(parents=True, exist_ok=True)
        readable_path.write_text(
            "# Speech conversation\n\nEvaluation: not performed.\n\n"
            + "\n\n".join(turns[:-2])
            + "\n",
            encoding="utf-8",
        )
        writer.emit(
            EpisodeEnded,
            runtime.now_ms,
            status="completed",
            reason=ended.reason,
            duration_ms=ended.duration_ms,
        )
        return {
            "episode_id": episode_id,
            "status": "completed",
            "interaction_mode": INTERACTION_MODE,
            "env_id": resolved.env_profile.pk,
            "agent_ids": episode.agents,
            "turns": runtime.state.semantic_turn_number,
            "budget_turns": runtime.state.budget_turns,
            "action_counts": dict(runtime.state.action_counts),
            "opener": "agent1",
            "end_reason": ended.reason,
            "duration_ms": ended.duration_ms,
            "latency": runtime.latency_summary(),
            "latency_report": str(latency_path.relative_to(run_dir)),
            "wall_seconds": time.monotonic() - started,
            "episode_pk": episode.pk or None,
            "evaluation_status": "not_performed",
            "original": str(original_path.relative_to(run_dir)),
            "readable": str(readable_path.relative_to(run_dir)),
            "speech": str(speech_path.relative_to(run_dir)),
            "events": str(events_path.relative_to(run_dir)),
            "conversation_audio": str(wav_path.relative_to(run_dir)),
        }
    except BaseException as exc:
        timed_out = (
            isinstance(exc, asyncio.CancelledError)
            and deadline is not None
            and time.monotonic() >= deadline
        )
        write_json(
            timeout_path,
            {
                "episode_id": episode_id,
                "timeout_seconds": EPISODE_TIMEOUT_S,
                "runtime": runtime.liveness_snapshot().model_dump(mode="json"),
                "budget_turns": runtime.state.budget_turns,
            },
        )
        writer.emit(
            EpisodeEnded,
            runtime.now_ms,
            status="cancelled"
            if isinstance(exc, asyncio.CancelledError) and not timed_out
            else "failed",
            reason="episode_timeout" if timed_out else "episode_error",
            duration_ms=runtime.now_ms,
            error_type=type(exc).__name__,
            error_message=safe_error(exc),
        )
        # Failed attempts retain their delivered audio and transcript for inspection.
        write_speech_log(
            speech_path,
            TranscriptBuilder.from_events(read_events(events_path)).build(),
            agents,
            args,
        )
        raise
    finally:
        await asyncio.gather(
            *(agent.stop_session() for agent in agents), return_exceptions=True
        )
        if opened:
            session.close()
        stereo.close()
        writer.close()


def build_episode(record, args, asr_client, tts_client, tts_semaphore=None):
    build = (
        build_round_robin_episode
        if args.interaction_mode == "round-robin"
        else build_duplex_episode
    )
    return build(record, args, asr_client, tts_client, tts_semaphore)


async def run_one_episode(
    resolved, agents, args, run_dir, episode_id, *, artifact_dir=None
):
    run = (
        run_round_robin_episode
        if args.interaction_mode == "round-robin"
        else run_duplex_episode
    )
    timeout = asyncio.timeout(EPISODE_TIMEOUT_S)
    try:
        deadline = time.monotonic() + EPISODE_TIMEOUT_S
        options = {"artifact_dir": artifact_dir}
        if args.interaction_mode == "surface5-full-duplex":
            options["deadline"] = deadline
        async with timeout:
            return await run(resolved, agents, args, run_dir, episode_id, **options)
    except TimeoutError:
        if not timeout.expired():
            raise
        path = (
            (artifact_dir or run_dir)
            / "simulation"
            / "diagnostics"
            / f"{episode_id}.json"
        )
        diagnostic = json.loads(path.read_text()) if path.exists() else {}
        write_json(
            path,
            {
                **diagnostic,
                "episode_id": episode_id,
                "interaction_mode": args.interaction_mode,
                "timeout_seconds": EPISODE_TIMEOUT_S,
                "reason": "episode_timeout",
                "turns": getattr(resolved, "turn_number", None),
            },
        )
        raise TimeoutError(f"Episode exceeded {EPISODE_TIMEOUT_S:g} seconds") from None


async def stage_3_simulate(
    records: Sequence[dict[str, Any]], args: argparse.Namespace, run_dir: Path
) -> int:
    from openai import AsyncOpenAI
    from talktopia.speech_agent import SpeechServerPool

    records = [
        {**record, "episode_id": record.get("episode_id", f"episode_{index:04d}")}
        for index, record in enumerate(records, start=1)
    ]
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
    if args.worker_gpu:
        workers = speech_worker_keys(args.worker_gpu)
    elif args.asr_base_url == args.tts_base_url == SPEECH_BASE_URL:
        workers = speech_worker_keys()
    else:
        workers = []  # Explicit external APIs retain their configured routing.
    if workers:
        pool = SpeechServerPool(workers)

        async def run(record, artifact_dir, result_path):
            async with pool.lease(record["episode_id"]) as (worker, url):
                record["speech_worker"] = worker
                async with (
                    AsyncOpenAI(
                        base_url=url, api_key=asr_key, timeout=120, max_retries=0
                    ) as asr_client,
                    AsyncOpenAI(
                        base_url=url, api_key=tts_key, timeout=120, max_retries=0
                    ) as tts_client,
                ):
                    resolved, agents = build_episode(
                        record, args, asr_client, tts_client
                    )
                    return await run_one_episode(
                        resolved,
                        agents,
                        args,
                        run_dir,
                        record["episode_id"],
                        artifact_dir=artifact_dir,
                    )

        return await run_simulation_batch(
            records, args, run_dir, run, concurrency_limit=len(workers)
        )
    async with (
        AsyncOpenAI(
            base_url=args.asr_base_url, api_key=asr_key, timeout=120, max_retries=2
        ) as asr_client,
        AsyncOpenAI(
            base_url=args.tts_base_url, api_key=tts_key, timeout=120, max_retries=0
        ) as tts_client,
    ):
        # Wait here, before the HTTP timeout starts, instead of filling the TTS
        # server queue with more requests than it can generate in one batch.
        tts_semaphore = asyncio.Semaphore(TTS_BATCH_SIZE)

        async def run(record, artifact_dir, result_path):
            resolved, agents = build_episode(
                record, args, asr_client, tts_client, tts_semaphore
            )
            return await run_one_episode(
                resolved,
                agents,
                args,
                run_dir,
                record["episode_id"],
                artifact_dir=artifact_dir,
            )

        return await run_simulation_batch(records, args, run_dir, run)


def resolve_run_args(args):
    if args.resume_run:
        args = utils.restore_run(
            args,
            path_fields=("out_dir", "sample_manifest", *evaluation.PATH_OPTIONS),
            overrides=evaluation.resume_overrides(args),
        )
    elif args.stage == "reevaluate":
        from talktopia.evaluation.cli import resolve_source_context

        resolve_source_context(args)
    if args.database_path:
        explicit_db = os.environ.get("TALKTOPIA_DB_DIR")
        if explicit_db and Path(explicit_db).expanduser().resolve() != Path(
            args.database_path
        ):
            raise ValueError("TALKTOPIA_DB_DIR differs from the saved/source database")
        os.environ["TALKTOPIA_DB_DIR"] = str(args.database_path)
    return args


def preflight(args):
    """Validate before server startup, and recheck in the process doing the run."""
    os.environ["SOTOPIA_STORAGE_BACKEND"] = args.storage_backend
    from talktopia.task_space import configure_database

    db_path = configure_database(args.storage_backend)
    args.database_path = str(db_path)
    if args.stage != "reevaluate":
        validate_dataset(db_path)
        if args.resume_run:
            manifest_path = args.resume_run / "02_sampled_characters.json"
            validate_run_inputs(args.resume_run, db_path, manifest_path)
            validate_manifest(read_manifest(manifest_path))
        else:
            manifest = (
                read_manifest(args.sample_manifest) if args.sample_manifest else None
            )
            profiles = stage_1_sample_env_profiles(args, manifest)
            stage_2_sample_characters(profiles, args, manifest)
    return db_path


def run_pipeline(args: argparse.Namespace) -> int:
    args = resolve_run_args(args)
    db_path = preflight(args)
    print(f"Talktopia DB: {db_path}")
    os.environ.setdefault("CUSTOM_API_KEY", "EMPTY")
    records = None
    if args.resume_run:
        run_dir = args.resume_run
    elif args.stage != "reevaluate":
        manifest = read_manifest(args.sample_manifest) if args.sample_manifest else None
        profiles = stage_1_sample_env_profiles(args, manifest)
        records = stage_2_sample_characters(profiles, args, manifest)
        for index, record in enumerate(records, start=1):
            record.setdefault("episode_id", f"episode_{index:04d}")
    if not args.resume_run:
        run_dir = create_run_directory(
            args.out_dir,
            args.interaction_mode + ("_eval" if args.stage == "reevaluate" else ""),
        )
    if not args.resume_run:
        args.tag = args.reeval_tag = run_dir.name
    print(f"Run started; output: {run_dir}", flush=True)
    try:
        with lock_run(run_dir):
            return asyncio.run(run_locked_pipeline(args, run_dir, db_path, records))
    finally:
        print(f"Run finished; output: {run_dir}", flush=True)


async def run_locked_pipeline(args, run_dir, db_path, records=None) -> int:
    if args.stage == "reevaluate":
        return await evaluation.run_evaluation(args, run_dir, db_path)
    records = prepare_simulation_run(args, run_dir, db_path, records)
    if getattr(args, "agent1_models", None):
        return await run_matrix(args, run_dir, db_path, records)
    if args.stage in {"all", "simulate"}:
        status = await evaluation.resume_linked_evaluation(args, run_dir, db_path)
        if status is not None:
            return status
    if not args.dry_run and args.stage != "sample":
        configure_generation(args)
    if args.dry_run and args.resume_run:
        return await run_simulation_batch(records, args, run_dir, None)
    if args.stage == "sample" or args.dry_run:
        print(
            "Sampled %s pairs. No inference requests or DB writes performed."
            % len(records)
        )
        return 0
    from talktopia.models.servers import release_worker_models

    await asyncio.to_thread(
        release_worker_models,
        args,
        keep_models=[
            args.agent1_model,
            args.agent2_model,
            args.bad_output_process_model,
        ],
    )
    status = await stage_3_simulate(records, args, run_dir)
    return await evaluation.finish_simulation(args, run_dir, db_path, status)


def gpu_snapshot() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    rows = []
    for line in output.splitlines():
        index, uuid, utilization, used, total = [
            value.strip() for value in line.split(",")
        ]
        rows.append(
            dict(
                index=int(index),
                uuid=uuid,
                utilization=int(utilization),
                used_mib=int(used),
                total_mib=int(total),
            )
        )
    return rows


def matrix_runtime(args) -> dict[str, Any]:
    """Verify every selected tag on both servers; freeze actual model digests."""
    from talktopia.models.servers import (
        get_json,
        speech_health,
        validate_ollama_settings,
    )

    aliases = [
        *args.agent1_models,
        *args.agent2_models,
        local_alias(args.evaluator_model),
        local_alias(args.bad_output_process_model),
    ]
    required = {MODEL_ALIASES[alias]["model"] for alias in aliases}
    digests = {}
    for endpoint, spec in OLLAMA_ENDPOINTS.items():
        validate_ollama_settings(endpoint)
        for worker in speech_worker_keys(endpoint):
            speech_health(worker)
        available = {
            item["name"]: item for item in get_json(f"{spec['url']}/api/tags")["models"]
        }
        missing = required - available.keys()
        if missing:
            raise ValueError(f"{endpoint}: missing selected models {sorted(missing)}")
        digests[endpoint] = {
            name: available[name]["digest"] for name in sorted(required)
        }
    if len({json.dumps(value, sort_keys=True) for value in digests.values()}) != 1:
        raise ValueError("GPU servers have different model digests")
    return dict(
        model_digests=digests,
        ollama_num_parallel=OLLAMA_NUM_PARALLEL,
        ollama_context_length=OLLAMA_CONTEXT_LENGTH,
        tts_batch_size=TTS_BATCH_SIZE,
        speech_workers_per_gpu=SPEECH_WORKERS_PER_GPU,
        ollama_endpoints=OLLAMA_ENDPOINTS,
        speech_endpoints=SPEECH_ENDPOINTS,
        gpu_uuids={str(row["index"]): row["uuid"] for row in gpu_snapshot()},
    )


def check_matrix_resources(
    run_dir: Path, *, strict: bool = True
) -> list[dict[str, Any]]:
    from talktopia.models.servers import get_json, speech_health

    if shutil.disk_usage(run_dir).free < 20 * 1024**3:
        raise RuntimeError(
            "Less than 20 GiB free; matrix paused before filling the disk"
        )
    warnings = []
    for worker in SPEECH_ENDPOINTS:
        try:
            speech_health(worker)
        except Exception as exc:
            warnings.append(f"{worker}: {safe_error(exc)}")
    for endpoint, spec in OLLAMA_ENDPOINTS.items():
        try:
            for model in get_json(f"{spec['url']}/api/ps").get("models", []):
                if (
                    model.get("size", 0)
                    and model.get("size_vram", 0) < model["size"] * 0.98
                ):
                    warnings.append(
                        f"{endpoint}: {model['name']} is partly offloaded to CPU"
                    )
        except Exception as exc:
            warnings.append(f"{endpoint}: {safe_error(exc)}")
    try:
        rows = gpu_snapshot()
    except Exception as exc:
        rows = []
        warnings.append(f"GPU monitoring: {safe_error(exc)}")
    for row in rows:
        if (
            f"gpu{row['index']}" in OLLAMA_ENDPOINTS
            and row["used_mib"] > row["total_mib"] * 0.9
        ):
            warnings.append(f"GPU {row['index']} exceeds the 90% memory budget")
    if warnings:
        message = "; ".join(warnings)
        if strict:
            raise RuntimeError(message)
        print(
            f"Resource warning; other work continues: {message}",
            file=sys.stderr,
            flush=True,
        )
    return rows


async def run_matrix(args, run_dir: Path, db_path: Path, records: list) -> int:
    """Two GPU workers, each reusing the existing single-pair subprocess workflow."""
    matrix_path = run_dir / "matrix_manifest.json"
    state_path = run_dir / "matrix_progress.json"
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text())
    expected = [
        dict(pair_id=f"pair_{index:02d}", agent1_model=left, agent2_model=right)
        for index, (left, right) in enumerate(
            (
                (left, right)
                for left in args.agent1_models
                for right in args.agent2_models
            ),
            1,
        )
    ]
    if matrix_path.exists():
        if (
            json.loads(matrix_path.read_text()) != expected
            or file_hash(matrix_path) != config["matrix_manifest_sha256"]
        ):
            raise ValueError("Frozen model matrix changed; resume refused")
    else:
        write_json(matrix_path, expected)
        config.update(
            kind="model_matrix", matrix_manifest_sha256=file_hash(matrix_path)
        )
        write_json(config_path, config)
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else dict(
            status="prepared",
            episodes_per_pair=len(records),
            interaction_mode=args.interaction_mode,
            coverage={
                **validate_manifest(records),
                "model_pairs": len(expected),
                "planned_episodes": len(records) * len(expected),
            },
            pairs=[{**pair, "status": "pending"} for pair in expected],
            wall_seconds=0,
        )
    )
    if state.get("interaction_mode") != args.interaction_mode:
        raise ValueError("Saved matrix has a different interaction mode")
    if [
        {key: pair[key] for key in ("pair_id", "agent1_model", "agent2_model")}
        for pair in state["pairs"]
    ] != expected or state["episodes_per_pair"] != len(records):
        raise ValueError("Matrix progress does not match its frozen manifest")
    if args.dry_run or args.stage == "sample":
        if not state_path.exists():
            write_json(state_path, state)
            matrix_report(run_dir, state)
        print(
            f"Prepared {len(expected)} model pairs x {len(records)} episodes = {len(expected) * len(records)}; no inference.",
            flush=True,
        )
        return 0
    runtime = await asyncio.to_thread(matrix_runtime, args)
    if config.get("runtime") and config["runtime"] != runtime:
        raise ValueError(
            "Models, GPU identities or server settings changed; resume refused"
        )
    config["runtime"] = runtime
    write_json(config_path, config)
    await asyncio.to_thread(check_matrix_resources, run_dir)
    for pair in state["pairs"]:
        if pair.get("run_dir"):
            # An orphaned child may survive an ungraceful parent kill. Do not duplicate it.
            with lock_run(Path(pair["run_dir"])):
                pass
            if pair["status"] in {"completed", "completed_with_failures"}:
                await asyncio.to_thread(validate_matrix_pair, pair, records)
                simulation = json.loads(
                    (Path(pair["run_dir"]) / "03_simulation.json").read_text()
                )
                if evaluation.pair_needs_retry(args, simulation):
                    pair["status"] = "pending"
    started = time.monotonic()
    previous_wall = state.get("wall_seconds", 0)
    state["status"] = "running"

    def checkpoint():
        state["wall_seconds"] = previous_wall + time.monotonic() - started
        write_json(state_path, state)

    processed = set()

    def claim(endpoint):
        for pair in state["pairs"]:
            if pair["pair_id"] in processed:
                continue
            if pair["status"] in {"completed", "completed_with_failures", "running"}:
                continue
            if pair.get("gpu") not in (None, endpoint):
                continue
            pair.update(gpu=endpoint, status="running")
            checkpoint()
            return pair
        return None

    # Only an in-flight state is reset. Terminal per-episode failures keep their
    # frozen evaluation selection; explicit single-pair retry remains available.
    for pair in state["pairs"]:
        if pair["status"] == "running":
            pair["status"] = "interrupted"
    checkpoint()

    async def worker(endpoint):
        while (pair := claim(endpoint)) is not None:
            if (
                await asyncio.to_thread(input_fingerprints, db_path)
                != config["input_fingerprints"]
            ):
                raise ValueError(
                    "Code or data changed while matrix was running; stopped before the next pair"
                )
            if not pair.get("run_dir"):
                pair_root = run_dir / "pairs" / pair["pair_id"]
                values = {
                    **vars(args),
                    "agent1_models": None,
                    "agent2_models": None,
                    "worker_gpu": endpoint,
                    "sample_manifest": run_dir / "02_sampled_characters.json",
                    "resume_run": None,
                    "out_dir": pair_root,
                    **evaluation.pair_settings(args, pair["pair_id"], endpoint),
                    "asr_base_url": speech_url(endpoint),
                    "tts_base_url": speech_url(endpoint),
                }
                for role in ("agent1", "agent2"):
                    values[f"{role}_model"] = model_on_gpu(
                        pair[f"{role}_model"], endpoint
                    )
                for role in ("env", "bad_output_process"):
                    values[f"{role}_model"] = model_on_gpu(
                        getattr(args, f"{role}_model"), endpoint
                    )
                child_args = argparse.Namespace(**values)
                child_dir = create_run_directory(
                    pair_root, args.interaction_mode + "_" + pair["pair_id"]
                )
                child_args.tag = child_args.reeval_tag = child_dir.name
                stage_1_sample_env_profiles(child_args, records)
                prepare_simulation_run(child_args, child_dir, db_path, records)
                pair["run_dir"] = str(child_dir)
                checkpoint()
            child_dir = Path(pair["run_dir"])
            child_config = json.loads((child_dir / "run_config.json").read_text())
            if child_config["interaction_mode"] != args.interaction_mode:
                raise ValueError("Matrix child has a different interaction mode")
            command = [
                sys.executable,
                "-m",
                "talktopia.pipeline",
                "--resume-run",
                str(child_dir),
                "--episode-limit",
                str(args.episode_limit),
            ]
            command.append(evaluation.automatic_switch(args))
            process = None
            try:
                with (child_dir / "worker.log").open("a") as output:
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=REPO_ROOT,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    pair["pid"] = process.pid
                    checkpoint()
                    code = await process.wait()
                sim = json.loads((child_dir / "03_simulation.json").read_text())
                if sim["status"] in {"running", "interrupted"} or code not in (0, 1):
                    raise RuntimeError(
                        f"{pair['pair_id']}: worker stopped unexpectedly; see {child_dir / 'worker.log'}"
                    )
                if sim["pending"]:
                    if not args.episode_limit:
                        raise RuntimeError(
                            f"{pair['pair_id']}: unexpected unfinished simulations"
                        )
                    pair["status"] = "partial"
                else:
                    evaluation.check_pair_finished(args, pair["pair_id"], sim)
                    pair["status"] = "completed_with_failures" if code else "completed"
                pair["exit_code"] = code
                pair.pop("error", None)
                saved_args = argparse.Namespace(
                    **json.loads((child_dir / "run_config.json").read_text())
                )
                from talktopia.models.servers import release_worker_models

                await asyncio.to_thread(
                    release_worker_models, saved_args, keep_models=[]
                )
                print(f"{endpoint} {pair['pair_id']}: {pair['status']}", flush=True)
            except Exception as exc:
                # A dead child must not cancel the other GPU or be claimed in
                # an endless loop. Keep its unfinished episodes for explicit resume.
                pair.update(status="failed", exit_code=1, error=safe_error(exc))
                processed.add(pair["pair_id"])
                print(
                    f"{endpoint} {pair['pair_id']}: {pair['error']}; continuing",
                    file=sys.stderr,
                    flush=True,
                )
            except BaseException as exc:
                pair.update(status="interrupted", error=safe_error(exc))
                raise
            finally:
                if process and process.returncode is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=15)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                pair.pop("pid", None)
                checkpoint()
                matrix_report(run_dir, state)
            if args.episode_limit:
                # Partial pairs are resumed only on the next invocation.
                processed.add(pair["pair_id"])

    async def monitor():
        failures = 0
        checks = 0
        while True:
            try:
                resources = await asyncio.to_thread(
                    check_matrix_resources, run_dir, strict=False
                )
                failures = 0
                with (run_dir / "gpu_metrics.jsonl").open("a") as output:
                    output.write(
                        json.dumps(
                            dict(time=datetime.now(UTC).isoformat(), gpus=resources)
                        )
                        + "\n"
                    )
            except Exception:
                failures += 1
                if failures >= 3:
                    raise
            checkpoint()
            checks += 1
            if checks % 6 == 0:
                matrix_report(run_dir, state)
            await asyncio.sleep(5)

    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    workers = [asyncio.create_task(worker(endpoint)) for endpoint in OLLAMA_ENDPOINTS]
    monitoring = asyncio.create_task(monitor())
    group = asyncio.gather(*workers)
    try:
        await asyncio.wait([group, monitoring], return_when=asyncio.FIRST_COMPLETED)
        if monitoring.done():
            monitoring.result()
        await group
        state["status"] = (
            "partial"
            if any(pair["status"] == "partial" for pair in state["pairs"])
            else (
                "completed_with_failures"
                if any(
                    pair["status"] in {"completed_with_failures", "failed"}
                    for pair in state["pairs"]
                )
                else "completed"
            )
        )
    except BaseException as exc:
        state.update(status="interrupted", error=safe_error(exc))
        raise
    finally:
        for pending in [*workers, monitoring]:
            pending.cancel()
        await asyncio.gather(*workers, monitoring, return_exceptions=True)
        await asyncio.gather(group, return_exceptions=True)
        loop.remove_signal_handler(signal.SIGTERM)
        checkpoint()
        matrix_report(run_dir, state)
    return int(any(pair.get("exit_code", 0) for pair in state["pairs"]))


def validate_matrix_pair(pair: dict, records: list) -> None:
    path = Path(pair["run_dir"])
    config = json.loads((path / "run_config.json").read_text())
    if config["worker_gpu"] != pair["gpu"] or any(
        config[f"agent{index}_model"]
        != model_on_gpu(pair[f"agent{index}_model"], pair["gpu"])
        for index in (1, 2)
    ):
        raise ValueError("Saved pair model or GPU assignment changed")
    manifest = path / "02_sampled_characters.json"
    if (
        json.loads(manifest.read_text()) != records
        or file_hash(manifest) != config["manifest_sha256"]
    ):
        raise ValueError("Saved pair no longer uses the common episode manifest")
    sim = json.loads((path / "03_simulation.json").read_text())
    if sim["pending"]:
        raise ValueError("Finished pair has pending episodes")
    for row in sim["episodes"]:
        if row["status"] == "completed" and completed_result(row, path) is None:
            raise ValueError(
                f"Saved completed artifact changed: {path} / {row['episode_id']}"
            )
    if config["evaluate_after_simulation"]:
        evaluation.validate_linked_evaluation(sim)


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
