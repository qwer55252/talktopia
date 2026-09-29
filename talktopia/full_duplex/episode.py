"""Run one duplex conversation inside Talktopia's episode worker lease."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from talktopia.utils import write_json

from .config import RuntimeConfig as RuntimeConfig


def profile_name(profile) -> str:
    return f"{profile.first_name} {profile.last_name}".strip()


def profile_background(profile) -> str:
    fields = (
        ("name", profile_name(profile)),
        ("age", profile.age),
        ("occupation", profile.occupation),
        ("gender", profile.gender),
        ("pronouns", profile.gender_pronoun),
        ("public information", profile.public_info),
        ("personality and values", profile.personality_and_values),
        ("decision style", profile.decision_making_style),
        ("secret", profile.secret),
    )
    return "\n".join(f"{label}: {value}" for label, value in fields)


async def run_with_timeout(runtime, snapshot, seconds: float, diagnostic_path: Path):
    task = asyncio.create_task(runtime.run(snapshot))
    try:
        done, _ = await asyncio.wait({task}, timeout=seconds)
        if task in done:
            return task.result()
        # Capture the work in progress before cancellation clears agent state.
        write_json(
            diagnostic_path,
            {
                "episode_id": runtime.state.episode_id,
                "timeout_seconds": seconds,
                "runtime": runtime.liveness_snapshot().model_dump(mode="json"),
            },
        )
        raise TimeoutError(f"Episode exceeded {seconds:g} seconds")
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def write_speech_log(path, entries, agents, args):
    """Keep existing speech field names alongside the duplex timing and identity."""
    voices = {agent.agent_name: agent.profile.voice_id for agent in agents}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for entry in entries:
            row = {
                **entry.model_dump(mode="json"),
                "llm_text": entry.generated_text,
                "tts_text": entry.synthesized_text,
                "asr_text": entry.received_text,
                "voice": voices[entry.speaker],
                "tts_model": args.tts_model,
                "asr_model": args.asr_model,
                "wav_path": None,
                "status": "committed" if entry.commit_id else "not_committed",
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_latencies(summary: dict) -> None:
    """Use observation counts, not an unweighted mean of episode means."""
    summary["latency"] = {}
    for kind in ("normal_response", "backchannel"):
        values = [
            row["latency"][kind]
            for row in summary["episodes"]
            if row["status"] == "completed" and "latency" in row
        ]
        count = sum(value["count"] for value in values)
        total = sum(
            value["mean_ms"] * value["count"] for value in values if value["count"]
        )
        summary["latency"][kind] = {
            "count": count,
            "mean_ms": total / count if count else None,
        }
    summary["latency_scope"] = "completed_episodes"
