"""Fixed comparison settings and the complete GeminiLight experiment population."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from talktopia.models.config import REPO_ROOT

INTERACTION_MODES = ("round-robin", "surface5-full-duplex")
MAX_TURNS = 12
EPISODE_TIMEOUT_S = 120.0
UNCOUNTED_ACTIONS = frozenset({"none", "backchanneling"})


def settings(mode: str) -> dict:
    if mode not in INTERACTION_MODES:
        raise ValueError(f"Unknown interaction mode: {mode}")
    return {
        "interaction_mode": mode,
        "max_turns": MAX_TURNS,
        "episode_timeout_s": EPISODE_TIMEOUT_S,
        "turn_budget": "confirmed_actions_except_none_and_backchanneling_v1",
        "opener": "agent1",
        "prompt_max_words": 40,
        "validated_max_words": 50 if mode == "surface5-full-duplex" else None,
    }


def dataset_directory() -> Path:
    return (
        Path(
            os.environ.get(
                "TALKTOPIA_GEMINILIGHT_DATA_DIR",
                str(REPO_ROOT / "data" / "geminilight_sotopia_dataset"),
            )
        )
        .expanduser()
        .resolve()
    )


def canonical_dataset() -> tuple[dict[str, list[dict]], dict]:
    """Read only the pinned source files; never silently download or replace data."""
    lock = json.loads((REPO_ROOT / "dataset.lock.json").read_text())
    rows = {}
    directory = dataset_directory()
    for name, spec in lock.items():
        path = directory / spec["file"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError(f"Canonical dataset hash mismatch: {path}")
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(records) != spec["count"] or len({row["pk"] for row in records}) != len(
            records
        ):
            raise ValueError(f"Invalid canonical collection: {name}")
        rows[name] = records
    return rows, lock


def validate_dataset(db_path: Path) -> dict:
    """Compare primary keys and every source field, not just collection sizes."""
    source, lock = canonical_dataset()
    for name, records in source.items():
        saved_rows = [
            json.loads(path.read_text()) for path in (db_path / name).glob("*.json")
        ]
        saved = {row["pk"]: row for row in saved_rows}
        if len(saved_rows) != len(saved) or set(saved) != {
            row["pk"] for row in records
        }:
            raise ValueError(f"{name}: database IDs differ from the canonical dataset")
        for row in records:
            changed = [
                key for key, value in row.items() if saved[row["pk"]].get(key) != value
            ]
            if changed:
                raise ValueError(
                    f"{name} {row['pk']}: source fields changed: {', '.join(changed)}"
                )
    # Voice files and metadata must agree, too. They are fingerprinted with the run.
    from talktopia.speech_agent import load_voice_registry

    load_voice_registry(db_path)
    return {name: spec["sha256"] for name, spec in lock.items()}


def validate_manifest(records: list[dict]) -> dict:
    source, _ = canonical_dataset()
    expected = {
        row["pk"]: (row["env_id"], tuple(row["agent_ids"]))
        for row in source["EnvAgentComboStorage"]
    }
    observed = {}
    for row in records:
        combo_id = row.get("combo_id")
        identity = (row["env_id"], tuple(row["agent_ids"]))
        if combo_id in observed or expected.get(combo_id) != identity:
            raise ValueError(
                "Manifest contains a duplicate, unknown or altered canonical combo"
            )
        observed[combo_id] = identity
    if observed != expected or len(set(observed.values())) != 450:
        raise ValueError(
            "Manifest must contain all 450 unique canonical combos exactly once"
        )
    by_env = Counter(env_id for env_id, _ in observed.values())
    agent_ids = {pk for _, agents in observed.values() for pk in agents}
    if len(by_env) != 90 or set(by_env.values()) != {5} or len(agent_ids) != 40:
        raise ValueError(
            "Manifest must cover 90 environments, 5 combos per environment and 40 profiles"
        )
    # IDs and order must be identical across modes, including when importing a manifest.
    ordered = sorted(expected, key=lambda pk: (expected[pk][0], pk))
    if [row["combo_id"] for row in records] != ordered or any(
        row.get("episode_id") != f"episode_{index:04d}"
        for index, row in enumerate(records, 1)
    ):
        raise ValueError(
            "Manifest order or episode IDs differ from the canonical order"
        )
    return {
        "profiles": 40,
        "environments": 90,
        "relationships": 120,
        "combos": 450,
        "combos_per_environment": 5,
    }
