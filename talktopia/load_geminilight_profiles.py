from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from talktopia.task_space import (
    CORE_MODELS,
    assert_separate,
    configure_database,
    database_path,
    require_local,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "geminilight_sotopia_dataset"
BASE_URL = "https://huggingface.co/datasets/GeminiLight/sotopia-dataset/resolve/main"

FILES = {
    "AgentProfile": "agent_profile.jsonl",
    "EnvironmentProfile": "environment_profile.jsonl",
    "RelationshipProfile": "relationship_profile.jsonl",
    "EnvAgentComboStorage": "env_agent_combo_storage.jsonl",
}

EXPECTED_COUNTS = {
    "AgentProfile": 40,
    "EnvironmentProfile": 90,
    "RelationshipProfile": 120,
    "EnvAgentComboStorage": 450,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Talktopia profiles and their surface5 reference voices."
    )
    parser.add_argument(
        "--storage-backend", choices=["local", "redis"], default="local"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--source-db",
        type=Path,
        help="Copy the five core collections from a local DB instead of downloading JSONL",
    )
    parser.add_argument(
        "--voice-source",
        type=Path,
        default=REPO_ROOT.parent / "sotopia" / "surface5" / "data" / "voices",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def download(url: str, path: Path, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not force:
        return

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response:
        with tmp_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    tmp_path.replace(path)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def count_existing(model_classes: dict[str, Any]) -> dict[str, int]:
    return {name: len(cls.all_pks()) for name, cls in model_classes.items()}


def assert_safe_to_load(model_classes: dict[str, Any], overwrite: bool) -> None:
    existing = count_existing(model_classes)
    non_empty = {name: count for name, count in existing.items() if count}
    if non_empty and not overwrite:
        raise SystemExit(
            "Talktopia storage already has records; rerun with --overwrite if this is intentional: "
            + ", ".join(f"{name}={count}" for name, count in non_empty.items())
        )


def clear_existing(model_classes: dict[str, Any]) -> None:
    for cls in model_classes.values():
        for pk in cls.all_pks():
            if pk:
                cls.delete(pk)


def validate_links(rows: dict[str, list[dict[str, Any]]]) -> None:
    ids = {}
    for name, records in rows.items():
        keys = [row.get("pk") for row in records]
        if any(
            not isinstance(pk, str)
            or not pk
            or Path(pk).name != pk
            or pk in {".", ".."}
            for pk in keys
        ):
            raise ValueError(f"{name}: invalid PK")
        if len(keys) != len(set(keys)):
            raise ValueError(f"{name}: duplicate PK")
        ids[name] = set(keys)
    for row in rows["RelationshipProfile"]:
        if not {row["agent_1_id"], row["agent_2_id"]} <= ids["AgentProfile"]:
            raise ValueError(f"RelationshipProfile {row['pk']}: missing AgentProfile")
    for row in rows["EnvAgentComboStorage"]:
        if (
            row["env_id"] not in ids["EnvironmentProfile"]
            or not set(row["agent_ids"]) <= ids["AgentProfile"]
        ):
            raise ValueError(
                f"EnvAgentComboStorage {row['pk']}: missing environment or agent"
            )
    for row in rows["EnvironmentList"]:
        if not set(row["environments"]) <= ids["EnvironmentProfile"]:
            raise ValueError(f"EnvironmentList {row['pk']}: missing environment")


def prepare(args: argparse.Namespace) -> dict[str, int]:
    require_local(args.storage_backend)
    target = database_path()
    if args.source_db:
        assert_separate(target, args.source_db.expanduser())
        if args.force_download:
            raise ValueError("--force-download cannot be used with --source-db")

    from sotopia.database import (
        EnvironmentProfile,
        EnvAgentComboStorage,
        RelationshipProfile,
    )
    from sotopia.database.persistent_profile import EnvironmentList
    from talktopia.speech_agent import AgentProfile, reference_profile

    model_classes = {
        "AgentProfile": AgentProfile,
        "EnvironmentProfile": EnvironmentProfile,
        "RelationshipProfile": RelationshipProfile,
        "EnvAgentComboStorage": EnvAgentComboStorage,
    }
    storage_model_classes = {**model_classes, "EnvironmentList": EnvironmentList}

    if args.source_db:
        source = args.source_db.expanduser().resolve()
        rows_by_model = {}
        for name in CORE_MODELS:
            paths = sorted((source / name).glob("*.json"))
            if not paths:
                raise ValueError(f"Missing source collection: {source / name}")
            rows_by_model[name] = [
                json.loads(path.read_text(encoding="utf-8")) for path in paths
            ]
    else:
        for filename in FILES.values():
            download(
                f"{BASE_URL}/{filename}", args.data_dir / filename, args.force_download
            )
        rows_by_model = {
            name: iter_jsonl(args.data_dir / filename)
            for name, filename in FILES.items()
        }
        rows_by_model["EnvironmentList"] = [
            dict(
                pk="geminilight_all_environments",
                name="GeminiLight SOTOPIA all environments",
                environments=sorted(
                    {row["pk"] for row in rows_by_model["EnvironmentProfile"]}
                ),
            )
        ]
    for name, rows in rows_by_model.items():
        expected = EXPECTED_COUNTS.get(name)
        if expected is not None and len(rows) != expected:
            raise SystemExit(f"{name}: expected {expected} rows, got {len(rows)}")

    validate_links(rows_by_model)
    voice_source = args.voice_source.expanduser().resolve()
    assert_separate(target, voice_source)
    voice_ids = set()
    for row in rows_by_model["AgentProfile"]:
        row.update(reference_profile(row["pk"], voice_source))
        if row["voice_id"] in voice_ids:
            raise ValueError(f"Duplicate voice_id: {row['voice_id']}")
        voice_ids.add(row["voice_id"])

    # Validate every row through the repository's own pydantic/redis-om models.
    objects_by_model = {
        name: [storage_model_classes[name](**row) for row in rows]
        for name, rows in rows_by_model.items()
    }

    if not args.dry_run:
        configure_database(args.storage_backend, require_profiles=False)
        assert_safe_to_load(storage_model_classes, args.overwrite)
        for row in rows_by_model["AgentProfile"]:
            destination = target / "voices" / row["pk"]
            if destination.is_symlink() or any(
                (destination / name).is_symlink()
                for name in ("reference.wav", "reference.txt", "design.json")
            ):
                raise ValueError(f"Refusing symlink voice destination: {destination}")
        if args.overwrite:
            clear_existing(storage_model_classes)
        for row in rows_by_model["AgentProfile"]:
            destination = target / "voices" / row["pk"]
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ("reference.wav", "reference.txt", "design.json"):
                shutil.copy2(
                    voice_source / row["pk"] / filename, destination / filename
                )

    saved_counts = {}
    for name, objects in objects_by_model.items():
        if not args.dry_run:
            for obj in objects:
                obj.save()
        saved_counts[name] = len(objects)

    print(json.dumps(saved_counts, indent=2, sort_keys=True))
    if args.dry_run:
        print(f"Dry run: validated profiles and voices; did not write {target}.")
    else:
        print(f"Loaded profiles and {len(voice_ids)} voices into {target}.")
    return saved_counts


def main(argv: list[str] | None = None) -> None:
    try:
        prepare(parse_args(argv))
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
