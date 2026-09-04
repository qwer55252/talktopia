from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load GeminiLight/sotopia-dataset profiles into SOTOPIA storage."
    )
    parser.add_argument("--storage-backend", choices=["local", "redis"], default="local")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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
    return {name: len(cls.all()) for name, cls in model_classes.items()}


def assert_safe_to_load(model_classes: dict[str, Any], overwrite: bool) -> None:
    existing = count_existing(model_classes)
    non_empty = {name: count for name, count in existing.items() if count}
    if non_empty and not overwrite:
        raise SystemExit(
            "SOTOPIA storage already has records; rerun with --overwrite if this is intentional: "
            + ", ".join(f"{name}={count}" for name, count in non_empty.items())
        )


def clear_existing(model_classes: dict[str, Any]) -> None:
    for cls in model_classes.values():
        for obj in cls.all():
            if obj.pk:
                cls.delete(obj.pk)


def load_environment_lists(env_rows: list[dict[str, Any]], dry_run: bool) -> int:
    from sotopia.database.persistent_profile import EnvironmentList

    # A small convenience index lets pipeline.py run with --environment-list-pk
    # while still using official EnvironmentProfile records for the scenario data.
    env_ids = sorted({str(row["pk"]) for row in env_rows})
    env_list = EnvironmentList(
        pk="geminilight_all_environments",
        name="GeminiLight SOTOPIA all environments",
        environments=env_ids,
    )
    if not dry_run:
        env_list.save()
    return 1


def main() -> None:
    args = parse_args()
    os.environ["SOTOPIA_STORAGE_BACKEND"] = args.storage_backend

    from sotopia.database import (
        AgentProfile,
        EnvironmentProfile,
        EnvAgentComboStorage,
        RelationshipProfile,
    )
    from sotopia.database.persistent_profile import EnvironmentList

    model_classes = {
        "AgentProfile": AgentProfile,
        "EnvironmentProfile": EnvironmentProfile,
        "RelationshipProfile": RelationshipProfile,
        "EnvAgentComboStorage": EnvAgentComboStorage,
    }
    storage_model_classes = {**model_classes, "EnvironmentList": EnvironmentList}

    for filename in FILES.values():
        download(f"{BASE_URL}/{filename}", args.data_dir / filename, args.force_download)

    rows_by_model = {
        name: iter_jsonl(args.data_dir / filename) for name, filename in FILES.items()
    }
    for name, rows in rows_by_model.items():
        expected = EXPECTED_COUNTS[name]
        if len(rows) != expected:
            raise SystemExit(f"{name}: expected {expected} rows, got {len(rows)}")

    # Validate every row through the repository's own pydantic/redis-om models.
    objects_by_model = {
        name: [model_classes[name](**row) for row in rows]
        for name, rows in rows_by_model.items()
    }

    if not args.dry_run:
        assert_safe_to_load(storage_model_classes, args.overwrite)
        if args.overwrite:
            clear_existing(storage_model_classes)

    saved_counts = {}
    for name, objects in objects_by_model.items():
        if not args.dry_run:
            for obj in objects:
                obj.save()
        saved_counts[name] = len(objects)

    saved_counts["EnvironmentList"] = load_environment_lists(
        rows_by_model["EnvironmentProfile"], args.dry_run
    )

    print(json.dumps(saved_counts, indent=2, sort_keys=True))
    if args.dry_run:
        print("Dry run: validated downloads but did not write SOTOPIA storage.")
    else:
        print(f"Loaded GeminiLight/sotopia-dataset into {args.storage_backend} storage.")


if __name__ == "__main__":
    main()
