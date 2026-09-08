"""Select Talktopia's local task space, following Poketopia's database setup."""

from __future__ import annotations

import os
from pathlib import Path

from talktopia.models.config import REPO_ROOT

CORE_MODELS = (
    "AgentProfile",
    "EnvironmentProfile",
    "RelationshipProfile",
    "EnvAgentComboStorage",
    "EnvironmentList",
)
SOURCE_DB = REPO_ROOT / "data" / "sotopia_db"
DEFAULT_DB = Path.home() / ".sotopia" / "talktopia" / "data"


def assert_separate(target: Path, source: Path) -> None:
    target, source = target.resolve(), source.resolve()
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError(
            f"Talktopia DB must not overlap source DB: {target} / {source}"
        )


def database_path() -> Path:
    path = (
        Path(os.environ.get("TALKTOPIA_DB_DIR", str(DEFAULT_DB))).expanduser().resolve()
    )
    assert_separate(path, SOURCE_DB)
    assert_separate(path, Path.home() / ".sotopia" / "data")
    return path


def require_local(backend: str) -> None:
    if backend != "local":
        raise ValueError(
            "Talktopia requires --storage-backend local; Redis is not supported"
        )
    os.environ["SOTOPIA_STORAGE_BACKEND"] = "local"


def configure_database(
    backend: str = "local", *, require_profiles: bool = True
) -> Path:
    require_local(backend)
    path = database_path()
    for name in (*CORE_MODELS, "EpisodeLog", "voices"):
        if (path / name).is_symlink():
            raise ValueError(f"Refusing symlink inside Talktopia DB: {path / name}")
    if require_profiles:
        missing = [
            name for name in CORE_MODELS if not any((path / name).glob("*.json"))
        ]
        if missing:
            raise ValueError(
                f"Talktopia DB at {path} is not prepared ({', '.join(missing)}); run ./load_profiles.sh"
            )
    from sotopia.database import storage_backend as storage

    if not storage.is_local_backend():
        raise ValueError(
            "SOTOPIA was already imported with Redis; start a fresh local process"
        )
    storage._storage_backend = storage.LocalJSONBackend(str(path))
    return path
