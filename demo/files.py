"""Small, read-only helpers shared by the artifact readers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ResultError(ValueError):
    """An input cannot be displayed safely or accurately."""


def read_json(path: Path, kind: type = dict):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ResultError(f"{path.name}: 파일이 없거나 JSON을 읽을 수 없습니다.") from error
    if not isinstance(data, kind):
        raise ResultError(f"{path.name}: 예상한 {kind.__name__} 형식이 아닙니다.")
    return data


def optional_json(path: Path, kind: type, warnings: list[str]):
    try:
        return read_json(path, kind)
    except ResultError as error:
        warnings.append(str(error))
        return kind()


def artifact_path(root: Path, name: str | None, suffix: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ResultError("결과 파일의 상대경로가 없거나 잘못되었습니다.")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path.suffix != suffix:
        raise ResultError("등록된 결과 디렉터리 밖의 파일은 읽을 수 없습니다.")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_json(root: Path, row: dict, field: str, warnings: list[str]) -> dict:
    try:
        return read_json(artifact_path(root, row.get(field), ".json"))
    except ResultError as error:
        warnings.append(f"{field}: {error}")
        return {}


def available(root: Path, name: str | None, suffix: str) -> bool:
    try:
        return artifact_path(root, name, suffix).is_file()
    except ResultError:
        return False
