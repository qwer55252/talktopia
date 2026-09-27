"""Read the exact profile files fingerprinted by the simulation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .files import ResultError


RELATIONSHIPS = {
    0: "stranger",
    1: "know_by_name",
    2: "acquaintance",
    3: "friend",
    4: "romantic_relationship",
    5: "family_member",
}


def load_profiles(config: dict, agents: list[str], environment: str, override: Path | None):
    warnings = []
    fingerprints = config.get("input_fingerprints", {})
    if not isinstance(fingerprints, dict):
        fingerprints = {}
    entries = {}
    for filename, expected in fingerprints.items():
        original = Path(filename)
        collection = original.parent.name
        if collection not in ("AgentProfile", "EnvironmentProfile", "RelationshipProfile"):
            continue
        path = (override / collection / original.name).resolve() if override else original
        if override and not path.is_relative_to(override.resolve()):
            warnings.append("프로필 경로가 지정한 DB 디렉터리 밖을 가리킵니다.")
            continue
        entries.setdefault(collection, []).append((original.stem, path, expected))

    def read_entry(entry):
        key, path, expected = entry
        try:
            raw = path.read_bytes()
            data = json.loads(raw)
        except (OSError, ValueError) as error:
            raise ResultError(f"{path.name}: 실행 당시 프로필을 읽을 수 없습니다.") from error
        if not isinstance(data, dict) or data.get("pk") != key:
            raise ResultError(f"{path.name}: 프로필 ID가 일치하지 않습니다.")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ResultError(f"{path.name}: 실행 당시 프로필과 내용이 다릅니다.")
        return data

    def selected(collection, key):
        matches = [item for item in entries.get(collection, []) if item[0] == key]
        if len(matches) != 1:
            warnings.append(f"{collection}: 실행 당시 프로필을 확인할 수 없습니다. ({key})")
            return None
        try:
            return read_entry(matches[0])
        except ResultError as error:
            warnings.append(str(error))
            return None

    agent_profiles = [selected("AgentProfile", key) for key in agents]
    environment_profile = selected("EnvironmentProfile", environment)
    relationships = []
    if environment_profile:
        for entry in entries.get("RelationshipProfile", []):
            try:
                candidate = read_entry(entry)
            except ResultError:
                continue
            if (
                {candidate.get("agent_1_id"), candidate.get("agent_2_id")} == set(agents)
                and candidate.get("relationship") == environment_profile.get("relationship")
            ):
                relationships.append(candidate)
    relationship = relationships[0] if len(relationships) == 1 else None
    if relationship is None:
        warnings.append("에이전트와 관계 유형이 일치하는 검증된 Relationship profile을 하나로 확인할 수 없습니다.")
    return {
        "agents": agent_profiles,
        "environment": environment_profile,
        "relationship": relationship,
        "relationship_label": RELATIONSHIPS.get(
            environment_profile.get("relationship") if environment_profile else None, ""
        ),
        "warnings": warnings,
    }
