"""Surface5 adapters over the existing SOTOPIA storage and environment APIs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from sotopia.agents.llm_agent import Agents
from sotopia.database import (
    EnvironmentProfile,
)
from talktopia.speech_agent import AgentProfile
from gymnasium.spaces.dict import Dict as GymDict
from gymnasium.spaces.text import Text
from sotopia.envs.parallel import LiteralSpace, ParallelSotopiaEnv
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator

from .actions import DuplexAction, DuplexActionType, DuplexObservation

if TYPE_CHECKING:
    from .agent import CascadedDuplexAgent


_ALL_ACTION_TYPES: tuple[DuplexActionType, ...] = (
    "none",
    "speak",
    "non-verbal communication",
    "action",
    "leave",
    "hesitation",
    "backchanneling",
    "correction",
    "interruption",
)
_BASE_ACTION_TYPES = {
    "none",
    "speak",
    "non-verbal communication",
    "action",
    "leave",
}


@dataclass(frozen=True, slots=True)
class ResolvedEpisode:
    combo_pk: str
    env_profile: EnvironmentProfile
    agent_profiles: tuple[AgentProfile, AgentProfile]

    def __post_init__(self) -> None:
        env_pk = str(self.env_profile.pk or "").strip()
        agent_pks = tuple(
            str(profile.pk or "").strip() for profile in self.agent_profiles
        )
        if not self.combo_pk or not env_pk or any(not pk for pk in agent_pks):
            raise ValueError("resolved episode records must have primary keys")
        if len(set(agent_pks)) != 2:
            raise ValueError("resolved episode requires two distinct agents")


@dataclass(frozen=True, slots=True)
class SemanticSnapshot:
    turn_number: int
    observations: Mapping[str, DuplexObservation]

    def __post_init__(self) -> None:
        if self.turn_number < 0:
            raise ValueError("turn_number must be non-negative")
        if len(self.observations) != 2:
            raise ValueError("semantic snapshot requires exactly two observations")
        object.__setattr__(
            self,
            "observations",
            MappingProxyType(dict(self.observations)),
        )


@dataclass(frozen=True, slots=True)
class SemanticCommit:
    actions: Mapping[str, DuplexAction]
    expected_turn_number: int
    timestamp_ms: float
    origin: str
    utterance_ids: Mapping[str, str | None]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.expected_turn_number < 0 or self.timestamp_ms < 0:
            raise ValueError("semantic commit counters must be non-negative")
        if len(self.actions) != 2:
            raise ValueError("semantic commit requires exactly two actions")
        if set(self.utterance_ids) != set(self.actions):
            raise ValueError("utterance_ids must have the same agents as actions")
        object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))
        object.__setattr__(
            self,
            "utterance_ids",
            MappingProxyType(dict(self.utterance_ids)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class DuplexSotopiaEnv(ParallelSotopiaEnv):
    """ParallelSotopiaEnv with Surface5 action and observation types."""

    def __init__(self, env_profile: EnvironmentProfile) -> None:
        super().__init__(
            env_profile=env_profile,
            available_action_types=cast(Any, _BASE_ACTION_TYPES),
            action_order="simultaneous",
            evaluators=[],
            terminal_evaluators=[],
            model_name="talktopia-full-duplex",
            hide_unknown=False,
            include_turn_marker=False,
        )

    def reset(  # type: ignore[override]
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
        agents: Agents | None = None,
        omniscient: bool = False,
        lite: bool = False,
        include_background_observations: bool | None = True,
    ) -> dict[str, DuplexObservation]:
        observations = super().reset(
            seed=seed,
            options=options,
            agents=agents,
            omniscient=omniscient,
            lite=lite,
            include_background_observations=include_background_observations,
        )
        self._expose_duplex_action_space()
        return self._wrap_observations(observations, source="reset")

    async def astep(  # type: ignore[override]
        self, actions: Mapping[str, DuplexAction]
    ) -> tuple[
        dict[str, DuplexObservation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        if set(actions) != set(self.agents):
            raise ValueError("semantic commit must contain exactly both agents")
        if any(not isinstance(action, DuplexAction) for action in actions.values()):
            raise TypeError("DuplexSotopiaEnv accepts only DuplexAction values")
        self.available_action_types = cast(Any, list(_BASE_ACTION_TYPES))
        try:
            observations, rewards, terminated, truncated, info = await super().astep(
                cast(Any, dict(actions))
            )
        finally:
            self._expose_duplex_action_space()
        return (
            self._wrap_observations(observations, source="commit"),
            rewards,
            terminated,
            truncated,
            info,
        )

    def _wrap_observations(
        self,
        observations: Mapping[str, object],
        *,
        source: str,
    ) -> dict[str, DuplexObservation]:
        if set(observations) != set(self.agents):
            raise RuntimeError("SOTOPIA returned an invalid observation mapping")
        wrapped: dict[str, DuplexObservation] = {}
        for index, agent_name in enumerate(self.agents):
            observation = observations[agent_name]
            base_available = list(observation.available_actions)  # type: ignore[attr-defined]
            available_actions = (
                ["none"] if base_available == ["none"] else list(_ALL_ACTION_TYPES)
            )
            wrapped[agent_name] = DuplexObservation(
                last_turn=observation.last_turn,  # type: ignore[attr-defined]
                turn_number=observation.turn_number,  # type: ignore[attr-defined]
                available_actions=cast(list[DuplexActionType], available_actions),
                action_instruction=observation.action_instruction,  # type: ignore[attr-defined]
                observation_id=(
                    f"sotopia-{observation.turn_number:06d}-{index}-{source}"  # type: ignore[attr-defined]
                ),
            )
        return wrapped

    def _expose_duplex_action_space(self) -> None:
        self.available_action_types = cast(Any, list(_ALL_ACTION_TYPES))
        self.action_spaces = {
            agent: GymDict(
                {
                    "action_type": LiteralSpace(list(_ALL_ACTION_TYPES)),
                    "argument": Text(256),
                }
            )
            for agent in self.agents
        }


class SotopiaSession:
    """Own one reset environment and serialize every semantic commit."""

    def __init__(self, resolved: ResolvedEpisode, *, seed: int = 0) -> None:
        if seed < 0:
            raise ValueError("SOTOPIA seed must be non-negative")
        self._resolved = resolved
        self._seed = seed
        self._environment = DuplexSotopiaEnv(resolved.env_profile)
        self._stale_evaluator = RuleBasedTerminatedEvaluator(max_stale_turn=2)
        self._lock = asyncio.Lock()
        self._latest: SemanticSnapshot | None = None

    def open(
        self,
        agents: tuple["CascadedDuplexAgent", "CascadedDuplexAgent"],
    ) -> SemanticSnapshot:
        if self._latest is not None:
            raise RuntimeError("SOTOPIA session is already open")
        if self._lock.locked():
            raise RuntimeError("cannot open SOTOPIA session during a commit")
        expected_pks = tuple(
            str(profile.pk or "") for profile in self._resolved.agent_profiles
        )
        actual_pks = tuple(str(agent.profile.pk or "") for agent in agents)
        if actual_pks != expected_pks:
            raise ValueError("agent profile order does not match resolved combo")
        if agents[0].agent_name == agents[1].agent_name:
            raise ValueError("SOTOPIA participant names must be distinct")

        agent_map = Agents(cast(Any, {agent.agent_name: agent for agent in agents}))
        observations = self._environment.reset(
            seed=self._seed,
            agents=agent_map,
            omniscient=False,
            lite=False,
            include_background_observations=True,
        )
        self._latest = SemanticSnapshot(
            turn_number=0,
            observations=observations,
        )
        return self._latest

    async def commit(self, commit: SemanticCommit) -> SemanticSnapshot:
        async with self._lock:
            if self._latest is None:
                raise RuntimeError("SOTOPIA session is not open")
            if commit.expected_turn_number != self._latest.turn_number:
                raise ValueError(
                    "stale semantic commit: expected "
                    f"{commit.expected_turn_number}, current "
                    f"{self._latest.turn_number}"
                )
            if set(commit.actions) != set(self._latest.observations):
                raise ValueError("semantic commit agents do not match session")
            (
                observations,
                _rewards,
                _terminated,
                _truncated,
                _info,
            ) = await self._environment.astep(commit.actions)
            turn_numbers = {item.turn_number for item in observations.values()}
            if len(turn_numbers) != 1:
                raise RuntimeError("SOTOPIA returned inconsistent turn numbers")
            self._latest = SemanticSnapshot(
                turn_number=next(iter(turn_numbers)),
                observations=observations,
            )
            return self._latest

    def is_stale(self, actions: list[tuple[str, DuplexAction]]) -> bool:
        """Apply SOTOPIA's stale-turn rule to actual actions, not peer placeholders."""
        if self._latest is None:
            raise RuntimeError("SOTOPIA session is not open")
        result = self._stale_evaluator(
            turn_number=0,
            messages=[(speaker, action) for speaker, action in actions],
            env=self._environment,
        )
        return bool(result[0][1][0][1])

    def close(self) -> None:
        if self._lock.locked():
            raise RuntimeError("cannot close SOTOPIA session during a commit")
        self._environment.close()
        self._latest = None
