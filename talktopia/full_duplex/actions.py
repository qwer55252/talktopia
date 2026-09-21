"""Typed actions and observations used by the Surface5 duplex runtime."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator
from sotopia.messages import AgentAction, Observation


DuplexActionType = Literal[
    "none",
    "speak",
    "non-verbal communication",
    "action",
    "leave",
    "hesitation",
    "backchanneling",
    "correction",
    "interruption",
]

MAX_GENERATED_WORDS = 40

_BASE_ACTION_TYPES = frozenset(
    {"none", "speak", "non-verbal communication", "action", "leave"}
)
_EMPTY_ARGUMENT_ACTION_TYPES = frozenset({"none", "leave"})
_NON_AUDIO_ARGUMENT_ACTION_TYPES = frozenset({"non-verbal communication", "action"})
_TARGETED_ACTION_TYPES = frozenset({"correction", "interruption"})
_CONTROLLER_METADATA_PHRASES = (
    "private goal",
    "hidden goal",
    "social goal",
    "system prompt",
    "developer message",
    "controller instruction",
    "evaluator",
    "evaluation score",
    "benchmark score",
    "reward function",
)


def _validate_generated_text(text: str, *, field_name: str) -> None:
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    if len(normalized.split()) > MAX_GENERATED_WORDS:
        raise ValueError(
            f"{field_name} must contain at most {MAX_GENERATED_WORDS} words"
        )
    folded = normalized.casefold()
    leaked = next(
        (phrase for phrase in _CONTROLLER_METADATA_PHRASES if phrase in folded),
        None,
    )
    if leaked is not None:
        raise ValueError(f"{field_name} contains controller metadata: {leaked!r}")


class DuplexAction(AgentAction):
    """A SOTOPIA action that preserves all nine Surface5 action types."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Surface5 intentionally widens SOTOPIA's five base action literals.
    action_type: DuplexActionType  # type: ignore[assignment]
    argument: str = ""
    to: list[str] = Field(default_factory=list)
    target_utterance_id: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "DuplexAction":
        argument = self.argument.strip()
        if self.action_type in _EMPTY_ARGUMENT_ACTION_TYPES:
            if argument:
                raise ValueError(f"{self.action_type} requires an empty argument")
        elif not argument:
            raise ValueError(f"{self.action_type} requires a non-empty argument")
        if self.action_type in _NON_AUDIO_ARGUMENT_ACTION_TYPES:
            _validate_generated_text(self.argument, field_name="argument")

        if self.action_type in _TARGETED_ACTION_TYPES:
            if not (self.target_utterance_id or "").strip():
                raise ValueError(f"{self.action_type} requires target_utterance_id")
        elif self.target_utterance_id is not None:
            raise ValueError(
                "target_utterance_id is only valid for correction or interruption"
            )
        return self

    def to_natural_language(self) -> str:
        if self.action_type in _BASE_ACTION_TYPES:
            return super().to_natural_language()

        recipients_prefix = "" if not self.to else f"[private to {self.to}] "
        match self.action_type:
            case "hesitation":
                rendered = f"[hesitation] {self.argument}"
            case "backchanneling":
                rendered = f'backchanneled: "{self.argument}"'
            case "correction":
                rendered = f'corrected: "{self.argument}"'
            case "interruption":
                rendered = f'interrupted: "{self.argument}"'
            case _:  # pragma: no cover - exhaustive Literal match
                raise AssertionError(f"unhandled action type: {self.action_type}")
        return f"{recipients_prefix} {rendered}"


class DuplexObservation(Observation):
    """Canonical SOTOPIA observation plus a stable ID and duplex action mask."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str = Field(min_length=1)
    # Surface5 intentionally widens SOTOPIA's five base action literals.
    available_actions: list[DuplexActionType]  # type: ignore[assignment]


class StreamingObservation(BaseModel):
    """Transient ASR/floor state layered over one canonical observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical: DuplexObservation
    source: Literal["reset", "asr_partial", "asr_final", "commit", "peer_left"]
    stable_text: str = ""
    new_stable_text: str = ""
    peer_utterance_id: str | None = None
    asr_revision_id: int = Field(default=0, ge=0)
    peer_speaking: bool = False
    self_speaking: bool = False
    self_active_action_type: DuplexActionType | None = None
    target_utterance_id: str | None = None


class DuplexActionDecision(BaseModel):
    """Structured first-stage decision without generated speech text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str = ""
    action_type: DuplexActionType
    non_audio_argument: str = ""
    to: list[str] = Field(default_factory=list)
    target_utterance_id: str | None = None

    @model_validator(mode="after")
    def validate_decision(self, info: ValidationInfo) -> "DuplexActionDecision":
        decision_id = self.decision_id.strip()
        if self.decision_id and not decision_id:
            raise ValueError("decision_id must be empty or non-blank")

        non_audio_argument = self.non_audio_argument.strip()
        if self.action_type in _NON_AUDIO_ARGUMENT_ACTION_TYPES:
            if not non_audio_argument:
                raise ValueError(f"{self.action_type} requires non_audio_argument")
            _validate_generated_text(
                self.non_audio_argument,
                field_name="non_audio_argument",
            )
        elif non_audio_argument:
            raise ValueError(
                "non_audio_argument is only valid for action or "
                "non-verbal communication"
            )

        if self.action_type in _TARGETED_ACTION_TYPES:
            if not (self.target_utterance_id or "").strip():
                raise ValueError(f"{self.action_type} requires target_utterance_id")
        elif self.target_utterance_id is not None:
            raise ValueError(
                "target_utterance_id is only valid for correction or interruption"
            )

        context = info.context or {}
        if context.get("require_decision_id") and not decision_id:
            raise ValueError("decision_id must be assigned before emission")

        available = context.get("available_actions")
        if available is not None and self.action_type not in available:
            raise ValueError(f"unavailable action: {self.action_type}")

        agent_names = set(context.get("agent_names", ()))
        sender = context.get("sender")
        invalid = [
            recipient
            for recipient in self.to
            if agent_names and (recipient not in agent_names or recipient == sender)
        ]
        if invalid:
            raise ValueError(f"invalid recipients: {invalid}")
        if len(set(self.to)) != len(self.to):
            raise ValueError("duplicate recipients are not allowed")
        return self


class HiddenSaid(BaseModel):
    """Private complete utterance before its audio is revealed to the peer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hidden_said_id: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_generated_text(self) -> "HiddenSaid":
        _validate_generated_text(self.text, field_name="text")
        return self


class SpeechChunk(BaseModel):
    """One complete sentence of a private utterance sent to TTS."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    utterance_id: str = Field(min_length=1)
    hidden_said_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    total_chunks: int = Field(gt=0)
    text: str = Field(min_length=1)
    is_final: bool

    @model_validator(mode="after")
    def validate_position(self) -> "SpeechChunk":
        if self.chunk_index >= self.total_chunks:
            raise ValueError("chunk_index must be less than total_chunks")
        if self.is_final != (self.chunk_index == self.total_chunks - 1):
            raise ValueError("is_final must identify exactly the final chunk")
        _validate_generated_text(self.text, field_name="text")
        return self


__all__ = [
    "DuplexAction",
    "DuplexActionDecision",
    "DuplexActionType",
    "DuplexObservation",
    "HiddenSaid",
    "MAX_GENERATED_WORDS",
    "SpeechChunk",
    "StreamingObservation",
]
