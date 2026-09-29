"""Append-only Surface5 event schema and JSONL journal."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .actions import (
    DuplexAction,
    DuplexActionDecision,
    DuplexActionType,
    DuplexObservation,
    HiddenSaid,
    SpeechChunk,
    StreamingObservation,
)


class EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["surface5-event/v1", "surface5-event/v2"] = (
        "surface5-event/v2"
    )
    event_type: str
    event_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0)
    causation_id: str | None = None
    correlation_id: str | None = None


class EpisodeStarted(EventBase):
    event_type: Literal["episode_started"] = "episode_started"
    environment_id: str = Field(min_length=1)
    agent_ids: tuple[str, str]
    agent_names: tuple[str, str]
    models: tuple[str, ...]
    initial_observations: dict[str, DuplexObservation]
    run_config: dict[str, object]


class EpisodeEnded(EventBase):
    event_type: Literal["episode_ended"] = "episode_ended"
    status: Literal["completed", "cancelled", "failed"]
    reason: str
    duration_ms: float = Field(ge=0)
    error_type: str | None = None
    error_message: str | None = None


class DecisionEvent(EventBase):
    event_type: Literal["decision"] = "decision"
    status: Literal["selected", "rejected", "stale", "cancelled"]
    agent: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    observation: StreamingObservation
    decision: DuplexActionDecision | None
    attempt: int = Field(ge=1)
    validation_errors: tuple[str, ...] = ()
    request_started_ms: float | None = None


class HiddenSaidCreated(EventBase):
    event_type: Literal["hidden_said_created"] = "hidden_said_created"
    agent: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    hidden_said: HiddenSaid
    attempt: int = Field(ge=1)


class SpeechChunkSynthesized(EventBase):
    event_type: Literal["speech_chunk_synthesized"] = "speech_chunk_synthesized"
    speaker: str = Field(min_length=1)
    chunk: SpeechChunk
    pcm_duration_ms: float = Field(gt=0)
    pcm_sha256: str = Field(min_length=1)
    attempt: int = Field(default=1, ge=1)


class SpeechSynthesisFailed(EventBase):
    event_type: Literal["speech_synthesis_failed"] = "speech_synthesis_failed"
    speaker: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    hidden_said_id: str = Field(min_length=1)
    chunk: SpeechChunk
    attempts: int = Field(ge=1)
    reason: Literal["empty_audio"]
    error_detail: str = Field(min_length=1)


class AudioDeliveryEvent(EventBase):
    event_type: Literal["audio_delivered"] = "audio_delivered"
    utterance_id: str = Field(min_length=1)
    source_agent: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    delivered_frames: int = Field(ge=0)
    planned_frames: int = Field(ge=0)
    start_ms: float = Field(ge=0)
    end_ms: float = Field(ge=0)
    delivered_pcm_sha256: str = Field(min_length=1)
    frame_spans: tuple[dict[str, float | int | str], ...] = ()


class SpeechLifecycleEvent(EventBase):
    event_type: Literal["speech_lifecycle"] = "speech_lifecycle"
    phase: Literal["started", "finished", "cancelled"]
    utterance_id: str = Field(min_length=1)
    speaker: str = Field(min_length=1)
    action_type: DuplexActionType
    target_utterance_id: str | None = None
    reason: str | None = None


class FloorEvent(EventBase):
    event_type: Literal["floor"] = "floor"
    change: Literal["reserved", "released", "transferred", "rejected", "collision"]
    owner_before: str | None
    owner_after: str | None
    utterance_ids: tuple[str, ...]
    reason: str | None = None


class ASRUpdateEvent(EventBase):
    event_type: Literal["asr_update"] = "asr_update"
    utterance_id: str = Field(min_length=1)
    listener: str = Field(min_length=1)
    text: str
    is_final: bool
    is_stable: bool
    revision_id: int = Field(ge=0)
    sentence_texts: dict[int, str] = Field(default_factory=dict)


class ResponseLatencyEvent(EventBase):
    event_type: Literal["response_latency"] = "response_latency"
    kind: Literal["normal_response", "backchannel"]
    speaker: str
    utterance_id: str
    decision_id: str
    observation_id: str
    peer_utterance_id: str
    origin_ms: float = Field(ge=0)
    first_audio_ms: float = Field(ge=0)
    latency_ms: float = Field(ge=0)


class ActionCommitted(EventBase):
    event_type: Literal["action_committed"] = "action_committed"
    commit_id: str = Field(min_length=1)
    turn_number: int = Field(ge=1)
    actions: dict[str, DuplexAction]
    observations_after: dict[str, DuplexObservation]
    origin: str
    trigger_event_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


SurfaceEvent = (
    EpisodeStarted
    | EpisodeEnded
    | DecisionEvent
    | HiddenSaidCreated
    | SpeechChunkSynthesized
    | SpeechSynthesisFailed
    | AudioDeliveryEvent
    | SpeechLifecycleEvent
    | FloorEvent
    | ASRUpdateEvent
    | ActionCommitted
    | ResponseLatencyEvent
)

_EVENT_CLASSES: dict[str, type[EventBase]] = {
    event_class.model_fields["event_type"].default: event_class
    for event_class in (
        EpisodeStarted,
        EpisodeEnded,
        DecisionEvent,
        HiddenSaidCreated,
        SpeechChunkSynthesized,
        SpeechSynthesisFailed,
        AudioDeliveryEvent,
        SpeechLifecycleEvent,
        FloorEvent,
        ASRUpdateEvent,
        ActionCommitted,
        ResponseLatencyEvent,
    )
}
_EventT = TypeVar("_EventT", bound=EventBase)


class EventWriter:
    """Assign the global envelope and append each event immediately."""

    def __init__(self, path: Path, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode_id must not be blank")
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id
        self.started_ns = time.monotonic_ns()
        self._sequence = 0
        self._file = self.path.open("x", encoding="utf-8")
        self._closed = False

    def now_ms(self) -> float:
        return (time.monotonic_ns() - self.started_ns) / 1_000_000

    def emit(
        self,
        event_class: type[_EventT],
        timestamp_ms: float,
        **payload: object,
    ) -> _EventT:
        if self._closed:
            raise RuntimeError("event writer is closed")
        event = event_class(
            event_id=f"{self.episode_id}-event-{uuid4().hex}",
            episode_id=self.episode_id,
            sequence=self._sequence,
            timestamp_ms=timestamp_ms,
            **payload,
        )
        self.append(cast(SurfaceEvent, event))
        return event

    def append(self, event: SurfaceEvent) -> None:
        if self._closed:
            raise RuntimeError("event writer is closed")
        if event.episode_id != self.episode_id:
            raise ValueError("event episode_id does not match writer")
        if event.sequence != self._sequence:
            raise ValueError(
                f"event sequence {event.sequence} does not match next sequence "
                f"{self._sequence}"
            )
        self._file.write(event.model_dump_json() + "\n")
        self._file.flush()
        self._sequence += 1

    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True


def read_events(path: Path) -> list[SurfaceEvent]:
    events: list[SurfaceEvent] = []
    with path.expanduser().open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                event_type = payload["event_type"]
                event_class = _EVENT_CLASSES[event_type]
                event = event_class.model_validate(payload)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid Surface5 event at {path}:{line_number}"
                ) from error
            events.append(event)  # type: ignore[arg-type]

    for expected, event in enumerate(events):
        if event.sequence != expected:
            raise ValueError(
                f"non-contiguous event sequence at {path}: expected {expected}, "
                f"got {event.sequence}"
            )
    return events


__all__ = [
    "ASRUpdateEvent",
    "ActionCommitted",
    "AudioDeliveryEvent",
    "DecisionEvent",
    "EpisodeEnded",
    "EpisodeStarted",
    "EventBase",
    "EventWriter",
    "FloorEvent",
    "HiddenSaidCreated",
    "ResponseLatencyEvent",
    "SpeechChunkSynthesized",
    "SpeechSynthesisFailed",
    "SpeechLifecycleEvent",
    "SurfaceEvent",
    "read_events",
]
