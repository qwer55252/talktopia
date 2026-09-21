"""Derive utterance-level transcript records from the append-only event stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict

from .actions import DuplexActionType, HiddenSaid
from .events import (
    ASRUpdateEvent,
    ActionCommitted,
    AudioDeliveryEvent,
    HiddenSaidCreated,
    SpeechChunkSynthesized,
    SpeechLifecycleEvent,
    SurfaceEvent,
)


class TranscriptEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    utterance_id: str
    speaker: str
    listener: str
    start_ms: int | None
    end_ms: int | None
    action_type: DuplexActionType
    generated_text: str
    synthesized_text: str
    received_text: str
    origin: str
    completed: bool
    interrupted: bool
    cancellation_reason: str | None
    decision_id: str | None
    hidden_said_id: str | None
    commit_id: str | None


@dataclass(slots=True)
class _TranscriptDraft:
    episode_id: str
    utterance_id: str
    speaker: str = ""
    listener: str = ""
    start_ms: int | None = None
    end_ms: int | None = None
    action_type: DuplexActionType = "speak"
    hidden_said_id: str | None = None
    hidden_said: HiddenSaid | None = None
    synthesized_chunks: dict[int, str] = field(default_factory=dict)
    received_text: str = ""
    origin: str = "agent"
    completed: bool = False
    interrupted: bool = False
    cancellation_reason: str | None = None
    commit_id: str | None = None

    def to_entry(self) -> TranscriptEntry:
        hidden = self.hidden_said
        return TranscriptEntry(
            episode_id=self.episode_id,
            utterance_id=self.utterance_id,
            speaker=self.speaker,
            listener=self.listener,
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            action_type=self.action_type,
            generated_text=hidden.text if hidden else "",
            synthesized_text=" ".join(
                text for _, text in sorted(self.synthesized_chunks.items())
            ),
            received_text=self.received_text,
            origin=self.origin,
            completed=self.completed,
            interrupted=self.interrupted,
            cancellation_reason=self.cancellation_reason,
            decision_id=hidden.decision_id if hidden else None,
            hidden_said_id=hidden.hidden_said_id if hidden else None,
            commit_id=self.commit_id,
        )


class TranscriptBuilder:
    def __init__(self) -> None:
        self._drafts: dict[str, _TranscriptDraft] = {}
        self._hidden_by_id: dict[str, HiddenSaid] = {}
        self._utterance_by_event_id: dict[str, str] = {}

    def consume(self, event: SurfaceEvent) -> None:
        if isinstance(event, HiddenSaidCreated):
            self._hidden_by_id[event.hidden_said.hidden_said_id] = event.hidden_said
            for draft in self._drafts.values():
                if (
                    draft.hidden_said is None
                    and draft.hidden_said_id == event.hidden_said.hidden_said_id
                ):
                    draft.hidden_said = event.hidden_said
            return

        if isinstance(event, SpeechChunkSynthesized):
            draft = self._draft(event.episode_id, event.chunk.utterance_id)
            draft.hidden_said_id = event.chunk.hidden_said_id
            hidden = self._hidden_by_id.get(event.chunk.hidden_said_id)
            if hidden is not None:
                draft.hidden_said = hidden
            draft.speaker = event.speaker
            draft.synthesized_chunks[event.chunk.chunk_index] = event.chunk.text
            return

        if isinstance(event, SpeechLifecycleEvent):
            draft = self._draft(event.episode_id, event.utterance_id)
            draft.speaker = event.speaker
            draft.action_type = event.action_type
            if event.phase == "started":
                draft.start_ms = event.timestamp_ms
            elif event.phase == "finished":
                draft.end_ms = event.timestamp_ms
                draft.completed = True
            else:
                draft.end_ms = event.timestamp_ms
                draft.completed = False
                draft.interrupted = True
                draft.cancellation_reason = event.reason
            return

        if isinstance(event, AudioDeliveryEvent):
            draft = self._draft(event.episode_id, event.utterance_id)
            draft.speaker = event.source_agent
            draft.start_ms = (
                event.start_ms
                if draft.start_ms is None
                else min(draft.start_ms, event.start_ms)
            )
            draft.end_ms = (
                event.end_ms
                if draft.end_ms is None
                else max(draft.end_ms, event.end_ms)
            )
            return

        if isinstance(event, ASRUpdateEvent):
            self._utterance_by_event_id[event.event_id] = event.utterance_id
            draft = self._draft(event.episode_id, event.utterance_id)
            draft.listener = event.listener
            if event.is_final:
                draft.received_text = event.text
            return

        if isinstance(event, ActionCommitted):
            if event.trigger_event_id is None:
                return
            utterance_id = self._utterance_by_event_id.get(event.trigger_event_id)
            if utterance_id is None:
                return
            draft = self._draft(event.episode_id, utterance_id)
            draft.commit_id = event.commit_id
            draft.origin = event.origin

    def build(self) -> list[TranscriptEntry]:
        entries = [draft.to_entry() for draft in self._drafts.values()]
        return sorted(
            entries,
            key=lambda entry: (
                entry.start_ms is None,
                entry.start_ms if entry.start_ms is not None else 0,
                entry.utterance_id,
            ),
        )

    @classmethod
    def from_events(cls, events: Iterable[SurfaceEvent]) -> "TranscriptBuilder":
        builder = cls()
        for event in sorted(events, key=lambda item: item.sequence):
            builder.consume(event)
        return builder

    def write_jsonl(self, path: Path) -> None:
        output_path = path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output:
            for entry in self.build():
                output.write(entry.model_dump_json() + "\n")

    def _draft(self, episode_id: str, utterance_id: str) -> _TranscriptDraft:
        draft = self._drafts.get(utterance_id)
        if draft is None:
            draft = _TranscriptDraft(
                episode_id=episode_id,
                utterance_id=utterance_id,
            )
            self._drafts[utterance_id] = draft
        elif draft.episode_id != episode_id:
            raise ValueError(f"utterance ID reused across episodes: {utterance_id}")
        return draft


def read_transcript(path: Path) -> list[TranscriptEntry]:
    entries: list[TranscriptEntry] = []
    with path.expanduser().open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                entries.append(TranscriptEntry.model_validate_json(line))
            except ValueError as error:
                raise ValueError(
                    f"invalid transcript entry at {path}:{line_number}"
                ) from error
    return entries


__all__ = ["TranscriptBuilder", "TranscriptEntry", "read_transcript"]
