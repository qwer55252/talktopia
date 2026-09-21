"""Pure renderers for human, agent-private, and SOTOPIA views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .actions import DuplexAction, DuplexObservation
from .events import ActionCommitted
from .transcript import TranscriptEntry


_AUDIBLE_ACTION_TYPES = frozenset(
    {"speak", "hesitation", "backchanneling", "correction", "interruption"}
)


def _timestamp(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return "--:--.---"
    minutes, remainder = divmod(timestamp_ms, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def render_markdown(entries: Sequence[TranscriptEntry]) -> str:
    blocks: list[str] = []
    for entry in entries:
        if entry.completed:
            status = "completed"
        elif entry.interrupted:
            status = "interrupted"
        else:
            status = "incomplete"
        if entry.cancellation_reason:
            status += f" ({entry.cancellation_reason})"

        blocks.append(
            "\n".join(
                [
                    f"[{_timestamp(entry.start_ms)} - {_timestamp(entry.end_ms)}] "
                    f"{entry.speaker} -> {entry.listener}",
                    f"Action: {entry.action_type} ({entry.origin})",
                    f"Generated: {entry.generated_text}",
                    f"Synthesized: {entry.synthesized_text}",
                    f"Received:  {entry.received_text}",
                    f"Status: {status}",
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_agent_history(
    entries: Sequence[TranscriptEntry], viewer: str, limit: int
) -> str:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    selected = entries[-limit:] if limit else []
    lines: list[str] = []
    for entry in selected:
        if entry.speaker == viewer:
            text = entry.generated_text
        elif entry.listener == viewer:
            text = entry.received_text
        else:
            continue
        if not text and entry.action_type in _AUDIBLE_ACTION_TYPES:
            continue
        rendered = (
            DuplexAction(
                action_type=entry.action_type,
                argument=text,
                to=[entry.listener] if entry.listener else [],
            ).to_natural_language()
            if entry.action_type not in {"correction", "interruption"}
            else f'[{entry.action_type}] "{text}"'
        )
        lines.append(f"{entry.speaker} {rendered}")
    return "\n".join(lines)


def render_sotopia_messages(
    initial_observations: Mapping[str, DuplexObservation],
    committed_events: Sequence[ActionCommitted],
    entries: Sequence[TranscriptEntry],
) -> list[list[tuple[str, str, str]]]:
    messages: list[list[tuple[str, str, str]]] = [
        [
            ("Environment", agent, observation.to_natural_language())
            for agent, observation in initial_observations.items()
        ]
    ]
    entries_by_commit_and_speaker = {
        (entry.commit_id, entry.speaker): entry
        for entry in entries
        if entry.commit_id is not None
    }

    for commit in sorted(committed_events, key=lambda event: event.sequence):
        turn: list[tuple[str, str, str]] = []
        for actor, action in commit.actions.items():
            if action.action_type == "none":
                continue
            rendered_action = action
            if action.action_type in _AUDIBLE_ACTION_TYPES:
                entry = entries_by_commit_and_speaker.get((commit.commit_id, actor))
                if entry is None:
                    raise ValueError(
                        f"audible commit {commit.commit_id} has no transcript entry "
                        f"for {actor}"
                    )
                if not entry.received_text.strip():
                    raise ValueError(
                        f"audible commit {commit.commit_id} has empty ASR final text"
                    )
                rendered_action = action.model_copy(
                    update={"argument": entry.received_text}
                )
            turn.append((actor, "Environment", rendered_action.to_natural_language()))
        if turn:
            messages.append(turn)
    return messages


__all__ = [
    "render_agent_history",
    "render_markdown",
    "render_sotopia_messages",
]
