"""Pure renderers for human, agent-private, and SOTOPIA views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sotopia.database import AgentProfile, EpisodeLog

from .actions import DuplexAction, DuplexObservation
from .events import ActionCommitted
from .transcript import TranscriptEntry


_AUDIBLE_ACTION_TYPES = frozenset(
    {"speak", "hesitation", "backchanneling", "correction", "interruption"}
)


def _timestamp(timestamp_ms: float | None) -> str:
    if timestamp_ms is None:
        return "--:--.---"
    minutes, remainder = divmod(round(timestamp_ms), 60_000)
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
        if entry.action_type == "backchanneling":
            text = entry.received_text
        nonverbal_backchannel = (
            entry.action_type == "backchanneling"
            and not entry.received_text.strip()
            and bool(entry.sentences)
        )
        if nonverbal_backchannel:
            text = ""
        if (
            not text
            and entry.action_type in _AUDIBLE_ACTION_TYPES
            and not nonverbal_backchannel
        ):
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
            if action.action_type == "none" and commit.metadata.get("actor") != actor:
                continue
            rendered_action = action
            if action.action_type in _AUDIBLE_ACTION_TYPES:
                entry = entries_by_commit_and_speaker.get((commit.commit_id, actor))
                if entry is None:
                    raise ValueError(
                        f"audible commit {commit.commit_id} has no transcript entry "
                        f"for {actor}"
                    )
                if not entry.sentences:
                    raise ValueError(
                        f"audible commit {commit.commit_id} has no delivered sentence evidence"
                    )
                rendered_action = action.model_copy(
                    update={"argument": entry.received_text}
                )
            turn.append((actor, "Environment", rendered_action.to_natural_language()))
        if turn:
            messages.append(turn)
    return messages


def render_episode_for_humans(
    source: EpisodeLog,
) -> tuple[list[AgentProfile], list[str]]:
    """Keep SOTOPIA's presentation without dropping real Surface5 pass actions."""
    profiles, rendered = source.render_for_humans()
    turns = [rendered[0]]
    for turn in source.messages[1:]:
        lines: list[str] = []
        for sender, receiver, message in turn:
            if receiver != "Environment":
                continue
            if sender == "Environment":
                lines.append(message)
            elif "said:" in message:
                lines.append(f"{sender} {message}")
            else:
                lines.append(f"{sender}: {message}")
        turns.append("\n".join(lines))
    return profiles, turns + rendered[-2:]


__all__ = [
    "render_agent_history",
    "render_episode_for_humans",
    "render_markdown",
    "render_sotopia_messages",
]
