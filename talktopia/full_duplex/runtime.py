"""Surface5's 40 ms duplex clock and explicit leave-handshake controller."""

from __future__ import annotations

from talktopia.experiment import UNCOUNTED_ACTIONS

import asyncio
import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .actions import (
    DuplexAction,
    DuplexActionDecision,
    DuplexActionType,
    DuplexObservation,
    HiddenSaid,
    SpeechChunk,
    StreamingObservation,
)
from .agent import ActiveUtterance, AgentOutput, CascadedDuplexAgent
from .audio import AudioChunk, AudioRouter, StereoWavWriter
from .config import RuntimeConfig
from .events import (
    ASRUpdateEvent,
    ActionCommitted,
    AudioDeliveryEvent,
    DecisionEvent,
    EpisodeEnded,
    EpisodeStarted,
    EventWriter,
    FloorEvent,
    HiddenSaidCreated,
    SpeechChunkSynthesized,
    SpeechLifecycleEvent,
    SpeechSynthesisFailed,
)
from .generation import GenerationFailure
from .sotopia_adapter import (
    ResolvedEpisode,
    SemanticCommit,
    SemanticSnapshot,
    SotopiaSession,
)
from .speech_backends import ASRUpdate, SpeechSynthesisFailure


LeavePhase = Literal[
    "active",
    "first_leave",
    "peer_closing",
    "draining",
    "completed",
]

_SPOKEN_ACTIONS = frozenset(
    {"speak", "hesitation", "backchanneling", "correction", "interruption"}
)
_TARGETED_ACTIONS = frozenset({"correction", "interruption"})
_AUXILIARY_ACTIONS = frozenset({"backchanneling", "hesitation"})
_BACKCHANNEL_MIN_STABLE_WORDS = 2
_PARTIAL_DECISION_INTERVAL_MS = 1_000


@dataclass(frozen=True, slots=True)
class DeferredObservation:
    listener: str
    trigger_speaker: str
    source: Literal["asr_final", "commit"]
    stable_text: str
    peer_utterance_id: str
    canonical: DuplexObservation

    def __post_init__(self) -> None:
        if not self.listener or not self.trigger_speaker:
            raise ValueError("deferred observation participants must not be blank")
        if not self.peer_utterance_id:
            raise ValueError("deferred observation peer utterance must not be blank")


@dataclass(slots=True)
class RuntimeState:
    episode_id: str
    now_ms: int
    opener_agent: str
    floor_utterances: dict[str, str] = field(default_factory=dict)
    next_floor_available_ms: int = 0
    active_utterances: dict[str, str] = field(default_factory=dict)
    left_agents: set[str] = field(default_factory=set)
    latest_observations: dict[str, StreamingObservation] = field(default_factory=dict)
    decision_revisions: dict[str, int] = field(default_factory=dict)
    leave_phase: LeavePhase = "active"
    pending_cancellations: set[str] = field(default_factory=set)
    deferred_observations: dict[str, DeferredObservation] = field(default_factory=dict)
    semantic_turn_number: int = 0
    budget_turns: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    stopping: bool = False


class AgentLivenessSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: str
    session_active: bool
    left: bool
    work_stage: Literal["idle", "decision", "hidden_said", "closing", "tts"]
    pending_generation: bool
    pending_output: bool
    active_utterance_id: str | None
    active_action_type: DuplexActionType | None
    latest_observation_id: str | None
    latest_observation_source: str | None


class RuntimeLivenessSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    now_ms: int
    leave_phase: LeavePhase
    stopping: bool
    audio_pending: bool
    floor_utterances: dict[str, str]
    active_utterances: dict[str, str]
    deferred_observation_listeners: tuple[str, ...]
    pending_decisions: dict[str, DuplexActionType]
    output_waiter_agents: tuple[str, ...]
    ready_output_count: int
    agents: tuple[AgentLivenessSnapshot, ...]


class FloorController:
    """Pure floor policy over the single mutable ``RuntimeState``."""

    def __init__(self, *, minimum_gap_ms: int = 200) -> None:
        if minimum_gap_ms < 0:
            raise ValueError("minimum_gap_ms must be non-negative")
        self.minimum_gap_ms = minimum_gap_ms

    def available_actions(
        self,
        agent: str,
        state: RuntimeState,
    ) -> list[DuplexActionType]:
        if agent in state.left_agents:
            return ["none"]
        peers = set(state.latest_observations) - {agent}
        if peers and peers.issubset(state.left_agents):
            return ["leave"]
        self_speaking = agent in state.active_utterances.values()
        peer_speaking = any(
            owner != agent for owner in state.active_utterances.values()
        )
        if self_speaking and peer_speaking:
            return [
                "none",
                "non-verbal communication",
                "action",
                "hesitation",
                "correction",
                "interruption",
                "leave",
            ]
        if self_speaking:
            return [
                "none",
                "non-verbal communication",
                "action",
                "hesitation",
                "leave",
            ]
        if peer_speaking:
            return [
                "none",
                "non-verbal communication",
                "action",
                "backchanneling",
                "correction",
                "interruption",
                "leave",
            ]
        return [
            "none",
            "speak",
            "non-verbal communication",
            "action",
            "leave",
            "hesitation",
        ]

    def reserve(
        self,
        state: RuntimeState,
        agent: str,
        utterance_id: str,
        action_type: DuplexActionType,
    ) -> FloorEvent:
        before = next(iter(state.floor_utterances.values()), None)
        if action_type == "backchanneling":
            change: Literal["reserved", "collision"] = (
                "collision" if state.floor_utterances else "reserved"
            )
            state.floor_utterances[utterance_id] = agent
        elif state.floor_utterances and agent not in state.floor_utterances.values():
            state.floor_utterances[utterance_id] = agent
            change = "collision"
        else:
            state.floor_utterances[utterance_id] = agent
            change = "reserved"
        state.active_utterances[utterance_id] = agent
        return self._event(
            state,
            change=change,
            owner_before=before,
            owner_after=agent,
            utterance_ids=tuple(state.floor_utterances),
            reason=action_type,
        )

    def release(
        self,
        state: RuntimeState,
        utterance_id: str,
        reason: str,
    ) -> FloorEvent:
        floor_owner = state.floor_utterances.pop(utterance_id, None)
        active_owner = state.active_utterances.pop(utterance_id, None)
        before = floor_owner if floor_owner is not None else active_owner
        after = next(iter(state.floor_utterances.values()), None)
        if floor_owner is not None and after is None:
            state.next_floor_available_ms = max(
                state.next_floor_available_ms,
                state.now_ms + self.minimum_gap_ms,
            )
        return self._event(
            state,
            change="released",
            owner_before=before,
            owner_after=after,
            utterance_ids=(utterance_id,),
            reason=reason,
        )

    def transfer(
        self,
        state: RuntimeState,
        source_id: str,
        target_id: str,
        action_type: DuplexActionType,
    ) -> FloorEvent:
        before = state.floor_utterances.pop(source_id, None)
        if before is None:
            raise ValueError(f"target utterance does not hold the floor: {source_id}")
        owner = state.active_utterances.get(target_id)
        if owner is None:
            raise ValueError(f"target utterance is not active: {target_id}")
        state.floor_utterances[target_id] = owner
        state.pending_cancellations.add(source_id)
        return self._event(
            state,
            change="transferred",
            owner_before=before,
            owner_after=owner,
            utterance_ids=(source_id, target_id),
            reason=action_type,
        )

    @staticmethod
    def _event(
        state: RuntimeState,
        *,
        change: Literal["reserved", "released", "transferred", "rejected", "collision"],
        owner_before: str | None,
        owner_after: str | None,
        utterance_ids: tuple[str, ...],
        reason: str | None,
    ) -> FloorEvent:
        # The controller returns a typed policy result. Runtime re-envelopes it
        # through EventWriter so sequence and globally unique IDs stay centralized.
        return FloorEvent(
            event_id=f"{state.episode_id}-floor-policy",
            episode_id=state.episode_id,
            sequence=0,
            timestamp_ms=state.now_ms,
            change=change,
            owner_before=owner_before,
            owner_after=owner_after,
            utterance_ids=utterance_ids,
            reason=reason,
        )


@dataclass(slots=True)
class _UtteranceRuntime:
    speaker: str
    listener: str
    decision: DuplexActionDecision
    action_type: DuplexActionType
    closing: bool
    hidden_said_id: str
    started: bool = False
    floor_claimed: bool = False


@dataclass(slots=True)
class _ChunkDelivery:
    planned_frames: int
    start_ms: int | None = None
    end_ms: int | None = None
    delivered_frames: int = 0
    pcm: bytearray = field(default_factory=bytearray)


class TurnLimitReached(Exception):
    """The last permitted semantic action has been committed."""


class DuplexRuntime:
    """Own global timing, journal envelopes, floor, and semantic commits."""

    def __init__(
        self,
        *,
        resolved: ResolvedEpisode,
        agents: tuple[CascadedDuplexAgent, CascadedDuplexAgent],
        session: SotopiaSession,
        event_writer: EventWriter,
        audio_router: AudioRouter,
        stereo_writer: StereoWavWriter,
        config: RuntimeConfig,
        model_names: tuple[str, str],
        seed: int = 0,
        sample_rate_hz: int = 24_000,
        frame_ms: int = 40,
    ) -> None:
        names = tuple(agent.agent_name for agent in agents)
        if len(set(names)) != 2:
            raise ValueError("duplex runtime requires two distinct agent names")
        self.resolved = resolved
        self.agents = agents
        self._agents = dict(zip(names, agents))
        self.session = session
        self.event_writer = event_writer
        self.audio_router = audio_router
        self.stereo_writer = stereo_writer
        self.config = config
        self.model_names = model_names
        self.seed = seed
        self.sample_rate_hz = sample_rate_hz
        self.frame_ms = frame_ms
        opener = select_opening_agent(resolved, event_writer.episode_id)
        self.state = RuntimeState(
            episode_id=event_writer.episode_id,
            now_ms=0,
            opener_agent=opener,
        )
        self.floor = FloorController(minimum_gap_ms=config.minimum_floor_gap_ms)
        self._snapshot: SemanticSnapshot | None = None
        self._pending_decisions: dict[str, DuplexActionDecision] = {}
        self._decision_observations: dict[str, StreamingObservation] = {}
        self._hidden_by_decision: dict[str, HiddenSaid] = {}
        self._utterances: dict[str, _UtteranceRuntime] = {}
        self._deliveries: dict[tuple[str, int], _ChunkDelivery] = {}
        self._first_leaver: str | None = None
        self._closing_agent: str | None = None
        self._latest_asr_event: dict[str, ASRUpdateEvent] = {}
        self._commit_sequence = 0
        self._output_tasks: dict[str, asyncio.Task[AgentOutput]] = {}
        self._ready_outputs: deque[tuple[str, AgentOutput]] = deque()
        self._last_dispatched_stable: dict[str, str] = {}
        self._last_partial_dispatch_ms: dict[str, int] = {}
        self._backchanneled_utterances: set[str] = set()
        self._partial_decision_suppressed_utterances: set[str] = set()
        self._cancelled_utterances: set[str] = set()
        self._discarded_utterances: set[str] = set()
        self._audio_finished_utterances: set[str] = set()
        self._closed_deliveries: set[tuple[str, int]] = set()
        self._discarded_decisions: set[str] = set()

    async def run(self, initial_snapshot: SemanticSnapshot) -> EpisodeEnded:
        self._snapshot = initial_snapshot
        self.state.semantic_turn_number = initial_snapshot.turn_number
        initial_streams = {
            name: self._streaming_observation(
                name,
                source="reset",
                canonical=observation,
                available_actions=(
                    ["speak", "non-verbal communication", "action"]
                    if name == self.state.opener_agent
                    else ["none"]
                ),
            )
            for name, observation in initial_snapshot.observations.items()
        }
        self.state.latest_observations.update(initial_streams)
        self.event_writer.emit(
            EpisodeStarted,
            self.state.now_ms,
            environment_id=str(self.resolved.env_profile.pk or ""),
            agent_ids=tuple(
                str(profile.pk or "") for profile in self.resolved.agent_profiles
            ),
            agent_names=tuple(self._agents),
            models=self.model_names,
            initial_observations=dict(initial_snapshot.observations),
            run_config={
                "turn_taking_policy": self.config.turn_taking_policy,
                "opener_agent": self.state.opener_agent,
                "opener_policy": "agent1",
                "minimum_floor_gap_ms": self.config.minimum_floor_gap_ms,
                "realtime": self.config.realtime,
                "seed": self.seed,
                "max_turns": self.config.max_turns,
            },
        )
        try:
            await self._agents[self.state.opener_agent].submit_observation(
                initial_streams[self.state.opener_agent]
            )
            while self.state.leave_phase != "completed":
                if self.audio_router.has_pending():
                    ready = self._try_next_output()
                    if ready is not None:
                        await self.handle_agent_output(*ready)
                    await self.tick_audio()
                else:
                    agent_name, output = await self._next_output()
                    await self.handle_agent_output(agent_name, output)
            # The episode runner writes the terminal event after saving artifacts.
            # Evaluation is a separate Talktopia stage.
            ended = EpisodeEnded(
                event_id=f"{self.state.episode_id}-runtime-completed",
                episode_id=self.state.episode_id,
                sequence=0,
                timestamp_ms=self.state.now_ms,
                status="completed",
                reason="explicit_leave_handshake",
                duration_ms=self.state.now_ms,
            )
            return ended
        except TurnLimitReached:
            await self._stop_at_turn_limit()
            return EpisodeEnded(
                event_id=f"{self.state.episode_id}-runtime-completed",
                episode_id=self.state.episode_id,
                sequence=0,
                timestamp_ms=self.state.now_ms,
                status="completed",
                reason="max_turns",
                duration_ms=self.state.now_ms,
            )
        except BaseException as error:
            reason = (
                "external_cancellation"
                if isinstance(error, asyncio.CancelledError)
                else "runtime_error"
            )
            self._record_stopped_audio(reason)
            raise
        finally:
            await self._cancel_output_waiters()

    async def handle_agent_output(self, agent: str, output: AgentOutput) -> None:
        if isinstance(output, DuplexActionDecision):
            observation = self._agents[agent].decision_observation(output.decision_id)
            attempt, validation_errors = self._agents[agent].generation.decision_audit(
                output.decision_id
            )
            for rejected_attempt, validation_error in enumerate(
                validation_errors,
                start=1,
            ):
                self.event_writer.emit(
                    DecisionEvent,
                    self.state.now_ms,
                    status="rejected",
                    agent=agent,
                    decision_id=output.decision_id,
                    observation=observation,
                    decision=None,
                    attempt=rejected_attempt,
                    validation_errors=(validation_error,),
                )
            latest = self.state.latest_observations[agent]
            if latest.canonical.observation_id != observation.canonical.observation_id:
                self.event_writer.emit(
                    DecisionEvent,
                    self.state.now_ms,
                    status="stale",
                    agent=agent,
                    decision_id=output.decision_id,
                    observation=observation,
                    decision=output,
                    attempt=attempt,
                    validation_errors=("a newer observation superseded this decision",),
                )
                self._discarded_decisions.add(output.decision_id)
                return
            self._pending_decisions[agent] = output
            self._decision_observations[output.decision_id] = observation
            self.event_writer.emit(
                DecisionEvent,
                self.state.now_ms,
                status="selected",
                agent=agent,
                decision_id=output.decision_id,
                observation=observation,
                decision=output,
                attempt=attempt,
                validation_errors=validation_errors,
            )
            await self.commit_selected_action(agent, output)
            return
        if isinstance(output, HiddenSaid):
            if output.decision_id in self._discarded_decisions:
                for chunk in self._agents[agent].generation.split_into_sentence_chunks(
                    output
                ):
                    self._discarded_utterances.add(chunk.utterance_id)
                return
            observation = self._decision_observations[output.decision_id]
            self._hidden_by_decision[output.decision_id] = output
            self.event_writer.emit(
                HiddenSaidCreated,
                self.state.now_ms,
                agent=agent,
                observation_id=observation.canonical.observation_id,
                hidden_said=output,
                attempt=self._agents[agent].generation.hidden_said_attempt(
                    output.hidden_said_id
                ),
            )
            return
        if isinstance(output, AudioChunk):
            if output.utterance_id in self._discarded_utterances:
                return
            await self._handle_audio_chunk(agent, output)
            return
        if isinstance(output, SpeechSynthesisFailure):
            await self._handle_synthesis_failure(agent, output)
            return
        if isinstance(output, ASRUpdate):
            event = self.event_writer.emit(
                ASRUpdateEvent,
                self.state.now_ms,
                utterance_id=output.utterance_id,
                listener=output.listener,
                text=output.text,
                is_final=output.is_final,
                is_stable=output.is_stable,
                revision_id=output.revision_id,
            )
            self._latest_asr_event[output.utterance_id] = event
            if output.is_final:
                await self.commit_asr_final(output)
            else:
                await self._handle_asr_partial(output)
            return
        raise TypeError(f"unsupported Surface5 AgentOutput: {type(output).__name__}")

    async def commit_selected_action(
        self,
        agent: str,
        decision: DuplexActionDecision,
    ) -> None:
        if decision.action_type in _SPOKEN_ACTIONS:
            observation = self._decision_observations[decision.decision_id]
            if (
                decision.action_type == "backchanneling"
                and observation.peer_utterance_id is not None
            ):
                self._backchanneled_utterances.add(observation.peer_utterance_id)
            return
        if decision.action_type == "none":
            return
        if decision.action_type == "leave":
            if self.state.leave_phase == "active":
                await self._commit_leave(agent, decision, first=True)
                return
            if self.state.leave_phase == "peer_closing" and agent != self._first_leaver:
                self._closing_agent = agent
                self.state.leave_phase = "draining"
                return
            raise RuntimeError("invalid leave decision for current handshake phase")
        action = DuplexAction(
            action_type=decision.action_type,
            argument=decision.non_audio_argument,
            to=decision.to,
        )
        metadata: dict[str, object] = {"decision_id": decision.decision_id}
        if decision.action_type in {"action", "non-verbal communication"}:
            metadata["non_audio_argument_attempt"] = self._agents[
                agent
            ].generation.non_audio_argument_attempt(decision.decision_id)
        await self._commit_joint(
            actor=agent,
            action=action,
            origin="agent",
            trigger_event_id=None,
            utterance_id=None,
            metadata=metadata,
        )
        if not self.state.active_utterances:
            await self._prompt_peer(agent, source="commit")

    async def commit_asr_final(self, update: ASRUpdate) -> None:
        utterance = self._utterances.get(update.utterance_id)
        if utterance is None:
            raise ValueError(
                f"ASR final has no active utterance: {update.utterance_id}"
            )
        if update.listener != utterance.listener:
            raise ValueError("ASR final listener does not match utterance")
        received = update.text.strip()
        if not received:
            if utterance.action_type == "backchanneling":
                floor_event = self.floor.release(
                    self.state,
                    update.utterance_id,
                    "blank_backchannel_asr_drop",
                )
                self._emit_floor(floor_event)
                self._agents[utterance.speaker].finish_speech(update.utterance_id)
                self._clear_utterance_state(update.utterance_id, utterance)
                await self._resume_deferred_observation(utterance.speaker)
                return
            raise RuntimeError(
                f"audible utterance has blank ASR final: {update.utterance_id}"
            )
        action = DuplexAction(
            action_type=utterance.action_type,
            argument=received,
            to=[utterance.listener],
            target_utterance_id=(
                utterance.decision.target_utterance_id
                if utterance.action_type in {"correction", "interruption"}
                else None
            ),
        )
        trigger = self._latest_asr_event.get(update.utterance_id)
        await self._commit_joint(
            actor=utterance.speaker,
            action=action,
            origin="environment_closing" if utterance.closing else "agent",
            trigger_event_id=trigger.event_id if trigger is not None else None,
            utterance_id=update.utterance_id,
            metadata={
                "decision_id": utterance.decision.decision_id,
                "hidden_said_id": utterance.hidden_said_id,
                "closing": utterance.closing,
            },
        )
        was_cancelled = update.utterance_id in self._cancelled_utterances
        floor_event = self.floor.release(
            self.state,
            update.utterance_id,
            "cancelled_asr_final_commit" if was_cancelled else "asr_final_commit",
        )
        self._emit_floor(floor_event)
        self._agents[utterance.speaker].finish_speech(update.utterance_id)
        self._clear_utterance_state(update.utterance_id, utterance)
        if utterance.closing:
            assert self._closing_agent == utterance.speaker
            decision = utterance.decision
            await self._commit_leave(utterance.speaker, decision, first=False)
        elif utterance.action_type in _AUXILIARY_ACTIONS:
            resumed = await self._resume_deferred_observation(utterance.speaker)
            if utterance.action_type == "backchanneling" or resumed:
                # A micro-listener response acknowledges the active speaker; it
                # must not manufacture a new substantive response opportunity.
                return
            await self._prompt_peer(
                utterance.speaker,
                source="asr_final",
                stable_text=received,
                peer_utterance_id=update.utterance_id,
            )
        elif was_cancelled:
            # A successfully claimed correction/interruption already answers
            # this cancelled prefix; do not create a duplicate prompt.
            return
        else:
            await self._prompt_peer(
                utterance.speaker,
                source="asr_final",
                stable_text=received,
                peer_utterance_id=update.utterance_id,
            )

    def _clear_utterance_state(
        self,
        utterance_id: str,
        utterance: _UtteranceRuntime,
    ) -> None:
        self._utterances.pop(utterance_id, None)
        self._cancelled_utterances.discard(utterance_id)
        self._audio_finished_utterances.discard(utterance_id)
        self._last_dispatched_stable.pop(utterance_id, None)
        self._partial_decision_suppressed_utterances.discard(utterance_id)
        self._pending_decisions.pop(utterance.speaker, None)
        self._hidden_by_decision.pop(utterance.decision.decision_id, None)

    async def tick_audio(self) -> None:
        frames = self.audio_router.pop_tick()
        if not any(frame is not None for frame in frames.values()):
            return
        self.stereo_writer.write_tick(frames)
        tick_start = self.state.now_ms
        deliveries: list[tuple[str, object]] = []
        for source, frame in frames.items():
            if frame is None:
                continue
            utterance = self._utterances[frame.utterance_id]
            delivery = self._deliveries[(frame.utterance_id, frame.chunk_index)]
            if delivery.start_ms is None:
                delivery.start_ms = tick_start
            delivery.end_ms = tick_start + frame.duration_ms
            delivery.delivered_frames += 1
            delivery.pcm.extend(frame.pcm_s16le)
            if not utterance.started:
                utterance.started = True
                self.event_writer.emit(
                    SpeechLifecycleEvent,
                    tick_start,
                    phase="started",
                    utterance_id=frame.utterance_id,
                    speaker=source,
                    action_type=utterance.action_type,
                    target_utterance_id=utterance.decision.target_utterance_id,
                )
            self._agents[source].note_audio_played(frame.utterance_id)
            deliveries.append((utterance.listener, frame))
        if deliveries:
            await asyncio.gather(
                *(
                    self._agents[listener].receive_audio(frame)  # type: ignore[arg-type]
                    for listener, frame in deliveries
                )
            )
        for source, frame in frames.items():
            if frame is None:
                continue
            delivery = self._deliveries[(frame.utterance_id, frame.chunk_index)]
            if frame.is_chunk_end:
                self.event_writer.emit(
                    AudioDeliveryEvent,
                    delivery.end_ms or tick_start,
                    utterance_id=frame.utterance_id,
                    source_agent=source,
                    chunk_index=frame.chunk_index,
                    delivered_frames=delivery.delivered_frames,
                    planned_frames=delivery.planned_frames,
                    start_ms=(
                        delivery.start_ms
                        if delivery.start_ms is not None
                        else tick_start
                    ),
                    end_ms=(
                        delivery.end_ms if delivery.end_ms is not None else tick_start
                    ),
                    delivered_pcm_sha256=hashlib.sha256(delivery.pcm).hexdigest(),
                )
                self._closed_deliveries.add((frame.utterance_id, frame.chunk_index))
            if frame.is_utterance_end:
                utterance = self._utterances[frame.utterance_id]
                self.event_writer.emit(
                    SpeechLifecycleEvent,
                    delivery.end_ms or tick_start,
                    phase="finished",
                    utterance_id=frame.utterance_id,
                    speaker=source,
                    action_type=utterance.action_type,
                    target_utterance_id=utterance.decision.target_utterance_id,
                )
                self._audio_finished_utterances.add(frame.utterance_id)
        self.state.now_ms += self.frame_ms
        await self._apply_pending_cancellations()
        if self.config.realtime:
            await asyncio.sleep(self.frame_ms / 1000)

    async def stop(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("stop reason must not be blank")
        self.state.stopping = True
        for agent in self.agents:
            await agent.cancel_speech(reason)

    async def _handle_audio_chunk(self, agent: str, audio: AudioChunk) -> None:
        decision = self._pending_decisions.get(agent)
        if decision is None:
            raise RuntimeError("audio chunk has no selected decision")
        hidden = self._hidden_by_decision.get(decision.decision_id)
        if hidden is None:
            raise RuntimeError("audio chunk has no hidden-said provenance")
        chunks = self._agents[agent].generation.split_into_sentence_chunks(hidden)
        chunk = chunks[audio.chunk_index]
        if chunk.utterance_id != audio.utterance_id or chunk.text != audio.text:
            raise RuntimeError("TTS audio diverged from its speech chunk")
        pcm_duration_ms = round(len(audio.pcm_s16le) / 2 * 1000 / audio.sample_rate_hz)
        self.event_writer.emit(
            SpeechChunkSynthesized,
            self.state.now_ms,
            speaker=agent,
            chunk=chunk,
            pcm_duration_ms=max(1, pcm_duration_ms),
            pcm_sha256=hashlib.sha256(audio.pcm_s16le).hexdigest(),
            attempt=audio.synthesis_attempt,
        )

        target_id = decision.target_utterance_id
        utterance = self._utterances.get(audio.utterance_id)
        if utterance is None:
            if decision.action_type in _TARGETED_ACTIONS and not self._target_available(
                agent,
                target_id,
            ):
                await self._discard_targeted_before_floor_claim(
                    agent=agent,
                    decision=decision,
                    hidden=hidden,
                    chunk=chunk,
                )
                return
            if (
                not self.state.floor_utterances
                and self.state.now_ms < self.state.next_floor_available_ms
            ):
                await self._advance_floor_gap()
            listener = self._peer(agent)
            closing = (
                decision.action_type == "leave"
                and self.state.leave_phase == "draining"
                and agent == self._closing_agent
            )
            action_type: DuplexActionType = "speak" if closing else decision.action_type
            utterance = _UtteranceRuntime(
                speaker=agent,
                listener=listener,
                decision=decision,
                action_type=action_type,
                closing=closing,
                hidden_said_id=hidden.hidden_said_id,
            )
            self._utterances[audio.utterance_id] = utterance
            floor_event = self.floor.reserve(
                self.state,
                agent,
                audio.utterance_id,
                action_type,
            )
            self._emit_floor(floor_event)
            if decision.action_type in _TARGETED_ACTIONS:
                assert target_id is not None
                transfer_event = self.floor.transfer(
                    self.state,
                    target_id,
                    audio.utterance_id,
                    decision.action_type,
                )
                self._emit_floor(transfer_event)
            utterance.floor_claimed = True
        elif decision.action_type in _TARGETED_ACTIONS and not utterance.floor_claimed:
            raise RuntimeError("targeted utterance emitted audio before claiming floor")
        bytes_per_frame = self.sample_rate_hz * self.frame_ms // 1000 * 2
        planned_frames = (len(audio.pcm_s16le) + bytes_per_frame - 1) // bytes_per_frame
        self._deliveries[(audio.utterance_id, audio.chunk_index)] = _ChunkDelivery(
            planned_frames=planned_frames
        )
        self.audio_router.enqueue(audio)

    def _target_available(self, agent: str, target_id: str | None) -> bool:
        if target_id is None:
            return False
        target = self._utterances.get(target_id)
        if target is None or target.speaker != self._peer(agent):
            return False
        if target_id not in self.state.floor_utterances:
            return False
        if self.state.active_utterances.get(target_id) != target.speaker:
            return False
        return target_id not in (
            self._audio_finished_utterances
            | self._cancelled_utterances
            | self._discarded_utterances
        )

    async def _discard_targeted_before_floor_claim(
        self,
        *,
        agent: str,
        decision: DuplexActionDecision,
        hidden: HiddenSaid,
        chunk: SpeechChunk,
    ) -> None:
        observation = self._decision_observations[decision.decision_id]
        attempt, _errors = self._agents[agent].generation.decision_audit(
            decision.decision_id
        )
        reason = "target_unavailable_before_floor_claim"
        self.event_writer.emit(
            DecisionEvent,
            self.state.now_ms,
            status="stale",
            agent=agent,
            decision_id=decision.decision_id,
            observation=observation,
            decision=decision,
            attempt=attempt,
            validation_errors=(reason,),
        )
        self.event_writer.emit(
            SpeechLifecycleEvent,
            self.state.now_ms,
            phase="cancelled",
            utterance_id=chunk.utterance_id,
            speaker=agent,
            action_type=decision.action_type,
            target_utterance_id=decision.target_utterance_id,
            reason=reason,
        )
        await self._agents[agent].cancel_speech(reason)
        self._agents[agent].finish_speech(chunk.utterance_id)
        self._discarded_decisions.add(decision.decision_id)
        self._discarded_utterances.add(chunk.utterance_id)
        self._pending_decisions.pop(agent, None)
        self._hidden_by_decision.pop(hidden.decision_id, None)

    async def _handle_synthesis_failure(
        self,
        agent: str,
        failure: SpeechSynthesisFailure,
    ) -> None:
        decision = self._pending_decisions.get(agent)
        if decision is None:
            raise RuntimeError("TTS failure has no selected decision")
        hidden = self._hidden_by_decision.get(decision.decision_id)
        if hidden is None:
            raise RuntimeError("TTS failure has no hidden-said provenance")
        chunk = failure.chunk
        if (
            failure.source_agent != agent
            or chunk.hidden_said_id != hidden.hidden_said_id
            or chunk.utterance_id in self._utterances
            or decision.action_type != "backchanneling"
            or chunk.chunk_index != 0
            or not chunk.is_final
        ):
            raise RuntimeError(
                "recoverable TTS failure is only valid for an unstarted backchannel"
            )
        observation = self._decision_observations[decision.decision_id]
        attempt, _errors = self._agents[agent].generation.decision_audit(
            decision.decision_id
        )
        self.event_writer.emit(
            SpeechSynthesisFailed,
            self.state.now_ms,
            speaker=agent,
            decision_id=decision.decision_id,
            hidden_said_id=hidden.hidden_said_id,
            chunk=chunk,
            attempts=failure.attempts,
            reason=failure.reason,
            error_detail=failure.error_detail,
        )
        self.event_writer.emit(
            DecisionEvent,
            self.state.now_ms,
            status="cancelled",
            agent=agent,
            decision_id=decision.decision_id,
            observation=observation,
            decision=decision,
            attempt=attempt,
            validation_errors=("backchannel_tts_empty_audio",),
        )
        self._agents[agent].finish_speech(chunk.utterance_id)
        self._discarded_decisions.add(decision.decision_id)
        self._discarded_utterances.add(chunk.utterance_id)
        self._pending_decisions.pop(agent, None)
        self._hidden_by_decision.pop(decision.decision_id, None)
        if observation.peer_utterance_id is not None:
            self._backchanneled_utterances.discard(observation.peer_utterance_id)
        await self._resume_deferred_observation(agent)

    async def _apply_pending_cancellations(self) -> None:
        pending = tuple(self.state.pending_cancellations)
        self.state.pending_cancellations.clear()
        for utterance_id in pending:
            if utterance_id in self._audio_finished_utterances:
                continue
            utterance = self._utterances.get(utterance_id)
            if utterance is None:
                continue
            reason = "floor_taken_by_" + next(
                (
                    candidate.action_type
                    for candidate in self._utterances.values()
                    if candidate.decision.target_utterance_id == utterance_id
                ),
                "interruption",
            )
            await self._agents[utterance.speaker].cancel_speech(reason)
            self.audio_router.cancel(utterance_id)
            self._emit_open_delivery_prefixes(utterance_id)
            self.event_writer.emit(
                SpeechLifecycleEvent,
                self.state.now_ms,
                phase="cancelled",
                utterance_id=utterance_id,
                speaker=utterance.speaker,
                action_type=utterance.action_type,
                target_utterance_id=utterance.decision.target_utterance_id,
                reason=reason,
            )
            self._cancelled_utterances.add(utterance_id)
            self._discarded_utterances.add(utterance_id)
            final = await self._agents[utterance.listener].cancel_received_audio(
                utterance_id
            )
            await self.handle_agent_output(utterance.listener, final)

    def _emit_open_delivery_prefixes(self, utterance_id: str) -> None:
        for key, delivery in self._deliveries.items():
            if (
                key[0] != utterance_id
                or key in self._closed_deliveries
                or delivery.delivered_frames == 0
                or delivery.start_ms is None
                or delivery.end_ms is None
            ):
                continue
            self.event_writer.emit(
                AudioDeliveryEvent,
                self.state.now_ms,
                utterance_id=utterance_id,
                source_agent=self._utterances[utterance_id].speaker,
                chunk_index=key[1],
                delivered_frames=delivery.delivered_frames,
                planned_frames=delivery.planned_frames,
                start_ms=delivery.start_ms,
                end_ms=delivery.end_ms,
                delivered_pcm_sha256=hashlib.sha256(delivery.pcm).hexdigest(),
            )
            self._closed_deliveries.add(key)

    async def _handle_asr_partial(self, update: ASRUpdate) -> None:
        if not update.is_stable or update.utterance_id not in self._utterances:
            return
        utterance = self._utterances[update.utterance_id]
        if utterance.action_type == "backchanneling":
            return
        if update.utterance_id in self._partial_decision_suppressed_utterances:
            return
        stable_text = update.text.strip()
        if not stable_text:
            return
        previous = self._last_dispatched_stable.get(update.utterance_id, "")
        if stable_text == previous:
            return
        if previous and not stable_text.startswith(previous):
            return
        last_dispatch = self._last_partial_dispatch_ms.get(
            update.listener,
            -_PARTIAL_DECISION_INTERVAL_MS,
        )
        if self.state.now_ms - last_dispatch < _PARTIAL_DECISION_INTERVAL_MS:
            return
        if update.listener in self.state.active_utterances.values():
            return

        await self._discard_unstarted_superseded_speech(update.listener)

        available = self.floor.available_actions(update.listener, self.state)
        available = [action for action in available if action != "leave"]
        word_count = len(stable_text.split())
        minimums = {
            "correction": self.config.correction_min_stable_words,
            "interruption": self.config.interruption_min_stable_words,
        }
        available = [
            action
            for action in available
            if action not in _TARGETED_ACTIONS or word_count >= minimums[action]
        ]
        if (
            word_count < _BACKCHANNEL_MIN_STABLE_WORDS
            or update.utterance_id in self._backchanneled_utterances
        ):
            available = [action for action in available if action != "backchanneling"]
        if not available:
            available = ["none"]
        if not any(
            action in {"backchanneling", "correction", "interruption"}
            for action in available
        ):
            return
        new_stable_text = (
            stable_text[len(previous) :].strip() if previous else stable_text
        )
        assert self._snapshot is not None
        observation = self._streaming_observation(
            update.listener,
            source="asr_partial",
            canonical=self._snapshot.observations[update.listener],
            available_actions=available,
            stable_text=stable_text,
            new_stable_text=new_stable_text,
            peer_utterance_id=update.utterance_id,
            peer_speaking=True,
        )
        self._last_dispatched_stable[update.utterance_id] = stable_text
        self._last_partial_dispatch_ms[update.listener] = self.state.now_ms
        self.state.latest_observations[update.listener] = observation
        await self._agents[update.listener].submit_observation(observation)

    async def _discard_unstarted_superseded_speech(
        self,
        agent: str,
    ) -> bool:
        """Cancel only an auxiliary that a newer partial will supersede.

        Nothing is discarded once runtime audio, floor ownership, or first
        playback exists.  The current task is cancelled before a replacement
        task is created, so this cannot cancel a newer observation's work.
        """

        active: ActiveUtterance | None = self._agents[agent].state.active_utterance
        if (
            active is None
            or active.action_type not in _AUXILIARY_ACTIONS
            or active.first_audio_played
            or active.utterance_id in self._utterances
            or active.utterance_id in self.state.active_utterances
            or active.utterance_id in self.state.floor_utterances
        ):
            return False

        reason = "superseded_before_first_audio"
        await self._agents[agent].cancel_speech(reason)
        if self._agents[agent].state.active_utterance is not active:
            return False
        self._agents[agent].finish_speech(active.utterance_id)
        self._discarded_decisions.add(active.decision_id)
        self._discarded_utterances.add(active.utterance_id)
        hidden = self._hidden_by_decision.pop(active.decision_id, None)
        decision = self._pending_decisions.get(agent)
        if decision is not None and decision.decision_id == active.decision_id:
            observation = self._decision_observations[active.decision_id]
            attempt, _errors = self._agents[agent].generation.decision_audit(
                active.decision_id
            )
            self.event_writer.emit(
                DecisionEvent,
                self.state.now_ms,
                status="cancelled",
                agent=agent,
                decision_id=active.decision_id,
                observation=observation,
                decision=decision,
                attempt=attempt,
                validation_errors=(reason,),
            )
            self._pending_decisions.pop(agent, None)
            if observation.peer_utterance_id is not None:
                self._backchanneled_utterances.discard(observation.peer_utterance_id)
        if (
            hidden is not None
            and hidden.hidden_said_id != active.hidden_said.hidden_said_id
        ):
            raise RuntimeError("superseded auxiliary hidden-said provenance diverged")
        return True

    async def _advance_floor_gap(self) -> None:
        silence = {name: None for name in self._agents}
        while self.state.now_ms < self.state.next_floor_available_ms:
            self.stereo_writer.write_tick(silence)
            self.state.now_ms += self.frame_ms
            if self.config.realtime:
                await asyncio.sleep(self.frame_ms / 1000)

    async def _commit_leave(
        self,
        agent: str,
        decision: DuplexActionDecision,
        *,
        first: bool,
    ) -> None:
        await self._commit_joint(
            actor=agent,
            action=DuplexAction(action_type="leave", argument="", to=[]),
            origin="agent",
            trigger_event_id=None,
            utterance_id=None,
            metadata={
                "decision_id": decision.decision_id,
                "leave_phase": "first" if first else "second",
            },
        )
        self.state.left_agents.add(agent)
        self._agents[agent].state.left = True
        self._pending_decisions.pop(agent, None)
        self.state.deferred_observations.pop(agent, None)
        if first:
            self._first_leaver = agent
            self.state.leave_phase = "first_leave"
            peer = self._peer(agent)
            self.state.deferred_observations.pop(peer, None)
            self.state.leave_phase = "peer_closing"
            assert self._snapshot is not None
            observation = self._streaming_observation(
                peer,
                source="peer_left",
                canonical=self._snapshot.observations[peer],
                available_actions=["leave"],
            )
            self.state.latest_observations[peer] = observation
            await self._agents[peer].submit_observation(observation)
        else:
            self.state.left_agents.add(agent)
            self.state.leave_phase = "completed"

    async def _commit_joint(
        self,
        *,
        actor: str,
        action: DuplexAction,
        origin: str,
        trigger_event_id: str | None,
        utterance_id: str | None,
        metadata: dict[str, object],
    ) -> None:
        assert self._snapshot is not None
        if self.state.budget_turns >= self.config.max_turns:
            raise TurnLimitReached
        peer = self._peer(actor)
        actions = {
            actor: action,
            peer: DuplexAction(action_type="none", argument="", to=[]),
        }
        commit = SemanticCommit(
            actions=actions,
            expected_turn_number=self._snapshot.turn_number,
            timestamp_ms=self.state.now_ms,
            origin=origin,
            utterance_ids={actor: utterance_id, peer: None},
            metadata=metadata,
        )
        snapshot = await self.session.commit(commit)
        self._snapshot = snapshot
        self.state.semantic_turn_number = snapshot.turn_number
        self.state.action_counts[action.action_type] = (
            self.state.action_counts.get(action.action_type, 0) + 1
        )
        if action.action_type not in UNCOUNTED_ACTIONS:
            self.state.budget_turns += 1
        self._commit_sequence += 1
        commit_id = f"{self.state.episode_id}-commit-{self._commit_sequence:04d}"
        self.event_writer.emit(
            ActionCommitted,
            self.state.now_ms,
            causation_id=trigger_event_id,
            commit_id=commit_id,
            turn_number=snapshot.turn_number,
            actions=actions,
            observations_after=dict(snapshot.observations),
            origin=origin,
            trigger_event_id=trigger_event_id,
            metadata={
                **metadata,
                "utterance_ids": {actor: utterance_id, peer: None},
                "budget_turns": self.state.budget_turns,
            },
        )
        decision_id = metadata.get("decision_id")
        hidden = (
            self._hidden_by_decision.get(decision_id)
            if isinstance(decision_id, str)
            else None
        )
        for agent in self.agents:
            agent.record_committed_action(
                actor,
                action,
                own_generated_text=(
                    hidden.text
                    if hidden is not None and agent.agent_name == actor
                    else None
                ),
            )

        if self.state.budget_turns >= self.config.max_turns:
            raise TurnLimitReached

    def _record_stopped_audio(self, reason: str) -> None:
        for utterance_id, utterance in self._utterances.items():
            self.audio_router.cancel(utterance_id)
            self._emit_open_delivery_prefixes(utterance_id)
            if utterance_id not in self._audio_finished_utterances:
                self.event_writer.emit(
                    SpeechLifecycleEvent,
                    self.state.now_ms,
                    phase="cancelled",
                    utterance_id=utterance_id,
                    speaker=utterance.speaker,
                    action_type=utterance.action_type,
                    target_utterance_id=utterance.decision.target_utterance_id,
                    reason=reason,
                )

    async def _stop_at_turn_limit(self) -> None:
        """Keep delivered audio evidence without committing a thirteenth action."""
        await self.stop("max_turns")
        self._record_stopped_audio("max_turns")
        for utterance_id, utterance in tuple(self._utterances.items()):
            previous = self._latest_asr_event.get(utterance_id)
            if previous is None or not previous.is_final:
                final = await self._agents[utterance.listener].finalize_received_audio(
                    utterance_id
                )
                if final is not None:
                    self.event_writer.emit(
                        ASRUpdateEvent,
                        self.state.now_ms,
                        **final.model_dump(),
                    )
            self._agents[utterance.speaker].finish_speech(utterance_id)
        self.state.active_utterances.clear()
        self.state.floor_utterances.clear()
        self.state.pending_cancellations.clear()
        self.state.deferred_observations.clear()

    async def _prompt_peer(
        self,
        speaker: str,
        *,
        source: Literal["asr_final", "commit"],
        stable_text: str = "",
        peer_utterance_id: str | None = None,
    ) -> None:
        assert self._snapshot is not None
        listener = self._peer(speaker)
        if listener in self.state.left_agents:
            return
        canonical = self._snapshot.observations[listener]
        if self._has_active_auxiliary(listener):
            if peer_utterance_id is None:
                raise RuntimeError(
                    "an auxiliary can only defer an identified peer utterance"
                )
            deferred = DeferredObservation(
                listener=listener,
                trigger_speaker=speaker,
                source=source,
                stable_text=stable_text,
                peer_utterance_id=peer_utterance_id,
                canonical=canonical,
            )
            existing = self.state.deferred_observations.get(listener)
            if existing is not None and existing != deferred:
                raise RuntimeError(
                    f"multiple deferred observations for listener: {listener}"
                )
            self.state.deferred_observations[listener] = deferred
            return
        available = self._response_actions(listener, source)
        observation = self._streaming_observation(
            listener,
            source=source,
            canonical=canonical,
            available_actions=available,
            stable_text=stable_text,
            peer_utterance_id=peer_utterance_id,
            peer_speaking=speaker in self.state.active_utterances.values(),
        )
        self.state.latest_observations[listener] = observation
        await self._agents[listener].submit_observation(observation)

    def _has_active_auxiliary(self, agent: str) -> bool:
        pending = self._pending_decisions.get(agent)
        if pending is not None and pending.action_type in _AUXILIARY_ACTIONS:
            return True
        active = self._agents[agent].state.active_utterance
        if active is not None and active.action_type in _AUXILIARY_ACTIONS:
            return True
        return any(
            utterance.speaker == agent and utterance.action_type in _AUXILIARY_ACTIONS
            for utterance in self._utterances.values()
        )

    async def _resume_deferred_observation(self, agent: str) -> bool:
        deferred = self.state.deferred_observations.pop(agent, None)
        if deferred is None:
            return False
        if (
            agent in self.state.left_agents
            or self.state.stopping
            or self.state.leave_phase != "active"
        ):
            return False
        if self._has_active_auxiliary(agent):
            raise RuntimeError(
                f"deferred observation resumed before auxiliary cleanup: {agent}"
            )
        available = self._response_actions(agent, deferred.source)
        observation = self._streaming_observation(
            agent,
            source=deferred.source,
            canonical=deferred.canonical,
            available_actions=available,
            stable_text=deferred.stable_text,
            peer_utterance_id=deferred.peer_utterance_id,
            peer_speaking=(
                deferred.trigger_speaker in self.state.active_utterances.values()
            ),
        )
        self.state.latest_observations[agent] = observation
        await self._agents[agent].submit_observation(observation)
        return True

    def _response_actions(
        self,
        agent: str,
        source: Literal["asr_final", "commit"],
    ) -> list[DuplexActionType]:
        available = self.floor.available_actions(agent, self.state)
        if (
            source in {"asr_final", "commit"}
            and self.state.leave_phase == "active"
            and not self.state.active_utterances
        ):
            available = [action for action in available if action != "none"]
        if not available:
            raise RuntimeError("authoritative idle observation has no response action")
        return available

    def liveness_snapshot(self) -> RuntimeLivenessSnapshot:
        agents: list[AgentLivenessSnapshot] = []
        for name, agent in self._agents.items():
            active = agent.state.active_utterance
            latest = agent.state.latest_observation
            agents.append(
                AgentLivenessSnapshot(
                    agent=name,
                    session_active=agent.state.session_active,
                    left=agent.state.left,
                    work_stage=agent.state.work_stage,
                    pending_generation=agent.has_pending_generation,
                    pending_output=agent.has_pending_output,
                    active_utterance_id=(
                        active.utterance_id if active is not None else None
                    ),
                    active_action_type=(
                        active.action_type if active is not None else None
                    ),
                    latest_observation_id=(
                        latest.canonical.observation_id if latest is not None else None
                    ),
                    latest_observation_source=(
                        latest.source if latest is not None else None
                    ),
                )
            )
        return RuntimeLivenessSnapshot(
            episode_id=self.state.episode_id,
            now_ms=self.state.now_ms,
            leave_phase=self.state.leave_phase,
            stopping=self.state.stopping,
            audio_pending=self.audio_router.has_pending(),
            floor_utterances=dict(self.state.floor_utterances),
            active_utterances=dict(self.state.active_utterances),
            deferred_observation_listeners=tuple(
                sorted(self.state.deferred_observations)
            ),
            pending_decisions={
                name: decision.action_type
                for name, decision in self._pending_decisions.items()
            },
            output_waiter_agents=tuple(sorted(self._output_tasks)),
            ready_output_count=len(self._ready_outputs),
            agents=tuple(agents),
        )

    def _streaming_observation(
        self,
        agent: str,
        *,
        source: Literal["reset", "asr_partial", "asr_final", "commit", "peer_left"],
        canonical: object,
        available_actions: list[DuplexActionType],
        stable_text: str = "",
        new_stable_text: str = "",
        peer_utterance_id: str | None = None,
        peer_speaking: bool = False,
    ) -> StreamingObservation:
        if not isinstance(canonical, DuplexObservation):
            raise TypeError("runtime requires a DuplexObservation")
        revision = self.state.decision_revisions.get(agent, 0) + 1
        self.state.decision_revisions[agent] = revision
        updated = canonical.model_copy(
            update={
                "available_actions": available_actions,
                "observation_id": (
                    f"{self.state.episode_id}-observation-"
                    f"{canonical.turn_number:06d}-{revision}-{source}"
                ),
            }
        )
        active = self._agents[agent].state.active_utterance
        supports_target = any(
            action in {"correction", "interruption"} for action in available_actions
        )
        return StreamingObservation(
            canonical=updated,
            source=source,
            stable_text=stable_text,
            new_stable_text=new_stable_text or stable_text,
            peer_utterance_id=peer_utterance_id,
            asr_revision_id=revision,
            peer_speaking=peer_speaking,
            self_speaking=agent in self.state.active_utterances.values(),
            self_active_action_type=active.action_type if active is not None else None,
            target_utterance_id=(peer_utterance_id if supports_target else None),
        )

    async def _next_output(self) -> tuple[str, AgentOutput]:
        while not self._ready_outputs:
            self._ensure_output_tasks()
            done, _pending = await asyncio.wait(
                set(self._output_tasks.values()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            self._collect_output_tasks(done)
        return self._ready_outputs.popleft()

    def _try_next_output(self) -> tuple[str, AgentOutput] | None:
        if self._ready_outputs:
            return self._ready_outputs.popleft()
        self._ensure_output_tasks()
        done = {task for task in self._output_tasks.values() if task.done()}
        self._collect_output_tasks(done)
        if self._ready_outputs:
            return self._ready_outputs.popleft()
        return None

    def _ensure_output_tasks(self) -> None:
        for name, agent in self._agents.items():
            if name not in self._output_tasks:
                self._output_tasks[name] = asyncio.create_task(
                    agent.next_output(),
                    name=f"surface5-runtime-output-{name}",
                )

    def _collect_output_tasks(
        self,
        done: set[asyncio.Task[AgentOutput]],
    ) -> None:
        for name in self._agents:
            task = self._output_tasks.get(name)
            if task is not None and task in done:
                self._output_tasks.pop(name)
                try:
                    output = task.result()
                except GenerationFailure as error:
                    observation: StreamingObservation | None = None
                    if error.stage == "decision":
                        observation = self._agents[name].state.latest_observation
                        if observation is not None:
                            decision_id = (
                                f"{self.state.episode_id}-{_slug(name)}-"
                                "decision-failed-"
                                f"{self.state.decision_revisions.get(name, 0):04d}"
                            )
                            for attempt, validation_error in enumerate(
                                error.validation_errors,
                                start=1,
                            ):
                                self.event_writer.emit(
                                    DecisionEvent,
                                    self.state.now_ms,
                                    status="rejected",
                                    agent=name,
                                    decision_id=decision_id,
                                    observation=observation,
                                    decision=None,
                                    attempt=attempt,
                                    validation_errors=(validation_error,),
                                )
                    if (
                        error.stage == "decision"
                        and observation is not None
                        and observation.source == "asr_partial"
                        and observation.peer_utterance_id is not None
                    ):
                        self._partial_decision_suppressed_utterances.add(
                            observation.peer_utterance_id
                        )
                        continue
                    raise
                self._ready_outputs.append((name, output))

    async def _cancel_output_waiters(self) -> None:
        tasks = tuple(self._output_tasks.values())
        self._output_tasks.clear()
        self._ready_outputs.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _emit_floor(self, policy_event: FloorEvent) -> FloorEvent:
        return self.event_writer.emit(
            FloorEvent,
            self.state.now_ms,
            change=policy_event.change,
            owner_before=policy_event.owner_before,
            owner_after=policy_event.owner_after,
            utterance_ids=policy_event.utterance_ids,
            reason=policy_event.reason,
        )

    def _peer(self, agent: str) -> str:
        peers = [name for name in self._agents if name != agent]
        if len(peers) != 1:
            raise ValueError(f"unknown runtime agent: {agent}")
        return peers[0]


def _slug(value: str) -> str:
    characters = "".join(
        character.lower() if character.isalnum() else " " for character in value
    )
    return "-".join(characters.split()) or "agent"


def select_opening_agent(resolved: ResolvedEpisode, episode_id: str) -> str:
    profile = resolved.agent_profiles[0]
    return f"{profile.first_name} {profile.last_name}".strip()


__all__ = [
    "AgentLivenessSnapshot",
    "DeferredObservation",
    "DuplexRuntime",
    "FloorController",
    "RuntimeLivenessSnapshot",
    "RuntimeState",
    "select_opening_agent",
]
