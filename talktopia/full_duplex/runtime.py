"""Surface5's live audio delivery, floor control, and first-leave termination."""

from __future__ import annotations

from talktopia.experiment import UNCOUNTED_ACTIONS
from talktopia.models.config import backchannel_tts_settings

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
from .config import (
    RuntimeConfig,
    SIMULATION_PROMPT_VERSION,
)
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
    ResponseLatencyEvent,
)
from .generation import GeneratedAction
from .sotopia_adapter import (
    ResolvedEpisode,
    SemanticCommit,
    SemanticSnapshot,
    SotopiaSession,
)
from .speech_backends import ASRUpdate, SpeechSynthesisFailure


LeavePhase = Literal["active", "completed"]

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
    now_ms: float
    opener_agent: str
    floor_utterances: dict[str, str] = field(default_factory=dict)
    next_floor_available_ms: float = 0
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
    end_reason: str | None = None


class AgentLivenessSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: str
    session_active: bool
    left: bool
    work_stage: Literal["idle", "decision", "hidden_said", "tts"]
    pending_generation: bool
    pending_output: bool
    active_utterance_id: str | None
    active_action_type: DuplexActionType | None
    latest_observation_id: str | None
    latest_observation_source: str | None


class RuntimeLivenessSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    now_ms: float
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

    def __init__(
        self, *, minimum_gap_ms: int = 200, config: RuntimeConfig | None = None
    ) -> None:
        if minimum_gap_ms < 0:
            raise ValueError("minimum_gap_ms must be non-negative")
        self.minimum_gap_ms = minimum_gap_ms
        self.config = config or RuntimeConfig()

    def _available_actions(
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
                "correction",
                "interruption",
                "leave",
            ]
        if self_speaking:
            return [
                "none",
                "non-verbal communication",
                "action",
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
        ]

    def available_actions(
        self, agent: str, state: RuntimeState
    ) -> list[DuplexActionType]:
        allowed = self._available_actions(agent, state)
        disabled = set()
        if not self.config.allow_backchannels:
            disabled.add("backchanneling")
        if not self.config.allow_corrections:
            disabled.add("correction")
        if not self.config.allow_interruptions:
            disabled.add("interruption")
        return [action for action in allowed if action not in disabled]

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
    hidden_said_id: str
    started: bool = False
    floor_claimed: bool = False


@dataclass(slots=True)
class _ChunkDelivery:
    planned_frames: int
    start_ms: float | None = None
    end_ms: float | None = None
    delivered_frames: int = 0
    pcm: bytearray = field(default_factory=bytearray)
    frame_spans: list[dict] = field(default_factory=list)


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
        self.floor = FloorController(
            minimum_gap_ms=config.minimum_floor_gap_ms, config=config
        )
        self._snapshot: SemanticSnapshot | None = None
        self._pending_decisions: dict[str, DuplexActionDecision] = {}
        self._decision_observations: dict[str, StreamingObservation] = {}
        self._hidden_by_decision: dict[str, HiddenSaid] = {}
        self._utterances: dict[str, _UtteranceRuntime] = {}
        self._deliveries: dict[tuple[str, int], _ChunkDelivery] = {}
        self._latest_asr_event: dict[str, ASRUpdateEvent] = {}
        self._committed_actions: list[tuple[str, DuplexAction]] = []
        self._fallback_decisions: set[str] = set()
        self._commit_sequence = 0
        self._output_tasks: dict[str, asyncio.Task[AgentOutput]] = {}
        self._ready_outputs: deque[tuple[str, AgentOutput]] = deque()
        self._last_dispatched_stable: dict[str, str] = {}
        self._last_partial_dispatch_ms: dict[str, int] = {}
        self._backchanneled_utterances: set[str] = set()
        self._cancelled_utterances: set[str] = set()
        self._discarded_utterances: set[str] = set()
        self._audio_finished_utterances: set[str] = set()
        self._closed_deliveries: set[tuple[str, int]] = set()
        self._discarded_decisions: set[str] = set()
        self._audio_task: asyncio.Task[None] | None = None
        self._audio_stopping = False
        self._audio_end_ms: dict[str, float] = {}
        self.latencies: dict[str, list[float]] = {
            "normal_response": [],
            "backchannel": [],
        }

    @property
    def now_ms(self) -> float:
        self.state.now_ms = self.event_writer.now_ms()
        return self.state.now_ms

    def latency_summary(self) -> dict:
        return {
            kind: {
                "count": len(values),
                "mean_ms": sum(values) / len(values) if values else None,
            }
            for kind, values in self.latencies.items()
        }

    async def _pump_audio(self) -> None:
        while not self._audio_stopping:
            if self.audio_router.has_pending():
                await self.tick_audio()
            else:
                self.stereo_writer.silence_until(self.now_ms)
                await asyncio.sleep(self.frame_ms / 1000)

    async def _stop_audio(self, *, check_errors: bool = True) -> None:
        self._audio_stopping = True
        if self._audio_task is not None:
            # Drain the currently transmitted frame (at most one frame period).
            # Never retain a recorded frame with no completed delivery timestamp.
            task = self._audio_task
            drain = asyncio.gather(task, return_exceptions=True)
            cancellation = None
            while not drain.done():
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError as error:
                    cancellation = error
            self._audio_task = None
            if cancellation is not None:
                self.stereo_writer.silence_until(self.now_ms)
                raise cancellation
            if check_errors and not task.cancelled():
                task.result()
        self.stereo_writer.silence_until(self.now_ms)

    async def run(self, initial_snapshot: SemanticSnapshot) -> EpisodeEnded:
        self._snapshot = initial_snapshot
        self.state.semantic_turn_number = initial_snapshot.turn_number
        initial_streams = {
            name: self._streaming_observation(
                name,
                source="reset",
                canonical=observation,
                available_actions=(
                    ["none", "speak", "non-verbal communication", "action", "leave"]
                    if name == self.state.opener_agent
                    else ["none"]
                ),
            )
            for name, observation in initial_snapshot.observations.items()
        }
        self.state.latest_observations.update(initial_streams)
        for name, agent in self._agents.items():
            agent.record_observation(initial_snapshot.observations[name])
        self.event_writer.emit(
            EpisodeStarted,
            self.now_ms,
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
                "clock": "monotonic_elapsed_ms",
                "audio_capture": "live_pcm_sample_clock_v3",
                "audio_frame_ms": self.frame_ms,
                "simulation_prompt": SIMULATION_PROMPT_VERSION,
                "generation_mode": self.config.generation_mode,
                "history_policy": self.config.history_policy,
                "termination_policy": self.config.termination_policy,
                "allow_backchannels": self.config.allow_backchannels,
                "allow_corrections": self.config.allow_corrections,
                "allow_interruptions": self.config.allow_interruptions,
                "backchannel_tts": backchannel_tts_settings(),
                "seed": self.seed,
                "max_turns": self.config.max_turns,
            },
        )
        try:
            await self._agents[self.state.opener_agent].submit_observation(
                initial_streams[self.state.opener_agent]
            )
            self._audio_task = asyncio.create_task(
                self._pump_audio(), name="surface5-live-audio"
            )
            while self.state.leave_phase != "completed":
                agent_name, output = await self._next_output()
                await self.handle_agent_output(agent_name, output)
            await self._stop_audio()
            assert self.state.end_reason is not None
            await self._finish_delivered_audio(self.state.end_reason)
            # The episode runner writes the terminal event after saving artifacts.
            # Evaluation is a separate Talktopia stage.
            ended = EpisodeEnded(
                event_id=f"{self.state.episode_id}-runtime-completed",
                episode_id=self.state.episode_id,
                sequence=0,
                timestamp_ms=self.now_ms,
                status="completed",
                reason=self.state.end_reason,
                duration_ms=self.now_ms,
            )
            return ended
        except TurnLimitReached:
            await self._stop_audio()
            await self._finish_delivered_audio("max_turns")
            return EpisodeEnded(
                event_id=f"{self.state.episode_id}-runtime-completed",
                episode_id=self.state.episode_id,
                sequence=0,
                timestamp_ms=self.now_ms,
                status="completed",
                reason="max_turns",
                duration_ms=self.now_ms,
            )
        except BaseException as error:
            await self._stop_audio(check_errors=False)
            reason = (
                "external_cancellation"
                if isinstance(error, asyncio.CancelledError)
                else "runtime_error"
            )
            self._record_stopped_audio(reason)
            raise
        finally:
            await self._stop_audio(check_errors=False)
            await self._cancel_output_waiters()

    async def handle_agent_output(self, agent: str, output: AgentOutput) -> None:
        if isinstance(output, GeneratedAction):
            generated = output
            decision = generated.decision
            observation = self._agents[agent].decision_observation(decision.decision_id)
            attempt, validation_errors = self._agents[agent].generation.decision_audit(
                decision.decision_id
            )
            for rejected_attempt, validation_error in enumerate(validation_errors, 1):
                self.event_writer.emit(
                    DecisionEvent,
                    self.now_ms,
                    status="rejected",
                    agent=agent,
                    decision_id=decision.decision_id,
                    observation=observation,
                    decision=None,
                    attempt=rejected_attempt,
                    validation_errors=(validation_error,),
                )
            rejection = self._proposal_rejection(agent, decision, observation)
            if rejection:
                self.event_writer.emit(
                    DecisionEvent,
                    self.now_ms,
                    status="stale",
                    agent=agent,
                    decision_id=decision.decision_id,
                    observation=observation,
                    decision=decision,
                    attempt=attempt,
                    validation_errors=(rejection,),
                    generation_fallback=generated.fallback,
                    raw_responses=generated.raw_responses,
                )
                self._discarded_decisions.add(decision.decision_id)
                return
            self._pending_decisions[agent] = decision
            self._decision_observations[decision.decision_id] = observation
            if generated.fallback:
                self._fallback_decisions.add(decision.decision_id)
            self.event_writer.emit(
                DecisionEvent,
                self.now_ms,
                status="selected",
                agent=agent,
                decision_id=decision.decision_id,
                observation=observation,
                decision=decision,
                attempt=attempt,
                validation_errors=validation_errors,
                generation_fallback=generated.fallback,
                raw_responses=generated.raw_responses,
                request_started_ms=(
                    self._agents[agent].generation.decision_started_ns(
                        decision.decision_id
                    )
                    - self.event_writer.started_ns
                )
                / 1_000_000,
            )
            await self.commit_selected_action(agent, decision)
            if decision.action_type in _SPOKEN_ACTIONS:
                self._agents[agent].start_speech(generated)
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
                self.now_ms,
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
                self.now_ms,
                utterance_id=output.utterance_id,
                listener=output.listener,
                text=output.text,
                is_final=output.is_final,
                is_stable=output.is_stable,
                revision_id=output.revision_id,
                sentence_texts=output.sentence_texts,
                sentence_index=output.sentence_index,
                has_next_sentence=output.has_next_sentence,
            )
            self._latest_asr_event[output.utterance_id] = event
            if output.is_final:
                await self.commit_asr_final(output)
            else:
                await self._handle_asr_partial(output)
            return
        raise TypeError(f"unsupported Surface5 AgentOutput: {type(output).__name__}")

    def _proposal_rejection(
        self,
        agent: str,
        decision: DuplexActionDecision,
        observation: StreamingObservation,
    ) -> str | None:
        if self.state.stopping or self.state.leave_phase != "active":
            return "episode_stopping"
        latest = self.state.latest_observations[agent]
        if latest.canonical.observation_id != observation.canonical.observation_id:
            return "a newer observation superseded this decision"
        if decision.action_type not in observation.canonical.available_actions:
            return "action was unavailable in the generating observation"
        if decision.action_type == "backchanneling" and (
            observation.peer_utterance_id in self._audio_end_ms
            or observation.peer_utterance_id not in self.state.active_utterances
        ):
            return "peer_audio_already_finished"
        if decision.action_type not in self.floor.available_actions(agent, self.state):
            return "action is no longer available on the floor"
        if decision.action_type in _TARGETED_ACTIONS and not self._target_available(
            agent, decision.target_utterance_id
        ):
            return "target_unavailable_before_synthesis"
        return None

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
            observation = self._decision_observations[decision.decision_id]
            self._pending_decisions.pop(agent, None)
            if observation.source == "asr_partial" or self.state.active_utterances:
                return
            await self._commit_joint(
                actor=agent,
                action=DuplexAction(action_type="none", argument="", to=[]),
                origin="agent",
                trigger_event_id=None,
                utterance_id=None,
                metadata={"decision_id": decision.decision_id},
            )
            if self.session.is_stale(self._committed_actions):
                self.state.end_reason = "stale"
                self.state.leave_phase = "completed"
                await self.stop("stale")
            else:
                await self._prompt_peer(agent, source="commit")
            return
        if decision.action_type == "leave":
            await self._commit_leave(agent, decision)
            return
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
        nonverbal_backchannel = (
            utterance.action_type == "backchanneling" and not received
        )
        if not received and not nonverbal_backchannel:
            raise RuntimeError(
                f"audible utterance has blank ASR final: {update.utterance_id}"
            )
        if nonverbal_backchannel and not any(
            key[0] == update.utterance_id and delivery.delivered_frames > 0
            for key, delivery in self._deliveries.items()
        ):
            raise RuntimeError(
                "Cannot commit a nonverbal backchannel without delivered audio"
            )
        action = DuplexAction(
            action_type=utterance.action_type,
            argument=received,
            to=utterance.decision.to,
            target_utterance_id=(
                utterance.decision.target_utterance_id
                if utterance.action_type in {"correction", "interruption"}
                else None
            ),
        )
        trigger = self._latest_asr_event.get(update.utterance_id)
        metadata: dict[str, object] = {
            "decision_id": utterance.decision.decision_id,
            "hidden_said_id": utterance.hidden_said_id,
        }
        if nonverbal_backchannel:
            metadata["observed_nonverbal_backchannel"] = True
        await self._commit_joint(
            actor=utterance.speaker,
            action=action,
            origin="agent",
            trigger_event_id=trigger.event_id if trigger is not None else None,
            utterance_id=update.utterance_id,
            metadata=metadata,
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
        if utterance.action_type in _AUXILIARY_ACTIONS:
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
        self._pending_decisions.pop(utterance.speaker, None)
        self._hidden_by_decision.pop(utterance.decision.decision_id, None)

    async def tick_audio(self) -> None:
        frames = self.audio_router.pop_tick()
        for source, frame in tuple(frames.items()):
            if frame is None:
                continue
            utterance = self._utterances[frame.utterance_id]
            observation = self._decision_observations[utterance.decision.decision_id]
            if (
                utterance.action_type == "backchanneling"
                and not utterance.started
                and observation.peer_utterance_id in self._audio_end_ms
            ):
                await self._drop_late_backchannel(frame.utterance_id)
                frames[source] = None
        if not any(frames.values()):
            return
        tick_start = self.now_ms
        delivered = {}
        receive_error = None
        try:
            for source, frame in frames.items():
                if frame is None:
                    continue
                utterance = self._utterances[frame.utterance_id]
                # This only buffers PCM and starts background decoders; it never awaits inference.
                await self._agents[utterance.listener].receive_audio(frame)
                delivered[source] = frame
        except BaseException as error:
            # Preserve only acknowledged channels if a receiver fails mid-tick.
            receive_error = error
        if delivered:
            capture = self.stereo_writer.write_tick(
                delivered, timestamp_ms=tick_start
            )
        for source, frame in delivered.items():
            utterance = self._utterances[frame.utterance_id]
            delivery = self._deliveries[(frame.utterance_id, frame.chunk_index)]
            if delivery.start_ms is None:
                delivery.start_ms = tick_start
            delivery.delivered_frames += 1
            delivery.pcm.extend(frame.pcm_s16le)
            delivery.frame_spans.append(
                {
                    "start_ms": tick_start,
                    "start_sample": capture.start_sample,
                    "clock_anchor_ms": capture.clock_anchor_ms,
                    "clock_anchor_sample": capture.clock_anchor_sample,
                    "underrun_ms": capture.underrun_ms,
                    "samples": len(frame.pcm_s16le) // 2,
                    "pcm_sha256": hashlib.sha256(frame.pcm_s16le).hexdigest(),
                }
            )
            if not utterance.started:
                utterance.started = True
                self.event_writer.emit(
                    SpeechLifecycleEvent,
                    self.now_ms,
                    phase="started",
                    utterance_id=frame.utterance_id,
                    speaker=source,
                    action_type=utterance.action_type,
                    target_utterance_id=utterance.decision.target_utterance_id,
                )
                self._record_latency(frame.utterance_id, tick_start)
            self._agents[source].note_audio_played(frame.utterance_id)
        # Observe each channel's actual completion, including a short last frame.
        lengths = sorted({len(frame.pcm_s16le) for frame in delivered.values()})
        for length in lengths:
            duration_ms = length / 2 * 1000 / self.sample_rate_hz
            # Pace against the stream's absolute sample cursor. Relative sleeps
            # accumulate callback overhead and turn each packet into a dropout.
            deadline_ms = capture.start_sample * 1000 / self.sample_rate_hz + duration_ms
            await asyncio.sleep(max(0, (deadline_ms - self.now_ms) / 1000))
            frame_end = self.now_ms
            for source, frame in delivered.items():
                if len(frame.pcm_s16le) != length:
                    continue
                delivery = self._deliveries[(frame.utterance_id, frame.chunk_index)]
                delivery.end_ms = frame_end
                delivery.frame_spans[-1]["end_ms"] = frame_end
                if frame.is_chunk_end:
                    self._emit_open_delivery_prefixes(
                        frame.utterance_id, chunk_index=frame.chunk_index
                    )
                    listener = self._utterances[frame.utterance_id].listener
                    self._agents[listener].finish_received_sentence(frame)
                if not frame.is_utterance_end:
                    continue
                utterance = self._utterances[frame.utterance_id]
                self.event_writer.emit(
                    SpeechLifecycleEvent,
                    self.now_ms,
                    phase="finished",
                    utterance_id=frame.utterance_id,
                    speaker=source,
                    action_type=utterance.action_type,
                    target_utterance_id=utterance.decision.target_utterance_id,
                )
                self._audio_finished_utterances.add(frame.utterance_id)
                self._audio_end_ms[frame.utterance_id] = frame_end
                self._agents[utterance.listener].finish_received_audio(
                    frame.utterance_id
                )
        if receive_error is not None:
            raise receive_error
        await self._apply_pending_cancellations()

    def _record_latency(self, utterance_id: str, first_audio_ms: float) -> None:
        utterance = self._utterances[utterance_id]
        observation = self._decision_observations[utterance.decision.decision_id]
        peer_id = observation.peer_utterance_id
        if not peer_id:
            return
        if utterance.action_type == "backchanneling":
            kind = "backchannel"
            origin = (
                self._agents[utterance.speaker].generation.decision_started_ns(
                    utterance.decision.decision_id
                )
                - self.event_writer.started_ns
            ) / 1_000_000
        elif utterance.action_type == "speak" and observation.source == "asr_final":
            kind = "normal_response"
            origin = self._audio_end_ms.get(peer_id)
        else:
            return
        if origin is None or origin > first_audio_ms:
            raise ValueError("Missing or invalid measured latency origin")
        latency = first_audio_ms - origin
        self.latencies[kind].append(latency)
        self.event_writer.emit(
            ResponseLatencyEvent,
            self.now_ms,
            kind=kind,
            speaker=utterance.speaker,
            utterance_id=utterance_id,
            decision_id=utterance.decision.decision_id,
            observation_id=observation.canonical.observation_id,
            peer_utterance_id=peer_id,
            origin_ms=origin,
            first_audio_ms=first_audio_ms,
            latency_ms=latency,
        )

    async def _drop_late_backchannel(self, utterance_id: str) -> None:
        utterance = self._utterances[utterance_id]
        self.audio_router.cancel(utterance_id)
        await self._agents[utterance.speaker].cancel_speech(
            "peer_audio_already_finished"
        )
        self._discarded_utterances.add(utterance_id)
        self._discarded_decisions.add(utterance.decision.decision_id)
        self._emit_floor(
            self.floor.release(self.state, utterance_id, "late_backchannel_dropped")
        )
        self.event_writer.emit(
            SpeechLifecycleEvent,
            self.now_ms,
            phase="cancelled",
            utterance_id=utterance_id,
            speaker=utterance.speaker,
            action_type=utterance.action_type,
            reason="peer_audio_already_finished",
        )
        self._agents[utterance.speaker].finish_speech(utterance_id)
        self._clear_utterance_state(utterance_id, utterance)
        await self._resume_deferred_observation(utterance.speaker)

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
            self.now_ms,
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
                and self.now_ms < self.state.next_floor_available_ms
            ):
                await self._advance_floor_gap()
            listener = self._peer(agent)
            action_type = decision.action_type
            utterance = _UtteranceRuntime(
                speaker=agent,
                listener=listener,
                decision=decision,
                action_type=action_type,
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
            self.now_ms,
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
            self.now_ms,
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
            self.now_ms,
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
            self.now_ms,
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
                self.now_ms,
                phase="cancelled",
                utterance_id=utterance_id,
                speaker=utterance.speaker,
                action_type=utterance.action_type,
                target_utterance_id=utterance.decision.target_utterance_id,
                reason=reason,
            )
            self._cancelled_utterances.add(utterance_id)
            self._discarded_utterances.add(utterance_id)
            self._audio_end_ms[utterance_id] = max(
                delivery.end_ms
                for key, delivery in self._deliveries.items()
                if key[0] == utterance_id and delivery.end_ms is not None
            )
            self._agents[utterance.listener].finish_received_audio(
                utterance_id, cancelled=True
            )

    def _emit_open_delivery_prefixes(
        self, utterance_id: str, chunk_index: int | None = None
    ) -> None:
        for key, delivery in self._deliveries.items():
            if (
                key[0] != utterance_id
                or (chunk_index is not None and key[1] != chunk_index)
                or key in self._closed_deliveries
                or delivery.delivered_frames == 0
                or delivery.start_ms is None
                or delivery.end_ms is None
            ):
                continue
            self.event_writer.emit(
                AudioDeliveryEvent,
                self.now_ms,
                utterance_id=utterance_id,
                source_agent=self._utterances[utterance_id].speaker,
                chunk_index=key[1],
                delivered_frames=delivery.delivered_frames,
                planned_frames=delivery.planned_frames,
                start_ms=delivery.start_ms,
                end_ms=delivery.end_ms,
                delivered_pcm_sha256=hashlib.sha256(delivery.pcm).hexdigest(),
                frame_spans=tuple(delivery.frame_spans),
            )
            self._closed_deliveries.add(key)

    async def _handle_asr_partial(self, update: ASRUpdate) -> None:
        if (
            not update.is_stable
            or update.utterance_id not in self._utterances
            or update.utterance_id in self._audio_end_ms
        ):
            return
        utterance = self._utterances[update.utterance_id]
        if utterance.action_type == "backchanneling":
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
        if self.now_ms - last_dispatch < _PARTIAL_DECISION_INTERVAL_MS:
            return
        if (
            update.listener in self.state.active_utterances.values()
            or self._agents[update.listener].has_pending_generation
        ):
            return

        await self._discard_unstarted_superseded_speech(update.listener)

        available = self.floor.available_actions(update.listener, self.state)
        available = [
            action
            for action in available
            if action in {"none", "backchanneling", "correction", "interruption"}
        ]
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
        self._last_partial_dispatch_ms[update.listener] = self.now_ms
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
                self.now_ms,
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
        remaining = self.state.next_floor_available_ms - self.now_ms
        if remaining > 0:
            await asyncio.sleep(remaining / 1000)

    async def _commit_leave(
        self,
        agent: str,
        decision: DuplexActionDecision,
    ) -> None:
        if self.state.leave_phase != "active":
            raise RuntimeError("Cannot leave an already completed episode")
        self._audio_stopping = True
        await self.stop("agent_left")
        await self._commit_joint(
            actor=agent,
            action=DuplexAction(action_type="leave", argument="", to=[]),
            origin="agent",
            trigger_event_id=None,
            utterance_id=None,
            metadata={
                "decision_id": decision.decision_id,
                "termination_policy": self.config.termination_policy,
            },
        )
        self.state.left_agents.add(agent)
        self._agents[agent].state.left = True
        self._pending_decisions.pop(agent, None)
        self.state.deferred_observations.clear()
        self.state.leave_phase = "completed"
        self.state.end_reason = "agent_left"

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
        metadata = {
            **metadata,
            "actor": actor,
            "generation_fallback": metadata.get("decision_id")
            in self._fallback_decisions,
        }
        commit = SemanticCommit(
            actions=actions,
            expected_turn_number=self._snapshot.turn_number,
            timestamp_ms=self.now_ms,
            origin=origin,
            utterance_ids={actor: utterance_id, peer: None},
            metadata=metadata,
        )
        snapshot = await self.session.commit(commit)
        self._snapshot = snapshot
        self._committed_actions.append((actor, action))
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
            self.now_ms,
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
        for agent in self.agents:
            agent.record_observation(snapshot.observations[agent.agent_name])

        if (
            self.state.budget_turns >= self.config.max_turns
            and action.action_type != "leave"
        ):
            raise TurnLimitReached

    def _record_stopped_audio(self, reason: str) -> None:
        for utterance_id, utterance in self._utterances.items():
            self.audio_router.cancel(utterance_id)
            self._emit_open_delivery_prefixes(utterance_id)
            if (
                utterance_id not in self._audio_finished_utterances
                and utterance_id not in self._cancelled_utterances
            ):
                self.event_writer.emit(
                    SpeechLifecycleEvent,
                    self.now_ms,
                    phase="cancelled",
                    utterance_id=utterance_id,
                    speaker=utterance.speaker,
                    action_type=utterance.action_type,
                    target_utterance_id=utterance.decision.target_utterance_id,
                    reason=reason,
                )
                self._cancelled_utterances.add(utterance_id)

    async def _finish_delivered_audio(self, reason: str) -> None:
        """Preserve received audio and ASR without committing another action."""
        await self.stop(reason)
        self._record_stopped_audio(reason)
        for utterance_id, utterance in tuple(self._utterances.items()):
            previous = self._latest_asr_event.get(utterance_id)
            if previous is None or not previous.is_final:
                final = await self._agents[utterance.listener].finalize_received_audio(
                    utterance_id
                )
                if final is not None:
                    self.event_writer.emit(
                        ASRUpdateEvent,
                        self.now_ms,
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
        if self.state.stopping or listener in self.state.left_agents:
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
            now_ms=self.now_ms,
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
        source: Literal["reset", "asr_partial", "asr_final", "commit"],
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
            waiters = set(self._output_tasks.values())
            if self._audio_task is not None:
                waiters.add(self._audio_task)
            done, _pending = await asyncio.wait(
                waiters,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self._audio_task in done:
                self._audio_task.result()
                raise RuntimeError("Live audio loop ended unexpectedly")
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
                self._ready_outputs.append((name, task.result()))

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
            self.now_ms,
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
