"""Cascaded duplex agent that assembles generation, TTS, and online ASR."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from sotopia.agents.base_agent import BaseAgent
from sotopia.database import AgentProfile

from .actions import (
    DuplexAction,
    DuplexActionDecision,
    DuplexActionType,
    DuplexObservation,
    HiddenSaid,
    StreamingObservation,
)
from .audio import AudioChunk, AudioFrame
from .generation import AgentSessionContext, DuplexGenerationEngine
from .speech_backends import (
    ASRUpdate,
    IncrementalTTS,
    OnlineASR,
    SpeechSynthesisFailure,
)


@dataclass(slots=True)
class ActiveUtterance:
    utterance_id: str
    decision_id: str
    action_type: DuplexActionType
    hidden_said: HiddenSaid
    synthesized_chunk_indexes: set[int] = field(default_factory=set)
    first_audio_played: bool = False
    played_frames: int = 0
    cancelled: bool = False


@dataclass(slots=True)
class AgentState:
    session_active: bool = False
    left: bool = False
    latest_observation: StreamingObservation | None = None
    active_utterance: ActiveUtterance | None = None
    observation_revision: int = 0
    last_llm_latency_ms: int | None = None
    work_stage: Literal["idle", "decision", "hidden_said", "closing", "tts"] = "idle"


AgentOutput = (
    DuplexActionDecision | HiddenSaid | AudioChunk | SpeechSynthesisFailure | ASRUpdate
)


class CascadedDuplexAgent(BaseAgent[DuplexObservation, DuplexAction]):
    """One participant's private cascaded model path, without global control."""

    def __init__(
        self,
        *,
        profile: AgentProfile,
        generation: DuplexGenerationEngine,
        asr: OnlineASR,
        tts: IncrementalTTS,
        voice_reference: Path,
        history_entries: int,
    ) -> None:
        super().__init__(agent_profile=profile)
        self.model_name = generation.model_name
        self.generation = generation
        self.asr = asr
        self.tts = tts
        self.voice_reference = voice_reference.expanduser().resolve()
        if not self.voice_reference.is_file():
            raise FileNotFoundError(self.voice_reference)
        if history_entries < 1:
            raise ValueError("history_entries must be positive")
        self.history_entries = history_entries
        self.state = AgentState()
        self._context: AgentSessionContext | None = None
        self._outputs: asyncio.Queue[AgentOutput] = asyncio.Queue()
        self._decision_task: asyncio.Task[None] | None = None
        self._asr_relay_task: asyncio.Task[None] | None = None
        self._background_errors: deque[tuple[str, BaseException]] = deque()
        self._background_changed = asyncio.Event()
        self._incoming_utterances: set[str] = set()
        self._recent_turns: deque[str] = deque(maxlen=history_entries)
        self._decision_observations: dict[str, StreamingObservation] = {}
        self._backchannel_index = 0
        self._received_finals: dict[str, ASRUpdate] = {}

    def act(self, obs: DuplexObservation) -> DuplexAction:
        del obs
        raise RuntimeError("turn-based act() is not supported by Surface5")

    async def aact(self, obs: DuplexObservation) -> DuplexAction:
        del obs
        raise RuntimeError("turn-based aact() is not supported by Surface5")

    async def start_session(self, context: AgentSessionContext) -> None:
        if self.state.session_active:
            raise RuntimeError("agent session is already active")
        if context.agent_name != self.agent_name:
            raise ValueError("session context is bound to a different agent")
        self._context = context
        self.goal = context.private_goal
        self.state = AgentState(session_active=True)
        self._outputs = asyncio.Queue()
        self._background_errors.clear()
        self._background_changed = asyncio.Event()
        self._incoming_utterances.clear()
        self._received_finals.clear()
        self._recent_turns.clear()
        self._decision_observations.clear()
        self._backchannel_index = 0
        self._asr_relay_task = self._track_background_task(
            "asr",
            asyncio.create_task(
                self._relay_asr_updates(),
                name=f"surface5-asr-relay-{self.agent_name}",
            ),
        )

    async def submit_observation(self, observation: StreamingObservation) -> None:
        self._require_active()
        if self.state.left:
            return
        self.state.observation_revision += 1
        revision = self.state.observation_revision
        self.state.latest_observation = observation
        if observation.canonical.available_actions == ["none"]:
            return
        if self._decision_task is not None and not self._decision_task.done():
            self._decision_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._decision_task
        self._decision_task = self._track_background_task(
            "decision",
            asyncio.create_task(
                self._respond(observation, revision),
                name=f"surface5-decision-{self.agent_name}-{revision}",
            ),
        )

    async def receive_audio(self, frame: AudioFrame) -> None:
        self._require_active()
        if frame.source_agent == self.agent_name:
            raise ValueError("an agent cannot receive its own audio channel")
        if frame.utterance_id not in self._incoming_utterances:
            await self.asr.start_utterance(frame.utterance_id, self.agent_name)
            self._incoming_utterances.add(frame.utterance_id)
        await self.asr.push_audio(frame)
        if frame.is_utterance_end:
            final = await self.asr.finish_utterance(frame.utterance_id)
            self._incoming_utterances.remove(frame.utterance_id)
            self._received_finals[frame.utterance_id] = final
            await self._outputs.put(final)

    async def cancel_received_audio(self, utterance_id: str) -> ASRUpdate:
        self._require_active()
        if utterance_id not in self._incoming_utterances:
            raise ValueError(f"incoming utterance is not active: {utterance_id}")
        final = await self.asr.cancel_utterance(utterance_id)
        self._incoming_utterances.remove(utterance_id)
        self._received_finals[utterance_id] = final
        return final

    async def finalize_received_audio(self, utterance_id: str) -> ASRUpdate | None:
        if utterance_id in self._received_finals:
            return self._received_finals[utterance_id]
        if utterance_id in self._incoming_utterances:
            return await self.cancel_received_audio(utterance_id)
        return None

    async def next_output(self) -> AgentOutput:
        self._require_active()
        while True:
            self._raise_background_errors()
            try:
                return self._outputs.get_nowait()
            except asyncio.QueueEmpty:
                pass

            self._background_changed.clear()
            self._raise_background_errors()
            try:
                return self._outputs.get_nowait()
            except asyncio.QueueEmpty:
                pass

            get_task = asyncio.create_task(self._outputs.get())
            changed_task = asyncio.create_task(self._background_changed.wait())
            try:
                done, pending = await asyncio.wait(
                    {get_task, changed_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if get_task in done:
                    changed_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await changed_task
                    return get_task.result()
                for task in pending:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
            except asyncio.CancelledError:
                for task in (get_task, changed_task):
                    if task.done():
                        continue
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                raise

    def try_next_output(self) -> AgentOutput | None:
        self._require_active()
        self._raise_background_errors()
        try:
            return self._outputs.get_nowait()
        except asyncio.QueueEmpty:
            return None

    @property
    def has_pending_output(self) -> bool:
        return not self._outputs.empty()

    @property
    def has_pending_generation(self) -> bool:
        return self._decision_task is not None and not self._decision_task.done()

    async def cancel_speech(self, reason: str) -> None:
        self._require_active()
        if not reason.strip():
            raise ValueError("speech cancellation reason must not be blank")
        active = self.state.active_utterance
        if active is not None:
            active.cancelled = True
        if self._decision_task is not None and not self._decision_task.done():
            self._decision_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._decision_task

    def note_audio_played(self, utterance_id: str) -> None:
        active = self.state.active_utterance
        if active is None or active.utterance_id != utterance_id:
            return
        active.first_audio_played = True
        active.played_frames += 1

    def finish_speech(self, utterance_id: str) -> None:
        active = self.state.active_utterance
        if active is not None and active.utterance_id == utterance_id:
            self.state.active_utterance = None

    def decision_observation(self, decision_id: str) -> StreamingObservation:
        """Return the exact immutable observation used for a queued decision."""

        try:
            return self._decision_observations[decision_id]
        except KeyError as error:
            raise ValueError(f"unknown decision observation: {decision_id}") from error

    def record_committed_action(
        self,
        actor: str,
        action: DuplexAction,
        *,
        own_generated_text: str | None = None,
    ) -> None:
        """Append one complete commit using this agent's permitted text view."""

        self._require_active()
        visible = action
        if actor == self.agent_name and own_generated_text is not None:
            visible = action.model_copy(update={"argument": own_generated_text})
        self._recent_turns.append(f"{actor} {visible.to_natural_language()}")

    async def stop_session(self) -> None:
        if not self.state.session_active:
            return
        if self._decision_task is not None and not self._decision_task.done():
            self._decision_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._decision_task
        await self.asr.close()
        if self._asr_relay_task is not None and not self._asr_relay_task.done():
            with suppress(asyncio.CancelledError):
                await self._asr_relay_task
        await self.tts.close()
        self._incoming_utterances.clear()
        self._received_finals.clear()
        self._recent_turns.clear()
        self._decision_observations.clear()
        self._background_errors.clear()
        self._background_changed.set()
        self._context = None
        self.state.session_active = False
        self.state.active_utterance = None
        self.state.work_stage = "idle"

    async def _respond(
        self,
        observation: StreamingObservation,
        revision: int,
    ) -> None:
        assert self._context is not None
        loop = asyncio.get_running_loop()
        started = loop.time()
        recent_history = "\n\n".join(self._recent_turns)
        self.state.work_stage = "decision"
        try:
            decision = await self.generation.decide_action(
                self._context,
                observation,
                recent_history,
            )
            self.state.last_llm_latency_ms = round((loop.time() - started) * 1000)
            if revision != self.state.observation_revision or self.state.left:
                return
            self._decision_observations[decision.decision_id] = observation
            await self._outputs.put(decision)
            if decision.action_type in {
                "none",
                "action",
                "non-verbal communication",
            }:
                return
            if decision.action_type == "leave" and observation.source != "peer_left":
                return

            if decision.action_type == "backchanneling":
                hidden = self.generation.make_backchannel(
                    self._context,
                    self._backchannel_index,
                    decision,
                )
                self._backchannel_index += 1
                action_type: DuplexActionType = "backchanneling"
            elif decision.action_type == "leave":
                self.state.work_stage = "closing"
                hidden = await self.generation.generate_closing(
                    self._context,
                    recent_history,
                    decision.decision_id,
                )
                action_type = "speak"
            else:
                self.state.work_stage = "hidden_said"
                hidden = await self.generation.generate_hidden_said(
                    self._context,
                    observation,
                    decision,
                    recent_history,
                )
                action_type = decision.action_type

            if revision != self.state.observation_revision or self.state.left:
                return
            chunks = self.generation.split_into_sentence_chunks(hidden)
            active = ActiveUtterance(
                utterance_id=chunks[0].utterance_id,
                decision_id=decision.decision_id,
                action_type=action_type,
                hidden_said=hidden,
            )
            self.state.active_utterance = active
            await self._outputs.put(hidden)
            for chunk in chunks:
                if active.cancelled or revision != self.state.observation_revision:
                    return
                self.state.work_stage = "tts"
                audio = await self.tts.synthesize(chunk, self.voice_reference)
                if isinstance(audio, SpeechSynthesisFailure):
                    if (
                        decision.action_type != "backchanneling"
                        or active.first_audio_played
                        or active.synthesized_chunk_indexes
                    ):
                        raise RuntimeError(
                            "TTS synthesis failed outside the recoverable pre-audio "
                            f"backchannel case: {audio.error_detail}"
                        )
                    await self._outputs.put(audio)
                    return
                active.synthesized_chunk_indexes.add(chunk.chunk_index)
                await self._outputs.put(audio)
        except asyncio.CancelledError:
            raise
        finally:
            if asyncio.current_task() is self._decision_task:
                self.state.work_stage = "idle"

    async def _relay_asr_updates(self) -> None:
        async for update in self.asr.updates():
            await self._outputs.put(update)

    def _require_active(self) -> None:
        if not self.state.session_active:
            raise RuntimeError("agent session is not active")

    def _track_background_task(
        self,
        role: str,
        task: asyncio.Task[None],
    ) -> asyncio.Task[None]:
        def record_completion(completed: asyncio.Task[None]) -> None:
            if not completed.cancelled():
                error = completed.exception()
                if error is not None:
                    self._background_errors.append((role, error))
            self._background_changed.set()

        task.add_done_callback(record_completion)
        self._background_changed.set()
        return task

    def _raise_background_errors(self) -> None:
        if not self._background_errors:
            return
        role, error = self._background_errors.popleft()
        if role == "asr":
            raise RuntimeError("agent ASR relay failed") from error
        raise error


__all__ = [
    "ActiveUtterance",
    "AgentOutput",
    "AgentState",
    "CascadedDuplexAgent",
]
