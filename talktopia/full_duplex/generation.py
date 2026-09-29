"""Strict two-stage LLM generation for Surface5 duplex agents."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sotopia.generation_utils import PydanticOutputParser, StrOutputParser, agenerate

from .actions import (
    DuplexActionDecision,
    DuplexActionType,
    HiddenSaid,
    SpeechChunk,
    StreamingObservation,
    _validate_generated_text,
)
from .config import SIMULATION_PROMPT_VERSION


GenerationStage = Literal[
    "decision",
    "non_audio_argument",
    "hidden_said",
]

# Keep the requested length independent from the validation safety margin.
PROMPT_MAX_WORDS = 40

_BACKCHANNELS = (
    "yeah",
    "yep",
    "yes",
    "right",
    "ok",
    "got it",
    "I see",
    "exactly",
    "sure",
    "makes sense",
)
_SENTENCE_BOUNDARY = re.compile(
    r"(?P<terminal>[.!?。！？]+)(?P<closers>[\"”’')\]]*)(?P<space>\s+)"
)
_NON_TERMINAL_ABBREVIATIONS = frozenset(
    {
        "dr.",
        "e.g.",
        "gen.",
        "gov.",
        "i.e.",
        "jr.",
        "lt.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "rep.",
        "sen.",
        "sr.",
        "st.",
        "vs.",
    }
)
_INITIALISM = re.compile(r"(?:[A-Za-z]\.){1,5}$")

_PROMPT_PATH = Path(__file__).with_name("prompts") / f"{SIMULATION_PROMPT_VERSION}.txt"
_parts = re.split(r"(?m)^\[([A-Z_]+)\]\s*$", _PROMPT_PATH.read_text(encoding="utf-8"))
_PROMPTS = {name: text.strip() for name, text in zip(_parts[1::2], _parts[2::2])}
_COMMON_PROMPT = _PROMPTS["COMMON"]
_DECISION_PROMPT = _PROMPTS["DECISION"]
_NON_AUDIO_ARGUMENT_PROMPT = _PROMPTS["NON_AUDIO_ARGUMENT"]
_HIDDEN_SAID_PROMPT = _PROMPTS["HIDDEN_SAID"]


@dataclass(frozen=True, slots=True)
class AgentSessionContext:
    episode_id: str
    agent_name: str
    peer_name: str
    scenario: str
    self_background: str
    private_goal: str

    def __post_init__(self) -> None:
        for field_name in (
            "episode_id",
            "agent_name",
            "peer_name",
            "scenario",
            "self_background",
            "private_goal",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be blank")
        if self.agent_name == self.peer_name:
            raise ValueError("agent_name and peer_name must be distinct")


class GenerationFailure(RuntimeError):
    """All bounded attempts for one generation stage failed validation."""

    def __init__(
        self,
        *,
        stage: GenerationStage,
        attempts: int,
        validation_errors: tuple[str, ...],
        last_output: str | None = None,
    ) -> None:
        self.stage = stage
        self.attempts = attempts
        self.validation_errors = validation_errors
        self.last_output = last_output
        detail = validation_errors[-1] if validation_errors else "unknown error"
        super().__init__(
            f"Surface5 {stage} generation failed after {attempts} attempts: {detail}"
        )


class _ActionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_type: DuplexActionType


class _GeneratedText(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def accept_ollama_json_string(cls, value: object) -> object:
        # Ollama may collapse a one-property JSON schema to its string value.
        # This is transport normalization only; HiddenSaid applies the public
        # 50-word and controller-metadata contract immediately afterwards.
        if isinstance(value, str):
            return {"text": value}
        return value


class _SchemaOnlyOutputParser(PydanticOutputParser[BaseModel]):
    """Request a JSON schema while leaving validation to Surface5's retry loop."""

    def parse(  # type: ignore[override]
        self,
        result: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        del context
        return result


def _parse_generated_text(raw_text: str) -> _GeneratedText:
    candidate = _normalize_json_payload(raw_text)
    return _GeneratedText.model_validate(json.loads(candidate))


def _parse_non_audio_argument(raw_text: str) -> _GeneratedText:
    candidate = _normalize_json_payload(raw_text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        if candidate.startswith(("{", "[", '"')):
            raise
        payload = candidate
    return _GeneratedText.model_validate(payload)


def _parse_action_choice(raw_text: str) -> _ActionChoice:
    """Project an untrusted model object onto the controller-owned decision."""

    candidate = _normalize_json_payload(raw_text)
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise TypeError("decision response must be a JSON object")
    projected = (
        {"action_type": payload["action_type"]} if "action_type" in payload else {}
    )
    return _ActionChoice.model_validate(projected)


def _normalize_json_payload(raw_text: str) -> str:
    candidate = _strip_leading_thinking(raw_text)
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        if first_newline == -1:
            raise ValueError("fenced JSON response has no content line")
        # Models sometimes omit the closing fence. JSON parsing below still
        # rejects truncated objects, strings, and extra trailing content.
        candidate = candidate[first_newline + 1 :].strip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
    if not candidate:
        raise ValueError("JSON response has no content")
    return candidate


def _strip_leading_thinking(raw_text: str) -> str:
    """Apply Poketopia's bounded DeepSeek transport normalization locally."""

    candidate = raw_text.strip()
    if not candidate.casefold().startswith("<think>"):
        return candidate
    match = re.match(r"(?is)^<think>.*?</think>\s*(.*)$", candidate)
    if match is None:
        raise ValueError("incomplete leading <think> block")
    normalized = match.group(1).strip()
    if not normalized:
        raise ValueError("leading <think> block has no response content")
    return normalized


class DuplexGenerationEngine:
    """Generate a strict action decision and, separately, its audible text."""

    def __init__(self, model_name: str, *, max_attempts: int = 2) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.model_name = model_name
        self.max_attempts = max_attempts
        self._decision_sequence = 0
        self._hidden_sequence = 0
        self._decision_audits: dict[str, tuple[int, tuple[str, ...]]] = {}
        self._hidden_attempts: dict[str, int] = {}
        self._non_audio_attempts: dict[str, int] = {}
        self._decision_starts_ns: dict[str, int] = {}

    def decision_started_ns(self, decision_id: str) -> int:
        return self._decision_starts_ns[decision_id]

    async def decide_action(
        self,
        session: AgentSessionContext,
        observation: StreamingObservation,
        recent_history: str,
    ) -> DuplexActionDecision:
        errors: list[str] = []
        last_output: str | None = None
        schema_parser: PydanticOutputParser[_ActionChoice] = PydanticOutputParser(
            pydantic_object=_ActionChoice
        )
        transport_parser = _SchemaOnlyOutputParser(pydantic_object=_ActionChoice)
        selected: _ActionChoice | None = None
        selected_attempt = 0
        request_started_ns = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                if request_started_ns is None:
                    request_started_ns = time.monotonic_ns()
                generated = await agenerate(
                    model_name=self.model_name,
                    template=_DECISION_PROMPT,
                    input_values={
                        "common_prompt": self._common_prompt(
                            session, observation, recent_history
                        ),
                        "observation": self._decision_observation_json(observation),
                        "available_actions": json.dumps(
                            observation.canonical.available_actions
                        ),
                        "validation_feedback": json.dumps(errors[-1:]),
                        "format_instructions": self._decision_schema_json(
                            observation,
                            schema_parser,
                        ),
                    },
                    output_parser=transport_parser,
                    temperature=0.0,
                    structured_output=True,
                )
                if not isinstance(generated, str):
                    raise TypeError(
                        "decision generation returned "
                        f"{type(generated).__name__}, expected str"
                    )
                last_output = generated
                choice = _parse_action_choice(generated)
                if choice.action_type not in observation.canonical.available_actions:
                    raise ValueError(f"unavailable action: {choice.action_type}")
                self._derived_target(observation, choice.action_type)
                selected = choice
                selected_attempt = attempt
                break
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        if selected is None:
            raise GenerationFailure(
                stage="decision",
                attempts=self.max_attempts,
                validation_errors=tuple(errors),
                last_output=last_output,
            )

        self._decision_sequence += 1
        decision_id = (
            f"{session.episode_id}-{self._slug(session.agent_name)}-"
            f"decision-{self._decision_sequence:04d}"
        )
        self._decision_starts_ns[decision_id] = request_started_ns
        non_audio_argument = ""
        if selected.action_type in {"action", "non-verbal communication"}:
            non_audio_argument = await self._generate_non_audio_argument(
                session=session,
                observation=observation,
                recent_history=recent_history,
                decision_id=decision_id,
                action_type=selected.action_type,
            )
        recipients = (
            [] if selected.action_type in {"none", "leave"} else [session.peer_name]
        )
        parser_context = {
            "available_actions": observation.canonical.available_actions,
            "agent_names": (session.agent_name, session.peer_name),
            "sender": session.agent_name,
            "require_decision_id": True,
        }
        decision = DuplexActionDecision.model_validate(
            {
                "decision_id": decision_id,
                "action_type": selected.action_type,
                "non_audio_argument": non_audio_argument,
                "to": recipients,
                "target_utterance_id": self._derived_target(
                    observation,
                    selected.action_type,
                ),
            },
            context=parser_context,
        )
        self._decision_audits[decision_id] = (selected_attempt, tuple(errors))
        return decision

    async def generate_hidden_said(
        self,
        session: AgentSessionContext,
        observation: StreamingObservation,
        decision: DuplexActionDecision,
        recent_history: str,
    ) -> HiddenSaid:
        if decision.action_type not in {
            "speak",
            "hesitation",
            "correction",
            "interruption",
        }:
            raise ValueError(
                f"hidden said is not valid for action {decision.action_type!r}"
            )
        return await self._generate_text(
            stage="hidden_said",
            session=session,
            decision_id=decision.decision_id,
            template=_HIDDEN_SAID_PROMPT,
            input_values={
                "common_prompt": self._common_prompt(
                    session, observation, recent_history
                ),
                "observation": observation.model_dump_json(),
                "action_type": decision.action_type,
                "max_words": str(PROMPT_MAX_WORDS),
                "format_instructions": PydanticOutputParser(
                    pydantic_object=_GeneratedText
                ).get_format_instructions(),
            },
        )

    def make_backchannel(
        self,
        session: AgentSessionContext,
        success_index: int,
        decision: DuplexActionDecision,
    ) -> HiddenSaid:
        if success_index < 0:
            raise ValueError("success_index must be non-negative")
        if decision.action_type != "backchanneling":
            raise ValueError("backchannel text requires a backchanneling decision")
        identity = f"{session.episode_id}\0{session.agent_name}".encode("utf-8")
        start = hashlib.sha256(identity).digest()[0] % len(_BACKCHANNELS)
        text = _BACKCHANNELS[(start + success_index) % len(_BACKCHANNELS)]
        hidden = self._hidden_said(
            session=session,
            decision_id=decision.decision_id,
            text=text,
        )
        self._hidden_attempts[hidden.hidden_said_id] = 1
        return hidden

    def split_into_sentence_chunks(self, hidden_said: HiddenSaid) -> list[SpeechChunk]:
        sentences = self._spoken_sentences(hidden_said.text)
        if not sentences:
            raise ValueError("hidden said contains no sentence")
        utterance_id = hidden_said.hidden_said_id.replace("hidden-said", "utterance")
        chunks = [
            SpeechChunk(
                utterance_id=utterance_id,
                hidden_said_id=hidden_said.hidden_said_id,
                chunk_index=index,
                total_chunks=len(sentences),
                text=sentence,
                is_final=index == len(sentences) - 1,
            )
            for index, sentence in enumerate(sentences)
        ]
        if " ".join(chunk.text for chunk in chunks) != hidden_said.text:
            raise AssertionError("speech chunks must reconstruct hidden said text")
        return chunks

    def decision_audit(self, decision_id: str) -> tuple[int, tuple[str, ...]]:
        """Return the successful attempt and validation failures that preceded it."""

        try:
            return self._decision_audits[decision_id]
        except KeyError as error:
            raise ValueError(f"unknown decision audit: {decision_id}") from error

    def hidden_said_attempt(self, hidden_said_id: str) -> int:
        """Return the successful text-generation attempt for one private utterance."""

        try:
            return self._hidden_attempts[hidden_said_id]
        except KeyError as error:
            raise ValueError(f"unknown hidden-said audit: {hidden_said_id}") from error

    def non_audio_argument_attempt(self, decision_id: str) -> int:
        """Return the successful non-audio text attempt for one decision."""

        try:
            return self._non_audio_attempts[decision_id]
        except KeyError as error:
            raise ValueError(f"unknown non-audio audit: {decision_id}") from error

    async def _generate_non_audio_argument(
        self,
        *,
        session: AgentSessionContext,
        observation: StreamingObservation,
        recent_history: str,
        decision_id: str,
        action_type: Literal["action", "non-verbal communication"],
    ) -> str:
        errors: list[str] = []
        last_output: str | None = None
        parser: PydanticOutputParser[_GeneratedText] = PydanticOutputParser(
            pydantic_object=_GeneratedText
        )
        transport_parser = _SchemaOnlyOutputParser(pydantic_object=_GeneratedText)
        for attempt in range(1, self.max_attempts + 1):
            try:
                generated = await agenerate(
                    model_name=self.model_name,
                    template=_NON_AUDIO_ARGUMENT_PROMPT,
                    input_values={
                        "common_prompt": self._common_prompt(
                            session, observation, recent_history
                        ),
                        "observation": observation.model_dump_json(),
                        "action_type": action_type,
                        "max_words": str(PROMPT_MAX_WORDS),
                        "validation_feedback": json.dumps(errors[-1:]),
                        "format_instructions": parser.get_format_instructions(),
                    },
                    output_parser=transport_parser,
                    temperature=0.4,
                    structured_output=True,
                )
                if not isinstance(generated, str):
                    raise TypeError(
                        "non-audio generation returned "
                        f"{type(generated).__name__}, expected str"
                    )
                last_output = generated
                raw = _parse_non_audio_argument(generated)
                normalized = " ".join(raw.text.split())
                _validate_generated_text(
                    normalized,
                    field_name="non_audio_argument",
                )
                self._non_audio_attempts[decision_id] = attempt
                return normalized
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        raise GenerationFailure(
            stage="non_audio_argument",
            attempts=self.max_attempts,
            validation_errors=tuple(errors),
            last_output=last_output,
        )

    async def _generate_text(
        self,
        *,
        stage: Literal["hidden_said"],
        session: AgentSessionContext,
        decision_id: str,
        template: str,
        input_values: dict[str, str],
    ) -> HiddenSaid:
        errors: list[str] = []
        last_output: str | None = None
        parser = StrOutputParser()
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw_text = await agenerate(
                    model_name=self.model_name,
                    template=template,
                    input_values={
                        **input_values,
                        "validation_feedback": json.dumps(errors[-1:]),
                    },
                    output_parser=parser,
                    temperature=0.4,
                    structured_output=False,
                )
                last_output = raw_text
                raw = _parse_generated_text(raw_text)
                hidden = self._hidden_said(
                    session=session,
                    decision_id=decision_id,
                    text=" ".join(raw.text.split()),
                )
                self._hidden_attempts[hidden.hidden_said_id] = attempt
                return hidden
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        raise GenerationFailure(
            stage=stage,
            attempts=self.max_attempts,
            validation_errors=tuple(errors),
            last_output=last_output,
        )

    def _hidden_said(
        self,
        *,
        session: AgentSessionContext,
        decision_id: str,
        text: str,
    ) -> HiddenSaid:
        self._hidden_sequence += 1
        return HiddenSaid(
            hidden_said_id=(
                f"{session.episode_id}-{self._slug(session.agent_name)}-"
                f"hidden-said-{self._hidden_sequence:04d}"
            ),
            decision_id=decision_id,
            speaker=session.agent_name,
            text=text,
        )

    @staticmethod
    def _derived_target(
        observation: StreamingObservation,
        action_type: DuplexActionType,
    ) -> str | None:
        if action_type in {"correction", "interruption"}:
            expected = observation.target_utterance_id or observation.peer_utterance_id
            if not expected:
                raise ValueError(f"{action_type} requires a current peer utterance")
            return expected
        return None

    @staticmethod
    def _decision_observation_json(observation: StreamingObservation) -> str:
        payload = observation.model_dump(mode="json")
        if not any(
            action in {"correction", "interruption"}
            for action in observation.canonical.available_actions
        ):
            payload.pop("peer_utterance_id", None)
            payload.pop("target_utterance_id", None)
        return json.dumps(payload)

    @staticmethod
    def _decision_schema_json(
        observation: StreamingObservation,
        parser: PydanticOutputParser[_ActionChoice],
    ) -> str:
        schema = parser.pydantic_object.model_json_schema()
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("decision schema has no properties")
        action_type = properties.get("action_type")
        if not isinstance(action_type, dict):
            raise ValueError("decision schema has no action_type property")
        action_type["enum"] = list(observation.canonical.available_actions)
        return json.dumps(schema)

    @staticmethod
    def _common_prompt(
        session: AgentSessionContext,
        observation: StreamingObservation,
        recent_history: str,
    ) -> str:
        history = (
            "Here is the context of the interaction:\n"
            f"Participants: {session.agent_name}; {session.peer_name}\n"
            f"Situation: {session.scenario}\n"
            f"Your background: {session.self_background}\n"
            f"Your social goal: {session.private_goal}\n\n"
            "Conversation so far:\n" + (recent_history or "(none yet)")
        )
        return _COMMON_PROMPT.format(
            agent=session.agent_name,
            history=history,
            turn_number=observation.canonical.turn_number,
            action_list=json.dumps(observation.canonical.available_actions),
        ).strip()

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
        return slug or "agent"

    @classmethod
    def _spoken_sentences(cls, text: str) -> list[str]:
        normalized = " ".join(text.split())
        sentences: list[str] = []
        start = 0
        for match in _SENTENCE_BOUNDARY.finditer(normalized):
            end = match.end("closers")
            candidate = normalized[start:end].strip()
            remainder = normalized[match.end() :]
            if cls._non_terminal_boundary(
                candidate,
                terminal=match.group("terminal"),
                closers=match.group("closers"),
                remainder=remainder,
            ):
                continue
            sentences.append(candidate)
            start = match.end()
        tail = normalized[start:].strip()
        if tail:
            sentences.append(tail)
        return sentences

    @staticmethod
    def _non_terminal_boundary(
        candidate: str,
        *,
        terminal: str,
        closers: str,
        remainder: str,
    ) -> bool:
        if terminal == ".":
            without_closers = (
                candidate[: len(candidate) - len(closers)] if closers else candidate
            )
            token = without_closers.rsplit(maxsplit=1)[-1].casefold()
            if token in _NON_TERMINAL_ABBREVIATIONS or _INITIALISM.fullmatch(token):
                return True
        return bool(closers and remainder[:1].islower())


__all__ = [
    "AgentSessionContext",
    "DuplexGenerationEngine",
    "GenerationFailure",
]
