"""SOTOPIA single-call action generation with Surface5 observation controls."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import gin
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    create_model,
    model_validator,
)
from sotopia.generation_utils import PydanticOutputParser, agenerate
from sotopia.generation_utils import generate as sotopia_generation
from sotopia.messages import AgentAction

from talktopia.models.config import CONFIRMATION_TTS_TAG as BACKCHANNEL_TTS_TEXT
from talktopia.speech_agent import prepare_tts_text, resolve_recipient_names

from .actions import (
    DuplexActionDecision,
    DuplexActionType,
    HiddenSaid,
    SpeechChunk,
    StreamingObservation,
    _validate_generated_text,
)
from .config import SIMULATION_PROMPT_VERSION

# Keep the requested length independent from the validation safety margin.
PROMPT_MAX_WORDS = 40

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
_ACTION_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


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


@dataclass(frozen=True, slots=True)
class GeneratedAction:
    """One validated action and its text from the same model response."""

    decision: DuplexActionDecision
    argument: str
    fallback: bool = False


class _JointAction(AgentAction):
    """Keep SOTOPIA's wire format, adding only Surface5 action literals."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action_type: DuplexActionType  # type: ignore[assignment]
    argument: str = Field(
        description=(
            "For speak, hesitation, correction, or interruption: only the words "
            "spoken aloud, without asterisks, stage directions, speaker labels, "
            "or narration. For action or non-verbal communication: describe the "
            "behavior. For none, leave, or backchanneling: an empty string. "
            "Use at most 40 words for a non-empty argument."
        )
    )

    @model_validator(mode="after")
    def validate_surface5_contract(self, info: ValidationInfo) -> _JointAction:
        context = info.context or {}
        available = context.get("available_action_types")
        if available is not None and self.action_type not in available:
            raise ValueError(f"unavailable action: {self.action_type}")
        if self.action_type in {"none", "leave", "backchanneling"}:
            if self.argument.strip():
                raise ValueError(f"{self.action_type} requires an empty argument")
        else:
            _validate_generated_text(self.argument, field_name="argument")
            if self.action_type in {
                "speak",
                "hesitation",
                "correction",
                "interruption",
            }:
                # A starred span could be emphasis or a stage direction. Do not
                # guess and silently remove words from the participant's speech.
                if "*" in self.argument:
                    raise ValueError("speech argument must not contain asterisks")
                if not prepare_tts_text(self.argument):
                    raise ValueError("speech argument must contain audible words")
        if (
            info.context is not None
            and self.action_type in {"correction", "interruption"}
            and not context.get("target_utterance_id")
        ):
            raise ValueError(f"{self.action_type} requires a current peer utterance")
        if len(self.to) != len(set(self.to)):
            raise ValueError("duplicate recipients are not allowed")
        return self


def _action_model(available_actions: list[DuplexActionType]) -> type[_JointAction]:
    """Constrain both SOTOPIA requests to this observation's action mask."""
    if not available_actions:
        raise ValueError("An action request must have at least one available action")
    fields: dict[str, Any] = {
        "action_type": (Literal[tuple(available_actions)], ...),
    }
    if set(available_actions) <= {"none", "leave", "backchanneling"}:
        fields["argument"] = (Literal[""], ...)
    # A fresh subclass avoids mutating a schema used by another agent's request.
    return create_model("Surface5Action", __base__=_JointAction, **fields)


class Surface5ActionOutputParser(PydanticOutputParser[_JointAction]):
    """Use the shared JSON parser and name resolution, retaining a local audit."""

    _attempts: int = PrivateAttr(default=0)
    _errors: list[str] = PrivateAttr(default_factory=list)

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(self._errors)

    def parse(self, result: str, context: dict[str, Any] | None = None) -> _JointAction:
        self._attempts += 1
        context = context or {}
        try:
            action = super().parse(result, context={**context, "agent_names": []})
            recipients = resolve_recipient_names(
                action.to, context.get("agent_names", [])
            )
            validated = _JointAction.model_validate(
                {**action.model_dump(), "to": recipients}, context=context
            )
            if recipients != action.to:
                sotopia_generation.log.info(
                    f"Resolved action recipients: {action.to} -> {recipients}"
                )
            return validated
        except Exception as error:
            self._errors.append(f"{type(error).__name__}: {error}")
            raise


class DuplexGenerationEngine:
    """Generate action and argument together, with SOTOPIA's one repair attempt."""

    def __init__(self, model_name: str) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        self.model_name = model_name
        self._decision_sequence = 0
        self._hidden_sequence = 0
        self._decision_audits: dict[str, tuple[int, tuple[str, ...]]] = {}
        self._hidden_attempts: dict[str, int] = {}
        self._non_audio_attempts: dict[str, int] = {}
        self._decision_starts_ns: dict[str, int] = {}

    def decision_started_ns(self, decision_id: str) -> int:
        return self._decision_starts_ns[decision_id]

    async def generate_action(
        self,
        session: AgentSessionContext,
        observation: StreamingObservation,
        history: str,
    ) -> GeneratedAction:
        parser = Surface5ActionOutputParser(
            pydantic_object=_action_model(observation.canonical.available_actions)
        )
        context = {
            "agent_names": [session.agent_name, session.peer_name],
            "sender": session.agent_name,
            "available_action_types": observation.canonical.available_actions,
            "target_utterance_id": (
                observation.target_utterance_id or observation.peer_utterance_id
            ),
        }
        try:
            temperature = gin.query_parameter(
                "sotopia.generation_utils.generate.agenerate_action.temperature"
            )
        except ValueError:
            temperature = sotopia_generation.DEFAULT_TEMPERATURE
        request_started_ns = time.monotonic_ns()
        fallback = False
        errors: tuple[str, ...]
        try:
            action = await agenerate(
                model_name=self.model_name,
                template=_ACTION_PROMPT,
                input_values={
                    "agent": session.agent_name,
                    "history": history,
                    "turn_number": str(observation.canonical.turn_number),
                    "action_list": " ".join(observation.canonical.available_actions),
                    "observation": self._observation_json(observation),
                },
                output_parser=parser,
                temperature=temperature,
                structured_output=True,
                context=context,
            )
            if not isinstance(action, _JointAction):
                raise TypeError(
                    f"expected _JointAction, received {type(action).__name__}"
                )
            errors = parser.errors
        except Exception as error:  # noqa: BLE001 - Match SOTOPIA request/parse fallback.
            # Match the Round-robin agent: failed generation skips this action.
            # asyncio.CancelledError inherits BaseException and propagates.
            sotopia_generation.log.warning(f"Failed to generate action due to {error}")
            fallback = True
            detail = f"{type(error).__name__}: {error}"
            errors = parser.errors
            if not errors or errors[-1] != detail:
                errors += (detail,)
            action = _JointAction(action_type="none", argument="", to=[])

        self._decision_sequence += 1
        decision_id = (
            f"{session.episode_id}-{self._slug(session.agent_name)}-"
            f"decision-{self._decision_sequence:04d}"
        )
        self._decision_starts_ns[decision_id] = request_started_ns
        argument = " ".join(action.argument.split())
        non_audio = action.action_type in {"action", "non-verbal communication"}
        decision = DuplexActionDecision.model_validate(
            {
                "decision_id": decision_id,
                "action_type": action.action_type,
                "non_audio_argument": argument if non_audio else "",
                "to": action.to,
                "target_utterance_id": self._derived_target(
                    observation, action.action_type
                ),
            },
            context={
                "available_actions": (
                    ["none"] if fallback else observation.canonical.available_actions
                ),
                "agent_names": context["agent_names"],
                "sender": session.agent_name,
                "require_decision_id": True,
            },
        )
        attempts = max(1, parser.attempts)
        self._decision_audits[decision_id] = (attempts, errors)
        if non_audio:
            self._non_audio_attempts[decision_id] = attempts
        return GeneratedAction(decision=decision, argument=argument, fallback=fallback)

    def make_hidden_said(
        self, session: AgentSessionContext, generated: GeneratedAction
    ) -> HiddenSaid:
        decision = generated.decision
        if decision.action_type not in {
            "speak",
            "hesitation",
            "correction",
            "interruption",
        }:
            raise ValueError(
                f"hidden said is not valid for action {decision.action_type!r}"
            )
        hidden = self._hidden_said(
            session=session, decision_id=decision.decision_id, text=generated.argument
        )
        self._hidden_attempts[hidden.hidden_said_id] = self.decision_audit(
            decision.decision_id
        )[0]
        return hidden

    def make_backchannel(
        self,
        session: AgentSessionContext,
        decision: DuplexActionDecision,
    ) -> HiddenSaid:
        if decision.action_type != "backchanneling":
            raise ValueError("backchannel text requires a backchanneling decision")
        hidden = self._hidden_said(
            session=session,
            decision_id=decision.decision_id,
            text=BACKCHANNEL_TTS_TEXT,
        )
        self._hidden_attempts[hidden.hidden_said_id] = self.decision_audit(
            decision.decision_id
        )[0]
        return hidden

    def split_into_sentence_chunks(self, hidden_said: HiddenSaid) -> list[SpeechChunk]:
        # Spoken actions reject asterisks before reaching this preparation, so
        # round-robin's star-span removal cannot silently discard spoken words.
        spoken_text = prepare_tts_text(hidden_said.text)
        sentences = self._spoken_sentences(spoken_text)
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
        if " ".join(chunk.text for chunk in chunks) != spoken_text:
            raise AssertionError("speech chunks must reconstruct prepared TTS text")
        return chunks

    def decision_audit(self, decision_id: str) -> tuple[int, tuple[str, ...]]:
        """Return parse attempts and validation or request failures for this action."""
        return self._decision_audits[decision_id]

    def hidden_said_attempt(self, hidden_said_id: str) -> int:
        """The text came from the same parse attempt as the action."""
        return self._hidden_attempts[hidden_said_id]

    def non_audio_argument_attempt(self, decision_id: str) -> int:
        return self._non_audio_attempts[decision_id]

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
    def _observation_json(observation: StreamingObservation) -> str:
        # Controller identifiers are never model-generated action arguments.
        return json.dumps(
            observation.model_dump(
                mode="json",
                exclude={"canonical", "peer_utterance_id", "target_utterance_id"},
            )
        )

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


__all__ = ["AgentSessionContext", "DuplexGenerationEngine", "GeneratedAction"]
