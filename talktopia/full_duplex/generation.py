"""SOTOPIA single-call action generation with Surface5 observation controls."""

from __future__ import annotations

import json
import random
import re
import time
from copy import deepcopy
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
from sotopia.generation_utils import PydanticOutputParser
from sotopia.generation_utils import generate as sotopia_generation
from sotopia.messages import AgentAction

from talktopia.models.config import BACKCHANNEL_TTS_INPUTS

from .actions import (
    DuplexActionDecision,
    DuplexActionType,
    HiddenSaid,
    SpeechChunk,
    StreamingObservation,
    _validate_generated_text,
)
from .config import BACKCHANNEL_PROMPT_VERSION, SIMULATION_PROMPT_VERSION
from .requests import generate_structured_action

# Keep the requested length independent from the validation safety margin.
PROMPT_MAX_WORDS = 40
_EMPTY_ARGUMENT_ACTIONS = frozenset({"none", "leave", "backchanneling"})
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
_BACKCHANNEL_PROMPT_PATH = _PROMPT_PATH.with_name(f"{BACKCHANNEL_PROMPT_VERSION}.txt")
_BACKCHANNEL_PROMPT = _BACKCHANNEL_PROMPT_PATH.read_text(encoding="utf-8")


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
    raw_responses: tuple[str | None, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationFailure:
    decision_id: str
    error: str
    raw_responses: tuple[str | None, ...]


class _JointAction(AgentAction):
    """Read the action payload; the two-agent runtime owns recipient routing."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    action_type: Literal["speak", "leave", "backchanneling", "none"]
    argument: str = Field(
        description="Words to speak, at most 40 words. Ignored for other actions."
    )

    @model_validator(mode="after")
    def validate_surface5_contract(self, info: ValidationInfo) -> _JointAction:
        available = (info.context or {}).get("available_action_types")
        if available is not None and self.action_type not in available:
            raise ValueError(f"unavailable action: {self.action_type}")
        if self.action_type == "speak":
            _validate_generated_text(self.argument, field_name="argument")
        # Both participants receive public speech. Model-written names do not route audio.
        return self.model_copy(
            update={
                "to": [],
                "argument": self.argument if self.action_type == "speak" else "",
            }
        )


def _action_json_schema(schema: dict[str, Any]) -> None:
    """Expose argument length per action without changing the parsed model."""
    action_schema = schema["properties"]["action_type"]
    available = action_schema.get("enum", [action_schema.get("const")])
    branches = []
    for empty_argument in (True, False):
        actions = [
            action
            for action in available
            if (action in _EMPTY_ARGUMENT_ACTIONS) == empty_argument
        ]
        if not actions:
            continue
        branch = deepcopy(schema)
        branch["properties"]["action_type"] = {"type": "string", "enum": actions}
        if not empty_argument:
            branch["properties"]["argument"]["minLength"] = 1
        branches.append(branch)

    # Ollama's grammar converter does not intersect root properties with anyOf.
    # Each alternative must retain the complete object and recipient contract.
    # Keep the first request's string grammar intact: a native pattern can
    # permit invalid JSON escapes unless it also narrows the allowed alphabet.
    title = schema.get("title", "Surface5Action")
    schema.clear()
    schema.update(
        branches[0] if len(branches) == 1 else {"title": title, "anyOf": branches}
    )


def _action_model(available_actions: list[DuplexActionType]) -> type[_JointAction]:
    """Constrain the model request to this observation's action mask."""
    if not available_actions or not set(available_actions) <= {
        "speak",
        "leave",
        "backchanneling",
        "none",
    }:
        raise ValueError("Unsupported live action mask")
    fields: dict[str, Any] = {
        "action_type": (Literal[tuple(available_actions)], ...),
    }
    # A fresh subclass avoids mutating a schema used by another agent's request.
    return create_model(
        "Surface5Action",
        __base__=_JointAction,
        __config__=ConfigDict(json_schema_extra=_action_json_schema),
        **fields,
    )


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
            return _JointAction.model_validate(
                action.model_dump(), context={**context, "agent_names": []}
            )
        except Exception as error:
            self._errors.append(f"{type(error).__name__}: {error}")
            raise


class DuplexGenerationEngine:
    """Generate action and argument together, without LLM output repair."""

    def __init__(self, model_name: str, *, seed: int = 0) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        self.model_name = model_name
        self.seed = seed
        self._decision_sequence = 0
        self._hidden_sequence = 0
        self._decision_audits: dict[str, tuple[int, tuple[str, ...]]] = {}
        self._hidden_attempts: dict[str, int] = {}
        self._decision_starts_ns: dict[str, int] = {}

    def decision_started_ns(self, decision_id: str) -> int:
        return self._decision_starts_ns[decision_id]

    async def generate_action(
        self,
        session: AgentSessionContext,
        observation: StreamingObservation,
        history: str,
    ) -> GeneratedAction | GenerationFailure:
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
        self._decision_sequence += 1
        decision_id = f"{session.episode_id}-{self._slug(session.agent_name)}-decision-{self._decision_sequence:04d}"
        self._decision_starts_ns[decision_id] = request_started_ns
        raw_responses: list[str | None] = []
        errors: tuple[str, ...]
        try:
            action = await generate_structured_action(
                model_name=self.model_name,
                template=(
                    _BACKCHANNEL_PROMPT
                    if observation.source == "asr_partial"
                    else _ACTION_PROMPT
                ),
                input_values={
                    "agent": session.agent_name,
                    "history": history,
                    "turn_number": str(observation.canonical.turn_number),
                    "action_list": " ".join(observation.canonical.available_actions),
                    "observation": self._observation_json(observation),
                },
                output_parser=parser,
                temperature=temperature,
                context=context,
                responses=raw_responses,
            )
            if not isinstance(action, _JointAction):
                raise TypeError(
                    f"expected _JointAction, received {type(action).__name__}"
                )
            errors = parser.errors
        except Exception as error:
            # Cancellation is a BaseException and propagates to the caller.
            detail = f"{type(error).__name__}: {error}"
            self._decision_audits[decision_id] = (1, (detail,))
            return GenerationFailure(decision_id, detail, tuple(raw_responses))

        argument = " ".join(action.argument.split())
        decision = DuplexActionDecision.model_validate(
            {
                "decision_id": decision_id,
                "action_type": action.action_type,
                "non_audio_argument": "",
                "to": action.to,
            },
            context={
                "available_actions": observation.canonical.available_actions,
                "agent_names": context["agent_names"],
                "sender": session.agent_name,
                "require_decision_id": True,
            },
        )
        attempts = max(1, parser.attempts)
        self._decision_audits[decision_id] = (attempts, errors)
        return GeneratedAction(
            decision=decision,
            argument=argument,
            raw_responses=tuple(raw_responses),
        )

    def make_hidden_said(
        self, session: AgentSessionContext, generated: GeneratedAction
    ) -> HiddenSaid:
        decision = generated.decision
        if decision.action_type != "speak":
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
        # Pick independently of dialogue meaning and other agents' scheduling.
        # The same run/episode/agent/decision reproduces the same uniform draw.
        identity = (
            f"surface5-backchannel-v1\0{self.seed}\0{session.episode_id}\0"
            f"{session.agent_name}\0{decision.decision_id}"
        )
        text = random.Random(identity).choice(BACKCHANNEL_TTS_INPUTS)
        hidden = self._hidden_said(
            session=session,
            decision_id=decision.decision_id,
            text=text,
        )
        self._hidden_attempts[hidden.hidden_said_id] = self.decision_audit(
            decision.decision_id
        )[0]
        return hidden

    def split_into_sentence_chunks(self, hidden_said: HiddenSaid) -> list[SpeechChunk]:
        # Strip formatting symbols only; never remove the enclosed words.
        # OmniVoice backchannel tags such as [confirmation-en] remain intact.
        spoken_text = " ".join(hidden_said.text.replace("*", "").split())
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
