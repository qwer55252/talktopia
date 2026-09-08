"""Reference voices, profile fields, and the ASR/LLM/TTS conversation agent."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import wave
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from openai import APIError, AsyncOpenAI
from pydantic import Field

from talktopia.task_space import require_local

# SOTOPIA selects its model classes at import time. Set the backend first.
require_local(os.environ.get("SOTOPIA_STORAGE_BACKEND", "local"))

from sotopia.agents import LLMAgent
from sotopia.database import AgentProfile as SotopiaAgentProfile


# WAV validation: API output may use other PCM widths; ASR references require PCM16.


def validate_wav(audio: bytes) -> None:
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getcomptype() != "NONE":
                raise ValueError("TTS must return an uncompressed mono WAV")
            frame_count = wav.getnframes()
            if frame_count <= 0:
                raise ValueError("TTS returned an empty WAV")
            frames = wav.readframes(frame_count)
            if len(frames) != frame_count * wav.getsampwidth():
                raise ValueError("TTS returned a truncated WAV")
    except (wave.Error, EOFError) as exc:
        raise ValueError("TTS returned invalid WAV audio") from exc


def read_pcm_wav(audio: bytes) -> tuple[bytes, int]:
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if (
                wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or wav.getcomptype() != "NONE"
            ):
                raise ValueError("Expected an uncompressed mono PCM16 WAV")
            count = wav.getnframes()
            pcm = wav.readframes(count)
            if count <= 0 or len(pcm) != count * 2 or wav.getframerate() <= 0:
                raise ValueError("Empty or truncated WAV")
            return pcm, wav.getframerate()
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid WAV") from exc


def combine_conversation_audio(
    utterances: Sequence[tuple[Path, int]], destination: Path
) -> Path | None:
    """Join (mono WAV, channel) pairs: agent1 left (0), agent2 right (1)."""
    if not utterances:
        return None

    # Validate all inputs before creating the conversation file.
    audio_format = None
    chunks = []
    for path, channel in utterances:
        if channel not in (0, 1):
            raise ValueError(f"Conversation channel must be 0 or 1: {channel}")
        audio = path.read_bytes()
        validate_wav(audio)
        with wave.open(io.BytesIO(audio), "rb") as wav:
            current_format = (
                wav.getnchannels(),
                wav.getsampwidth(),
                wav.getframerate(),
            )
            if audio_format is not None and current_format != audio_format:
                raise ValueError(f"Conversation WAV formats do not match: {path}")
            audio_format = current_format
            chunks.append((wav.readframes(wav.getnframes()), channel))

    assert audio_format is not None
    _, width, rate = audio_format
    # WAV PCM8 is unsigned; zero amplitude is 128, not 0.
    silent_sample = b"\x80" if width == 1 else b"\x00" * width
    silence = silent_sample * 2 * round(rate * 0.3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output, wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        for index, (frames, channel) in enumerate(chunks):
            if index:
                wav.writeframes(silence)
            stereo = bytearray(silent_sample * 2 * (len(frames) // width))
            # Interleave the original sample bytes; the other channel stays silent.
            for byte_offset in range(width):
                stereo[channel * width + byte_offset :: 2 * width] = frames[
                    byte_offset::width
                ]
            wav.writeframes(stereo)
    return destination


# Reference voices: shared by dataset preparation and the speech server.


def reference_profile(pk: str, source: Path) -> dict[str, str]:
    if not pk or Path(pk).name != pk or pk in {".", ".."}:
        raise ValueError(f"Invalid Agent PK: {pk!r}")
    directory = source / pk
    try:
        audio = (directory / "reference.wav").read_bytes()
        read_pcm_wav(audio)
        text = (directory / "reference.txt").read_text(encoding="utf-8").strip()
        design = json.loads((directory / "design.json").read_text(encoding="utf-8"))
        voice = design["voice"]
        if (
            design["agent_pk"] != pk
            or not text
            or voice["reference_text"].strip() != text
        ):
            raise ValueError("Agent PK or reference text mismatch")
        if hashlib.sha256(audio).hexdigest() != voice["reference_audio_checksum"]:
            raise ValueError("reference.wav checksum mismatch")
        if not isinstance(voice["voice_id"], str) or not voice["voice_id"].strip():
            raise ValueError("Missing voice_id")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Agent {pk}, voice files at {directory}: {exc}") from exc
    return {
        "voice_id": voice["voice_id"],
        "voice_reference_wav": f"voices/{pk}/reference.wav",
        "voice_reference_text": text,
    }


def load_voice_registry(db: Path) -> dict[str, dict[str, Any]]:
    registry = {}
    for path in sorted((db / "AgentProfile").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        pk = row["pk"]
        expected = reference_profile(pk, db / "voices")
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(
                    f"Agent {pk}: {field} does not match copied reference files"
                )
        voice_id = expected["voice_id"]
        if voice_id in registry:
            raise ValueError(f"Duplicate voice_id: {voice_id}")
        registry[voice_id] = {
            **expected,
            "agent_pk": pk,
            "wav_path": db / expected["voice_reference_wav"],
        }
    if not registry:
        raise ValueError(f"No profile voices at {db}; run ./load_profiles.sh first")
    return registry


# Agent profile: preserve the original fields and add the character's voice.


class AgentProfile(SotopiaAgentProfile):
    voice_id: str = Field(min_length=1)
    voice_reference_wav: str = Field(min_length=1)
    voice_reference_text: str = Field(min_length=1)


# SOTOPIA adds query descriptors as class attributes after defining its model.
# Pydantic mistakes those attributes for defaults when creating a subclass.
# Restore the original field definitions, without changing the engine's class.
for _name, _field in SotopiaAgentProfile.model_fields.items():
    AgentProfile.model_fields[_name] = deepcopy(_field)
AgentProfile.model_rebuild(force=True)


# Conversation agent: synthesize its own speech and transcribe the other agent.


def prepare_tts_text(text: str) -> str:
    """Remove starred spans only from speech input, not from the saved LLM text."""
    # A run of stars opens/closes a span. An unclosed span consumes the remainder.
    cleaned = re.sub(r"\*+[^*]*(?:\*+|$)", " ", text)
    cleaned = " ".join(cleaned.split())
    return cleaned if any(character.isalnum() for character in cleaned) else ""


def speech_api_error(operation: str, exc: APIError) -> RuntimeError:
    # Response bodies may echo credentials; do not persist them in run artifacts.
    status = getattr(exc, "status_code", None)
    detail = f", HTTP {status}" if status is not None else ""
    return RuntimeError(f"{operation} request failed ({type(exc).__name__}{detail})")


class CascadedSpeechAgent(LLMAgent):
    def __init__(
        self,
        *,
        agent_profile: AgentProfile,
        model_name: str,
        asr_client: AsyncOpenAI,
        tts_client: AsyncOpenAI,
        asr_model: str,
        tts_model: str,
        asr_language: str,
    ) -> None:
        super().__init__(agent_profile=agent_profile, model_name=model_name)
        self.asr_client = asr_client
        self.tts_client = tts_client
        self.asr_model = asr_model
        self.tts_model = tts_model
        self.asr_language = asr_language
        self.voice = agent_profile.voice_id

    async def synthesize(self, text: str) -> bytes:
        text = prepare_tts_text(text)
        if not text:
            raise ValueError("Cannot synthesize an empty speak action")
        try:
            response = await self.tts_client.audio.speech.create(
                model=self.tts_model,
                input=text,
                voice=self.voice,
                response_format="wav",
            )
        except APIError as exc:
            raise speech_api_error("TTS", exc) from exc
        audio = response.content
        validate_wav(audio)
        return audio

    async def transcribe(self, audio: bytes, filename: str) -> str:
        try:
            response = await self.asr_client.audio.transcriptions.create(
                model=self.asr_model,
                file=(filename, audio, "audio/wav"),
                language=self.asr_language,
                response_format="json",
            )
        except APIError as exc:
            raise speech_api_error("ASR", exc) from exc
        text = response.text
        if not isinstance(text, str) or not text.strip():
            raise ValueError("ASR returned an empty or invalid transcript")
        return text.strip()
