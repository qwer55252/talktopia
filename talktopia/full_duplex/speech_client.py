"""PCM operations over Talktopia's existing WAV speech endpoints."""

from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path

from openai import APIError, AsyncOpenAI

from talktopia.speech_agent import read_pcm_wav
from talktopia.utils import safe_error


class SpeechClientError(RuntimeError):
    def __init__(self, message: str, *, code: str = "backend_error", error_type=None):
        self.code = code
        self.error_type = error_type
        super().__init__(message)


def pcm_wav(pcm: bytes, sample_rate_hz: int) -> bytes:
    if not pcm or len(pcm) % 2:
        raise ValueError("ASR input must contain complete PCM16 samples")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm)
    return output.getvalue()


class SpeechClient:
    """An episode owns the HTTP clients; each agent owns its voice binding."""

    def __init__(
        self,
        asr_client: AsyncOpenAI,
        tts_client: AsyncOpenAI,
        *,
        voice_id: str,
        voice_reference: Path,
        asr_model: str,
        tts_model: str,
        asr_language: str,
        tts_semaphore: asyncio.Semaphore | None = None,
    ):
        self.asr_client = asr_client
        self.tts_client = tts_client
        self.voice_id = voice_id
        self.voice_reference = voice_reference.resolve()
        self.asr_model = asr_model
        self.tts_model = tts_model
        self.asr_language = asr_language
        self.tts_semaphore = tts_semaphore or asyncio.Semaphore(1)

    async def decode(self, pcm: bytes, sample_rate_hz: int) -> str:
        try:
            response = await self.asr_client.audio.transcriptions.create(
                model=self.asr_model,
                file=("received.wav", pcm_wav(pcm, sample_rate_hz), "audio/wav"),
                language=self.asr_language,
                response_format="json",
            )
        except APIError as exc:
            raise SpeechClientError(safe_error(exc)) from exc
        if not isinstance(response.text, str):
            raise SpeechClientError("ASR returned a non-text transcript")
        # Silence and short backchannels can legitimately decode to empty text.
        return response.text.strip()

    async def synthesize(self, text: str, voice_reference: Path, seed: int) -> bytes:
        if voice_reference.resolve() != self.voice_reference:
            raise ValueError("TTS reference differs from the agent's profile voice")
        try:
            async with self.tts_semaphore:
                response = await self.tts_client.audio.speech.create(
                    model=self.tts_model,
                    input=text,
                    voice=self.voice_id,
                    response_format="wav",
                    extra_body={"seed": seed},
                )
        except APIError as exc:
            body = exc.body if isinstance(exc.body, dict) else {}
            detail = body.get("detail", body)
            code = detail.get("code") if isinstance(detail, dict) else None
            raise SpeechClientError(
                safe_error(exc),
                code="empty_audio" if code == "empty_audio" else "backend_error",
            ) from exc
        pcm, rate = read_pcm_wav(response.content)
        if rate != 24_000:
            raise SpeechClientError("Full-duplex TTS requires 24000 Hz PCM16 WAV")
        return pcm
