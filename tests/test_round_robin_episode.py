import io
import json
import wave
from types import SimpleNamespace

import pytest

from talktopia import pipeline
from talktopia.speech_agent import (
    AgentProfile,
    CascadedSpeechAgent,
    resolve_recipient_names,
)
from sotopia.database import EnvironmentProfile, EpisodeLog
from sotopia.messages import AgentAction


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x01\x00" * 240)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_real_round_robin_environment_uses_asr_and_twelve_actions(
    tmp_path, monkeypatch
):
    from sotopia.database import storage_backend

    monkeypatch.setattr(
        storage_backend,
        "_storage_backend",
        storage_backend.LocalJSONBackend(str(tmp_path / "db")),
    )
    profiles = [
        AgentProfile(
            pk=f"agent-{i}",
            first_name=name,
            last_name="Test",
            age=30,
            voice_id=f"voice-{i}",
            voice_reference_wav="reference.wav",
            voice_reference_text="Hello",
        )
        for i, name in enumerate(("Alice", "Bob"))
    ]
    for profile in profiles:
        profile.save()
    env_profile = EnvironmentProfile(
        pk="environment-test",
        scenario="Arrange a meeting.",
        agent_goals=["Agree on a time.", "Agree on a time."],
    )
    env_profile.save()
    order = []
    audio_inputs = []

    async def aact(self, observation):
        if observation.available_actions == ["none"]:
            return AgentAction(action_type="none", argument="", to=[])
        order.append(self.agent_name)
        return AgentAction(
            action_type="speak", argument="*smiles* GENERATED words", to=[]
        )

    async def synthesize(self, text):
        audio_inputs.append(text)
        return wav_bytes()

    async def transcribe(self, audio, filename):
        return "RECOGNIZED speech"

    monkeypatch.setattr(CascadedSpeechAgent, "aact", aact)
    monkeypatch.setattr(CascadedSpeechAgent, "synthesize", synthesize)
    monkeypatch.setattr(CascadedSpeechAgent, "transcribe", transcribe)
    args = pipeline.parse_args([])
    args.tag = "test-run"
    record = {
        "env_id": env_profile.pk,
        "agent_ids": [profile.pk for profile in profiles],
    }
    env, agents = pipeline.build_episode(record, args, None, None)
    result = await pipeline.run_one_episode(env, agents, args, tmp_path, "episode_0001")
    assert order == ["Alice Test", "Bob Test"] * 6
    assert result["budget_turns"] == result["turns"] == 12
    assert result["action_counts"] == {"speak": 12}
    assert audio_inputs == ["GENERATED words"] * 12
    source = EpisodeLog.model_validate_json((tmp_path / result["original"]).read_text())
    text = json.dumps(source.messages)
    assert "RECOGNIZED speech" in text and "GENERATED words" not in text
    with wave.open(str(tmp_path / result["conversation_audio"]), "rb") as wav:
        assert wav.getnchannels() == 2
        assert wav.getnframes() == 240 * 12 + 7200 * 11
    assert resolve_recipient_names(["Bob"], ["Alice Test", "Bob Test"]) == ["Bob Test"]


@pytest.mark.asyncio
async def test_backend_timeout_is_not_misreported_as_episode_deadline(
    tmp_path, monkeypatch
):
    async def failing(*args, **kwargs):
        raise TimeoutError("ASR backend unavailable")

    monkeypatch.setattr(pipeline, "run_round_robin_episode", failing)
    with pytest.raises(TimeoutError, match="ASR backend unavailable"):
        await pipeline.run_one_episode(
            SimpleNamespace(), [], pipeline.parse_args([]), tmp_path, "episode_0001"
        )
    assert not (tmp_path / "simulation/diagnostics/episode_0001.json").exists()
