"""Only exact nonverbal tags change duration and boost quiet audio."""

import io
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from talktopia.full_duplex.config import runtime_settings
from talktopia.models.config import (
    BACKCHANNEL_TAG_DURATION_S,
    BACKCHANNEL_TTS_INPUTS,
    BACKCHANNEL_TTS_TAGS,
    SPEECH_PROTOCOL,
)
from talktopia.speech_agent import EmptyAudioError, SpeechBackend, TTSRequest


def mock_backend(*, empty=False):
    backend = object.__new__(SpeechBackend)
    backend.np = np
    backend.torch = SimpleNamespace(manual_seed=Mock())
    backend.voices = {
        "test": {"wav_path": "unused.wav", "voice_reference_text": "Reference."}
    }
    backend.prompts = {"test": object()}

    def generate(**kwargs):
        count = len(kwargs["text"]) if isinstance(kwargs["text"], list) else 1
        return [
            np.zeros(0, dtype=np.float32)
            if empty
            else np.full(2400, 0.1, dtype=np.float32)
            for _ in range(count)
        ]

    backend.tts = SimpleNamespace(generate=Mock(side_effect=generate))
    return backend


@pytest.mark.parametrize(
    "text,seed",
    [
        ("[confirmation-en]", 123),
        ("[dissatisfaction-hnn]", None),  # Removed from the pool.
        ("[question-ah]", None),  # Removed from the pool.
        ("[question-ei]", None),  # Removed from the pool: no special TTS settings.
        ("yeah", 123),
        ("A normal sentence.", None),
        ("[confirmation-en] is a tag.", 123),
        (" [confirmation-en] ", None),
    ],
)
def test_tag_duration_preserves_text_voice_language_and_normal_defaults(text, seed):
    backend = mock_backend()
    result = backend._synthesize_batch([TTSRequest(text, "test", seed=seed)])
    assert backend.tts.generate.call_count == 1
    kwargs = backend.tts.generate.call_args.kwargs
    assert kwargs["text"] == (text if seed is not None else [text])
    assert kwargs["language"] == "English"
    assert kwargs["voice_clone_prompt"] == (
        backend.prompts["test"] if seed is not None else [backend.prompts["test"]]
    )
    if text in BACKCHANNEL_TTS_TAGS:
        assert kwargs["duration"] == (0.6 if seed is not None else [0.6])
        assert set(kwargs) == {"text", "language", "voice_clone_prompt", "duration"}
    else:
        assert set(kwargs) == {"text", "language", "voice_clone_prompt"}
    if seed is None:
        backend.torch.manual_seed.assert_not_called()
    else:
        backend.torch.manual_seed.assert_called_once_with(seed)
    with wave.open(io.BytesIO(result[0])) as wav:
        assert wav.getnframes() == 2400  # No duration enforcement by cropping PCM.
        assert wav.getframerate() == 24000
        assert (
            wav.readframes(2400)
            == np.full(2400, int(0.1 * 32767), dtype="<i2").tobytes()
        )


def test_mixed_batch_sets_only_nonverbal_tag_duration():
    backend = mock_backend()
    texts = [*BACKCHANNEL_TTS_INPUTS, "A normal sentence."]
    result = backend._synthesize_batch([TTSRequest(text, "test") for text in texts])
    kwargs = backend.tts.generate.call_args.kwargs
    assert kwargs["text"] == texts
    assert kwargs["duration"] == [None, 0.6, None, None, None, None]
    assert len(result) == len(texts) and backend.tts.generate.call_count == 1
    assert set(kwargs) == {"text", "language", "voice_clone_prompt", "duration"}


def test_empty_backchannel_output_does_not_add_backend_retries():
    backend = mock_backend(empty=True)
    with pytest.raises(EmptyAudioError):
        backend._synthesize_batch([TTSRequest("[confirmation-en]", "test", seed=123)])
    assert backend.tts.generate.call_count == 1


def test_frozen_backchannel_settings_use_the_backend_constants():
    assert BACKCHANNEL_TTS_INPUTS == (
        "yeah", "[confirmation-en]", "Uh-huh", "Mm-hmm", "Yep"
    )
    assert runtime_settings()["backchannel_tts"] == {
        "inputs": list(BACKCHANNEL_TTS_INPUTS),
        "selection": "uniform_per_decision_v1",
        "seed_fields": ["run_seed", "episode_id", "agent_name", "decision_id"],
        "tag_duration_s": BACKCHANNEL_TAG_DURATION_S,
        "volume": {
            "policy": "whole_clip_rms_boost_v1",
            "applies_to": "exact_nonverbal_tags",
            "target_rms": 0.05,
            "max_gain": 8.0,
            "gain_peak_ceiling": 0.8,
            "minimum_rms": 0.001,
        },
    }
    assert SPEECH_PROTOCOL == "surface5-http-v6"


def decode_pcm(audio):
    with wave.open(io.BytesIO(audio)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


@pytest.mark.parametrize(
    "samples,expected_gain",
    [
        (np.full(2400, 0.01, dtype=np.float32), 5.0),  # Target RMS.
        (np.full(2400, 0.005, dtype=np.float32), 8.0),  # Gain cap.
        (np.tile([0.4, 0.0], 1200).astype(np.float32), 1.0),  # Already loud.
        (np.full(2400, 0.1, dtype=np.float32), 1.0),
        (np.array([0.9] + [0.001] * 2399, dtype=np.float32), 1.0),
        (np.array([-0.4] + [0.002] * 2399, dtype=np.float32), 2.0),  # Peak cap.
    ],
)
def test_backchannel_scalar_gain_preserves_duration(samples, expected_gain):
    backend = mock_backend()
    original = samples.copy()
    backend.tts.generate.return_value = [samples]
    backend.tts.generate.side_effect = None
    result = backend._synthesize_batch([TTSRequest("[confirmation-en]", "test", seed=7)])
    pcm = decode_pcm(result[0])
    expected = (original * expected_gain * 32767).astype("<i2")
    np.testing.assert_allclose(pcm.astype(float), expected.astype(float), atol=1)
    np.testing.assert_array_equal(samples, original)  # Do not mutate model output.
    assert len(pcm) == len(samples)
    assert np.max(np.abs(pcm.astype(int))) < 32767
    backend.torch.manual_seed.assert_called_once_with(7)
    assert backend.tts.generate.call_count == 1


@pytest.mark.parametrize("amplitude", [0.0, 0.0009])
def test_near_silent_backchannel_uses_existing_empty_audio_error(amplitude):
    backend = mock_backend()
    backend.tts.generate.side_effect = None
    backend.tts.generate.return_value = [np.full(2400, amplitude, dtype=np.float32)]
    with pytest.raises(EmptyAudioError, match="backchannel is near-silent"):
        backend._synthesize_batch([TTSRequest("[confirmation-en]", "test", seed=7)])
    assert backend.tts.generate.call_count == 1


@pytest.mark.parametrize("bad_sample", [float("nan"), float("inf")])
def test_nonfinite_backchannel_rejected_before_volume_adjustment(bad_sample):
    backend = mock_backend()
    backend.tts.generate.side_effect = None
    backend.tts.generate.return_value = [np.array([0.01, bad_sample])]
    with pytest.raises(RuntimeError, match="non-finite audio"):
        backend._synthesize_batch([TTSRequest("[confirmation-en]", "test")])


def test_mixed_batch_boosts_only_exact_tag_and_preserves_ordinary_pcm():
    backend = mock_backend()
    texts = [*BACKCHANNEL_TTS_INPUTS, "Hello.", "[confirmation-en] is a tag.", " [confirmation-en] "]
    samples = np.linspace(-0.02, 0.02, 2400, dtype=np.float32)
    backend.tts.generate.side_effect = None
    backend.tts.generate.return_value = [samples.copy() for _ in texts]
    result = backend._synthesize_batch([TTSRequest(text, "test") for text in texts])
    original_pcm = (samples * 32767).astype("<i2")
    for text, audio in zip(texts, result, strict=True):
        pcm = decode_pcm(audio)
        assert len(pcm) == len(samples)
        if text in BACKCHANNEL_TTS_TAGS:
            boosted = pcm.astype(float) / 32767
            assert np.sqrt(np.mean(boosted ** 2)) == pytest.approx(0.05, abs=1 / 32767)
        else:
            assert pcm.tobytes() == original_pcm.tobytes()
    assert backend.tts.generate.call_count == 1
