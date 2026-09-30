"""The Surface5 dialogue settings, frozen into each Talktopia run."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from talktopia.models.config import backchannel_tts_settings


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    realtime: Literal[True] = True
    allow_backchannels: bool = True
    turn_taking_policy: str = "surface5_speech_or_leave_v1"
    termination_policy: Literal["first_leave"] = "first_leave"
    generation_mode: Literal["sotopia_single_call"] = "sotopia_single_call"
    validation_policy: Literal["essential_v1"] = "essential_v1"
    episode_max_attempts: Literal[1] = 1
    output_repair_policy: Literal["disabled"] = "disabled"
    history_policy: Literal["full_canonical_asr"] = "full_canonical_asr"
    minimum_floor_gap_ms: int = Field(default=200, ge=0)
    max_turns: int = Field(default=12, ge=1)


SAMPLE_RATE_HZ = 24_000
FRAME_MS = 40
ASR_DECODE_INTERVAL_MS = 400
ASR_WINDOW_MS = 3000
INTERACTION_MODE = "surface5-full-duplex"
SIMULATION_PROMPT_VERSION = "simulation_action_general_v1"
BACKCHANNEL_PROMPT_VERSION = "simulation_action_FDB_v1"
BACKCHANNEL_TRIGGER = "received_sentence_asr_with_next_sentence_v1"


def runtime_options(values) -> dict:
    get = (
        values.get
        if isinstance(values, dict)
        else lambda key, default: getattr(values, key, default)
    )
    return {
        "max_turns": get("max_turns", 12),
        "allow_backchannels": get("duplex_backchannels", True),
    }


def runtime_settings(max_turns: int = 12, **options) -> dict:
    return {
        **RuntimeConfig(max_turns=max_turns, **options).model_dump(),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "frame_ms": FRAME_MS,
        "asr_decode_interval_ms": ASR_DECODE_INTERVAL_MS,
        "asr_window_ms": ASR_WINDOW_MS,
        "prompt_max_words": 40,
        "max_generated_words": 50,
        "turn_budget": "confirmed_actions_except_none_and_backchanneling_v1",
        "opener": "agent1",
        "clock": "monotonic_elapsed_ms",
        "audio_capture": "live_pcm_sample_clock_v3",
        "simulation_prompt": SIMULATION_PROMPT_VERSION,
        "backchannel_prompt": BACKCHANNEL_PROMPT_VERSION,
        "backchannel_trigger": BACKCHANNEL_TRIGGER,
        "backchannel_limit_per_utterance": 1,
        "evaluation_prompt": "evaluation_FDB_v1",
        "temporal_evaluation": "sentence_asr_delivery_v1",
        "backchannel_tts": backchannel_tts_settings(),
    }
