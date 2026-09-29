"""The Surface5 dialogue settings, frozen into each Talktopia run."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    realtime: Literal[True] = True
    allow_backchannels: bool = True
    allow_corrections: bool = False
    allow_interruptions: bool = False
    turn_taking_policy: str = "surface5_realtime_v2.1"
    generation_max_attempts: int = Field(default=2, ge=1)
    history_entries: int = Field(default=8, ge=1)
    minimum_floor_gap_ms: int = Field(default=200, ge=0)
    correction_min_stable_words: int = Field(default=8, ge=1)
    interruption_min_stable_words: int = Field(default=12, ge=1)
    max_turns: int = Field(default=12, ge=1)


SAMPLE_RATE_HZ = 24_000
FRAME_MS = 40
ASR_DECODE_INTERVAL_MS = 400
ASR_WINDOW_MS = 3000
INTERACTION_MODE = "surface5-full-duplex"


def runtime_options(values) -> dict:
    get = (
        values.get
        if isinstance(values, dict)
        else lambda key, default: getattr(values, key, default)
    )
    return {
        "max_turns": get("max_turns", 12),
        "allow_backchannels": get("duplex_backchannels", True),
        "allow_corrections": get("duplex_corrections", False),
        "allow_interruptions": get("duplex_interruptions", False),
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
        "audio_capture": "live_delivered_pcm_v2",
        "simulation_prompt": "simulation_v2.1",
        "temporal_evaluation": "sentence_asr_delivery_v1",
    }
