"""The Surface5 dialogue settings, frozen into each Talktopia run."""

from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    realtime: bool = True
    turn_taking_policy: str = "natural_duplex_v1"
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


def runtime_settings(max_turns: int = 12) -> dict:
    return {
        **RuntimeConfig(max_turns=max_turns).model_dump(),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "frame_ms": FRAME_MS,
        "asr_decode_interval_ms": ASR_DECODE_INTERVAL_MS,
        "asr_window_ms": ASR_WINDOW_MS,
        "max_generated_words": 40,
    }
