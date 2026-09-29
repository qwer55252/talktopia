# Surface5: measured speech timing and backchannel controls

Run the implementation in `talktopia-dev`. The Surface5 default allows
backchannels and disables correction and interruption:

```sh
./run_pipeline.sh --interaction-mode surface5-full-duplex
```

For a comparison within the same Surface5 implementation, disable backchannels
with `--no-duplex-backchannels`. Correction and interruption remain disabled.
`--duplex-corrections` and `--duplex-interruptions` explicitly enable those actions
for a separate condition. The flags are saved in `run_config.json` and the episode
start event; resume uses the saved settings. A comparison with round-robin also
changes turn handling, generation, and observations, so it does not isolate the
effect of backchannels.

Both modes retain the common 12-action budget and 120-second episode deadline.
Backchannels and `none` do not consume the action budget. Shutdown drains the
in-flight audio frame before closing the capture; it does not start another frame.
No change to the canonical 450-combination population is made by these controls.

## Clock and audio

Surface5 v2 events use milliseconds measured from a shared monotonic clock. Model
waits do not stop that clock. `conversation.wav` captures the PCM delivered on the
live audio bus: agent 1 is the left channel, agent 2 is the right channel, with
24 kHz PCM16 samples and at most 40 ms per frame. It preserves initial silence,
model-processing gaps, sentence gaps, and overlapping voices. It is written during
the interaction, without rebuilding a conversation from utterance WAV files.

The audio pump runs independently of decisions and final ASR. Each delivery
records its measured start/end, its WAV sample offset, and the hash of the PCM
accepted by the receiver. Sample offsets are quantized to the nearest audio sample;
frame end times are observed after the paced transmission completes. Scheduling
jitter remains in the recording. A short final frame uses its actual sample
duration. Cancellation ASR also runs in the background. A backchannel that is ready only after its peer's
speech has ended is discarded before its first frame.

## Latency

`simulation/latency/episode_XXXX.json` contains the measured samples and separate
counts and arithmetic means for:

- **Normal response:** the peer utterance's last audio delivery end to the
  response's first audio delivery start. Final ASR, LLM, TTS, and scheduling waits
  are included.
- **Backchannel:** the first LLM request using the triggering partial ASR
  observation to the backchannel's first audio delivery start. Retries and TTS
  waits are included. The decision/observation/peer utterance identifiers are
  recorded in `response_latency` events.

The opener, the leave-handshake closing, and decisions with no delivered audio
are excluded from these averages. An empty group has `count: 0` and `mean_ms: null`.
Episode results contain the same statistics. `03_simulation.json` aggregates
completed episodes using measurement counts, rather than averaging episode means.

## Prompts and evaluation

The runtime loads the five sections of
`talktopia/full_duplex/prompts/simulation_v2.1.txt` directly. This is the appendix
version, rather than a copy of a separate inline prompt. Prompts request at most
40 words; the existing 50-word validation limit and JSON handling remain intact.
DeepSeek answer-only generation is unchanged.

For Surface5 evaluation, each delivered sentence chunk has its own ASR result.
The evaluator verifies event identity, frame positions and PCM hashes against the
recording, requires silence outside delivered frames, and checks that
whole-utterance ASR commits match the saved EpisodeLog. Batch evaluation freezes
the entire WAV hash before starting.
It then adds utterance and sentence start/end times to the evaluation history,
with `evaluation/prompts/temporal_v1.txt` as the timing instructions. Sentence ASR
can differ from whole-utterance ASR; both represent the same action, and generated
text is never substituted for received speech. The saved EpisodeLog is unchanged.
The committed whole-utterance ASR determines what the listener heard. Differences
in the auxiliary sentence ASR must not be scored as an agent contradicting itself.

Round-robin history and scoring instructions are unchanged. Surface5 logs made
with the old frame-count clock, or standalone logs without timing/audio sidecars,
cannot supply measured temporal evidence and are rejected for this evaluation.
They require a new measured simulation; the old timing is not retrospectively
estimated. The original seven score dimensions, ranges and evaluation temperature
are retained.
