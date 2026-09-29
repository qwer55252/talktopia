# Surface5: measured speech timing and backchannel controls

Run the implementation in `talktopia-dev`. The Surface5 default allows
backchannels and disables hesitation, correction, and interruption:

```sh
./run_pipeline.sh --interaction-mode surface5-full-duplex
```

For a comparison within the same Surface5 implementation, disable backchannels
with `--no-duplex-backchannels`. Correction and interruption remain disabled.
`--duplex-corrections` and `--duplex-interruptions` explicitly enable those actions
for a separate condition. The flags are saved in `run_config.json` and the episode
start event; resume uses the saved settings. A comparison with round-robin also
changes turn handling, audio delivery, and observations, so it does not isolate the
effect of backchannels.

Both modes retain the common 12-action budget and 120-second episode deadline.
Backchannels and `none` do not consume the action budget. Shutdown drains the
in-flight audio frame before closing the capture; it does not start another frame.
No change to the canonical 450-combination population is made by these controls.

The first selected `leave` ends a Surface5 episode with reason `agent_left`,
including when it is the twelfth counted action. No closing sentence or second
leave is generated. Shutdown preserves delivered PCM and final ASR evidence,
cancels pending generation, and does not commit further actions. The frozen
runtime settings record `termination_policy: first_leave`.

The opening and subsequent idle opportunities include SOTOPIA's five base actions.
An idle `none` commits a pass and gives the other participant an opportunity.
SOTOPIA's rule evaluator ends the episode as `stale` after three consecutive
actual passes (`max_stale_turn=2`). Peer placeholders and partial-ASR `none`
observations are not passes. `ActionCommitted.metadata.actor` identifies the
actual participant; new ordinary and timed histories retain that participant's
pass. A first leave or all-pass conversation may contain no delivered audio and
no latency samples. Existing evaluation rules exclude conversations with no
interaction; generation does not force speech to make them evaluable.

## Clock and audio

Surface5 v2 events use milliseconds measured from a shared monotonic clock. Model
waits do not stop that clock. `conversation.wav` captures the PCM delivered on the
live audio bus: agent 1 is the left channel, agent 2 is the right channel, with
24 kHz PCM16 samples and at most 40 ms per frame. It preserves initial silence,
model-processing gaps, sentence gaps, and overlapping voices. It is written during
the interaction, without rebuilding a conversation from utterance WAV files.

The audio pump runs independently of decisions and final ASR. Capture policy
`live_pcm_sample_clock_v3` keeps the PCM stream continuous between callbacks and
paces delivery with absolute deadlines. This prevents sub-frame scheduler delays
from becoming repeated silence inside speech. Every frame retains its actual
callback start/end, WAV sample offset, PCM hash, and measured stream anchor
(`clock_anchor_ms`, `clock_anchor_sample`). Callback times and PCM sample positions
are separate measurements; latency still uses the measured callback times.

An empty queue records real silence and re-anchors the next delivery. A callback
stall longer than one frame also records the gap and its `underrun_ms`. Within a
continuous stream, callback lateness is bounded against the absolute sample
cursor, not the previous callback, so it cannot accumulate across frames.
Evaluation verifies this bound, uninterrupted stereo sample positions, underrun
sizes, and PCM hashes. Existing `live_delivered_pcm_v2` artifacts retain their
original strict position checks. A short final frame uses its actual sample
duration. Cancellation ASR runs in the background. A backchannel ready only after
its peer's speech has ended is discarded before its first frame.

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

The opener and decisions with no delivered audio
are excluded from these averages. An empty group has `count: 0` and `mean_ms: null`.
Episode results contain the same statistics. `03_simulation.json` aggregates
completed episodes using measurement counts, rather than averaging episode means.

## Prompts and evaluation

The runtime loads `talktopia/full_duplex/prompts/simulation_FDB_v6.txt` directly.
It preserves the body of `sotopia_action_v1.txt`, including the participant's
freedom to leave, and adds the same 40-word and recipient instructions as the
round-robin speech prompt plus the partial-ASR and backchannel instructions.
One request generates `action_type`, `argument`, and `to` together. The runtime
validates the current observation and floor before allowing TTS to start, and
checks validity again before delivery. `HiddenSaid` records that same response's
private speech text; it is not a second LLM request.

Each request uses a separate SOTOPIA action-model subclass whose JSON schema
contains only the currently available action types. Complete object branches
distinguish empty arguments for controls from non-empty arguments for speech and
nonverbal behavior. The initial request and its possible repair use the same
schema. Runtime validation retains the blank-text and 50-word checks and still
checks the current floor before synthesis and delivery. The native decoder does
not enforce the semantic distinction between dialogue and stage narration.
The prompt distinguishes words spoken aloud from nonverbal action descriptions.
Spoken arguments containing asterisks or round, square, or curly brackets use
the existing repair/fallback path. The runtime does not guess whether a marked
span is emphasis, an aside, or a stage direction and then delete it. Ordinary
quotes, apostrophes, and other punctuation remain allowed. These spoken-field
rules do not change the shared relaxed JSON parser or nonverbal descriptions. The argument schema repeats the spoken-text rule for repair
requests. `HiddenSaid` preserves the accepted generated text, while each sentence
chunk records the actual synthesis input. Nonverbal action descriptions remain
non-audio actions. No narration-removal heuristic is applied.

For Qwen3.5:9b structured requests with reasoning disabled, the Ollama proxy uses
the native raw generation endpoint with the verified Qwen chat template and an
empty thinking prefix. This avoids a locally reproduced Ollama 0.31.1 issue where
chat requests ignored the supplied JSON schema. The single model call, messages,
schema, token limit, and OpenAI sampling options are preserved. Other models and
unsupported request shapes retain their previous routes. This transport fix also
applies to round-robin Qwen requests using this proxy; historical frozen runs are
unchanged, but newly generated outputs may differ.

Generation uses SOTOPIA's structured-output parser and JSON repair, recipient
name resolution, configured action temperature (1.0 in the pipeline), and the
configured bad-output processing model. A failed parse can issue one repair
request; final failure produces `none` and an error audit, not another Surface5
regeneration loop. Decision events and committed passes distinguish this fallback
with `generation_fallback`. Cancellation still propagates. All inference and
repair waits remain in measured latency and the live recording.

The agent history contains the initial SOTOPIA observation and every subsequent
canonical observation, without an eight-entry limit. Both participants see the
confirmed ASR text, including for their own earlier utterances. Private generated
speech is kept in artifacts, not substituted for ASR in that history. Only the
current partial ASR observation is appended to the request. Backchannel responses
have an empty model argument. After runtime validation, the exact OmniVoice tag
`[confirmation-en]` produces a short nonverbal acknowledgment in the agent's
voice. The tag is TTS input, not recognized speech. Backchannels with this tag use
OmniVoice's [`duration=0.6` generation parameter](https://github.com/k2-fsa/OmniVoice/blob/main/docs/generation-parameters.md)
uniformly across
voices; the actual delivered duration is measured and may differ. Ordinary
speech keeps its existing generation defaults and PCM encoding. Before encoding
the exact confirmation tag's output, the speech server raises quiet backchannels
toward a whole-clip RMS of 0.05 (about -26 dBFS), using one constant gain per clip.
Gain is at most 8 (about +18 dB) and is further limited so amplification does not
raise the peak above 0.8. Already-loud clips are unchanged, including those with
an original peak above 0.8. There is no cropping or per-frame gain adjustment.
The same adjusted PCM is delivered to the peer and saved in the live recording.
Inputs below RMS 0.001 are treated as empty audio and use the existing two-attempt
TTS retry, rather than amplifying near-silence. The input, duration, and volume
policy are recorded in `backchannel_tts`; the speech server logs actual RMS,
gain, and peak for each synthesized confirmation. This applies to new audio only.
Delivered backchannels with empty ASR are retained as nonverbal actions with their measured interval and
`[no recognized words]`; the raw ASR and transcript text remain empty. An empty
TTS result produces no committed backchannel and no latency sample.

The runtime settings identify `generation_mode: sotopia_single_call` and
`history_policy: full_canonical_asr`. Code and prompt fingerprints prevent resuming
an old run with these changes; use a new output directory and reuse its complete
canonical manifest to repeat a scenario. Prior prompts `simulation_FDB_v1.txt`
and `simulation_FDB_v2.txt`, along with `simulation_FDB_v3.txt`, retain their
contents. The 50-word validation limit and
DeepSeek answer-only generation remain intact. The shared engine's existing JSON
normalization, including removal of thinking tags and extraction of JSON from
surrounding text, applies to all joint responses.

For Surface5 evaluation, each delivered sentence chunk has its own ASR result.
The evaluator verifies event identity, frame positions and PCM hashes against the
recording, requires silence outside delivered frames, and checks that
whole-utterance ASR commits match the saved EpisodeLog. Batch evaluation freezes
the entire WAV hash before starting.
It then adds utterance and sentence start/end times to the evaluation history,
using the complete `evaluation/prompts/evaluation_FDB_v1.txt` template. It
preserves the wording and order of `evaluation_v1.txt` and adds only a speech
timing section. The seven score dimensions, ranges, validation, retries and
evaluation temperature are unchanged. Sentence ASR
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
