# Surface5: measured speech timing and backchannel controls

Run the implementation in `talktopia-dev`. The Surface5 default allows
backchannels and disables hesitation, correction, and interruption:

```sh
./run_pipeline.sh --interaction-mode surface5-full-duplex
```

For a comparison within the same Surface5 implementation, disable backchannels
with `--no-duplex-backchannels`. The live runtime supports only `speak`, `leave`,
`backchanneling`, and listening `none`. The correction/interruption switches and
execution branches have been removed. Legacy action and event types remain
readable for existing artifacts. A comparison with round-robin changes more than
backchannels, including turn handling, delivery, and observations.

Backchannel decisions are offered only after a non-final sentence has finished
audio delivery and its ASR result is ready. At that boundary, the audio frame's
`is_utterance_end` carries the sentence chunk's `is_final`, so
`has_next_sentence = not is_final`. A non-final sentence produces an `asr_partial`
observation containing ASR through that sentence, its `sentence_index`, and
`has_next_sentence: true`. Rolling ASR updates are still logged but do not trigger
LLM calls. ASR results are assembled in sentence order, without exposing the
speaker's generated text or later sentence results.

The listener chooses `backchanneling` or `none` under the default controls. The
runtime permits at most one played backchannel per peer utterance. Once its first
audio frame is delivered, later sentence ASR results for that utterance are still
logged but cannot trigger another listener LLM request. Choosing `none` leaves the
next non-final sentence eligible, and a new peer utterance gets a new opportunity.
A backchannel that fails synthesis or is discarded before playback does not use
this allowance. There is no minimum-word threshold. An already busy or speaking
listener does not start another concurrent decision. The speaker does not wait for
sentence ASR, the listener's decision, or backchannel synthesis before continuing.
Sentence results that arrive after the utterance ends cannot trigger a backchannel. Pending late
backchannels are rejected before their first frame. The final sentence instead
leads to whole-utterance ASR and one general response with
`has_next_sentence: false`.

Both modes retain the 12-action budget and the 120-second time setting.
Surface5 treats 120 seconds as the boundary for selecting new actions. At that
point it cancels unfinished LLM decisions, stops new response and backchannel
requests, and finishes speech already selected before the boundary. This includes
pending TTS, every remaining sentence, any already selected backchannel, and final
ASR commits. The complete final speech appears in the recording and evaluation
history. The episode is saved as `completed` with `end_reason: time_limit`, and its
measured duration can exceed 120 seconds. No extra goodbye or `leave` is generated.
Run settings record `time_limit_policy: finish_selected_speech_v1`.

`simulation/readable/episode_XXXX.md` uses the same measured utterance and sentence
timestamps as evaluation history. Its `[MM:SS.mmm - MM:SS.mmm]` spans come from
delivered audio, including the final utterance that finishes after the time limit.

Actual backend errors and request timeouts still fail the episode; external
cancellation still cancels it. Round-robin retains its existing hard timeout.
Backchannels and `none` do not consume the action budget. Ordinary shutdown drains
the in-flight audio frame before closing the capture. No change to the canonical
450-combination population is made by these controls.

The first selected `leave` ends a Surface5 episode with reason `agent_left`,
including when it is the twelfth counted action. No closing sentence or second
leave is generated. Shutdown preserves delivered PCM and final ASR evidence,
cancels pending generation, and does not commit further actions. The frozen
runtime settings record `termination_policy: first_leave`.

Opening and ordinary responses allow only `speak` and `leave`. This includes a
response deferred until the listener's backchannel finishes. Under the default
controls, non-final sentence decisions allow only `backchanneling` and `none`.
`action` and `non-verbal communication` are excluded from all live action masks,
request schemas, and runtime commits: action-description text cannot substitute
for delivered speech. Run settings record `surface5_speech_or_leave_v1` as the
turn-taking policy.

Listening `none` and peer placeholders are not committed passes. If generation
fails where `none` is unavailable, including the opening, the
episode fails with diagnostic events instead of consuming a silent turn. Existing
logs containing passes or non-audio actions remain readable and evaluable.
`ActionCommitted.metadata.actor` identifies the actual participant in those logs.
Existing evaluation rules exclude conversations with no interaction; an immediate
`leave` remains allowed.

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
- **Backchannel:** the first LLM request using the triggering completed-sentence ASR
  observation to the backchannel's first audio delivery start. Retries and TTS
  waits are included. The decision/observation/peer utterance identifiers are
  recorded in `response_latency` events.

The opener and decisions with no delivered audio
are excluded from these averages. An empty group has `count: 0` and `mean_ms: null`.
Episode results contain the same statistics. `03_simulation.json` aggregates
completed episodes using measurement counts, rather than averaging episode means.

## Prompts and evaluation

The runtime loads two templates from `talktopia/full_duplex/prompts/`:
`simulation_action_general_v1.txt` for opening and ordinary responses, and
`simulation_action_FDB_v1.txt` for non-final sentence decisions. The general
template preserves the body of `sotopia_action_v1.txt`, including the participant's
freedom to leave, and retains the 40-word and recipient instructions. It contains
no backchannel guidance. The FDB template keeps SOTOPIA's role, social goal,
naturalness, history, available-action list, and JSON output structure, adding
only listening and backchannel instructions. It does not instruct the listener
to generate spoken text or choose a backchannel sound by meaning. Both templates
receive the same canonical history and role context. Their versions and the
sentence-based trigger policy are saved in run settings and the start event.
One request generates `action_type`, `argument`, and `to` together. The runtime
validates the current observation and floor before allowing TTS to start, and
checks validity again before delivery. `HiddenSaid` records that same response's
private speech text; it is not a second LLM request.

Action validation retains required fields/types, the current action mask, and
nonblank speech with at most 50 whitespace-separated words. Extra JSON fields are
ignored. A control action's argument is ignored. The model's recipient names do
not route speech: in this two-agent simulation, audio always goes to the peer.

There are no content filters for controller-related phrases, non-Latin letters,
compact currency, emphasis, or brackets. The prompt still asks for spoken dialogue
in English, but formatting does not fail an episode. Before TTS, only `*` symbols
and redundant whitespace are removed; enclosed words are preserved. Parenthesized
text and stage directions are not guessed away. OmniVoice backchannel tags remain
intact. `HiddenSaid` preserves the generated words and sentence chunks record the
actual synthesis input. Currency and contractions are not expanded.

For Qwen3.5:9b structured requests with reasoning disabled, the Ollama proxy uses
the native raw generation endpoint with the verified Qwen chat template and an
empty thinking prefix. This avoids a locally reproduced Ollama 0.31.1 issue where
chat requests ignored the supplied JSON schema. The single model call, messages,
schema, token limit, and OpenAI sampling options are preserved. Other models and
unsupported request shapes retain their previous routes. This transport fix also
applies to round-robin Qwen requests using this proxy; historical frozen runs are
unchanged, but newly generated outputs may differ.

Generation never calls another LLM to repair an invalid response. A failed general
response records the original output and actual error and fails the attempt
without manufacturing a `none` action. A failed listening decision records the
error and skips that backchannel opportunity. Completed or failed attempts retain
their existing event/raw-response audit. Legacy repair logs remain readable.

Full-duplex episodes have one attempt; failures are not automatically restarted.
Only transient LLM/ASR/TTS transport errors get one request-level retry. Empty TTS
retains its existing one retry with a different deterministic seed, without a
nested HTTP retry for that same error. Invalid PCM or inconsistent internal state
still fails. External cancellation propagates.

An empty final ASR transcript is allowed when audio was actually delivered. Its
empty text, sentence timestamps, and PCM evidence are kept in agent/evaluation
history; generated text is never substituted for missing recognition.

Frozen run settings and start events record `validation_policy: essential_v1`,
`output_repair_policy: disabled`, and `episode_max_attempts: 1`. Old configurations
cannot silently resume under these rules. Round-robin and evaluation retain their
existing behavior, and the shared engine is unchanged.

The shared speech API uses `without_timestamps=True` when decoding Whisper text.
In a recorded 3.63-second utterance, timestamp-token decoding reproducibly added
unrelated repeated text; the same PCM and VAD preprocessing decoded correctly
with only this option changed. This reduces that observed failure without
guaranteeing error-free ASR. Silero VAD now checks only whether speech is present;
Whisper receives the complete resampled PCM, including leading padding and pauses.
In a separate recorded two-sentence utterance, the former VAD trimming caused
Whisper to return only the first sentence in eight replays. Keeping the original
waveform restored both sentences in all eight, without changing the decoder or
providing reference text. VAD detected both sentences; the words were lost during
decoding, not removed by VAD. This does not guarantee error-free recognition.
The ASR model, sampling, and full-utterance recognition policy remain unchanged.
Round-robin and Surface5 both record these settings under
`experiment.asr_decoding`; speech protocol v5 and health flags prevent reuse of
an older server. Future runs in both modes use this shared policy; previous runs
used different preprocessing and should not be treated as identical conditions.
All evaluation timestamps still come from
measured audio delivery, not the ASR decoder. Historical recordings and
transcripts are not rewritten.

The agent history contains the initial SOTOPIA observation and every subsequent
canonical observation, without an eight-entry limit. Both participants see the
confirmed ASR text, including for their own earlier utterances. Private generated
speech is kept in artifacts, not substituted for ASR in that history. Only the
current completed-sentence ASR observation is appended to the listening request.
Backchannel responses have an empty model argument. After runtime validation,
the runtime selects uniformly from `yeah`, `[confirmation-en]`, `Uh-huh`,
`Mm-hmm`, and `Yep`, independently of their meanings. The selected item is TTS
input, not recognized speech. Backchannels with `[confirmation-en]` use
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
