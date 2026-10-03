# EXP-001 — Audio Pipeline Foundation

**Status:** all modules implemented. 7 of 8 criteria measured on synthetic fixtures
(7 pass, 0 fail, 1 pending). Criterion 7 needs one real Tamil recording, which does
not exist yet; until then the criteria have **not** been verified on real speech
and EXP-001 is **not** accepted.

## Hypothesis

A single canonical audio path — load → resample to 16 kHz mono → quality analysis
→ VAD → 80-bin log-Mel — can be implemented as small independent modules, each
testable in isolation, and can be verified end to end on at least one real Tamil
recording before any model work begins.

## Why this comes before ASR

GUIDE section 31: before training on any real dataset, the model must be able to
overfit 10-30 minutes of speech. If it cannot, the fault is almost always in this
pipeline — audio loading, sample rate, features, tokenizer, padding, CTC lengths,
blank handling, loss, or decoder — not in the amount of data. Building this path
first is what makes those failures diagnosable.

## Scope

```text
src/tamil_voice/audio/io.py             load_audio, validation
src/tamil_voice/audio/resampling.py     resample_audio
src/tamil_voice/audio/normalization.py  peak/rms analysis, clipping, optional gain
src/tamil_voice/audio/features.py       STFT wrapper, mel filterbank, log-mel
src/tamil_voice/audio/quality.py        quality report
src/tamil_voice/vad/detector.py         energy + spectral + hangover
src/tamil_voice/vad/postprocess.py      segment merge, padding, minimum duration
```

Explicitly out of scope: neural VAD, speech enhancement, any model, any training.

## Acceptance criteria

The pipeline is accepted only when all of the following hold, measured on a real
Tamil recording:

1. A 48 kHz stereo file and a 22.05 kHz mono file both load and converge on the
   same 16 kHz mono representation.
2. 8 / 22.05 / 44.1 / 48 kHz inputs all resample correctly, and 16 kHz input is
   passed through untouched.
3. Resampling a sine wave preserves its dominant frequency to within a stated
   tolerance.
4. Corrupt, empty, NaN, infinite, clipped and near-silent files are detected and
   reported, not silently accepted.
5. Mel features have shape `(n_frames, 80)` with `n_frames` consistent with
   frame length 25 ms and hop 10 ms.
6. VAD segments cover the spoken regions of the fixture and nothing else, to a
   stated tolerance.
7. Waveform, spectrogram, mel spectrogram and VAD regions are plotted.
8. Every module has unit tests. `pytest`, `ruff` and `mypy` all pass.

## Expected result

All eight criteria met. Plots land in `artifacts/plots/`.

## Dataset

None required for the unit-test criteria. Criterion 7 onward needs one real Tamil
recording placed in `data/raw/speech/`, outside version control.

Which corpus to draw that first recording from is still open — see
`MEMORY.md` section 6.

## Files

```text
README.md            this file
config.yaml          fixture and tolerance parameters, read by verify_criteria.py
verify_criteria.py   runs the criteria and writes results.json
results.json         measured 2026-10-03 on synthetic fixtures; criterion 7 pending
notes.md             what worked, what failed, why, what to test next
```

`results.json` records only what was measured. Criteria 1-6 and 8 were executed on
synthetic fixtures by `verify_criteria.py`; criterion 7 is marked `pending` because
it needs a real Tamil recording. Every number in the file came from that run — none
is a placeholder. The criteria must be re-run on real speech before EXP-001 is
accepted.