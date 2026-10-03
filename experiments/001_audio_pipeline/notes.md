# notes.md

## What worked

- `audio/io.py`: `load_audio` reads any soundfile-supported format at its native
  rate and never resamples. `validate_audio` reports typed issues (empty,
  NaN/Inf, unsupported rate, too long, clipping, low amplitude, silence, DC
  offset) with a `ValidationReport`; clipping is a warning, the rest are errors.
- `audio/resampling.py`: 8/22.05/44.1/48 kHz -> 16 kHz via `librosa.resample`
  (soxr). A 440 Hz tone survives every conversion to within a couple of Hz, and
  already-16-kHz input is returned as the same object with no re-filtering.
- 45 new tests (91 total), covering mono/stereo, downmix, corrupt/missing/empty
  files, NaN/Inf, clipping, near-silence, unsupported rates and over-long input.

## What failed

- `validate_audio` divided by `sample_rate` to compute duration before checking
  that the rate was positive, raising `ZeroDivisionError` on a 0 Hz input. Caught
  by `test_validate_zero_sample_rate`; fixed by returning early when the rate is
  non-positive, since every rate-dependent metric is meaningless there.

## Why

`load_audio` and `resampling.py` are deliberately separate: nothing in the load
path changes the sample rate, so "what the file contained" stays separable from
"what we did to it". This is also what lets the 16 kHz no-op be a hard guarantee
rather than a hope.

## What should be tested next

`audio/normalization.py` (report-only loudness), then `audio/quality.py`. Then
`audio/features.py`, `vad/`. Full acceptance criteria are in `README.md`. Do not
start EXP-002 until every criterion is measured and recorded.