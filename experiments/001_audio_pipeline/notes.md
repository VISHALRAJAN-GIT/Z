# notes.md

## What worked

- `audio/io.py`: `load_audio` reads any soundfile-supported format at its native
  rate and never resamples. `validate_audio` reports typed issues (empty,
  NaN/Inf, unsupported rate, too long, clipping, low amplitude, silence, DC
  offset) with a `ValidationReport`; clipping is a warning, the rest are errors.
- `audio/resampling.py`: 8/22.05/44.1/48 kHz -> 16 kHz via `librosa.resample`
  (soxr). A 440 Hz tone survives every conversion to within a couple of Hz, and
  already-16-kHz input is returned as the same object with no re-filtering.
- `audio/normalization.py`: read-only `analyze_loudness` reporting peak, RMS, DC
  offset, crest factor and clipping, in linear and dBFS units, and classifying the
  recording as too_quiet / normal / too_loud / clipped. Gain is opt-in only
  (`apply_gain`, `normalize_peak`, `normalize_rms`); nothing normalizes implicitly.
- `audio/features.py`: an own torch STFT wrapper (time-major `(frames, freqs)`)
  exposing magnitude, phase and power, plus a unit-peak mel filterbank and
  `log_mel_spectrogram` at `(frames, 80)`. The filterbank matches
  `librosa.filters.mel(htk=True, norm=None)` to 1e-5, and the STFT round-trips
  through `torch.istft` to 1e-4.
- `audio/quality.py`: `analyze_quality` aggregating the level metrics with an
  estimated SNR, silence ratio, zero-crossing rate and spectral
  centroid/bandwidth/rolloff/flatness.
- 114 new tests across the five modules (160 total), covering mono/stereo, downmix,
  corrupt/missing/empty files, NaN/Inf, clipping, near-silence, unsupported rates,
  over-long input, dBFS conversions, classification boundaries, gain, STFT geometry,
  mel correctness, and the quality report.

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

`vad/detector.py` (energy + spectral + hangover) and `vad/postprocess.py`
(segment merge, padding, minimum duration). Then run all eight acceptance
criteria in `README.md`, which needs one real Tamil recording for criterion 7.
Do not start EXP-002 until every criterion is measured and recorded.

## Note on ordering

`features.py` was built before `quality.py` even though the module list orders
them the other way. Quality's spectral statistics need an STFT, and GUIDE section
15 places the STFT wrapper in `features.py`; implementing a second STFT inside
`quality.py` would have duplicated the framing and windowing logic.