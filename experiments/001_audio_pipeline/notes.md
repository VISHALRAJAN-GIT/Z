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
- `vad/detector.py`: a non-neural VAD — energy above a percentile noise floor, a
  spectral-flatness gate, median smoothing, hangover and short-run removal, all on
  one STFT grid so the energy and spectral views share frames. On a tone embedded
  in silence it detects the tone and not the silence; embedded broadband noise is
  rejected by the flatness gate (measured: median flatness ~0.56 for white noise
  versus ~0.00 for a 300 Hz tone).
- `vad/postprocess.py`: `build_segments` runs merge -> pad -> clamp -> re-merge ->
  drop-short in that documented order, turning the frame mask into speech spans.
- 33 new tests across the two VAD modules (193 total), covering hangover,
  median smoothing, short-run removal, detected-in-place tones, noise rejection,
  flatness separation, config validation, and the segment pipeline.

## What failed

- `validate_audio` divided by `sample_rate` to compute duration before checking
  that the rate was positive, raising `ZeroDivisionError` on a 0 Hz input. Caught
  by `test_validate_zero_sample_rate`; fixed by returning early when the rate is
  non-positive, since every rate-dependent metric is meaningless there.
- A percentile noise floor cannot see a clip that is loud from start to finish:
  with no quiet frames the floor is estimated at the signal level, so a steady
  tone or continuous noise yields little or no speech. This is inherent to the
  method, not a bug; it is recorded in `MEMORY.md` as a candidate improvement, and
  the VAD tests use signals embedded in silence, which is the realistic case.

## Why

`load_audio` and `resampling.py` are deliberately separate: nothing in the load
path changes the sample rate, so "what the file contained" stays separable from
"what we did to it". This is also what lets the 16 kHz no-op be a hard guarantee
rather than a hope.

## What should be tested next

All modules of EXP-001 now exist. Next is to run the eight acceptance criteria in
`README.md` and record the measured numbers in `results.json`; criterion 7 needs
one real Tamil recording. Do not start EXP-002 until every criterion is measured
and recorded.

## Note on ordering

`features.py` was built before `quality.py` even though the module list orders
them the other way. Quality's spectral statistics need an STFT, and GUIDE section
15 places the STFT wrapper in `features.py`; implementing a second STFT inside
`quality.py` would have duplicated the framing and windowing logic.