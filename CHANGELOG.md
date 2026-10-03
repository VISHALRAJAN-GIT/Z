# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
semantic versioning once behaviour is public.

The record of record for engineering decisions is `MEMORY.md`. This file records
what changed.

## [Unreleased]

Phase 01 / EXP-001, complete and verified. `verify_criteria.py` reported 8 pass,
0 fail, 0 pending: criteria 1-6 and 8 on synthetic fixtures with known ground
truth, criterion 7 and the `real_recording` section on the supplied real Tamil
recording `data/raw/speech/Tamil voice sample.mp3` (48 kHz stereo, 47.49 s).

### Data

- Obtained the first ASR corpus: **IISc-MILE Tamil ASR Corpus** (OpenSLR SLR127,
  **CC BY 2.0**) — 89,401 utterances (77,314 train / 12,087 test), ~150 h read
  speech, 16 kHz mono PCM, 531 speakers. Stored at `data/raw/iisc_mile_ta/`
  (outside git, 16.125 GB). No manifest or speaker-disjoint split exists yet.
- `docs/02_data/licensing.md` — the dataset licensing register, first entry
  `iisc-mile-ta` with license, source, retrieval date, permitted use and attribution.
- `docs/02_data/dataset_sources.md` — where the corpus was obtained (Kaggle mirror)
  and the authoritative OpenSLR SLR127 fallback.

### Added

- `src/tamil_voice/audio/io.py` — `AudioData` container; `load_audio` (reads at
  the file's native rate, never resamples); `audio_info` header probe;
  `validate_audio` with `ValidationReport` / `ValidationLimits` and typed issues
  (empty, NaN, infinite, unsupported rate, too long, clipped, low amplitude,
  silent, DC offset); `AudioLoadError` / `AudioValidationError`.
- `src/tamil_voice/audio/resampling.py` — `resample_waveform`, `resample_audio`,
  `resample_to_canonical`. Converts 8/22.05/44.1/48 kHz to 16 kHz; already-16-kHz
  input is returned untouched.
- `src/tamil_voice/audio/normalization.py` — `analyze_loudness` / `LoudnessReport`
  (peak, RMS, DC offset, crest factor, clipping, in linear and dBFS) and the
  too_quiet / normal / too_loud / clipped classification. Gain is opt-in:
  `apply_gain`, `normalize_peak`, `normalize_rms`. Nothing is normalized
  automatically.
- `src/tamil_voice/audio/features.py` — torch STFT wrapper (`stft`,
  `magnitude_spectrogram`, `phase_spectrogram`, `power_spectrogram`), a unit-peak
  `mel_filterbank`, and `mel_spectrogram` / `log_mel_spectrogram` at 80 bins.
  Spectrograms are time-major `(frames, freqs)`.
- `src/tamil_voice/audio/quality.py` — `analyze_quality` / `QualityReport`:
  duration, level metrics, *estimated* SNR, silence ratio, zero-crossing rate and
  spectral centroid / bandwidth / roll-off / flatness.
- `src/tamil_voice/vad/detector.py` — `VadConfig`, `VadResult`, `detect_speech`:
  non-neural VAD combining energy above a percentile noise floor, a spectral
  flatness gate, median smoothing, hangover and short-run removal. Energy and
  spectral cues share one STFT grid; the energy level is relative, not dBFS.
- `src/tamil_voice/vad/postprocess.py` — `Segment`, `SegmentConfig` and
  `frames_to_segments` / `merge_segments` / `pad_segments` /
  `filter_short_segments` / `build_segments` (merge, pad, clamp, re-merge, drop
  segments shorter than the minimum).
- Tests: `test_audio_io.py` (28), `test_audio_resampling.py` (17),
  `test_audio_normalization.py` (25), `test_audio_features.py` (30),
  `test_audio_quality.py` (14), `test_vad_detector.py` (19) and
  `test_vad_postprocess.py` (14).
- `experiments/001_audio_pipeline/verify_criteria.py` — runs the EXP-001
  acceptance criteria and writes measured numbers to `results.json`; `config.yaml`
  holds the fixture and tolerance parameters. Criteria 1-6 and 8 use synthetic
  fixtures; criterion 7 and the `real_recording` section use the real Tamil
  recording. Current run: 8 pass, 0 fail, 0 pending.

### Fixed

- `validate_audio` raised `ZeroDivisionError` on a non-positive sample rate; it
  now returns the rate error before computing any rate-dependent metric.

## [0.1.0] — Phase 00, Project Engineering

### Added

- Repository skeleton per `GUIDE.MD`: `src/`, `training/`, `evaluation/`,
  `data/`, `configs/`, `experiments/`, `checkpoints/`, `artifacts/`, `scripts/`,
  `tests/`, `deployment/`, `docs/`, `notebooks/`.
- `src/tamil_voice.common.config` — project root and directory resolution with
  `TVF_*` environment overrides, YAML loading with real type errors, and the
  canonical audio spec: mono, 16 kHz, float32.
- `src/tamil_voice.common.logging` — JSON-lines logging on stderr. One logger for
  the whole project, idempotent setup, level from `TVF_LOG_LEVEL`.
- `src/tamil_voice.common.seed` — `seed_everything` across Python, NumPy and
  torch, plus a DataLoader worker initialiser. Reports what was actually seeded;
  when torch is absent it does not claim determinism it does not have.
- Test suite: 46 tests covering canonical audio constants, path resolution,
  YAML errors, logging format and idempotency, reproducibility, and package
  structure.
- `README.md`, `AGENTS.md`, `MEMORY.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `LICENSE`.
- `requirements.txt`, `requirements-dev.txt`, `.env.example`, `.gitignore`,
  `.gitattributes`.
- `experiments/001_audio_pipeline/` with acceptance criteria and an explicit
  record of what has *not* been measured.

### Configuration

- pytest, Ruff and mypy configured. All three are required to pass before a
  commit.
- CPU-first dependencies. torch and torchaudio are an optional extra, installable
  from an explicitly chosen wheel index.

### Known gaps

- torch is not installed yet. Phase 01 needs it and the wheel index must be
  chosen deliberately.
- No audio module exists. No dataset exists. No model exists.