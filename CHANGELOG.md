# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
semantic versioning once behaviour is public.

The record of record for engineering decisions is `MEMORY.md`. This file records
what changed.

## [Unreleased]

Phase 01 / EXP-001, in progress. No acceptance criterion has been measured yet.

### Added

- `src/tamil_voice/audio/io.py` — `AudioData` container; `load_audio` (reads at
  the file's native rate, never resamples); `audio_info` header probe;
  `validate_audio` with `ValidationReport` / `ValidationLimits` and typed issues
  (empty, NaN, infinite, unsupported rate, too long, clipped, low amplitude,
  silent, DC offset); `AudioLoadError` / `AudioValidationError`.
- `src/tamil_voice/audio/resampling.py` — `resample_waveform`, `resample_audio`,
  `resample_to_canonical`. Converts 8/22.05/44.1/48 kHz to 16 kHz; already-16-kHz
  input is returned untouched.
- Tests: `test_audio_io.py` (28) and `test_audio_resampling.py` (17).

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