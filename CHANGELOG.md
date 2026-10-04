# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
semantic versioning once behaviour is public.

The record of record for engineering decisions is `MEMORY.md`. This file records
what changed.

## [Unreleased]

### Model - EXP-003 complete: tokenizer measured and tiny CTC overfit test accepted

Both steps of Phase 02's first experiment are done. The overfit test is the
project's first model of any kind. Its metrics are on its own training subset;
**no WER has ever been measured**.

Step 1, `measure_tokenizer.py` (committed earlier as `dab1687`):

- Word-level whitespace tokenisation over `dataset_v001`.
- Train vocabulary 138047 words over 676524 tokens (4.9 tokens per type).
- OOV against train: dev 13.78 %, test 14.38 %, train 0 %.
- 15.01 min subset of 661 utterances selected shortest-first.

Step 2, `train_overfit.py` (this change):

- `Conv1d(80->32->64, stride 2, no BatchNorm) -> GRU(256, 2, unidirectional) ->
  Linear`, AdamW, 200 epochs, `train.jsonl` only. 1006002 parameters.
- Measured: `train_loss` 0.0947, `token_error_rate` 0.0386, `exact_match_rate`
  0.8956, 271.3 s on one RTX 2050. Acceptance PASSED against thresholds that were
  written into `config.yaml` before the run.
- `dev.jsonl` and `test.jsonl` were never opened by the training script.
- Checkpoints go to the gitignored `checkpoints/exp003_overfit/`.

### Added

- `experiments/003_tokenizer_baseline/train_overfit.py`, `config.yaml`,
  `notes.md`, `subset_vocab.json`, `results.json`. `config.yaml` now supplies
  every model, feature, data and training value; nothing is hardcoded in the
  script, and it holds the acceptance thresholds.
- `verify_length_math()` checks the encoder-length arithmetic against what
  `nn.Conv1d` actually returns, on a padded batch and on single utterances at
  both ends of the length range. A runtime assertion refuses to start if any
  subset row has fewer encoder frames than labels.
- `notes.md` records the full research history, including the first failed run.

### Fixed

Three bugs in the first draft of the training script, each found by running it
rather than by reading it:

- CTC targets were the padded `(B, S)` label block flattened, 40 entries where
  `sum(target_lengths)` was 26. Beyond the shape mismatch, `CTCLoss` counts blank
  as a target symbol, so the padding would have been trained on as if it were
  text. Now masked to `sum(target_lengths)` before flattening.
- Encoder length used `floor(T / 4)` and was wrong on **497 of 661** subset rows;
  two stride-2 convolutions give `ceil(ceil(T/2)/2)`. Replaced with exact
  per-layer arithmetic plus the verification above.
- The output layer was sized for the full vocabulary. At the measured 138047-word
  train vocabulary the linear layer alone is **35478593** parameters at hidden
  256, so "tiny" was never true and the project target of < 100 MB of weights is
  unreachable word-level at that size. The overfit test now builds its
  1360-word vocabulary from the subset and trains a real 1.0M-parameter model,
  and `results.json` records the full-vocab figure rather than dropping it.

### Changed

- `MEMORY.md`: corrected the torch record. It said `2.14.1+cpu` with CUDA
  unavailable; the actual install is **`torch 2.6.0+cu124`** /
  **`torchaudio 2.6.0+cu124`** with `torch.cuda.is_available()` returning `True`.
  The earlier statement was true when written and is now false, so it was
  replaced rather than appended.
- `MEMORY.md` open question 2 ("is word-level adequate?") is **answered: no, not
  as-is**, given 13.78 % dev OOV. It is replaced by the actual open decision,
  which is what to do instead, measured rather than assumed.
- The overfit experiment runs with `dropout: 0.0`. Measured: at 0.1 the run
  stalled at token error rate 0.3838 and failed acceptance. Regularisation's only
  effect on a memorisation test is to prevent memorisation.
- `MEMORY.md` section 1 no longer claims "no tokenizer exists", which stopped
  being true at `dab1687`.

### Known gaps

- The vocabulary scheme for the real baseline is undecided and now blocks it.
  Word-level is measured inadequate at 13.78 % dev OOV; the OOV-versus-N curve
  and the character inventory have not been measured yet.
- The overfit subset spans 0.243 s to 1.59 s against a corpus maximum of
  38.85 s, so the pipeline check has not covered long or fast speech.

### Data — EXP-002 complete and verified

`dataset_v001` built over the full IISc-MILE corpus and independently verified:
**14 pass, 0 fail**.

- `data/manifests/dataset_v001/{train,dev,test}.jsonl` + `metadata.json`, produced
  by `experiments/002_data_split/build_manifests.py` at ratios 0.90/0.05/0.05, seed
  `20261003`. Manifests are committed; no audio is.
- Measured split: train 80304 utts / 476 speakers / 135.4202 h, dev 4526 / 26 /
  7.3988 h, test 4571 / 29 / 7.2812 h, total 89401 / 531 / 150.1002 h.
- Speaker overlap 0 for all three pairs. 0 empty transcripts, 0 duplicate utterance
  ids, 0 absolute paths, 0 missing audio files of 89401, 16 kHz throughout,
  durations 0.2427 s to 38.8509 s.
- Built twice and compared by SHA-256: `train.jsonl`, `dev.jsonl`, `test.jsonl` and
  `metadata.json` are byte-identical across runs.

### Added

- `experiments/002_data_split/verify_manifests.py` — independent verifier. Re-reads
  the written JSONL and re-derives every claim in `results.json` from the bytes and
  from real file headers, including a 2000-utterance duration sample. Prints
  `14 pass, 0 fail`; exit 0 on success.
- `experiments/002_data_split/README.md` and `notes.md` — hypothesis, scope,
  measured results, acceptance table, the reproducibility check, and an honest
  record of what went wrong on the way.
- `HEADER_READ_WORKERS = 8` in `src/tamil_voice/data/manifest.py`, with the measured
  numbers behind it in the docstring.

### Fixed

- `build_records` read 89401 audio headers serially and did not finish inside 30
  minutes. Header and transcript reads now run on an 8-thread pool — libsndfile
  releases the GIL, so these are overlapping disk waits. Measured 13.9 ms/file
  single-threaded against 0.27 ms with four workers; eight is on the plateau.
  Results are reassembled in input order before sorting, so output is unchanged,
  which the SHA-256 comparison confirms.

### Session memory

- `../AGENTS.md` and `../START-HERE.md` — the workspace-root entry points. The first
  is loaded automatically by opencode when a session starts one folder up, so a new
  agent is told to read `MEMORY.md` and `git log` before any code is touched.
- `../scripts/update-status.ps1` — regenerates the `AUTO-STATUS` block in both entry
  files from live git output: last commit, branch, ahead/behind `origin/main`,
  working-tree entries, recent commits, last recorded gate results. It also mirrors
  both entry files into `.session/` with the workspace path replaced by
  `<workspace>` so nothing machine-specific is committed.
- `../scripts/run-gates.ps1` — runs pytest, ruff and mypy inside the repo virtual
  environment and records the real result of each to `.session/gates/*.txt`.
- `../scripts/install-hooks.ps1` and `../scripts/hooks/post-commit` — installs the
  git `post-commit` hook that refreshes the status block after every commit. Hooks
  live in `.git` and are not tracked, so they must be reinstalled after a fresh clone.
- `MEMORY.md` section 11 documents the whole mechanism and its accepted failure modes.

### Data layer, committed at 3988e32

- `src/tamil_voice/data/corpus.py` — `Utterance`, `parse_iisc_mile_name`,
  `read_transcript`, `discover_iisc_mile`, `CorpusError`. The prefix is a recording
  condition, not part of the speaker key; the union over prefixes is 531 speakers,
  matching OpenSLR's published count.
- `src/tamil_voice/data/manifest.py` — `ManifestRecord`, `build_records`,
  `write_manifest`, `read_manifest`.
- `src/tamil_voice/data/splits.py` — `SplitRatios`, `SplitConfig`,
  `plan_speaker_split`, `SpeakerSplitPlan`, `check_speaker_disjoint`, `SplitError`.
- `experiments/002_data_split/build_manifests.py` and `config.yaml`.
- `tests/unit/test_data_corpus.py`, `test_data_manifest.py`, `test_data_splits.py`.

The shipped `train/` and `test/` folders are **not** speaker-disjoint and are never
used for splitting. They are recorded on each row as `shipped_split` provenance only,
so filtering a manifest by that field does not give a disjoint set.

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