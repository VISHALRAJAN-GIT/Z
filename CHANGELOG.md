# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
semantic versioning once behaviour is public.

The record of record for engineering decisions is `MEMORY.md`. This file records
what changed.

## [Unreleased]

### Research - EXP-005 complete: character targets overfit, and the whitespace default is settled

`experiments/005_character_ctc_overfit/` trains the EXP-003 model on character
targets from `src/tamil_voice/text/tokenizer.py`, with architecture, features,
subset, hyperparameters, seed and device all held identical to EXP-003 so the two
experiments are directly comparable. Both whitespace variants ran. cuda, 200 epochs.

| | with_space | without_space | EXP-003 word |
|---|---:|---:|---:|
| real symbols | 48 | 47 | 1360 words |
| parameters | 668818 | 668561 | 1006002 |
| rows violating frames >= labels | 12 | 6 | 0 |
| rows trained | 649 | 655 | 661 |
| train loss | 0.002765 | 0.000690 | 0.094745 |
| token error rate | 0.000190 | 0.0000674 | 0.038647 |
| exact match rate | 0.995378 | 0.998473 | 0.895613 |
| word error rate | 0.001647 | not computable | not computable |
| acceptance | passed | passed | passed |

- The tokenizer is proven usable as CTC targets end to end. Character TER 0.000190
  over 15822 target labels is about 3 wrong characters in the whole subset.
- **First WER in the project: 0.001647** — on the training subset, so it measures
  memorisation, not generalisation. `dev` and `test` were never opened.
- `include_space=True` is confirmed on measurement rather than argument. Excluding
  whitespace gives a lower token error rate and half the invalid rows, and in
  exchange word error rate becomes undefined, which EXP-005 verified empirically.
- The 5 corrupt `MILE_0000289` rows **cannot be trained on at all** at character
  targets. Character targets are what exposed this: word targets fit inside 7-12
  encoder frames. This makes the `dataset_v002` disposition blocking, not tidying.
- The comparison is not a clean A/B: the variants train on different row sets
  (649 vs 655) because CTC validity forces it. The token error rate gap is
  confounded and is not claimed as a controlled result.
- Added `train_char_overfit.py`, `config.yaml`, `results.json`, `notes.md` and the
  two saved tokenizer artifacts.

### Added - the character tokenizer that EXP-004 decided on

The `text` package, which has been an empty placeholder since Phase 00, now holds
the implementation the EXP-004 decision was waiting for.

- `src/tamil_voice/text/unicode.py` — `normalize_text` / `is_normalized` over
  NFC/NFD/NFKC/NFKD, `Script` with `script_of` / `scripts_in` / `script_histogram`,
  `UnicodeError`. Script classification exists so a future corpus containing Latin
  or digits is a measurement rather than a silent distribution shift.
- `src/tamil_voice/text/tokenizer.py` — `TokenizerConfig`, `CharacterTokenizer`
  (`build`, `encode`, `encode_with_report`, `decode`, `decode_words`,
  `unknown_rate`, `save`, `load`), `build_from_jsonl`, `sequence_length_report`,
  `prepare_text`, `TokenizerError`.
- `tests/unit/test_text_unicode.py`, `tests/unit/test_text_tokenizer.py` — 85 new
  tests. pytest 220 -> 305.

Design points that are decisions rather than defaults:

- **The inventory is derived from data.** Nothing carries a 48-symbol table,
  because `dataset_v001` contains no Latin letters and no ASCII digits and so
  exercises none of what GUIDE section 28 requires.
- **Whitespace is kept by default**, deviating from the EXP-004 headline in favour
  of being able to compute WER at all. The space-free variant is one flag away and
  the measured cost of keeping spaces is 24 utterances out of 89401.
- **Unseen codepoints become `<unk>`, never dropped** — dropping shortens the
  target and could satisfy CTC's frames >= labels constraint for the wrong reason.
- **Text preparation has one definition**, shared by building and encoding, so the
  two cannot drift apart.

### Fixed - EXP-004 miscounted its own violations

The speaking-rate audit said 6 of the 9 stride-4 CTC violations were corrupt rows
belonging to speaker `0000289`, and listed 3 as legitimate fast speech. Re-checked
against `vocabulary_measurements.json`: **5** rows are corrupt
(`MILE_0000289_0000067`..`_0000071`, 98.9–107.4 chars/s) and **4** are legitimate.
`MILE_0000232_0000013` was filed under the corrupt cluster but belongs to a
different speaker and runs 196 characters over 3.98 s — fast, not impossible. The
corpus-wide count of 6 utterances above 30 chars/s is unchanged and is what the
sixth row actually is. Corrected in `experiments/004_character_tokenizer/README.md`,
`notes.md`, `MEMORY.md` and `CHANGELOG.md`.

### Research - EXP-004 complete: the tokenizer scheme is decided from measurement

Character-level, chosen by measuring the alternatives rather than by preference.
`experiments/004_character_tokenizer/` produces every number below.

| Option | Vocab | dev OOV | test OOV | fc params | fc MB fp32 |
|---|---:|---:|---:|---:|---:|
| **character** | **48** | **0.000000** | **0.000000** | **12 593** | **0.05** |
| word top 10 000 | 10 000 | 0.3000 | 0.3058 | 2 570 257 | 10.28 |
| word full | 138 047 | 0.1378 | 0.1438 | 35 478 336 | 141.91 |

Three findings that were not predictable in advance:

- **Restricting the word vocabulary does not fix OOV.** The dev OOV curve against
  most-frequent-N is nearly flat at the top: N=100 000 gives 0.1570, barely
  better than the full vocabulary's 0.1378. The top-N strategy is refuted.
- **The character inventory is 48 codepoints and closed** - 47 in the Tamil block
  plus the space - with 0 unseen characters on dev and 0 on test.
- **Encoder stride 4 is now a hard architectural constraint.** Utterances where
  encoder frames fall below the character label count, out of 89 401: stride 4
  gives 9, stride 8 gives **35 148**, stride 16 gives 88 458. Stride 8 is the
  usual conv CTC choice and it invalidates 39 % of this corpus. Mean utterance is
  69.41 characters over 6.07 s, i.e. 11.9 characters per second.

Also measured: NFC changes 0 of 89 401 transcripts, so the corpus is already NFC;
normalization is still required, since that is a property of this corpus and not
of Tamil.

### Added

- `experiments/004_character_tokenizer/measure_vocabulary.py`,
  `vocabulary_measurements.json`, `README.md`, `notes.md`. The script decides
  nothing and asserts no verdict; it emits the numbers the decision rests on.
- A speaking-rate audit that separates two populations hiding in the 9 CTC
  violations: **5 rows are corrupt** (speaker `0000289`, 26–47 characters in
  0.24–0.45 s, i.e. 98.9–107.4 characters per second, which no human can produce),
  while **4 are legitimate fast speech** — `MILE_0000232_0000013` at 49.2
  characters per second over 3.98 s, and three more at 26.1–26.6 characters per
  second — and must be kept. All 9 are among the 200 fastest utterances in the
  corpus; only 6 utterances corpus-wide exceed 30 characters per second against a
  median of 11.9.

### Known gaps

- **Five corrupt rows are identified but not dispositioned.** No row was removed.
  If any are dropped that creates `dataset_v002`, with the reason and utterance
  ids recorded.
- **BPE is unmeasured.** GUIDE section 28 says to benchmark it rather than assume
  it, and EXP-004 does neither. Character-level is the specified starting point,
  so this does not block, and no claim is made about BPE in either direction.
- **`dataset_v001` contains no ASCII digits and no Latin letters.** Numbers are
  written in Tamil script. GUIDE section 28 requires English, digit and
  punctuation support in the tokenizer and this corpus exercises none of it, so
  the 48-symbol inventory must not be hardcoded.

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