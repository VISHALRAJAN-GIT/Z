# MEMORY.md

Live project state for the Tamil Voice Foundation. **Read this file first** in any
new session, before touching code. It is updated before every commit so that a new
model resumes with full context instead of starting from zero.

This file is authoritative. Frolic telemetry is secondary and may be rotated.

---

## 1. Current status

**Phase 00 — complete and committed at `35765c4`.
Phase 01 / EXP-001 — complete and verified.
Phase 02 / EXP-002 — complete and verified. `dataset_v001` exists on disk.
Phase 02 / EXP-003 — both steps complete. Tokenizer measured, tiny CTC overfit
test accepted.
Phase 02 / EXP-004 — complete. Character-level decided from measurement, and
`text/unicode.py` + `text/tokenizer.py` now implement it.
Phase 02 / EXP-005 — complete. Character targets overfit to TER 0.000190, first WER
computed (0.001647, training data only), whitespace default confirmed by
measurement.**

Implemented in Phase 01: `audio/io.py` (loading, canonical conversion, real
validation), `audio/resampling.py` (8/22.05/44.1/48 kHz -> 16 kHz, with a
no-op for already-16-kHz input), `audio/normalization.py` (read-only loudness
analysis and classification, plus opt-in gain), `audio/features.py` (STFT wrapper
on torch, mel filterbank, log-mel), `audio/quality.py` (aggregate diagnostic
report), and `vad/detector.py` + `vad/postprocess.py` (non-neural energy +
spectral-flatness VAD, and segment cleanup). All are tested.

EXP-001 is **accepted**: `experiments/001_audio_pipeline/verify_criteria.py`
reported **8 pass, 0 fail, 0 pending**. Criteria 1-6 and 8 ran on synthetic
fixtures with known ground truth; criterion 7 and the `real_recording` section ran
on a real Tamil recording the user supplied, `data/raw/speech/Tamil voice
sample.mp3` (48 kHz stereo, 47.49 s). On it the pipeline produced 16 kHz mono
(759 886 frames, duration preserved), `normal` loudness (peak -14.0 dBFS,
RMS -35.2 dBFS, est. SNR 23.3 dB, 24.6 % silence, no clipping), log-Mel
`(4750, 80)`, and VAD 13 segments / 39.6 s speech (79.3 %). Plots:
`artifacts/plots/exp001_Tamil_voice_sample_overview.png`. The real-recording check
is a behaviour check with statistics — there is no reference transcript or VAD
annotation, so no accuracy figure is claimed.

One dataset has been **obtained** and is now **manifested**: the **IISc-MILE Tamil
ASR Corpus** (OpenSLR SLR127, **CC BY 2.0**), at
`data/raw/iisc_mile_ta/mile_tamil_asr_corpus/` — 89,401 utterances, 531 speakers,
150.1002 h measured, 16 kHz mono PCM, 16.125 GB on disk. License and source are
recorded in `docs/02_data/licensing.md` and `docs/02_data/dataset_sources.md`.

EXP-002 is **accepted**. `experiments/002_data_split/build_manifests.py` produced
`data/manifests/dataset_v001/{train,dev,test}.jsonl` plus `metadata.json`, and
`verify_manifests.py` re-derived every claim from the written bytes:
**14 pass, 0 fail**. Measured split:

```text
split   utterances  speakers     hours
train       80304       476  135.4202
dev          4526        26    7.3988
test         4571        29    7.2812
total       89401       531  150.1002
```

Speaker overlap is 0 for all three pairs. 0 empty transcripts, 0 duplicate
utterance ids, 0 absolute paths, 0 missing audio files of 89401, 16 kHz
everywhere. Durations were re-measured against file headers on a 2000-utterance
sample: 0 mismatches. The manifests were built twice and are byte-identical by
SHA-256. Ratios 0.90/0.05/0.05, seed `20261003`.

**No model has been trained on the full corpus and no WER has ever been
measured.** The only model that has ever run is the EXP-003 overfit test, whose
metrics are on its own training subset. Any statement claiming a WER is false.

Gates, run in `.venv` on Python 3.11.9 and recorded by `scripts/run-gates.ps1` into
`.session/gates/`:

```text
pytest   220 passed
ruff     All checks passed!
mypy     Success: no issues found in 26 source files
```

The exact wall time of the pytest run varies between roughly 10 and 60 s depending
on cache state. The verbatim result of the latest run, with its exit code, is in
`.session/gates/pytest.txt` and is re-recorded by every `run-gates.ps1`.

torch is installed as **`torch 2.6.0+cu124`** with **`torchaudio 2.6.0+cu124`**,
from the cu124 index. `torch.cuda.is_available()` is **`True`** on this machine and
the EXP-003 run trained on the RTX 2050. This corrects the earlier record in this
file, which said `2.14.1+cpu` with CUDA unavailable; that was true when written
and is now false. `seed_everything` still reports `fully_deterministic` from the
Python/NumPy/torch seeding alone — its `cuda_deterministic` field reflects
`cudnn.deterministic = True`, which is requested but not a guarantee of bitwise
reproducibility across GPU kernel selections. Do not overstate it.

Session memory is now backed by committed files plus an automatic refresher. See
section 11.

### EXP-003, in detail

**Step 1, tokenizer** (`experiments/003_tokenizer_baseline/measure_tokenizer.py`,
committed at `dab1687`). Whitespace word-level tokenisation over
`dataset_v001`: train vocabulary **138047** words over 676524 tokens (4.9 tokens
per type). OOV against train: dev **13.78 %**, test **14.38 %**, train 0 %.
A 15.01 min subset of 661 utterances was selected, shortest first.

**Step 2, overfit test** (`train_overfit.py`, this session). Architecture
`Conv1d(80→32→64, stride 2, no BatchNorm) → GRU(256, 2, unidirectional) →
Linear`, AdamW, 200 epochs, train split only, 271.3 s on the RTX 2050:

```text
parameters            1006002
train_loss            0.0947
token_error_rate      0.0386
exact_match_rate      0.8956
acceptance            PASSED
```

These are **training-subset** metrics. dev and test were never opened by that
script. No WER.

Three bugs were caught by running it, all in the first draft, none visible by
reading: CTC targets were the padded `(B, S)` block flattened (40 entries where
`sum(target_lengths)` was 26); encoder length used `floor(T/4)` and was wrong on
**497 of 661** rows; and the full-vocabulary output layer is **35478593**
parameters at hidden 256, so "tiny" was never true at full vocab. A fourth
finding: the first run used `dropout: 0.1` and stalled at token error rate
0.3838, failing acceptance — dropout is regularisation and this step tests
memorisation. At 0.0 it reaches 0.0386.

No new `src/` module was added this session. The CTC decoder and the token error
rate function live in `experiments/` and graduate to `src/` when a real baseline
needs them.

### EXP-004, in detail — the tokenizer scheme is decided

**The vocabulary question is closed: character-level**, chosen from measurement in
`experiments/004_character_tokenizer/measure_vocabulary.py`.

| Option | Vocab | dev OOV | test OOV | fc params | fc MB fp32 |
|---|---:|---:|---:|---:|---:|
| **character** | **48** | **0.000000** | **0.000000** | **12 593** | **0.05** |
| word top 10 000 | 10 000 | 0.3000 | 0.3058 | 2 570 257 | 10.28 |
| word full | 138 047 | 0.1378 | 0.1438 | 35 478 336 | 141.91 |

Three findings that were not predictable in advance:

1. **Restricting the word vocabulary does not fix OOV.** The dev OOV curve against
   most-frequent-N is nearly flat at the top: N=100 000 still gives 0.1570, barely
   better than the full vocabulary's 0.1378. There is no cheap operating point.
   The top-N strategy is refuted by measurement.
2. **The character inventory is 48 codepoints and closed** — 47 in the Tamil
   block plus the space. Unseen characters on dev and test: **0** each.
3. **Encoder stride 4 is now a hard architectural constraint.** CTC needs encoder
   frames >= labels. Utterances violating that, out of 89 401: stride 4 → **9**,
   stride 8 → **35 148**, stride 16 → 88 458. Stride 8, the usual conv CTC choice,
   breaks 39 % of the corpus. Mean utterance is 69.41 characters over 6.07 s =
   11.9 characters per second. **Do not deepen the conv subsampling in the real
   baseline.** A deeper encoder requires a subword or phoneme unit, not a deeper
   stack.

Also measured: NFC changes **0 of 89 401** transcripts, so the corpus is already
NFC (normalize anyway — that is a property of this corpus, not of Tamil). Speaking
rate p50 11.9 / p99 19.1 / max 107.4 characters per second, and only **6**
utterances corpus-wide exceed 30/s.

**Constraint discovered for later phases:** `dataset_v001` contains **no ASCII
digits and no Latin letters at all** — numbers are written in Tamil script. GUIDE
section 28 requires English, digit and punctuation support in the tokenizer, and
this corpus cannot exercise any of it. That must be validated against the Phase 06
corpora, and a tokenizer that hardcodes 48 symbols would be a bug.

### EXP-005, in detail — the tokenizer works, and the whitespace default is now measured

`experiments/005_character_ctc_overfit/` trained the same tiny CTC model on
character targets, using `src/tamil_voice/text/tokenizer.py`. Everything except the
targets is identical to EXP-003 step 2: same architecture, same features, same
shortest-first subset (661 utts, 900.7 s, 148 speakers), same hyperparameters, same
seed, same device (cuda). So the two experiments are directly comparable.

| | with_space | without_space | EXP-003 word |
|---|---:|---:|---:|
| real symbols | 48 | 47 | 1360 words |
| num classes | 50 | 49 | 1362 |
| parameters | 668 818 | 668 561 | 1 006 002 |
| rows violating frames >= labels | **12** | **6** | 0 |
| rows trained | 649 | 655 | 661 |
| train loss | 0.002765 | 0.000690 | 0.094745 |
| token error rate | 0.000190 | **0.0000674** | 0.038647 |
| exact match rate | 0.995378 | 0.998473 | 0.895613 |
| word error rate | **0.001647** | *not computable* | *not computable* |
| acceptance | passed | passed | passed |

Four things this settled:

1. **The tokenizer works as CTC targets.** Character TER 0.000190 over 15 822 target
   labels is about 3 wrong characters in the whole subset. That is all an overfit
   test can establish, and it is established.
2. **The project's first WER exists: 0.001647**, on training data. It is a
   memorisation number and is not a generalisation result. `dev` and `test` were
   never opened.
3. **`include_space=True` is confirmed, on measurement.** Excluding whitespace gives
   a lower token error rate (0.0000674 against 0.000190) and half the invalid rows
   (6 against 12), and in exchange word error rate becomes *undefined* — confirmed
   empirically, not argued. Since the next step exists to produce a WER, the spaces
   stay. This closes the deviation recorded when the tokenizer was written.
4. **The 5 corrupt `MILE_0000289` rows cannot be trained on at all**, at either
   whitespace setting. They are not awkward examples, they are unusable ones.
   Character targets made this visible for the first time; word targets were short
   enough to fit inside 7–12 encoder frames, which is why EXP-003 consumed them
   without ever being blocked. Because the subset is shortest-first, they are the
   first rows selected.

The comparison is **not** a clean A/B: the two variants train on different row sets
(649 against 655) because CTC validity forces it. The token error rate gap is
confounded by the row set and is not claimed as a controlled result. Holding the
row set constant means dropping 6 rows from both, which means `dataset_v002`.

## 2. The project

Offline, low-latency, CPU-first Tamil voice intelligence foundation. Handles
formal and colloquial Tamil, Tanglish and Tamil-English code switching, regional
pronunciation variation, and degraded audio: noisy, reverberant, far-field, phone
microphone, low volume, fast speech.

Built bottom-up. Each stage independently measurable before anything is connected.
Dependency direction is one-way: `notebooks -> scripts -> training/evaluation ->
src/tamil_voice`.

Specification: `C:\ANTI_GRAVITY\PROJECT Z\GUIDE.MD`. It is authoritative and
unmodified.

## 3. Repository

```text
C:\ANTI_GRAVITY\PROJECT Z\tamil-voice-foundation\
```

git repository, branch `main`, remote `origin` =
`https://github.com/VISHALRAJAN-GIT/Z.git`. Initialized 2026-10-03. Pushed:
Phase 00 skeleton, the audio io/resampling work, and a non-destructive merge of
GitHub's placeholder README (its commit remains an ancestor; the project README
was kept).

Structure: `src/tamil_voice/` (9 subpackages + `common`), `training/`,
`evaluation/`, `data/`, `configs/`, `experiments/`, `checkpoints/`,
`artifacts/`, `scripts/`, `tests/`, `deployment/`, `docs/`, `notebooks/`.

Empty directories are preserved with `.gitkeep`. `.venv`, caches, `checkpoints/`,
`artifacts/` and `data/raw|interim|processed|benchmarks` are gitignored.

## 4. What exists in code

| Module | Purpose |
| --- | --- |
| `common/config.py` | project root, directory resolution with `TVF_*` overrides, YAML loading with real type errors, `AudioSpec`, canonical audio constants |
| `common/logging.py` | JSON-lines logging on stderr, one logger, idempotent setup, level from `TVF_LOG_LEVEL` |
| `common/seed.py` | `seed_everything` across Python/NumPy/torch, DataLoader worker init, `SeedReport` |
| `audio/io.py` | `AudioData` container, `load_audio` (no resampling), `audio_info` header probe, `validate_audio` / `ValidationReport` / `ValidationLimits`, `AudioLoadError` / `AudioValidationError` |
| `audio/resampling.py` | `resample_waveform`, `resample_audio`, `resample_to_canonical`; identical-rate input is returned untouched |
| `audio/normalization.py` | `analyze_loudness` / `LoudnessReport` (peak, RMS, DC, crest, clipping, dBFS), classification too_quiet/normal/too_loud/clipped; opt-in `apply_gain`, `normalize_peak`, `normalize_rms` |
| `audio/features.py` | `StftConfig`, `stft` (torch, time-major `(frames, freqs)`), magnitude/phase/power, `mel_filterbank`, `mel_spectrogram`, `log_mel_spectrogram` (80 bins) |
| `audio/quality.py` | `analyze_quality` / `QualityReport`: duration, peak/RMS/crest/clipping (from normalization), estimated SNR, silence ratio, ZCR, spectral centroid/bandwidth/rolloff/flatness |
| `vad/detector.py` | `VadConfig`, `VadResult`, `detect_speech`: energy above a percentile noise floor + spectral flatness gate + median smoothing + hangover + short-run removal, all on one STFT grid |
| `vad/postprocess.py` | `Segment`, `SegmentConfig`, `frames_to_segments`, `merge_segments`, `pad_segments`, `filter_short_segments`, `build_segments` (merge -> pad -> clamp -> re-merge -> drop short) |
| `data/corpus.py` | `Utterance`, `parse_iisc_mile_name`, `read_transcript`, `discover_iisc_mile`, `CorpusError` — turns the corpus directory into typed utterance records |
| `data/manifest.py` | `ManifestRecord`, `build_records`, `write_manifest`, `read_manifest` — JSONL manifests with durations measured from the audio; header reads run on an 8-thread pool because the corpus is 89k files |
| `data/splits.py` | `SplitRatios`, `SplitConfig`, `plan_speaker_split`, `SpeakerSplitPlan`, `check_speaker_disjoint`, `SplitError` — speaker-disjoint split planning with a disjointness assertion |
| `text/unicode.py` | `NormalizationForm`, `SUPPORTED_FORMS`, `normalize_text`, `is_normalized`, `Script`, `script_of`, `scripts_in`, `script_histogram`, `UnicodeError` — NFC normalization and script classification |
| `text/tokenizer.py` | `TokenizerConfig`, `CharacterTokenizer` (build / encode / encode_with_report / decode / decode_words / unknown_rate / save / load), `EncodingReport`, `build_from_jsonl`, `sequence_length_report`, `prepare_text`, `TokenizerError` |

Tests: `tests/unit/test_config.py`, `test_common.py`, `test_package.py`,
`test_audio_io.py`, `test_audio_resampling.py`, `test_audio_normalization.py`,
`test_audio_features.py`, `test_audio_quality.py`, `test_vad_detector.py`,
`test_vad_postprocess.py`, `test_data_corpus.py`, `test_data_manifest.py`,
`test_data_splits.py`, `test_text_unicode.py`, `test_text_tokenizer.py`.

The `text` package has held a tracked empty `__init__.py` placeholder since the
Phase 00 skeleton (`35765c4`). Every subpackage does; there are no PEP 420
namespace packages in `src`.

Canonical audio: mono, 16 kHz, float32 — defined once in `config.py` and imported
everywhere else (`CANONICAL_SAMPLE_RATE`, re-exported by `audio/io.py`).

`load_audio` does not resample: it reads at the file's native rate. Converting to
16 kHz is `resampling.py`'s job, and it is skipped when the input is already 16
kHz. Validation reports problems (empty, NaN/Inf, unsupported rate, too long,
clipping, low amplitude, silence, DC offset) as typed issues; clipping is a
warning and does not prevent loading, while the rest are errors.

## 5. Decisions made

- **Full Phase 00 first.** No Phase 01 modules written before the skeleton is
  committed and gated.
- **Repo in a subfolder.** `GUIDE.MD` stays outside the repo, unmodified.
- **CPU-first.** torch is an optional extra so the audio foundation can be built
  without a multi-gigabyte install. CUDA is deferred until training needs it.
- **torch wheel index is explicit.** Installed from
  `https://download.pytorch.org/whl/cpu` (`torch 2.14.1+cpu`,
  `torchaudio 2.11.0+cpu`). Never the default PyPI index, which would pull a CUDA
  build. cu124 is a separate, later install for the RTX 2050.
- **Loading and resampling are separate modules.** `io.py` never changes the
  sample rate; `resampling.py` never repairs defects. This keeps "what the file
  contained" separable from "what we did to it".
- **Resampling is skipped at 16 kHz.** An already-canonical waveform is returned
  as the same object rather than passed through the interpolator again.
- **The STFT lives in `features.py` and works in torch.** GUIDE section 15 wants
  an own wrapper over tensor ops; section 18's spectral statistics reuse it rather
  than reimplementing an STFT. Spectrograms are time-major `(frames, freqs)` and
  log-mel is `(frames, 80)`, matching EXP-001 criterion 5.
- **Quality SNR is explicitly *estimated*.** There is no reference signal, so the
  noise floor is a low percentile of frame energies and the signal a high
  percentile. It is named `estimated_snr_db` so it is never mistaken for measured
  SNR.
- **VAD is non-neural and shares one STFT grid.** GUIDE section 17 asks for energy
  plus spectral cues before any model exists. Energy and spectral flatness are read
  from the same STFT so frames align; the energy value is a relative windowed-power
  level, not calibrated dBFS, and is used comparatively.
- **Known VAD limitation (documented, not hidden).** Using a percentile noise
  floor means a clip that is loud from start to finish (no quiet frames) has its
  floor estimated at the signal level, so it yields little or no speech. Real
  speech contains pauses; this is acceptable for the first VAD and is a candidate
  improvement, not a silent bug.
- **Acceptance is split by what each criterion can prove.** `verify_criteria.py`
  runs criteria 1-6 and 8 on synthetic fixtures, because those carry known ground
  truth (a resample target, a tone frequency, a planted defect, a known VAD burst);
  a real recording cannot supply an "expected" answer for them. Criterion 7 and the
  `real_recording` section run on the supplied recording. EXP-001 is accepted on
  that basis: 8 pass, 0 fail. The real-recording checks assert behaviour (canonical
  16 kHz mono, duration preserved, feature shape, valid segments, plots written)
  and report statistics; with no transcript or VAD annotation, no accuracy figure
  is claimed. This gap is recorded in section 6.
- **Peak normalization is not automatic.** Loudness carries information. Silence,
  clipping and near-silence are detected and reported instead.
- **Progressive implementation.** Directories now, files when their phase starts.
  A tree of `pass` bodies returning plausible values is the exact failure this
  project already had to undo once; `tests/unit/test_package.py` guards against it.
- **Docs are written per phase.** Folder skeletons exist; documents appear when
  they have real content.
- **docs/ follows GUIDE, not invention.** An 18-folder scheme was drafted and
  rejected in favour of the 10-folder GUIDE structure.
- **Session memory is committed and partially automatic.** `MEMORY.md` stays the
  authoritative narrative. Around it: `AGENTS.md` and `START-HERE.md` in the
  workspace root carry an auto-generated status block written from real git state
  by `scripts/update-status.ps1`, a git `post-commit` hook refreshes that block
  after every commit, `scripts/run-gates.ps1` records real gate output to
  `.session/gates/`, and copies of the two entry files live in `.session/` so git
  history preserves them. The script can only report git facts; it can never know
  what was decided or why, so the narrative stays a manual duty.
- **The shipped IISc-MILE split is ignored for splitting.** The corpus's own
  `train/` and `test/` folders put the same speaker on both sides. Each manifest row
  keeps `shipped_split` as provenance, and `splits.py` never plans from it. Consequence
  to remember: dev and test rows have audio under the shipped `train/` folder, so
  nobody may filter manifests by `shipped_split` and assume disjointness. Disjointness
  lives in which file a row is in.
- **Header reads go through an 8-thread pool.** libsndfile releases the GIL, so the
  89401 `soundfile.info` calls are overlapping disk waits. Measured 13.9 ms/file
  single-threaded against 0.27 ms with four workers; eight is on the plateau and
  sixteen measured no better. Results are reassembled in input order before sorting,
  so scheduling cannot change the output. Without this the first full run did not
  finish in 30 minutes; with it, 375.9 s.
- **Durations come from the audio header, never from file size.** 16-bit PCM at
  16 kHz would have made size inference nearly free and removed the disk cost. A
  shortcut that is right 99.9 % of the time is exactly what silently poisons a
  dataset version later, so `verify_manifests.py` re-measures a sample against real
  headers.
- **The builder's own report is never accepted as proof.** `build_manifests.py`
  prints counts; `verify_manifests.py` re-derives all of them from the written bytes
  and the file headers. Same split as EXP-001: the verifier is a separate program
  that could disagree.
- **Manifest reproducibility is measured, not asserted.** `config.yaml` claims the
  manifests rebuild byte-for-byte, so the build was run twice and compared by
  SHA-256. All four files matched.
- **No manifest filtering.** All 89401 utterances are in `dataset_v001`, including
  the 0.2427 s minimum and the 38.8509 s maximum, with 0 empty transcripts found.
  Dropping outliers is a decision for a measured baseline, not for the builder.
- **The overfit test uses a subset vocabulary, and that is stated, not hidden.**
  The subset holds 1360 distinct words, 0.99 % of the 138047-word train
  vocabulary. The full-vocab linear layer is 35478593 parameters at hidden 256 —
  35x the model that was trained. Rebuilding the vocabulary from the subset is
  what makes this a pipeline test instead of a vocabulary-coverage test, so the
  full-vocab figure is computed and stored in `results.json` rather than dropped.
  Anyone reading only the passing acceptance number must still see it.
- **Acceptance thresholds are written before the run and checked explicitly.**
  `config.yaml` carries `accept_max_token_error_rate` and `accept_max_loss`; the
  script exits non-zero if they are missed. The first run failed acceptance at
  token error rate 0.3838 and said so instead of being retuned until it passed.
- **No BatchNorm and no bidirectional GRU in the CTC baseline.** Both are padding
  correctness, not taste. BatchNorm averages the zero-padded tail into the
  statistics of real frames; a bidirectional GRU reads that tail backwards into
  the last valid frames. With a unidirectional stack and true CTC input lengths,
  zero-padding cannot influence any valid output.
- **Encoder lengths are exact per-layer arithmetic, verified against the real op.**
  `floor(T / 4)` is wrong on 497 of 661 subset rows; two stride-2 convs give
  `ceil(ceil(T/2)/2)`. `verify_length_math()` compares the arithmetic against what
  `nn.Conv1d` actually returns, on a padded batch and on single utterances.
- **CTC runs with `zero_infinity=False`.** The flag is commonly `True`, and it
  would silently swallow the input-shorter-than-target condition this step exists
  to detect. The condition is asserted instead, before training starts.
- **Dropout is 0 for the overfit test.** Regularisation's only effect on a
  memorisation test is to prevent memorisation. This is scoped to this
  experiment, not a claim about the real baseline.
- **The overfit subset stays the shortest-first subset from step 1.** Changing the
  selection rule in step 2 alone would silently decouple the two steps'
  measurements. Its unrepresentativeness is recorded as a limitation instead.
- **The tokenizer is character-level, decided by measurement, not preference.**
  EXP-004 measured the alternative before choosing: word top-N across 11 sizes
  (which refutes the truncation strategy, since the OOV curve is nearly flat at
  the top), the character inventory and its OOV, and the output-layer cost of each
  option. GUIDE section 28 also specifies character-level as the starting point,
  and the measurements do not contradict it.
- **Encoder stride 4 is a measured constraint, not a default.** Character-level
  CTC on this corpus is invalid at stride 8 for 35148 of 89401 utterances. Any
  proposal to deepen the conv subsampling must be measured against that table
  first. This is the finding most likely to be rediscovered the hard way.
- **The character vocabulary is measured, not hardcoded.** 48 codepoints is what
  `dataset_v001` contains. The implementation must derive its inventory from
  whichever manifest is current, because `dataset_v001` exercises neither Latin
  letters nor ASCII digits and GUIDE section 28 requires both.
- **BPE stays open and unmeasured.** GUIDE says benchmark it, and EXP-004 does
  not. No claim is made in either direction.
- **The tokenizer keeps whitespace, and EXP-005 settled it by measurement rather
  than argument.** EXP-004 recorded character-level *with whitespace excluded* as
  the decision, on the arithmetic that it leaves 9 invalid utterances against 33.
  EXP-005 trained both: without spaces the token error rate is lower (0.0000674
  against 0.000190) and half the rows are invalid (6 against 12), and word error
  rate becomes undefined because there are no boundaries to recover. That last
  point was confirmed empirically, not assumed — `decode_words` returns a single
  token in that configuration. The next step exists to produce a WER, so the
  spaces stay and the deviation from the EXP-004 headline stands. The space-free
  variant remains available through one flag.
- **Text preparation has exactly one definition.** `prepare_text` is shared by
  `CharacterTokenizer.prepare`, `CharacterTokenizer.build` and `build_from_jsonl`.
  A vocabulary counted over differently-prepared text than the text later encoded
  would be a silent total mismatch, and the mismatch would only show up as a
  mysterious training failure.
- **Unknown codepoints become `<unk>` and are never dropped.** Dropping them
  would shorten the target sequence, which could make an utterance satisfy CTC's
  frames >= labels constraint for entirely the wrong reason.

## 6. Open questions

1. **No WER has ever been measured.** Nothing has been trained on the full
   corpus. Every number in this file is a data, signal or memorisation
   measurement, not a generalisation result. The first real WER will come from
   the baseline evaluated on `dev.jsonl`.
2. **BPE is unmeasured, and that is deliberate.** GUIDE section 28 says not to
   assume BPE is better, only to benchmark it. EXP-004 did not benchmark it, and
   makes no claim either way. Character-level is the specified *starting point*,
   so this does not block. If a later experiment asserts BPE is worse, that
   assertion needs its own measurements. BPE is also the one option that could
   justify a deeper encoder, since subword targets are shorter than characters.
3. **Corpus coverage is narrow.** IISc-MILE is read, studio-clean, single-condition
   speech. It cannot cover colloquial, code-switched, regional or noisy audio.
   AI4Bharat IndicVoices (Tamil, CC BY 4.0) and Kathbath (Tamil, conversational)
   are the planned additions; each license must be recorded in
   `docs/02_data/licensing.md` before use. Undecided.
4. **Real-recording VAD has no ground truth.** EXP-001's supplied recording has no
   transcript or time-marked speech, so its VAD statistics (79.3 % speech, 13
   segments) are sanity-checked, not scored. Open, not blocking.
5. **Build wall time is unstable.** The same manifest build measured 375.9 s and
   750.6 s on this machine. If a future step is time-boxed, measure twice first.
6. **The overfit test covers only very short utterances.** The subset runs
   0.243 s to 1.59 s against a corpus maximum of 38.85 s. It proves the pipeline
   is internally consistent; it says nothing about long or fast speech. A
   duration-representative overfit run is still wanted.
7. **Five corpus rows are corrupt, and EXP-005 upgraded this from observation to
   blocker.** Speaker `0000289` rows `_0000067` to `_0000071` carry 26–47
   characters in 0.24–0.45 s, which is 98.9–107.4 characters per second and
   physically impossible; the transcripts cannot belong to that audio. EXP-005
   showed that at character targets these rows **cannot be trained on at all**, at
   either whitespace setting — they are unusable examples, not awkward ones.
   Character targets are what exposed this: word targets fit inside 7–12 encoder
   frames, which is why EXP-003 consumed them without ever being blocked. Four
   further rows violate CTC's frames >= labels while speaking fast but humanly, so
   they are legitimate hard cases rather than defects: `MILE_0000232_0000013` at
   49.2 characters per second over 3.98 s, and `MILE_0000133_0000144`,
   `MILE_0000137_0000011`, `MILE_0000167_0000024` at 26.1–26.6 characters per
   second over 1.0–2.9 s. All 9 are among the 200 fastest utterances in the
   corpus. **No row has been removed from `dataset_v001`.** EXP-005 excluded the
   invalid rows per variant and listed them by id, but that is a filter on the
   overfit subset, not a disposition. `dataset_v002` must happen before the full
   baseline and must record which rows were removed and why.
8. **`.session/AGENTS.md` and `.session/START-HERE.md` are always one commit
   stale, and this cannot be fixed by committing.** The auto-status block reports
   the last commit's hash, and the post-commit hook rewrites the workspace entry
   files immediately after a commit lands. Mirroring them into `.session/` then
   produces a block describing a commit that the mirroring commit itself cannot
   contain. So the tracked copies differ from the workspace copies immediately
   after every commit, and committing them just starts the cycle again. Left
   alone deliberately: it is cosmetic, the workspace copies are correct, and
   redesigning the hook is not Phase 02 work. If it is ever fixed, the fix is for
   the post-commit hook to write `.session/` and for that path to be
   `git update-index --assume-unchanged`, or for the block to omit its own hash.

## 7. Next actions

EXP-002 is **complete and verified** (14 pass, 0 fail). EXP-003 is **complete**:
step 1 measured the tokenizer, step 2's overfit test is accepted. EXP-004 is
**complete**: the tokenizer scheme is decided — character-level, from measurement.
EXP-005 is **complete**: the character targets overfit, the tokenizer is proven
inside a training loop, and the whitespace default is settled by measurement. The
GPU is installed and working.

Items 1 and 3 below are **done** and are kept as the record of what was decided, not
as work to do.

1. ~~**Build the character tokenizer as a `src/tamil_voice/text/` module with
   tests.**~~ **Done** in `80fc716`: `text/unicode.py` and `text/tokenizer.py`, 85
   tests. The inventory is data-driven, as required.
2. **Create `dataset_v002` by dispositioning the CTC-invalid rows.** This is now
   **blocking the baseline**, not optional tidying. EXP-005 showed the 5 corrupt
   rows cannot be trained on at all; the other 4 are legitimate fast speech and
   should be kept. Record the reason and the utterance ids, per the rule against
   silent dataset changes. EXP-005 found 12 rows invalid with whitespace and 6
   without on the overfit subset; the corpus-wide counts are 33 and 9.
3. ~~**Re-run the overfit test with character targets.**~~ **Done** in EXP-005.
   Both whitespace variants trained; acceptance passed for both.
4. **Move the CTC model into `src/tamil_voice/asr/`.** The model currently exists
   twice, as a copy in EXP-003 and in EXP-005, because a committed experiment must
   stay runnable on its own. The baseline is the first thing that needs a model for
   a reason other than a test, and it should import one implementation.
5. **Then** the real CTC baseline on `train.jsonl`, evaluated on `dev.jsonl`, for
   the first genuine WER in this project. `test.jsonl` stays untouched until the
   baseline is fixed. Keep encoder stride 4: do not deepen the conv subsampling,
   because stride 8 invalidates 39 % of the corpus.

If a session ends before step 1 finishes, the exact starting point is: `dataset_v001`
is built and verified, EXP-003 and EXP-004 are complete, torch 2.6.0+cu124 has CUDA
available, the tokenizer decision is character-level, and no model has ever been
trained on the full corpus.

**Uncommitted state at the end of this session:** only `.session/AGENTS.md` and
`.session/START-HERE.md`, which the post-commit hook rewrites after every commit.
Open question 7 explains why committing them cannot fix it. Nothing else is
uncommitted and nothing is half-finished.

## 8. Rules that must survive every session

From `AGENTS.md`, condensed. Read `AGENTS.md` for the full text.

1. No fabricated datasets, results, benchmarks, or actions. Ever.
2. No placeholder values presented as results.
3. Build one component, measure it, then build the next.
4. All three gates pass before a commit: pytest, ruff, mypy.
5. If a gate cannot run, say which and why. Never report it as passed.
6. Split by speaker, never by recording.
7. Version every dataset and manifest: `dataset_v001`, `v002`.
8. Record every dataset's license before use.
9. Audio never enters git. Manifests do.
10. Update `MEMORY.md` before every commit.
11. No secrets, no tokens, no machine-specific paths.

## 9. Session protocol

At the start of every session:

1. Read this file.
2. Read `AGENTS.md` (this repository's), and the workspace-root `AGENTS.md`, which
   is the file opencode auto-loads when the session starts one folder up.
3. `git log --oneline -10`
4. `git status`
5. Verify the environment matches section 1 before claiming any gate result.

Before the end of every session:

1. `powershell -ExecutionPolicy Bypass -File ..\scripts\run-gates.ps1` — runs all
   three gates, records the real output, refreshes the status block.
2. `MEMORY.md` updated: status, decisions, open questions, next actions.
3. `CHANGELOG.md` appended.
4. Copy the workspace-root `AGENTS.md` and `START-HERE.md` into `.session/`.
5. Committed, or the uncommitted state stated explicitly.

## 10. Provenance

- `C:\ANTI_GRAVITY\PROJECT Z\GUIDE.MD` — authoritative specification, unmodified.
- `C:\ANTI_GRAVITY\PROJECT Z\.frolic-session.json` — automatic editor telemetry
  from the Frolic extension. Secondary. Outside this repository.
- `C:\ANTI_GRAVITY\PROJECT Z\chrome-browser-agent\` — separate project, unrelated
  to this one. Not part of this repository.
- GitHub remote `origin` — `https://github.com/VISHALRAJAN-GIT/Z.git`. The
  published copy of this repository. Never force-push `main`.

## 11. How the session backup works

The user lost session context once when opencode restarted. This is the fix, and
it has four parts. None of them replaces `MEMORY.md`; they keep it reachable and
current.

```text
PART 1  auto-loaded entry point
        PROJECT Z\AGENTS.md is loaded by opencode automatically at session start.
        It says, in the first ten lines: read MEMORY.md, then git log, then
        git status, then state the plan.

PART 2  authoritative memory, committed
        This file. Sections 1 (status), 5 (decisions), 6 (open questions) and
        7 (next actions) are the ones that matter. Updated before every commit.

PART 3  automatic status refresh
        PROJECT Z\scripts\update-status.ps1 rewrites the AUTO-STATUS block in
        PROJECT Z\AGENTS.md and PROJECT Z\START-HERE.md from live git output:
        last commit, branch, ahead/behind origin, working-tree files, recent
        commits, and the last recorded gate results.
        A git post-commit hook runs it, so the block refreshes on every commit
        without anyone remembering. Re-install it after a fresh clone with
        PROJECT Z\scripts\install-hooks.ps1, because .git\hooks is not tracked.
        PROJECT Z\scripts\run-gates.ps1 writes the real pytest/ruff/mypy output
        to .session\gates\ so the block can report it.

PART 4  git-tracked copies
        .session\AGENTS.md and .session\START-HERE.md are copies of the two entry
        files, refreshed at session end, so even the entry point itself survives
        in git history.

Failure modes that are accepted on purpose: the script reports git facts only, so
it can never record *why* something was decided — that is the agent's job. If the
block ever disagrees with this file, this file and the repository win, and the
block is regenerated.