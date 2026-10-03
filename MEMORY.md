# MEMORY.md

Live project state for the Tamil Voice Foundation. **Read this file first** in any
new session, before touching code. It is updated before every commit so that a new
model resumes with full context instead of starting from zero.

This file is authoritative. Frolic telemetry is secondary and may be rotated.

---

## 1. Current status

**Phase 00 — complete and committed at `35765c4`.
Phase 01 / EXP-001 — complete and verified.
Phase 02 / EXP-002 — complete and verified. `dataset_v001` exists on disk.**

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

**No model, tokenizer or training code exists.** No WER has ever been measured on
this corpus. Any statement claiming otherwise is false.

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

torch is **installed**: `torch 2.14.1+cpu`, `torchaudio 2.11.0+cpu`, from the
explicitly chosen CPU wheel index `https://download.pytorch.org/whl/cpu`. CUDA is
unavailable in this environment and `seed_everything` reports
`fully_deterministic=True` honestly because of that. The RTX 2050 still requires a
separate cu124 install when training starts.

Session memory is now backed by committed files plus an automatic refresher. See
section 11.

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

Tests: `tests/unit/test_config.py`, `test_common.py`, `test_package.py`,
`test_audio_io.py`, `test_audio_resampling.py`, `test_audio_normalization.py`,
`test_audio_features.py`, `test_audio_quality.py`, `test_vad_detector.py`,
`test_vad_postprocess.py`, `test_data_corpus.py`, `test_data_manifest.py`,
`test_data_splits.py`.

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

## 6. Open questions

1. **No WER has ever been measured.** Nothing has been trained. Every number in
   this file is a data or signal measurement, not a model result. The first model
   number will come from EXP-003's overfit test.
2. **Tokenizer vocabulary size is unmeasured.** Word-level on 135.42 h of read
   formal Tamil will produce a large vocabulary, and the out-of-vocabulary rate on
   held-out speakers is unknown. Measure before assuming word-level is adequate.
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

## 7. Next actions

EXP-002 is **complete and verified** (14 pass, 0 fail). `dataset_v001` is built,
verified and reproducible. The next work is the first thing that trains a model.

1. Install the cu124 torch build for the RTX 2050. This is the first step that
   needs a GPU and the last piece of Phase 00/01 setup left open:
   `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`
   Verify afterwards and record the real versions here. Until then, CPU only.
2. EXP-003: define the tokenizer. Word-level first for the CTC baseline, per
   GUIDE section 31, and only reconsider a subword scheme from measured results
   rather than preference. The corpus is read, formal Tamil, so a word-level
   vocabulary's size and out-of-vocabulary rate on this corpus are the numbers to
   measure before anything else.
3. Then the tiny CTC baseline as its own experiment, trained to **overfit 10-30
   minutes of speech**, using only rows from `train.jsonl`. Never dev or test for
   that check. A model that cannot overfit a small set has a data, tokenizer,
   padding, CTC-length, blank or decoder bug, and more data will not reveal it.

If a session ends before step 1, the exact starting point is: `dataset_v001`
exists and is verified, and the first training run has not been started.

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