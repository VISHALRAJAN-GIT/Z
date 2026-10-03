# MEMORY.md

Live project state for the Tamil Voice Foundation. **Read this file first** in any
new session, before touching code. It is updated before every commit so that a new
model resumes with full context instead of starting from zero.

This file is authoritative. Frolic telemetry is secondary and may be rotated.

---

## 1. Current status

**Phase 00 — complete and committed at `35765c4`.
Phase 01 / EXP-001 — complete and verified.
Phase 02 — the data layer is written but NOT committed and NOT yet run.**

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

One dataset has been **obtained**: the **IISc-MILE Tamil ASR Corpus** (OpenSLR
SLR127, **CC BY 2.0**), at `data/raw/iisc_mile_ta/mile_tamil_asr_corpus/` —
89,401 utterances (77,314 train / 12,087 test), ~150 h read speech, 16 kHz mono
PCM, 16.125 GB on disk. License and source are recorded in
`docs/02_data/licensing.md` and `docs/02_data/dataset_sources.md`. No manifest,
no speaker-disjoint split, and no model exist yet. Any statement to the contrary
is false.

**Uncommitted work present in the working tree right now** (git status, measured):
`src/tamil_voice/data/` (`corpus.py`, `manifest.py`, `splits.py`, `__init__.py`),
`experiments/002_data_split/` (`build_manifests.py`, `config.yaml`), and
`tests/unit/test_data_corpus.py`, `test_data_manifest.py`, `test_data_splits.py`.
The gates pass with those files present, but the module is not committed and
`build_manifests.py` has never been executed, so no `dataset_v001` manifest exists
on disk yet.

Verification, run in `.venv` on Python 3.11.9, with the uncommitted data layer
present, recorded by `scripts/run-gates.ps1` into `.session/gates/`:

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
| `data/corpus.py` | **uncommitted.** `Utterance`, `parse_iisc_mile_name`, `read_transcript`, `discover_iisc_mile`, `CorpusError` — turns the corpus directory into typed utterance records |
| `data/manifest.py` | **uncommitted.** `ManifestRecord`, `build_records`, `write_manifest`, `read_manifest` — JSONL manifests with durations measured from the audio |
| `data/splits.py` | **uncommitted.** `SplitRatios`, `SplitConfig`, `plan_speaker_split`, `SpeakerSplitPlan`, `check_speaker_disjoint`, `SplitError` — speaker-disjoint split planning with a disjointness assertion |

Tests: `tests/unit/test_config.py`, `test_common.py`, `test_package.py`,
`test_audio_io.py`, `test_audio_resampling.py`, `test_audio_normalization.py`,
`test_audio_features.py`, `test_audio_quality.py`, `test_vad_detector.py`,
`test_vad_postprocess.py`, and (uncommitted) `test_data_corpus.py`,
`test_data_manifest.py`, `test_data_splits.py`.

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

## 6. Open questions

1. **Real-recording VAD has no ground truth.** EXP-001's supplied recording has no
   transcript or time-marked speech, so the VAD statistics (79.3 % speech, 13
   segments) are sanity-checked, not scored. Getting even one annotated/timestamped
   Tamil recording would turn criterion 6/7 on real audio into a precision/recall
   measurement. Open, not blocking.
2. **Diversity corpora.** IISc-MILE Tamil (CC BY 2.0) is obtained and recorded. It
   is read, studio-clean speech, so it cannot cover colloquial, code-switched or
   noisy audio on its own. AI4Bharat IndicVoices (Tamil, CC BY 4.0) and Kathbath
   (Tamil, conversational) are the planned additions; each license must be recorded
   before use. Undecided.
3. **Tokenizer.** Word-level for the first CTC baseline, then decide on
   subword from measured results, not preference. No commitment yet.

## 7. Next actions

The immediate unfinished work is the **Phase 02 data layer**, already written and
already gate-clean but still uncommitted and unexecuted. In order:

1. Run `experiments/002_data_split/build_manifests.py` over
   `data/raw/iisc_mile_ta/` with `config.yaml` to produce the real
   **speaker-disjoint** train/dev/test split and the JSONL manifests as
   `dataset_v001`. Record what the run actually reports: utterance counts, speaker
   counts, hours per split, and the disjointness check. Splitting by speaker, not
   by recording, is mandatory or the test numbers are worthless.
2. Complete the experiment record: `experiments/002_data_split/README.md`,
   `results.json`, `notes.md`. Commit the data layer and the manifests.
3. Then define the tokenizer and the tiny CTC baseline as its own experiment
   (`experiments/003_*`). Per GUIDE section 31, the first real milestone is
   overfitting 10-30 minutes of speech. Install the cu124 torch build for the
   RTX 2050 before that training starts.

If this session ends before step 1 finishes, the uncommitted data layer and the
missing `dataset_v001` manifests are the exact starting point of the next one.

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