# MEMORY.md

Live project state for the Tamil Voice Foundation. **Read this file first** in any
new session, before touching code. It is updated before every commit so that a new
model resumes with full context instead of starting from zero.

This file is authoritative. Frolic telemetry is secondary and may be rotated.

---

## 1. Current status

**Phase 00 — complete and committed at `35765c4`.
Phase 01 / EXP-001 — in progress.**

Implemented so far this phase: `audio/io.py` (loading, canonical conversion, real
validation), `audio/resampling.py` (8/22.05/44.1/48 kHz -> 16 kHz, with a
no-op for already-16-kHz input), `audio/normalization.py` (read-only loudness
analysis and classification, plus opt-in gain), `audio/features.py` (STFT wrapper
on torch, mel filterbank, log-mel), `audio/quality.py` (aggregate diagnostic
report), and `vad/detector.py` + `vad/postprocess.py` (non-neural energy +
spectral-flatness VAD, and segment cleanup). All are tested, and an end-to-end
load -> resample -> quality -> mel -> VAD smoke run returns sane numbers. The eight
EXP-001 acceptance criteria were run on **synthetic fixtures** by
`experiments/001_audio_pipeline/verify_criteria.py`: **7 pass, 0 fail, 1 pending**
(criterion 7 needs one real Tamil recording). The criteria have *not* been verified
on real speech, so EXP-001 is not yet accepted.

No dataset has been downloaded. No model has been trained. Any statement to the
contrary is false.

Verification, run in `.venv` on Python 3.11.9, after adding io.py, resampling.py,
normalization.py, features.py, quality.py, vad/detector.py and vad/postprocess.py:

```text
pytest   193 passed
ruff     All checks passed
mypy     Success: no issues found in 22 source files
```

torch is **installed**: `torch 2.14.1+cpu`, `torchaudio 2.11.0+cpu`, from the
explicitly chosen CPU wheel index `https://download.pytorch.org/whl/cpu`. CUDA is
unavailable in this environment and `seed_everything` reports
`fully_deterministic=True` honestly because of that. The RTX 2050 still requires a
separate cu124 install when training starts.

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

Tests: `tests/unit/test_config.py`, `test_common.py`, `test_package.py`,
`test_audio_io.py`, `test_audio_resampling.py`, `test_audio_normalization.py`,
`test_audio_features.py`, `test_audio_quality.py`, `test_vad_detector.py`,
`test_vad_postprocess.py`.

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
- **Acceptance is not claimed on synthetic fixtures.** `verify_criteria.py` runs
  criteria 1-6 and 8 on synthetic signals and writes measured numbers to
  `results.json`, but the README requires the criteria be met on a real Tamil
  recording. Criterion 7 is recorded as `pending`, and EXP-001 stays unaccepted
  until that run happens.
- **Peak normalization is not automatic.** Loudness carries information. Silence,
  clipping and near-silence are detected and reported instead.
- **Progressive implementation.** Directories now, files when their phase starts.
  A tree of `pass` bodies returning plausible values is the exact failure this
  project already had to undo once; `tests/unit/test_package.py` guards against it.
- **Docs are written per phase.** Folder skeletons exist; documents appear when
  they have real content.
- **docs/ follows GUIDE, not invention.** An 18-folder scheme was drafted and
  rejected in favour of the 10-folder GUIDE structure.

## 6. Open questions

1. **First real Tamil recording.** EXP-001 needs one for its plotting and
   end-to-end criteria. Undecided.
2. **First ASR corpus.** Candidates are Common Voice Tamil, OpenSLR, AI4Bharat
   ASR. Requires a licensing decision recorded in `docs/02_data/licensing.md`
   before use. Undecided.
3. **Tokenizer.** Word-level for the first CTC baseline, then decide on
   subword from measured results, not preference. No commitment yet.

## 7. Next actions

Phase 01 / EXP-001 continues. Done: torch installed (CPU), `audio/io.py`,
`audio/resampling.py`, `audio/normalization.py`, `audio/features.py`,
`audio/quality.py`, `vad/detector.py`, `vad/postprocess.py`, each with tests, plus
`experiments/001_audio_pipeline/verify_criteria.py` which measured 7 of 8 criteria
on synthetic fixtures (7 pass, 0 fail, 1 pending). Next:

1. Place one real Tamil recording in `data/raw/speech/` (see section 6), re-run
   `verify_criteria.py` on it, add the criterion-7 plots, and update this file.

Do not start EXP-002 or Phase 02 until the criteria are measured on real speech.

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
2. Read `AGENTS.md`.
3. `git log --oneline -10`
4. `git status`
5. Verify the environment matches section 1 before claiming any gate result.

Before the end of every session:

1. All touched gates run, with real output.
2. `MEMORY.md` updated: status, decisions, open questions, next actions.
3. Committed, or the uncommitted state stated explicitly.

## 10. Provenance

- `C:\ANTI_GRAVITY\PROJECT Z\GUIDE.MD` — authoritative specification, unmodified.
- `C:\ANTI_GRAVITY\PROJECT Z\.frolic-session.json` — automatic editor telemetry
  from the Frolic extension. Secondary. Outside this repository.
- `C:\ANTI_GRAVITY\PROJECT Z\chrome-browser-agent\` — separate project, unrelated
  to this one. Not part of this repository.
- GitHub remote `origin` — `https://github.com/VISHALRAJAN-GIT/Z.git`. The
  published copy of this repository. Never force-push `main`.