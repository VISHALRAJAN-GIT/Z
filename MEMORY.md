# MEMORY.md

Live project state for the Tamil Voice Foundation. **Read this file first** in any
new session, before touching code. It is updated before every commit so that a new
model resumes with full context instead of starting from zero.

This file is authoritative. Frolic telemetry is secondary and may be rotated.

---

## 1. Current status

**Phase 00 — Project Engineering — complete and committed.**

Nothing is implemented beyond shared infrastructure. No audio module exists. No
dataset has been downloaded. No model has been trained. No benchmark has been
measured. Any statement to the contrary is false.

Verification for the commit below, run in `.venv` on Python 3.11.9:

```text
pytest   46 passed
ruff     All checks passed
mypy     Success: no issues found in 15 source files
```

torch is **not installed**. Phase 01 needs it and the wheel index must be chosen
deliberately: CPU for development, cu124 for the RTX 2050.

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

git repository, branch `main`, initial commit only. Initialized 2026-10-03.

Structure: `src/tamil_voice/` (9 subpackages + `common`), `training/`,
`evaluation/`, `data/`, `configs/`, `experiments/`, `checkpoints/`,
`artifacts/`, `scripts/`, `tests/`, `deployment/`, `docs/`, `notebooks/`.

Empty directories are preserved with `.gitkeep`. `.venv`, caches, `checkpoints/`,
`artifacts/` and `data/raw|interim|processed|benchmarks` are gitignored.

## 4. What exists in code

Only `src/tamil_voice/common/`. Every other package is an empty namespace
declaration.

| Module | Purpose |
| --- | --- |
| `common/config.py` | project root, directory resolution with `TVF_*` overrides, YAML loading with real type errors, `AudioSpec`, canonical audio constants |
| `common/logging.py` | JSON-lines logging on stderr, one logger, idempotent setup, level from `TVF_LOG_LEVEL` |
| `common/seed.py` | `seed_everything` across Python/NumPy/torch, DataLoader worker init, `SeedReport` |

Tests: `tests/unit/test_config.py`, `test_common.py`, `test_package.py`.

Canonical audio: mono, 16 kHz, float32 — defined once in `config.py` and imported
everywhere else.

## 5. Decisions made

- **Full Phase 00 first.** No Phase 01 modules written before the skeleton is
  committed and gated.
- **Repo in a subfolder.** `GUIDE.MD` stays outside the repo, unmodified.
- **CPU-first.** torch is an optional extra so the audio foundation can be built
  without a multi-gigabyte install. CUDA is deferred until training needs it.
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

Phase 00 is closed. The next work is Phase 01 / EXP-001, in this order:

1. Install torch. Choose the wheel index explicitly and record the choice.
2. `src/tamil_voice/audio/io.py` — load any format, validate, convert to canonical.
   With tests.
3. `src/tamil_voice/audio/resampling.py`. With tests.
4. `audio/normalization.py`, `audio/quality.py`.
5. `audio/features.py` — mel filterbank, log-mel, 80 bins.
6. `vad/detector.py`, `vad/postprocess.py`.
7. Run all eight acceptance criteria in
   `experiments/001_audio_pipeline/README.md`, record measured numbers in
   `results.json`, and update this file.

Do not start EXP-002 or Phase 02 until every EXP-001 criterion is measured.

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