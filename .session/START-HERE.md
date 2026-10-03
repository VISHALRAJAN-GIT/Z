# START HERE — Tamil Voice Foundation

Plain-English status. Same facts as `AGENTS.md`, shorter.

---

## What am I looking at?

We are building an **offline Tamil voice assistant foundation**: speech recognition,
speech cleaning, Tamil text handling and speech output — all running on an ordinary
CPU, with no internet, low latency, and built to handle messy real-world audio and
real Tamil variety (colloquial Tamil, Tanglish, Tamil-English mixing, regional accents).

Nothing has been downloaded from a paid or closed service. Everything is measured
before it is claimed.

## Live status

<!-- AUTO-STATUS:BEGIN -->
_generated 2026-10-04 00:08 by scripts\update-status.ps1_

**Last commit:** `050b614 chore: durable session memory with an auto-refreshing status block`
**Branch:** `main`
**Sync:** 2 commit(s) AHEAD of origin/main (not pushed)
**Working tree:** 6 changed, 5 untracked
- `M .session/AGENTS.md`
- `M .session/START-HERE.md`
- `M .session/gates/pytest.txt`
- `M CHANGELOG.md`
- `M MEMORY.md`
- `M src/tamil_voice/data/manifest.py`
- `?? data/manifests/dataset_v001/`
- `?? experiments/002_data_split/README.md`
- `?? experiments/002_data_split/notes.md`
- `?? experiments/002_data_split/results.json`
- `?? experiments/002_data_split/verify_manifests.py`

**Recent commits**
- `050b614 chore: durable session memory with an auto-refreshing status block`
- `3988e32 feat: Phase 02 data layer for corpus discovery, manifests and speaker-disjoint splits`
- `32db8ae docs: record IISc-MILE Tamil corpus license and source`
- `1f19894 test: verify EXP-001 criteria on a real Tamil recording`
- `c4e1436 test: verify EXP-001 criteria on synthetic fixtures`
- `6f50cb6 feat: non-neural VAD and speech segment post-processing`
- `adc18bd refactor: share frame RMS and dBFS helpers across audio modules`
- `fe67953 feat: STFT/mel features and audio quality report`

**Gates, as last measured** (`tamil-voice-foundation\.session\gates\`)
- pytest: exit=0 | 220 passed in 6.24s
- ruff:   exit=0 | All checks passed!
- mypy:   exit=0 | Success: no issues found in 26 source files

_Numbers above come from git. Narrative status lives in `tamil-voice-foundation\MEMORY.md`._
<!-- AUTO-STATUS:END -->

## Where the project stands

```text
Phase 00  Engineering              DONE
Phase 01  Audio foundation         DONE      (loading, resampling, loudness,
                                            log-mel, quality report, VAD)
Phase 02  Tiny CTC ASR             IN PROGRESS (data layer written, not yet run)
Phase 03+ Real Tamil ASR, noise, diversity, streaming, TTS ... NOT STARTED
```

The 16-phase roadmap lives in `GUIDE.MD` and in `tamil-voice-foundation/AGENTS.md`.

## What is actually finished

1. Project skeleton, config system, logging, seeding, tests, git, GitHub remote.
2. **Audio foundation**, all tested: read any format, convert to the canonical
   mono / 16 kHz / float32, measure loudness and quality, log-Mel features, and a
   non-neural VAD that finds speech.
3. **Experiment 001 accepted** — 8 of 8 criteria pass, on synthetic fixtures with
   known answers plus one real Tamil recording the user supplied.
4. **First real corpus downloaded**: IISc-MILE Tamil ASR (OpenSLR SLR127, CC BY 2.0),
   ~150 hours, 89 401 utterances. License recorded in `docs/02_data/licensing.md`.

## What is half-finished (uncommitted right now)

The **Phase 02 data layer** exists as code but is not yet committed and has not
been run:

```text
src/tamil_voice/data/corpus.py     reads IISc-MILE filenames, transcripts
src/tamil_voice/data/manifest.py   builds and reads JSONL manifests
src/tamil_voice/data/splits.py     speaker-disjoint train/dev/test split
experiments/002_data_split/        build_manifests.py + config.yaml
tests/unit/test_data_*.py          3 test files
```

No manifests have been generated yet. No model exists yet.

## Next step

Build the manifests and the speaker-disjoint split for `dataset_v001`, measure
them, commit, then design the tiny CTC baseline as experiment 003.

Read `tamil-voice-foundation/MEMORY.md` section 7 for the authoritative version.

## If a new AI session starts here and forgets everything

It cannot afford to. Three things prevent it:

1. `AGENTS.md` in this folder is auto-loaded by opencode. It tells the agent to
   read `MEMORY.md` before touching anything.
2. `MEMORY.md` inside the repo is committed to git and is the single source of truth.
3. `scripts/update-status.ps1` refreshes the status blocks above automatically,
   and a git `post-commit` hook runs it on every commit.

If `MEMORY.md` and the repository ever disagree, the **repository wins**, and
`MEMORY.md` gets fixed first.