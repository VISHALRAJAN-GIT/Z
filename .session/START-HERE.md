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
_generated 2026-10-04 11:40 by scripts\update-status.ps1_

**Last commit:** `930c7d2 feat: EXP-004 tokenizer scheme decided from measurement, character-level`
**Branch:** `main`
**Sync:** 7 commit(s) AHEAD of origin/main (not pushed)
**Working tree:** 8 changed, 4 untracked
- `M .session/AGENTS.md`
- `M .session/START-HERE.md`
- `M .session/gates/mypy.txt`
- `M .session/gates/pytest.txt`
- `M CHANGELOG.md`
- `M MEMORY.md`
- `M experiments/004_character_tokenizer/README.md`
- `M experiments/004_character_tokenizer/notes.md`
- `?? src/tamil_voice/text/tokenizer.py`
- `?? src/tamil_voice/text/unicode.py`
- `?? tests/unit/test_text_tokenizer.py`
- `?? tests/unit/test_text_unicode.py`

**Recent commits**
- `930c7d2 feat: EXP-004 tokenizer scheme decided from measurement, character-level`
- `fc7409e docs: record that the .session entry copies cannot converge after a commit`
- `e3d56d3 feat: EXP-003 tiny CTC overfit test accepted, and the three bugs it caught`
- `dab1687 feat: EXP-003 word-level tokenizer measurements and subset`
- `4f9826c feat: EXP-002 speaker-disjoint manifests (dataset_v001) with independent verification`
- `050b614 chore: durable session memory with an auto-refreshing status block`
- `3988e32 feat: Phase 02 data layer for corpus discovery, manifests and speaker-disjoint splits`
- `32db8ae docs: record IISc-MILE Tamil corpus license and source`

**Gates, as last measured** (`tamil-voice-foundation\.session\gates\`)
- pytest: exit=0 | 305 passed in 11.48s
- ruff:   exit=0 | All checks passed!
- mypy:   exit=0 | Success: no issues found in 28 source files

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
5. **Experiment 002 accepted** — speaker-disjoint manifests for all 89 401
   utterances, independently verified 14 pass / 0 fail, and reproducible
   byte-for-byte across two builds.
6. **Experiment 003 accepted** — word-level tokenizer measured, then a tiny CTC
   model trained to overfit a 15-minute subset. Loss 6.15 to 0.09, token error rate
   on its own training data 97.5% to 3.9%. This is the project's first model. It
   proves the data, features, padding and CTC plumbing are sound; it says nothing
   about real-world accuracy.

## What is not started

- **No model has ever been trained on the full corpus. No WER has ever been
  measured.** Everything so far is data and signal measurement.
- The **vocabulary scheme is the open decision**. Word-level tokenisation leaves
  13.8% of dev words and 14.4% of test words with no entry in the vocabulary at
  all. That has to be fixed before a real baseline is worth training.
- No checkpoint is under version control, and no audio ever will be.

## Next step

Measure the vocabulary options and choose from the numbers: sweep the
most-frequent-N restriction to get an OOV-versus-N curve, and count the distinct
Tamil characters to size a character-level vocabulary. Then train the real CTC
baseline on the train split and get the first real WER from the dev split.

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