# EXP-006 — Disposition of the CTC-invalid rows, creating `dataset_v002`

## Why this step exists

EXP-004 found 9 utterances where character CTC cannot train at encoder stride 4.
EXP-005 showed that 5 of them are not merely awkward but **untrainable**: speaker
`0000289` rows `_0000067`..`_0000071` carry 26–47 characters in 0.24–0.45 s, which
is 98.9–107.4 characters per second. No human speaks that fast, so the transcript
cannot belong to the audio.

Those rows cannot be repaired. There is no offline way to recover the intended
transcript short of re-transcribing the audio, which is a different project. The
project rule is explicit that removing data creates a new dataset version, so this
step creates `dataset_v002`.

## The distinction this step exists to protect

"CTC cannot train on this row" is one fact with two causes, and giving them the same
answer is the mistake:

| cause | rows (with space) | answer |
|---|---:|---|
| transcript corrupt, cannot belong to its audio | 5 | **remove**, new version |
| valid audio, valid transcript, encoder too coarse | 28 | **keep**, filter at training time |

The 28 are genuine fast speech at 23.05–49.24 characters per second. A stride-4
encoder runs at 25 Hz, so it physically cannot emit 26 labels for 1.0 seconds of
audio. That is an architecture limit, not a data defect, and fast speech is an
explicit GUIDE target condition. Deleting them would throw away precisely the hard
cases the project exists to solve.

So `dataset_v002` = `dataset_v001` minus 5 rows, and the set CTC cannot train on is
published in metadata so the baseline filters by explicit id instead of silently
skipping rows.

## How the corrupt/fast boundary was chosen

Not chosen — measured. Every violating row's speaking rate was computed and sorted:

```text
MILE_0000289_0000068   107.40/s     9 frames   40 labels   corrupt
MILE_0000289_0000070   107.13/s     7 frames   28 labels   corrupt
MILE_0000289_0000067   106.37/s    11 frames   47 labels   corrupt
MILE_0000289_0000069   105.68/s    12 frames   50 labels   corrupt
MILE_0000289_0000071    98.94/s    12 frames   49 labels   corrupt
----------------------------------------------------------------- gap
MILE_0000232_0000013    49.24/s   100 frames  221 labels   fast, valid
MILE_0000137_0000011    26.64/s    54 frames   60 labels   fast, valid
   ... 26 more, down to 23.05/s
```

There is a factor-of-two gap between 49.24 and 98.94. **Zero rows corpus-wide fall
between 50 and 95 characters per second**, which the script asserts rather than
assumes. Any threshold in that interval produces the identical partition, so the
result does not depend on the exact number. Claiming more precision than that would
be false; the separation is this clean and no cleaner.

The corrupt set is also **identical with and without whitespace** (verified), so the
rows removed do not depend on the tokenizer configuration.

## Measurements

| | v001 | v002 |
|---|---:|---:|
| utterances | 89401 | **89396** |
| train | 80304 | 80299 |
| dev | 4526 | 4526 |
| test | 4571 | 4571 |
| train hours | 135.4202 | 135.4197 |
| speakers (train/dev/test) | 476 / 26 / 29 | 476 / 26 / 29 |
| speaker overlap | 0 | 0 |

| variant | violations in v001 | corrupt (removed) | fast, valid (kept) |
|---|---:|---:|---:|
| with_space | 33 | 5 | **28** |
| without_space | 9 | 5 | **4** |

## Verification (`verify_criteria.py`, 31 pass / 0 fail / 0 pending)

Deliberately shares no code with the builder, because a verifier built from the same
functions can only prove the code is self-consistent.

- `dataset_v001` unmodified, checked against git rather than a copy.
- Exactly the 5 corrupt ids absent from v002; no id added; no duplicate ids.
- Every kept row byte-identical to its v001 counterpart.
- No empty transcripts; every `shipped_split` a real corpus partition.
- Speaker overlap 0 in all three pairs; **no speaker moved splits**.
- Metadata counts and hours recomputed and matched.
- Both CTC-ineligible lists reproduced from scratch and matched.
- Audio for all 5 removed rows still on disk — removal is manifest-level only.

Determinism: all four files byte-identical across three consecutive builds.

```text
train.jsonl    0DC8ADD6DABD9350
dev.jsonl      5147475562C4F1C1
test.jsonl     EB26EF8F6CC30D32
metadata.json  DD14B97CF3705B94
```

Output is written with explicit `newline="\n"`. Python's default text mode emits CRLF on
Windows, and since `.gitattributes` declares `eol=lf` the committed blob would then be
LF while the working copy stayed CRLF — so a recorded SHA-256 would not reproduce after
a fresh checkout. `dataset_v001` is pure LF on disk, and this keeps the hashes above
equal to what git actually stores.

## Three bugs this step caught in itself

Worth recording, because none were visible to the count checks.

1. **The published CTC-ineligible list included the 5 removed rows.** The counts were
   right (28 and 4) while the sets were wrong, so any training filter built from that
   list would have referenced utterances `dataset_v002` no longer contains. Caught by
   the independent recomputation, not by the totals. The builder now filters and
   asserts this before writing.
2. **A verification check asserted a false invariant.** It required
   `shipped_split` to equal the file a row lives in. That is wrong by design: EXP-002
   re-split the corpus speaker-disjointly across the union of the corpus splits, so
   `shipped_split` records how IISc-MILE shipped the row and is deliberately not the
   project split. The check was measuring the project, not the data, and would have
   "failed" a perfectly correct dataset.
3. **Output line endings would have made the recorded hashes unreproducible.** The
   first build wrote CRLF, as Python's text mode does on Windows, while `.gitattributes`
   declares `eol=lf`. Git would have stored LF, so the on-disk file and the committed
   blob would have differed and no recorded SHA-256 would survive a fresh checkout.
   Found by reading git's own warnings rather than by any check, and fixed by writing
   with explicit `newline="\n"`.

## What this does not settle

- The 28 fast-but-valid rows are excluded at training time, not solved. A deeper
  encoder or a subword unit is the real answer, and EXP-004 already refuted stride 8.
- `dataset_v001` remains on disk and in git history. `dataset_v002` is additive.
- No model was trained. This step is data hygiene.

## Files

| File | What |
| --- | --- |
| `build_dataset_v002.py` | measures every violation, classifies, writes v002 |
| `verify_criteria.py` | 31 independent checks, exits non-zero on any failure |
| `results.json` | full violation tables for both variants |
| `data/manifests/dataset_v002/` | `train/dev/test.jsonl` + `metadata.json` |

```powershell
.venv\Scripts\python.exe experiments\006_dataset_v002\build_dataset_v002.py
.venv\Scripts\python.exe experiments\006_dataset_v002\verify_criteria.py
```