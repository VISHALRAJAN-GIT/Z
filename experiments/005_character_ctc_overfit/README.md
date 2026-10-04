# EXP-005 — Character-level CTC overfit, and the whitespace variant decided by measurement

## Hypothesis

Two things, one of which is a pipeline check and one of which is a decision.

1. `src/tamil_voice/text/tokenizer.py` has never been inside a training loop. Its
   ids must work as CTC targets, not merely round-trip through `encode`/`decode`.
2. EXP-004 chose whitespace-**excluded** character targets on CTC arithmetic (9
   invalid utterances against 33 corpus-wide). The implementation defaults to
   whitespace-**included**, because a target with no space symbol decodes with no
   word boundaries and word error rate becomes undefined. Both are now run on the
   same data with the same hyperparameters so the choice stops being an argument.

## Approach

Everything is held identical to EXP-003 step 2 — same architecture, same features,
same shortest-utterances-first subset rule, same hyperparameters, same seed, same
device. Only the targets and the vocabulary source change.

| | EXP-003 (word) | EXP-005 (character) |
| --- | --- | --- |
| subset | 661 utts, 900.7 s, 148 speakers | identical |
| device | cuda | cuda |
| epochs | 200 | 200 |
| vocabulary source | the subset | **the full train manifest** |
| targets | whitespace words | characters |

The vocabulary source changes because EXP-003's reason for rebuilding from the
subset no longer applies. A 138049-way softmax is not a tiny model, but a 50-way
softmax is, so building from the full train manifest costs nothing in parameters
and produces the tokenizer the baseline will actually use. The tokenizer is saved
as `tokenizer_with_space.json` and `tokenizer_without_space.json`.

The model in `train_char_overfit.py` is a deliberate copy of EXP-003's. The
research record stays runnable on its own; moving the model into
`src/tamil_voice/asr/` belongs to the baseline step, which is the first thing that
needs a model for a reason other than a test.

## Measurements (measured, `results.json`)

Both variants overfit. Acceptance thresholds were TER <= 0.10 and loss <= 1.0.

| | with_space | without_space | EXP-003 (word) |
| --- | ---: | ---: | ---: |
| real symbols | 48 | 47 | 1360 words |
| num classes | 50 | 49 | 1362 |
| parameters | 668818 | 668561 | 1006002 |
| rows violating frames >= labels | **12** | **6** | 0 |
| rows trained | 649 | 655 | 661 |
| total target labels | 15822 | 14831 | — |
| final train loss | 0.002765 | 0.000690 | 0.094745 |
| token error rate | 0.000190 | **0.0000674** | 0.038647 |
| exact match rate | 0.995378 | 0.998473 | 0.895613 |
| word error rate | **0.001647** | *not computable* | *not computable* |
| elapsed | 288.4 s | 280.4 s | 271.3 s |
| acceptance | **passed** | **passed** | passed |

### 1. The tokenizer works as CTC targets

Character TER of 0.00019 over 15822 target labels means about 3 wrong characters in
the whole 15-minute subset. The pipeline can drive CTC to zero on data it
memorises with these ids, which is the only thing an overfit test is supposed to
establish.

Character-level also overfits **more completely** than word-level did — loss 0.0028
against 0.0947, exact match 0.9954 against 0.8956. That is the expected direction:
a 48-way target space over short strings is a much smaller thing to memorise than
138047-way word vocabulary. It is not evidence that character-level will
generalise better.

### 2. The project's first WER, on training data

`with_space` reaches **WER 0.001647**. `without_space` cannot produce a WER at all,
because `decode_words` returns a single token when whitespace is not a target
symbol. The argument for including whitespace stopped being an argument.

**This is a memorisation number on the training subset. It is not a generalisation
result and must never be quoted as one.** `dev` and `test` were never opened by this
script.

### 3. Whitespace costs exactly what EXP-004 predicted

Excluding whitespace halves the CTC-invalid rows on this subset: **6 against 12**.
EXP-004 measured the same direction corpus-wide (9 against 33). The token error
rate is also better without spaces, 0.0000674 against 0.000190, which is expected
when the model has one fewer symbol to emit.

So the measured trade is: without spaces you get a slightly easier target and half
the invalid rows, and in exchange you cannot compute word error rate at all. Given
that the entire point of the next step is the first genuine WER, the spaces stay.

### 4. Every one of the 5 corrupt rows blocks training

`with_space` excluded these 12 utterance ids:

```text
MILE_0000289_0000067   MILE_0000289_0000068   MILE_0000289_0000069
MILE_0000289_0000070   MILE_0000289_0000071   MILE_0000167_0000024
MILE_0000152_0000135   MILE_0000291_0000093   MILE_0000137_0000022
MILE_0000137_0000025   MILE_0000260_0000217   MILE_0000260_0000232
```

`without_space` excluded the first 6 of those. The 5 `MILE_0000289` rows are the
corrupt cluster EXP-004 identified; EXP-005 confirms they cannot be trained on at
all, at either whitespace setting, because character targets make frames >= labels
impossible for them. Because the subset is shortest-utterances-first, they are the
first rows selected — EXP-003's word-level run spent its earliest batches on them
without ever being blocked, since word targets were short enough to fit.

`dataset_v001` is **not** modified and no manifest is rewritten. The rows are
excluded per variant and listed by id in `results.json`. The permanent disposition
is still an open decision.

### 5. The comparison is not perfectly controlled

The two variants train on **different row sets** (649 against 655) because CTC
validity forces it. Six rows are trainable without spaces and invalid with them.
So the TER gap between the variants is not a clean A/B on identical data, and is
not claimed as one. Holding the row set identical would mean dropping the 6 rows
from both, which is the `dataset_v002` decision that has not been made.

## Success criteria

- [x] Both variants drive CTC to zero on the training subset (TER <= 0.10).
- [x] Both variants drive train loss below 1.0.
- [x] The tokenizer's ids are usable as CTC targets end to end.
- [x] A WER is computable for the variant that keeps word boundaries.
- [x] Every row that could not be trained on is listed by utterance id.
- [x] `dev` and `test` never opened.

## What this does not establish

- No generalisation result of any kind. Every number here is on training data.
- No claim that character-level beats word-level on accuracy. It overfits more
  completely, which is a statement about model capacity versus subset size.
- BPE remains unmeasured.
- The 12 invalid rows are excluded, not dispositioned.

## Files

| File | What |
| --- | --- |
| `train_char_overfit.py` | trains both variants, emits `results.json` |
| `config.yaml` | every parameter; nothing hardcoded in the script |
| `results.json` | all measurements, per-variant history, excluded ids |
| `tokenizer_with_space.json` | 48 symbols + blank + unk, built from the full train manifest |
| `tokenizer_without_space.json` | 47 symbols + blank + unk |

Reproduce with:

```powershell
.venv\Scripts\python.exe experiments\005_character_ctc_overfit\train_char_overfit.py
```

An optional first argument selects a different config; outputs are written beside
that config, so a short smoke run cannot overwrite these results.