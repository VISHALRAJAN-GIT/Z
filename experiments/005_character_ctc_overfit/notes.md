# EXP-005 notes — what worked, what failed, what it means

## What was tested and what came out

Both whitespace variants trained to zero CTC loss on the 15-minute subset. The
tokenizer in `src/tamil_voice/text/` works as a CTC target source. The whitespace
default in `TokenizerConfig` is now supported by a measurement rather than by an
argument about word boundaries.

| | with_space | without_space |
| --- | ---: | ---: |
| rows excluded for frames < labels | 12 | 6 |
| rows trained | 649 | 655 |
| train loss | 0.002765 | 0.000690 |
| token error rate | 0.000190 | 0.0000674 |
| exact match | 0.995378 | 0.998473 |
| word error rate | 0.001647 | not computable |
| parameters | 668818 | 668561 |

## What worked

**Sharing one feature extraction between the variants.** 661 utterances of log-Mel
took 11.71 s to compute and are reused by both runs. This also guarantees the two
variants saw identical audio, which matters more than the 30 s it saved.

**Copying the model instead of importing it.** EXP-003's `TinyCTC` stays untouched
and runnable. The alternative — refactoring EXP-003 to expose its model — would
have edited a committed experiment for the convenience of a new one.

**Reporting excluded rows by id rather than filtering silently.** The 12 ids are in
`results.json` under `variants[].subset.violations`. Character targets made this
failure mode far more likely than word targets did, so it needed to be visible.

**An overridable config path.** `train_char_overfit.py` takes an optional config
path and writes its outputs beside that config. The 1-epoch smoke run that caught
the pipeline bugs therefore could not overwrite the real `results.json`. This was
added after the smoke run demonstrated the need, not before.

## What failed

**Nothing in the pipeline.** The smoke run passed on the first attempt and the real
run needed no fixes. That is unusual enough to be worth recording, but it is a
statement about this script, not a claim about the project.

**The controlled comparison, partly.** The two variants do not train on the same
rows. CTC validity excludes 12 rows with spaces and 6 without, so 6 rows are
trainable in one variant and not the other. The token error rate gap is therefore
confounded by the row set. Holding it constant requires dropping the 6
space-only-infeasible rows from both, which means creating `dataset_v002`, which
has not been decided. The gap is reported as measured and explicitly not claimed as
a clean A/B.

**Any attempt at a generalisation number.** None was attempted. Every metric is on
the training subset.

## The five corrupt rows are now a blocker, not an observation

EXP-004 identified speaker `0000289` rows `_0000067`..`_0000071` as corrupt: 26–47
characters in 0.24–0.45 s, which no human can speak. EXP-005 shows the consequence
that EXP-004 could only predict: **at character targets they cannot be trained on
at all**, at either whitespace setting. They are not awkward examples, they are
unusable examples.

Because the subset is shortest-utterances-first, these five rows are selected
before almost anything else. EXP-003's word-level run consumed them in its earliest
batches without ever being blocked, because 3–8 word targets fit inside 7–12
encoder frames comfortably. Moving to character targets is exactly what exposed
them.

So the open decision from EXP-004 is no longer theoretical. `dataset_v002` needs to
happen before the full baseline, and it should record at minimum: the 5 corrupt ids
removed, whether any of the 7 remaining excluded rows are also removed, and the
reason for each. The 4 legitimate fast-speech rows in the corpus-wide audit should
be kept — GUIDE lists fast speech as a target condition.

## What the WER number is and is not

**WER 0.001647 on 649 training utterances.** The model has memorised the subset;
that is the experiment's purpose. It says the decoder, the tokenizer, the word
boundary recovery and the edit-distance arithmetic all agree with each other. It
says nothing about speech recognition.

It is recorded because it is the first time word boundaries and WER have been
computable at all in this project, and because `without_space` demonstrates
concretely that the space-free configuration cannot produce one at all.

## Parameter count

The two variants share an identical conv front-end and GRU, so the whole difference
between them and EXP-003 is the output layer:

| part | parameters |
| --- | ---: |
| conv front-end (80->32->64, kernel 3) | 13920 |
| GRU (2 layers, hidden 256, unidirectional) | 642048 |
| shared subtotal | 655968 |
| output layer, 50 classes (`with_space`) | 12850 |
| output layer, 49 classes (`without_space`) | 12593 |
| output layer, 1362 classes (EXP-003) | 350034 |

Totals: 668818, 668561, 1006002. Each reconciles exactly as shared + output layer.
The 337184 parameter gap between EXP-005 and EXP-003 is **entirely** the output
layer — 350034 against 12850 — because everything upstream is identical.

For scale, EXP-004 measured the full 138047-word vocabulary at 35478336 output-layer
parameters against 12593 for a 49-class character head: **0.0355 %**. Against
EXP-003's subset-sized 1362-word head the character head is 3.67 %. The dramatic
ratio is only available at the full vocabulary, which is the case that matters —
it is why character-level can hold a closed vocabulary inside a CPU budget and
word-level cannot.

## What was not settled

1. **BPE is still unmeasured.** EXP-004 left it open and this step does not touch
   it. It remains the obvious tokenizer experiment that could legitimately argue
   against character-level on sequence-length grounds.
2. **The row-set confound.** Resolved only by the `dataset_v002` decision.
3. **Generalisation.** Untouched. Next step is the real baseline on `train.jsonl`
   evaluated on `dev.jsonl`.
4. **Long and fast speech.** This subset runs 0.243–1.59 s against a corpus maximum
   of 38.85 s, so it still says nothing about long utterances. The 7 non-corrupt
   excluded rows were fast, and the ones that survived the filter are short by
   construction.