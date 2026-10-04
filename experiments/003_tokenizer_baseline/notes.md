# EXP-003 notes — tokenizer and tiny CTC overfit test

Findings from actually running this, in the order they happened. Every number
below was printed by `train_overfit.py` and is stored in `results.json`.

## Step 1 — word-level tokenizer (`measure_tokenizer.py`)

Whitespace tokenisation over `dataset_v001`.

| Split | Utt | Tokens | OOV | OOV rate | Vocab |
|---|---:|---:|---:|---:|---:|
| train | 80304 | 676524 | 0 | 0.0000 | 138047 |
| dev | 4526 | 38621 | 5322 | 0.1378 | 16541 |
| test | 4571 | 38356 | 5514 | 0.1438 | 16945 |

Two things this settled, both of which were open questions in `MEMORY.md`:

1. **Word-level on this corpus is not viable as-is.** 13.8 % dev and 14.4 % test
   tokens are out-of-vocabulary. No amount of model quality recovers from a
   fourteenth of the words simply not existing in the vocabulary.
2. **The vocabulary is huge**: 138047 words for 676524 tokens, i.e. 4.9 tokens per
   type on average. Read formal Tamil with numerals ("ஆயிரத்தி", "இரண்டு") does
   that.

## Step 2 — tiny CTC overfit test (`train_overfit.py`)

Accepted. Final measured numbers, 200 epochs, 271.3 s on one RTX 2050:

```text
parameters            1006002
train_loss            0.0947
token_error_rate      0.0386
exact_match_rate      0.8956
acceptance            PASSED (thresholds were in config.yaml before the run)
```

The curve, from the recorded history:

| Epoch | CTC loss | Token error rate | Exact match |
|---:|---:|---:|---:|
| 10 | 6.1475 | 0.9753 | 0.0000 |
| 40 | 1.1295 | 0.2029 | 0.4992 |
| 80 | 0.2569 | 0.0548 | 0.8487 |
| 120 | 0.1526 | 0.0333 | 0.9107 |
| 180 | 0.0867 | 0.0140 | 0.9622 |
| 200 | 0.0947 | 0.0386 | 0.8956 |

**What this proves:** data loading, feature extraction, padding, encoder lengths,
blank handling, CTC targets and greedy decoding are all mutually consistent. A
model can drive its own loss to near zero on data it memorises, so when a full
baseline fails to train, the fault will be in the data volume, the vocabulary or
the architecture — not in a plumbing bug.

**What this does not prove:** anything about generalisation. Every metric here is
on the training subset. No WER is claimed and none should be quoted from this
experiment. dev and test were never opened.

## Three bugs this step caught before it could produce a number

These are the reason the step exists. All three were in the first draft of the
training script and all three were found by the script failing loudly rather than
by reading the code.

1. **CTC targets were padded.** The target tensor was the padded `(B, S)` label
   block flattened, giving 40 entries where `sum(target_lengths)` was 26.
   `RuntimeError: Expected tensor to have size 26 at dimension 0`. This is worse
   than a crash: CTCLoss counts blank as a target symbol, so had the shapes
   matched by luck, the padding would have been trained on as if it were text.
   Fixed by masking the label block to `sum(target_lengths)` before flattening.

2. **Encoder length was wrong on 497 of 661 rows.** The draft used
   `floor(T / 4)`. Two stride-2 convolutions with `padding=1` give
   `ceil(ceil(T / 2) / 2)`, which differs whenever the length is odd. Recorded in
   `results.json` as `rows_where_floor_over_stride_would_be_wrong: 497`. Replaced
   with exact per-layer arithmetic, and `verify_length_math()` now checks that
   arithmetic against what `nn.Conv1d` actually returns on this machine, on a
   padded batch and on single utterances, at both ends of the length range.

3. **The output layer could not be tiny.** With the measured 138047-word
   vocabulary the linear layer alone is **35478593 parameters** at hidden 256 —
   35x the whole model that was actually trained. The subset contains 1360
   distinct words, 0.99 % of the train vocabulary, so this step builds its
   vocabulary from the subset and the trained model is 1006002 parameters. The
   full-vocab figure is still computed and stored in `results.json` rather than
   quietly dropped, because it is the number that decides the real baseline's
   vocabulary scheme.

## What was wrong with the first attempt, and why it stalled

The first run used `dropout: 0.1` and 40 epochs. It reached token error rate
0.3838 and **failed acceptance**. That was the correct outcome, not a nuisance.

Dropout is regularisation; its entire effect on a memorisation test is to make
memorisation harder. At `dropout: 0.0` and 200 epochs the same model reaches
0.0386. The parameter count never changed — only whether the model was being
prevented from doing the one thing the test asks of it.

Two hypotheses were checked and rejected before touching the configuration:

- *"The subset is too short for CTC."* No. Frames-per-label ratio across the
  subset is min 2.20, median 12.67, max 39.00, and 0 rows fall below 2.0. The
  runtime assertion that every row has at least as many encoder frames as labels
  passes for all 661.
- *"The shortest-first subset rule is degenerate."* It is unrepresentative — max
  duration 1.59 s against a corpus maximum of 38.85 s — but it is not
  structurally harder for CTC, and it spans 148 speakers. This is a real
  limitation of the test's coverage, recorded below, not its cause.

## Padding: two decisions that are about correctness, not taste

- **No BatchNorm.** BatchNorm over a padded `(B, C, T)` tensor averages the
  zero-padded tail into the statistics of the real frames. Every batch in this
  run is padded, so this is not a hypothetical.
- **Unidirectional GRU.** A bidirectional GRU reads the padded tail backwards
  into the last valid frames. CTC supplies true input lengths, so with a
  unidirectional stack the padded region cannot influence any valid output.

Together they mean zero-padding is inert. That is worth stating explicitly,
because it is the assumption the whole variable-length batch path rests on.

## `zero_infinity=False` is deliberate

CTC is run with `zero_infinity=False`. That flag is commonly set to `True` to
avoid infinite loss, and it would silently swallow exactly the
input-shorter-than-target condition this step exists to detect. The run instead
asserts the condition explicitly and refuses to start if it does not hold.

## Known limitations of this step

1. **The subset is unrepresentative.** 661 shortest utterances, 0.243 s to
   1.59 s, against a corpus range up to 38.85 s. It proves the pipeline is sound;
   it says nothing about how the model handles long or fast speech.
2. **Token error rate is not WER.** It is token-level Levenshtein over words, so
   the two coincide here, but it is measured on training data only.
3. **Greedy decoding only.** No beam search, no language model. Fine for an
   overfit test; not a decoding study.
4. **Termination is not perfectly monotonic.** Epoch 180 reached 0.0140 and
   epoch 200 sat at 0.0386. Both pass acceptance, but the run is not converged to
   zero and a learning-rate schedule would tighten it.

## Next

1. **Decide the vocabulary scheme from the OOV measurement, not from preference.**
   13.8 % dev / 14.4 % test word-level OOV is the blocking number. Candidates:
   restrict to the most frequent N words and measure the OOV/N curve; or a
   character-level Tamil vocabulary, which is small and closed. Measure, then
   choose.
2. **Overfit a duration-representative subset too**, so the check covers long
   utterances and is not limited to the 1.59 s tail.
3. **Then** the real baseline on `train.jsonl`, evaluated on `dev.jsonl` for the
   first WER in this project.