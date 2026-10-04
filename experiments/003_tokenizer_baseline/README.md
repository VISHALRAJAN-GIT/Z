# EXP-003 — Tokenizer (word-level) and tiny CTC baseline (overfit test)

**Status:** Step 1 done (tokenizer measurements). Step 2 done and accepted (tiny CTC overfits the subset). Next: choose the vocabulary scheme from the measured OOV rate.

## Why

Guide section 31: before training on a large split, prove the pipeline can overfit 10-30 minutes of speech. If it cannot, the bug is in the data, tokenizer, feature extraction, padding, CTC lengths, blank, or decoder — not the amount of data. This experiment is exactly that sanity check.

`notes.md` holds the full record, including the three bugs this step caught before it could produce a number. Read that before changing anything here.

## Measurements

`measure_tokenizer.py` tokenizes by whitespace (simple, reasonable for Tamil reads) on `dataset_v001` manifests.

- `train_vocab_size = 138047` words
- `total_train_tokens = 676524`
- OOV: dev 13.78%, test 14.38%; train 0%
- Subset selected from train (sorted by duration, accumulated) to reach ~15.01 min: 661 utterances, total ~900.7 s

Files:
- `word_vocab.txt` (4.8 MB) — vocabulary sorted by frequency desc
- `token2id.json` (6.5 MB) — maps token and `<UNK>` to id (0 reserved for blank)
- `tokenizer_stats.json` — full stats (vocab, OOV, subset, top10)

Notes: The word-level vocabulary is large (138k). That’s expected for this corpus; we don’t switch to subwords yet — the overfit test will tell us if the model can learn. If not, reducing sequence length or switching to a smaller vocabulary (e.g. character-level or BPE) is the next move, not a guess.

## Step 2 result (measured, `results.json`)

Tiny CTC: `Conv1d(80→32→64, stride 2, no BatchNorm) → GRU(256, 2, unidirectional) → Linear`, trained with AdamW on the 661-utterance / 900.7 s subset, train split only.

```text
parameters            1006002
train_loss            0.0947
token_error_rate      0.0386
exact_match_rate      0.8956
elapsed               271.3 s on one RTX 2050 (torch 2.6.0+cu124)
acceptance            PASSED
```

Acceptance thresholds (`accept_max_token_error_rate: 0.10`, `accept_max_loss: 1.0`) were written into `config.yaml` before the run and are checked explicitly; the script exits non-zero if they are missed. The first attempt, at `dropout: 0.1` and 40 epochs, reached token error rate 0.3838 and failed. Dropout is regularisation and this step tests memorisation, so it is now 0.

These are **training-subset** metrics. No WER is claimed. dev and test were never opened by `train_overfit.py`.

## The vocabulary finding that blocks the next step

The subset contains **1360 distinct words**, 0.99 % of the 138047-word train vocabulary. With the full vocabulary the linear output layer alone is **35478593 parameters** at hidden 256 — 35x the entire model trained here, and far past the project's < 100 MB weight target. So this step builds its vocabulary from the subset and trains a genuine 1.0M-parameter model; the full-vocab figure is computed and stored in `results.json` rather than dropped, because it is the number that decides the real baseline's vocabulary scheme.

Measured from step 1 and not yet acted on: **dev OOV 13.78 %, test OOV 14.38 %** for word-level. That, not architecture preference, is what should pick the next vocabulary scheme.

## Approach for baseline

1. Vocabulary scheme chosen from the measured OOV curve, not from preference.
2. 80-bin log-Mel features (canonical, matches EXP-001) on 16 kHz audio, loaded from manifests.
3. Train on `train.jsonl`, evaluate on `dev.jsonl`. `test.jsonl` stays untouched until the baseline is fixed.
4. Record all hyperparams in config/results.

## Files

| File | Role |
|---|---|
| `measure_tokenizer.py` | step 1: word-level vocabulary, OOV rates, subset selection |
| `train_overfit.py` | step 2: the overfit run. Reads every value from `config.yaml` |
| `config.yaml` | all model, feature, data and training parameters, including the acceptance thresholds |
| `verify_criteria.py` | *not written* — step 2's checks are runtime assertions inside the trainer, because three of the bugs were only findable by running the real op |
| `tokenizer_stats.json` | step 1 measurements |
| `subset_vocab.json` | the 1360-word subset vocabulary actually used in step 2 |
| `results.json` | step 2 measured metrics, full per-epoch history, acceptance block |
| `notes.md` | the research record: what failed, why, what to test next |
| `word_vocab.txt`, `token2id.json` | full 138047-word train vocabulary from step 1 |

Checkpoints go to `checkpoints/exp003_overfit/`, which is gitignored. No model weights are committed.

## Measured numbers (for sanity)

| Split | Utt | Tokens | OOV | OOV% | Vocab |
|---|---|---|---|---|---|
| train | 80304 | 676524 | 0 | 0.0000 | 138047 |
| dev | 4526 | 38621 | 5322 | 0.1378 | 16541 |
| test | 4571 | 38356 | 5514 | 0.1438 | 16945 |
| subset (15.01 min) | 661 | 1863 | 0 | 0.0000 | 1360 |

The subset row uses the subset's own vocabulary, so its OOV is 0 by construction. That is not a property of the corpus; it is why step 2 is a pipeline test and not a vocabulary test.