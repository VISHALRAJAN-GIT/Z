# EXP-003 — Tokenizer (word-level) and tiny CTC baseline (overfit test)

**Status:** Step 1 done (tokenizer measurements). Step 2 pending: tiny CTC to overfit 10-30 min subset.

## Why

Guide section 31: before training on a large split, prove the pipeline can overfit 10-30 minutes of speech. If it cannot, the bug is in the data, tokenizer, feature extraction, padding, CTC lengths, blank, or decoder — not the amount of data. This experiment is exactly that sanity check.

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

## Approach for baseline

1. Build a tiny CTC model: small Conv+GRU encoder (or Transformer-lite), ~1–5M params, CPU-first, but with CUDA available.
2. Use 80-bin log-Mel features (canonical, matches EXP-001) on 16 kHz audio. Load from manifests, derive features per utterance.
3. Train on the 15-min subset only (never dev/test). Monitor CTC loss and see if it goes to very small (overfits) — target loss decreasing quickly, converging to near-zero or very low over many steps.
4. Keep batch small, deterministic-ish (seed) where easy. Record all hyperparams in config/results.

## Files to add

- `experiments/003_tokenizer_baseline/train_overfit.py` — training script
- `experiments/003_tokenizer_baseline/config.yaml` — model+training params
- `experiments/003_tokenizer_baseline/results.json` — actual measured metrics (loss curves, steps, final loss)
- `experiments/003_tokenizer_baseline/notes.md` — observations

## Success criteria

The subset trains and loss drops substantially (overfits). Specifically: CTC loss on the subset training set becomes < 1.0 (rough target) and clearly decreasing; model can be saved. (Exact target set once we see first run.)

## Known caveats

- Word-level vocab 138k means the output layer is large. That’s a tradeoff for this test; if memory/throughput is an issue, we can try character-level as fallback.
- Audio loading: use existing `audio.io.load_audio` + resampling (already canonical-ready).
- Features: reuse `audio.features.log_mel_spectrogram` (time-major). Handle variable lengths.

## Measured numbers (for sanity)

| Split | Utt | Tokens | OOV | OOV% | Vocab |
|---|---|---|---|---|---|
| train | 80304 | 676524 | 0 | 0.0000 | 138047 |
| dev | 4526 | 38621 | 5322 | 0.1378 | 16541 |
| test | 4571 | 38356 | 5514 | 0.1438 | 16945 |
| subset (15.01 min) | 661 | - | - | - | - |