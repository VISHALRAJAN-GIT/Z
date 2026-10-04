# EXP-004 notes — what the tokenizer measurements imply

## The short version

Character-level won, and not narrowly. But the interesting result is not that it
won. It is the second table: **stride 8 breaks character-level CTC on this corpus**,
and stride 8 is what a conv CTC encoder normally uses.

## The finding that constrains the architecture

EXP-003's front end runs the GRU at 25 Hz, which was chosen as a reasonable
default and never defended. It turns out to be load-bearing:

```text
stride 4  (25 Hz)     9 of 89401 utterances invalid
stride 8  (12.5 Hz)  35148 of 89401 invalid
stride 16 (6.25 Hz)  88458 of 89401 invalid
```

A mean train utterance is 69.41 non-space characters over 6.07 s, which is 11.9
characters per second. At 12.5 Hz that is 1.05 encoder frames per character, so
roughly half the corpus cannot be trained on. Stride 4 gives 2.1 frames per
character and survives.

The practical consequence: **do not deepen the conv subsampling when the real
baseline is built**, however standard that would feel. If a later architecture
wants stride 8, it needs character-level targets that are shorter than single
graphemes, which means a subword or phoneme unit, not a deeper conv stack. That
is a real trade and it should be measured if anyone proposes it.

## Five rows are corrupt and four are hard

The rate audit separated two populations that both look like "CTC violation":

- **Speaker `0000289`, five rows, 98.9–107.4 characters per second.** No human
  speaks at 107 characters per second. The transcripts cannot belong to those
  audio files, or the audio is truncated. These are defects.
- **Four rows at 26–49 characters per second.** Three of them run 1.0–2.9 s at
  26.1–26.6 characters per second; `MILE_0000232_0000013` is 3.98 s carrying 196
  characters at 49.2 characters per second. Fast, but human — and the 0000232 row
  is a different speaker from the corrupt cluster, which is why it belongs here.
  GUIDE explicitly lists fast speech as something this project must handle, so
  dropping these would be deleting the hard cases the project exists to solve.

Averaged over the corpus, 6 of 89 401 utterances exceed 30 characters per second
and 459 exceed 20. The median is 11.9. The tail is thin and concentrated.

Worth noting: `MILE_0000289_0000067`, `_0000068` and `_0000070` are the first three
rows of the EXP-003 overfit subset's recorded sample, because that subset selects
the *shortest* utterances and these are among the shortest in the corpus. The
overfit test therefore spent its first batches on some of the corpus's worst rows
and still reached token error rate 0.0386 — worth remembering when judging how
robust that result is.

## What was not settled

1. **BPE was not benchmarked.** GUIDE section 28 says not to assume BPE is
   better, and this experiment does not assume it either way. It is the obvious
   next tokenizer experiment, and it is the one that could legitimately argue for
   a deeper encoder, since subword targets are shorter than characters.
2. **Decoding is not settled.** With whitespace excluded from the target
   sequence, words are re-joined at decode time. How that is done, and whether
   Tamil needs a language model to fix word boundaries, is a Phase 08 question.
3. **The 9 rows are not dispositioned.** No row was removed. See the README.
4. **Only one corpus.** `dataset_v001` is read, formal Tamil from a single
   source. Every number here — the 48-symbol inventory, the zero OOV, the
   speaking-rate distribution — is a property of *this* corpus. The Phase 06
   corpora will contain Latin letters, digits and possibly decomposed Unicode,
   and the character inventory will have to be re-measured against whatever
   manifest is current at that time. A tokenizer that hardcodes 48 symbols would
   be a bug waiting for Phase 06.

## What this does not say about accuracy

Nothing here says a character-level CTC model will recognise Tamil well. It says
the output layer will not be the bottleneck and that held-out speakers will not
contain unseen symbols. Character-level ASR typically needs more acoustic capacity
per unit and a language model or lexicon to fix word boundaries; the first real
WER will tell us how much.

## Next

1. Build the character tokenizer as a `src/` module with tests: NFC
   normalisation, script-aware inventory, `<blank>` / `<unk>` handling, and a
   decode path that re-joins words. Its vocabulary must be data-driven, not the
   48 symbols measured here.
2. Disposition the 9 rows, and create `dataset_v002` if any are dropped, with the
   reason and the ids recorded.
3. Re-run the overfit test with character targets before the full baseline. This
   is cheap and it directly tests the new tokenizer end to end.
4. Then the real baseline on `train.jsonl`, evaluated on `dev.jsonl`.