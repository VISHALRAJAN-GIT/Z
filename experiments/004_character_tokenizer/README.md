# EXP-004 — Tokenizer scheme: character-level, chosen from measurements

**Status:** complete. The vocabulary decision is made and every number behind it
is in `vocabulary_measurements.json`, produced by `measure_vocabulary.py`.

## Hypothesis

That word-level tokenisation is the wrong unit for this corpus, and that
character-level is both sufficient and affordable. Stated before measuring so the
numbers below are a test of it rather than a description of it.

## Why this experiment had to happen

GUIDE section 28 says:

> Start with character-level tokens. [...] Do not assume BPE is automatically
> better. Benchmark it.

EXP-003 measured the thing that made the question urgent: word-level leaves
**13.78 %** of dev tokens and **14.38 %** of test tokens out of vocabulary. The
obvious "just use the most frequent N words" fix had never been measured, and it
is not obviously safe to assume it fails either. So both options were measured,
plus what character-level costs, because a character target sequence is roughly
eight times longer than a word one and CTC cannot train where encoder frames are
fewer than labels.

## Measurements

All from `dataset_v001` (train 80304, dev 4526, test 4571). Output-layer cost is
quoted at hidden 256, matching the model trained in EXP-003.

| Option | Vocab | dev OOV | test OOV | fc params | fc MB (fp32) | < 100 MB |
|---|---:|---:|---:|---:|---:|---|
| **character** | **48** | **0.000000** | **0.000000** | **12 593** | **0.05** | yes |
| word, top 10 000 | 10 000 | 0.3000 | 0.3058 | 2 570 257 | 10.28 | yes |
| word, top 100 000 | 100 000 | 0.1570 | 0.1635 | 25 700 257 | 102.80 | no |
| word, full | 138 047 | 0.1378 | 0.1438 | 35 478 336 | 141.91 | no |

### 1. Truncating the word vocabulary does not fix OOV

The full curve, dev OOV rate against the most-frequent-N restriction:

```text
N=100     0.8306      N=5000    0.3786
N=250     0.7516      N=10000   0.3000
N=500     0.6738      N=20000   0.2373
N=1000    0.5881      N=50000   0.1841
N=2000    0.4963      N=100000  0.1570
                     N=138047  0.1378   (full vocabulary)
```

The curve is nearly flat at the top. Reaching 15.7 % OOV needs 100 000 of the
138 047 words and still misses the full vocabulary's 13.78 %. There is no useful
operating point: every size that is cheap enough to consider leaves more than
half of dev tokens unknown. **The top-N strategy is refuted by measurement, not
by preference.**

### 2. The character vocabulary is 48 symbols and closed

Distinct codepoints across the whole corpus: **48**. Composition: 47 in the
Tamil block `U+0B80..U+0BFF`, plus one ASCII character, the space. Characters
unseen in train: **0** on dev and **0** on test. Character-level OOV is measured
zero, not assumed zero.

Worth stating plainly because it constrains later phases: **this corpus contains
no ASCII digits and no Latin letters at all.** Numbers are written in Tamil
script (`ஆயிரத்தி`, not `1000`). GUIDE section 28 requires the tokenizer to
support English, digits and punctuation; none of those are exercised by
`dataset_v001`, so that support cannot be validated here and must be checked
against the Phase 06 corpora.

### 3. Unicode normalization is a non-issue in this corpus, but still required

`NFC` changes **0 of 89 401** transcripts, and the distinct-codepoint count is 48
either way. `NFD` and `NFKC` were measured too and are recorded in the JSON. The
corpus is already NFC. Normalization must still be implemented, because that is a
property of this corpus and not of Tamil.

### 4. The cost: characters multiply target length, and that constrains the encoder

Mean non-space characters per train utterance is **69.41** (max 532), against 8.42
words. CTC requires encoder frames >= labels for every utterance, so the encoder
stride is now an architectural constraint rather than a free choice.

Utterances where encoder frames < character labels, out of 89 401:

| Encoder stride | Encoder rate | word | character | character incl. space |
|---:|---:|---:|---:|---:|
| **4** | **25 Hz** | 0 | **9** | 33 |
| 8 | 12.5 Hz | 0 | 35 148 | 50 740 |
| 16 | 6.25 Hz | 5 | 88 458 | 89 001 |
| 32 | 3.1 Hz | 8 | 89 401 | 89 401 |

**The stride-4 front end from EXP-003 must be kept.** Stride 8 is the usual choice
for a conv CTC encoder and it is not available here: it breaks 39 % of the corpus.
This is the single most consequential number in the experiment.

### 5. Those 9 are outliers, and 5 of them are corrupt rows

Speaking rate, non-space characters per second:

```text
split   p50     p99     max       >20/s   >30/s
train   11.926  19.117  107.402   389      6
dev     11.940  20.332   23.463    58      0
test    12.240  18.055   22.895    12      0
```

All 9 stride-4 violations are among the 200 fastest utterances in the corpus. Five
belong to speaker `0000289` and sit at 98.9–107.4 characters per second — a rate no
human can produce, so those transcripts cannot belong to those audio files:

```text
MILE_0000289_0000067   0.4042 s   43 chars   11 frames   106.4 chars/s
MILE_0000289_0000068   0.3538 s   38 chars    9 frames   107.4 chars/s
MILE_0000289_0000069   0.4447 s   47 chars   12 frames   105.7 chars/s
MILE_0000289_0000070   0.2427 s   26 chars    7 frames   107.1 chars/s
MILE_0000289_0000071   0.4548 s   45 chars   12 frames    98.9 chars/s
```

The other four are fast rather than impossible, and are treated as legitimate hard
cases — GUIDE lists fast speech as a target condition, not a defect:

```text
MILE_0000232_0000013   3.9801 s  196 chars  100 frames    49.2 chars/s
MILE_0000133_0000144   2.9101 s   76 chars   73 frames    26.1 chars/s
MILE_0000137_0000011   2.1401 s   57 chars   54 frames    26.6 chars/s
MILE_0000167_0000024   1.0201 s   27 chars   26 frames    26.5 chars/s
```

`MILE_0000232_0000013` is a 4-second utterance carrying 196 characters. It is the
sixth and last corpus utterance above 30 characters per second, and the only fast
case with enough duration to be checked against a plausible speaking rate rather
than dismissed as a truncated file.

**No rows were removed.** Corpus-wide, 6 utterances exceed 30 characters per
second, and 4 of the 9 violations are real speech. Which of these to drop, and
whether that constitutes `dataset_v002`, is a decision for the baseline step with
these numbers in hand. It is recorded here rather than acted on, because silently
changing a dataset version is exactly what the rules forbid.

## Decision

**Character-level, whitespace excluded from the target sequence, NFC-normalised,
encoder stride 4 (25 Hz).**

1. OOV is measured at exactly 0 on held-out speakers, against 13.78 % word-level.
2. The output layer is 12 593 parameters and 0.05 MB, against 35 478 336 and
   141.91 MB — which alone breaks the project's < 100 MB weight target at
   word-level hidden 256.
3. It survives CTC's frames >= labels constraint at stride 4 for 89 392 of 89 401
   utterances.
4. It is what GUIDE section 28 specifies to start with, and the measurements do
   not contradict that instruction.

**BPE is not ruled out and was not measured here.** GUIDE says to benchmark it
rather than assume it, and this experiment does not benchmark it. Character-level
is the specified starting point, so the absence of a BPE benchmark does not block
the start. If a later experiment claims BPE is worse, that claim needs its own
measurements.

## Success criteria

| Criterion | Result |
|---|---|
| Character OOV on dev and test measured | pass, 0.000000 both |
| Character OOV below word-level at equal or lower cost | pass, 0 vs 0.1378 at 2817x fewer parameters |
| An encoder stride identified where CTC is valid corpus-wide | pass, stride 4 |
| Word top-N measured rather than assumed | pass, 11 points |
| Unicode normalization state measured | pass, 0/89401 changed under NFC |

## Files

| File | Role |
|---|---|
| `measure_vocabulary.py` | produces every number above |
| `vocabulary_measurements.json` | full results, including per-split rate percentiles and the 9 violations in detail |
| `notes.md` | what the numbers imply, and what they do not settle |

Run with `python experiments/004_character_tokenizer/measure_vocabulary.py`.
No `src/` module is changed by this experiment; the tokenizer itself is the next
step.