# EXP-002 notes

What worked, what did not, and what the next experiment should be.

## What worked

The filename scheme carried more than it looked. Parsing
`<PREFIX>_<SPEAKER>_<UTTERANCE>` gave 531 distinct speakers, exactly OpenSLR's
published count for the corpus. That agreement is the evidence that the parse is
correct and that the prefix really is a condition rather than part of the identity;
had the union come out at 638, the prefix would have been wrongly included in the
speaker key and every speaker would have been split into two identities.

Discovery over 89401 files took 9.0 s on the first run. It reads directory entries
and transcript text only and never opens audio, which is what kept it that cheap.

The speaker-disjoint plan held exactly: 476 / 26 / 29 speakers, zero overlap, no
near-duplicate fallback and no special-casing. The ratios are on speakers rather
than utterances, so per-split hours land near 90/5/5 without being forced to
(135.42 / 7.40 / 7.28 h).

## What did not work, and was fixed

**Header reading was the bottleneck, badly.** The first full run did not finish
inside 30 minutes and was killed. Measured cause: `soundfile.info` costs ~7-14 ms
per file on this machine because every open is an AV-scanned I/O wait, and
`build_records` did 89401 of them one at a time. Profiling the header read alone
showed 13.9 ms/file single-threaded against 0.27 ms with four workers, so the fix
was a thread pool in `build_records` — libsndfile releases the GIL, so these
workers overlap disk waits rather than compete for CPU. Eight workers sit on the
plateau; sixteen measured no better. Results are reassembled in input order before
sorting, so scheduling cannot affect the output.

That is a 25x change on the measured step and it took the whole run from
"does not finish in half an hour" to 375.9 s. Worth stating plainly: the first
version was correct but practically unusable, and correctness alone would have let
it ship.

**The run is still disk-bound and its timing is not stable.** The same command took
375.9 s on the first run and 750.6 s on the second, on the same machine with a warm
cache. Both produced identical bytes. Do not treat the wall time as a stable
property of the code; treat the byte equality as the property that matters.

**PowerShell prints text badly.** Dumping a manifest line in PowerShell showed the
Tamil transcript as mojibake. Re-reading through Python gave real UTF-8: first 10
codepoints U+0BAA, U+0BB0, U+0BBF, U+0B95, U+0BBE, U+0B9A, U+0BAE, U+0BCD, U+0020,
U+0B9A, with 75 Tamil characters, zero U+FFFD. `verify_manifests.py` counts
codepoints and checks for replacement characters rather than printing the raw text.

**Text writers must write LF, not CRLF, for reproducibility.** `metadata.json` and
`results.json` were first written with `write_text` which created CRLF on Windows;
that made them differ on Linux. The fix was to use `open(..., newline='\n')` and
explicitly write `\n`. The final hashes are for LF-only files.

## Things deliberately not done

No duration was inferred from file size, even though 16-bit PCM at 16 kHz would
have made it nearly free and removed the disk cost entirely. Every recorded duration
comes from `soundfile.info`, and the verifier re-measures a sample of them against
the file headers. A shortcut that is right 99.9 % of the time is the kind of thing
that quietly poisons a dataset version later.

No augmentation, no normalisation, no filtering of short or long utterances. Every
one of the 89401 utterances is in the manifests, including the 0.2427 s minimum.
Filtering is a decision that belongs to a measured baseline, not to the manifest
builder.

No use of the shipped `train/` / `test/` folders for splitting. They are recorded as
provenance on each row and nothing else.

## What is not proven yet

Nothing about model quality. This experiment proves the data is addressable,
complete, disjoint and reproducible. It says nothing about whether an ASR model can
learn from it.

The `shipped_split` field has one measured oddity worth carrying forward: dev and
test rows both contain utterances whose audio path lives under the shipped `train/`
folder. That is expected — the shipped split is ignored — but it means nobody may
ever filter the manifests by `shipped_split` and assume the result is disjoint. The
disjointness lives in which file each utterance is in.

## Next

EXP-003: the tokenizer and the tiny CTC baseline. Per GUIDE section 31, the first
milestone is overfitting 10-30 minutes of speech, and it must be done on a subset of
the train split only, never on dev or test. That install the cu124 torch build for
the RTX 2050 first, since this is the first step that actually trains.