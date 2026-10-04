# EXP-006 notes

## What was decided

`dataset_v002` = `dataset_v001` minus 5 corrupt rows. The 28 fast-but-valid rows
stay and are filtered at training time by published id.

The decision this step had to get right was not "which rows are unusable" — that was
easy. It was refusing to treat *all* unusable-for-CTC rows as unusable data. Those
are different sets, they overlap, and conflating them would have deleted valid fast
speech to make a training loop simpler.

## Why removal, not repair

The 5 rows carry 26–47 characters in 0.24–0.45 seconds. The audio is present, the
duration is real, the transcript is text — but no human produces 99–107 characters
per second, so at least one of the two is wrong and there is no way to tell which
from the files alone. Re-transcribing would fix it, and re-transcription is not
something this offline project can do. So the pair is unusable and the honest action
is to remove it and say so.

Removing data is exactly the action the rules reserve a version bump for, which is
why this is `v002` and not an edit to `v001`. `dataset_v001` is untouched in git and
still on disk.

## Why the fast rows stay

The 28 fast rows fail for an architectural reason: a stride-4 encoder emits 25 frames
per second, and 23–49 characters per second of speech needs more labels than frames.
Nothing is wrong with the audio or the transcript.

Three arguments for keeping them:

1. Fast speech is an explicit GUIDE target condition. Removing the rows the model
   cannot currently handle removes the evidence that it cannot handle them.
2. The condition may be fixable by architecture later — a subword or phoneme unit
   shortens targets without deepening the encoder. Data removed now is data not
   available when that is tried.
3. The cost of keeping them is one published id list. The cost of deleting them is
   permanent and invisible.

The filter is published in `metadata.json` under
`changes.kept_fast_speech.ctc_ineligible_at_stride_4`, so the baseline skips exactly
those ids and reports the count. Silent skipping would make the training set smaller
than the manifest implies, which is the kind of thing that quietly invalidates a WER.

## What went wrong, and what it caught

**The published ineligible list was wrong while the counts were right.** The builder
published every violation it found in `dataset_v001`, which included the 5 rows it
was removing in the same breath. So `metadata.json` claimed 28 and 4 rows — correct
numbers — attached to 33 and 9 ids, of which 5 no longer existed. A training filter
built from that list would have tried to load removed utterances.

The totals agreed, so no count-based check would ever have found it. The independent
recomputation in `verify_criteria.py` found it immediately, because it compared sets
rather than lengths. The builder now filters the corrupt rows out and asserts that
the published list is a subset of v002 before writing.

**A verification check asserted something false.** It required `shipped_split` to
equal the manifest a row lives in. It is not, and must not be: EXP-002 built a
speaker-disjoint split across the union of IISc-MILE's own splits, so
`shipped_split` records the corpus's partition while the file records the project's.
The check would have failed a correct dataset. Replaced with a check that
`shipped_split` is preserved from v001 (already covered by the byte-identity check)
and holds a real corpus partition value.

Both were found by running the verification, not by reading the code. Neither was
visible in the summary numbers the builder printed.

## Honest limits

- **The threshold is a range, not a number.** 60 characters per second was used, but
  the answer is identical anywhere in 50–95 because no row falls in that interval.
  The script asserts the emptiness rather than trusting it.
- **"Corrupt" is inferred, not proven.** The evidence is that the transcript is
  impossible for the audio duration. The audio could instead be truncated with a
  correct transcript. Both readings mean the row is unusable; neither can be
  distinguished offline, and the distinction does not change the action.
- **28 rows of valid fast speech remain unhandled.** This step makes that explicit
  and countable. It does not solve it.
- **No model was trained.** Nothing here says anything about recognition quality.

## What is next

The baseline can now train on `dataset_v002` without hitting an untrainable row:
filter the published ineligible ids, train on `train.jsonl`, evaluate WER on
`dev.jsonl`. That is the first generalisation number this project will have, and it
should be read knowing that 28 rows of deliberately-hard fast speech are excluded
from it.