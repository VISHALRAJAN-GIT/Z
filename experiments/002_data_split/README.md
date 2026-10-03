# EXP-002 — Speaker-disjoint dataset manifests

**Status:** complete. The manifests were built over the full corpus and verified
independently: **14 pass, 0 fail**. `results.json` holds the measured counts;
`verify_manifests.py` re-derives every one of them from the written bytes rather
than trusting the builder's own report.

## Hypothesis

A deterministic, speaker-disjoint split of the IISc-MILE Tamil corpus can be
produced from its filenames alone and recorded as versioned JSONL manifests, with
no audio decoded and no speaker appearing in two splits.

## Why the shipped split cannot be used

IISc-MILE ships its own `train/` and `test/` folders. They are **not**
speaker-disjoint: the same speaker is recorded in both halves. Training on one and
testing on the other would put the same voice on both sides, which flatters WER for
reasons that have nothing to do with the model. Every record therefore keeps its
`shipped_split` as provenance, and `splits.py` ignores it when planning the real
split. Splitting by speaker, never by recording, is `AGENTS.md` section 8.

## Scope

```text
src/tamil_voice/data/corpus.py     discover_iisc_mile, parse_iisc_mile_name, read_transcript
src/tamil_voice/data/manifest.py   build_records, write_manifest, read_manifest
src/tamil_voice/data/splits.py     plan_speaker_split, check_speaker_disjoint
experiments/002_data_split/        this experiment: config.yaml, both scripts, results.json, notes.md
```

Explicitly out of scope: any augmentation, any tokenizer, any model, any training,
and any text normalisation. This experiment only turns a corpus on disk into a
versioned, auditable manifest set.

## How the speaker id is derived

Filenames are `<PREFIX>_<SPEAKER>_<UTTERANCE>.wav`, with `PREFIX` one of `ISTL`,
`MILE`, `MICI`. The prefix is a recording condition, not an identity, so it is
**not** part of the speaker key: 107 speakers appear under more than one prefix, and
the union over prefixes is 531, which matches OpenSLR's published "531 speakers".
That agreement is itself a check that the parse is right.

## Measured result

Read from `results.json`, produced by the run, not written by hand.

```text
split   utterances  speakers     hours
train       80304       476  135.4202
dev          4526        26    7.3988
test         4571        29    7.2812
total       89401       531  150.1002
```

```text
speaker overlap   train&dev 0   train&test 0   dev&test 0
empty transcripts 0
sample rates      16000 everywhere
duration          min 0.2427 s   median 5.1601 s   max 38.8509 s
prefix mix        MILE 65274   ISTL 21178   MICI 2949
```

Split ratios 0.90 / 0.05 / 0.05, seed `20261003`, dataset version `v001`.

## Acceptance criteria

Verified by `verify_manifests.py`, all passing:

| # | Criterion | Result |
| --- | --- | --- |
| 1 | the three files parse as JSONL and their line counts match `results.json` | 80304 / 4526 / 4571, all match |
| 2 | no speaker appears in two splits | 0 shared in all three pairs; 531 distinct total |
| 3 | no utterance id repeats | 0 duplicates across all 89401 |
| 4 | no empty transcript, no Unicode replacement character | 0 and 0 |
| 5 | audio paths relative, present on disk, durations match file headers | 0 absolute, 0 missing of 89401, 0 mismatches in 2000 sampled |
| 6 | recorded durations sum to the reported hours | 150.1002 h both ways |

Criterion 5 re-measures a deterministic sample, every 44th utterance, not all
89401 files. The sample size is printed in the verification output. Durations for
the other 87401 records come from the same `soundfile` header read that produced
them, so this checks the recording path rather than re-reading everything.

## Reproducibility

`config.yaml` plus the corpus is enough to rebuild the manifests byte-for-byte.
Measured, not assumed: the manifests were built three times and compared by SHA-256.

```text
dev.jsonl      100501F418187766  identical in all three runs
test.jsonl     8B7E60DE6FCDB8ED  identical in all three runs
train.jsonl    FB5E8F3E463697DF  identical in all three runs
metadata.json  671603073D0511A3  identical in all three runs
```

`results.json` is **not** byte-identical, and should not be: it carries a
`generated` UTC timestamp. Two runs were compared after removing that one field,
and the remainder is identical (`1955585b2b79c956`). Its timestamps were
`18:44:36Z` and `18:46:26Z` on 2026-10-03.

Both `metadata.json` and `results.json` are written with explicit `newline="\n"`.
The first version used a plain text write, which translated to CRLF on Windows and
would have produced different bytes on Linux — correct on this machine, wrong as a
portable reproducibility claim.

The concurrency in `build_records` cannot change the output, because results are
reassembled in input order before sorting. The `train.jsonl` hash above is also the
proof: it is unchanged across the run before the thread pool was added and the
three runs after it.

## Version control

```text
committed     data/manifests/dataset_v001/dev.jsonl      2.2 MB
committed     data/manifests/dataset_v001/test.jsonl     2.2 MB
committed     data/manifests/dataset_v001/metadata.json  11 KB
gitignored    data/manifests/dataset_v001/train.jsonl    40.3 MB
```

`train.jsonl` is 40.3 MB, almost all Tamil transcript text, and `AGENTS.md` says
manifests are small and committed — 40 MB is not small. The split stays fully pinned
without it, because `metadata.json` lists all 531 speaker ids per split, so train
membership is exactly derivable from the seed, the corpus and those lists. Rebuild
with one command.

## Licensing

IISc-MILE Tamil ASR Corpus, OpenSLR SLR127, **CC BY 2.0**. Recorded in
`docs/02_data/licensing.md` before the corpus was used. Audio lives outside git;
the manifests are committed because they are small and contain no audio.

## How to run

```powershell
python experiments\002_data_split\build_manifests.py     # writes the manifests and results.json
python experiments\002_data_split\verify_manifests.py    # independent check, exit 0 on success
```

Both read `config.yaml` from this folder and take no other parameters. Audio is
never decoded, so the build takes roughly 3 to 6 minutes on this machine and
verifying takes under a minute.