# data/

Audio lives here. Audio is never committed.

## Lifecycle

```text
raw          exact bytes as obtained from the source. read-only. never modified.
  -> interim  extracted, converted, cleaned. disposable. rebuilt from raw.
  -> processed  model-ready segments. derived from a specific dataset version.
  -> manifests  JSONL describing what exists. the only thing version-controlled.
  -> training
```

`raw/` is immutable. If something needs fixing, transform into `interim/` and
leave the original alone.

## Layout

```text
data/
  raw/
    speech/           downloaded recordings, untouched
    text/             transcripts and corpora
    noise/            background noise samples
    tts/              text and reference audio for TTS
    asr_india/        Indic ASR material
    common_voice/     Common Voice Tamil
    openslr/          OpenSLR material
  interim/            working space
  processed/          model-ready
  manifests/          *.jsonl, committed
  benchmarks/         published benchmark sets
```

## Manifests

Manifests are committed because they are the specification of what exists.

```json
{"id": "sample_000001", "audio": "raw/speech/file.wav", "text": "வணக்கம்", "speaker_id": "spk_001", "duration": 3.4, "language": "ta", "split": "train"}
```

One JSON object per line. Required keys: `id`, `audio`, `speaker_id`. Add
`text` for ASR, `language`, `split`, `duration` as available.

Every manifest is versioned:

```text
dataset_v001
dataset_v002
```

Never overwrite a manifest that has been trained against. A changed dataset means
a new version, and every existing measurement stays interpretable.

## Splits

**Split by speaker, never by individual recording.**

Random splitting by recording leaks speaker identity between train and test.
Character error rate then improves for reasons that have nothing to do with the
model, and the number is worthless.

```text
train   ~90%
dev     ~5%
test    ~5%
```

Stratify across speakers and recording conditions. Keep a held-out test set that
is touched only for final reporting.

## Licensing

Every dataset needs a recorded license, source URL, retrieval date and permitted
use **before** it is used. `docs/02_data/licensing.md` holds the register. A
dataset with no clear license does not get used.

## Rules

- No fabricated data. A synthetic sample is labelled synthetic, always.
- No audio in git. `.gitignore` enforces this; do not defeat it.
- Corpus preprocessing code belongs in `src/tamil_voice/audio/`, not in a script
  that runs once and is never revisited.
- Manifests point at paths relative to this directory.