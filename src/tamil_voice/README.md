# src/tamil_voice/

Reusable implementation. This is the only importable product code in the
repository.

## Dependency direction

One way only:

```text
notebooks -> scripts -> training / evaluation -> src/tamil_voice
```

`src/tamil_voice` imports nothing from `scripts`, `training`, `evaluation` or
`experiments`. If a reusable module needs something from a script, the logic
belongs in this package and the script should be calling it.

## Packages

```text
audio/           I/O, resampling, normalisation, features, quality analysis
vad/             voice activity detection
enhancement/     speech enhancement. classical baselines first.
asr/             speech recognition. conformer encoder, ctc + attention decoders.
text/            unicode normalisation, tanglish, code switching, tokenisation
language/        intent, entities, context, response generation
tts/             text to speech
optimization/    distillation, quantization, pruning, export
runtime/         streaming pipeline assembly. loads components, trains nothing.
common/          shared configuration, logging, determinism
```

## What exists right now

`common/` only:

```text
common/config.py    paths, yaml loading, canonical audio spec
common/logging.py   json-lines logging
common/seed.py      reproducibility
```

Every other package exists as a namespace declaration and nothing more. That is
deliberate. GUIDE rule 15 and section 2 are explicit: build one component, measure
it, then build the next. A file full of `pass` bodies that return plausible values
is the specific failure this project already had to undo once.

## Conventions

- `from __future__ import annotations` at the top of every module.
- Type hints on public functions. mypy is not optional.
- No module-level mutable state.
- No network access at import time.
- No printing. Use `tamil_voice.common.logging`.
- No placeholder values returned as if they were results.

## Canonical audio

Mono, 16 kHz, float32 — defined once in `common.config.CANONICAL_AUDIO` and
imported everywhere else. If a second copy of that constant appears anywhere in
this package, that is a bug.