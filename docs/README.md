# docs/

The repository's engineering knowledge base. Research history lives in
`experiments/`; durable knowledge lives here.

## Why this is organised by topic, not by time

Knowledge outlives the project structure it describes. Topic folders are stable
and linkable; dated documents rot. Current state goes in `MEMORY.md`, not here.

## Structure

```text
00_project_overview/    what is being built, for whom, and what is explicitly out of scope
01_architecture/        pipeline, contracts, data flow, resource budget
02_data/                sources, licensing, preprocessing, manifests, splits
03_speech_enhancement/  enhancement models and evaluation
04_asr/                 conformer, ctc, attention decoding, metrics, checkpoints
05_language/            intent, entities, memory, conversation
06_tts/                 text to speech, vocoders, speaker work
07_optimization/        distillation, quantization, pruning, export
08_evaluation/          benchmarks and measured results
09_deployment/          windows, linux, cpu, onnx, android, offline
```

Each folder holds a `README.md` index plus the documents it needs. Those documents
are written **when the phase that produces them starts**, not in advance. A tree of
empty files is navigation debt, not documentation.

## The documents that matter most

- `02_data/licensing.md` — every dataset's license, source, retrieval date and
  permitted use. Blocking: nothing gets used before it is recorded here.
- `08_evaluation/results.md` — measurement summaries with the setup that produced
  them.
- `03_speech_enhancement/README.md` — enhancement results, including the ones that
  did not work.
- `09_deployment/windows.md` and `09_deployment/linux.md` — target environments
  and their constraints.
- `01_architecture/contracts.md` — what each stage accepts and returns.

## Rule

If a document states a number, it states the command, config, commit and hardware
that produced it. Otherwise it states no number.

## Writing a document

- English. Tamil appears only where it is the subject.
- Concrete over aspirational. "WER on Common Voice v17 dev is 18.4% with the
  Phase 03 model" beats "we aim for low WER".
- Record negative results. A rejected approach with a reason is worth more than a
  silent one.
- Link to the experiment folder that produced the finding.