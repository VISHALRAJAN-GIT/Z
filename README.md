# Tamil Voice Foundation

A compact, offline, low-latency **Tamil voice intelligence foundation** built for
CPU-first inference on modest hardware.

This repository is a research and engineering record, not a product. Every
component is built and measured independently before anything is connected.

---

## Scope

The system is intended to handle:

- formal Tamil, colloquial Tamil, and slang
- regional pronunciation variation
- Tanglish (Romanized Tamil) and Tamil-English code switching
- noisy, reverberant, far-field and phone-microphone audio
- low-volume and fast speech
- CPU-only inference on low-RAM machines
- streaming, real-time interaction

## Architecture

Modular by design. Each stage is independently testable.

```text
Microphone
    -> Audio Capture
    -> Audio Quality Analysis
    -> VAD
    -> Noise Estimation
    -> Adaptive Speech Enhancement
    -> Feature Extraction
    -> Tamil ASR
    -> Tamil / Text Normalization
    -> Tanglish + Code-Switch Processing
    -> Tamil Language Core
    -> Response Generation
    -> Tamil TTS
    -> Vocoder
    -> Speaker
```

This pipeline is the destination, not the starting point. It is built bottom-up,
one phase at a time, and each phase must be measured before the next begins.

## Current status

**Phase 02 — Tiny CTC ASR, in progress.** Phase 00 (engineering) and Phase 01
(audio foundation) are complete.

Six experiments are committed and verified:

```text
EXP-001  audio pipeline          8 pass, 0 fail  on synthetic + one real recording
EXP-002  speaker-disjoint split  14 pass, 0 fail  dataset_v001
EXP-003  tokenizer baseline      overfit test accepted
EXP-004  tokenizer scheme        character-level, chosen by measurement
EXP-005  character CTC targets   overfit accepted, first WER computed
EXP-006  dataset_v002            31 pass, 0 fail  5 corrupt rows removed
```

What is implemented and tested: audio loading, validation and resampling to the
canonical 16 kHz mono format, loudness and quality analysis, log-mel features,
a non-neural energy + spectral-flatness VAD, corpus discovery, JSONL manifests,
speaker-disjoint split planning, Unicode NFC handling, a character tokenizer, and
a CTC model (`TinyCTC`).

The dataset is the **IISc-MILE Tamil ASR Corpus** (OpenSLR SLR127, CC BY 2.0),
89,396 utterances across 531 speakers and 150.1 hours, as `dataset_v002`, split
by speaker with zero speaker overlap between train, dev and test.

**No WER has ever been measured on held-out data.** No model has been trained on
the full corpus. The only WER in this repository, 0.001647, comes from an
overfit test scored on its own training subset and is a memorisation number, not
a generalisation result. The next step is the real CTC baseline, evaluated on
`dev.jsonl` for the first genuine WER.

[`MEMORY.md`](MEMORY.md) holds the live state and
[`AGENTS.md`](AGENTS.md) holds the rules any contributor or AI agent must follow.

## Repository layout

```text
src/tamil_voice/     reusable implementation (the only importable product code)
training/            training algorithms and scripts
evaluation/          measuring model quality
data/                dataset lifecycle: raw -> interim -> processed -> manifests
configs/             YAML experiment configuration
experiments/         research history, one folder per experiment
checkpoints/         model weights (gitignored)
artifacts/           generated outputs: reports, plots, ONNX, quantized (gitignored)
scripts/             command-line entry points
tests/               unit, model and integration tests
deployment/          packaging for Windows, Linux, Android, CPU, ONNX
docs/                engineering knowledge, one folder per subsystem
notebooks/           exploration only, never imported by src
```

Dependency direction, one way only:

```text
notebooks -> scripts -> training / evaluation -> src/tamil_voice
```

## Prerequisites

```text
Python 3.11
Windows 11 (primary dev platform)
Intel i5 12th Gen / RTX 2050 4 GB / ~16 GB RAM
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

torch and torchaudio are an optional extra so the audio foundation can be
developed without a multi-gigabyte install. Choose the wheel index deliberately:

```powershell
# CPU only: development, unit tests, inference benchmarking
.\.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.4: RTX 2050 Laptop
.\.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Or `pip install -e ".[torch]"` for the default PyPI build.

## Quality gates

Run all three before every commit:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy
```

All three must pass. If a gate cannot run, say so rather than reporting success.

## Canonical audio format

```text
mono
16 kHz
float32 waveform
```

Other formats are accepted at the boundary and converted internally. Peak
normalization is **not** applied automatically: loudness carries information.
Silence, clipping and near-silence are detected and reported instead.

## Dataset policy

```text
raw -> interim -> processed -> manifest -> training
```

- Manifests are JSONL and are committed. Audio is not.
- Splits are by **speaker**, never by individual recording.
- Datasets are versioned: `dataset_v001`, `dataset_v002`, ...
- Licenses and permitted uses are recorded in `docs/02_data/licensing.md`.
- No fabricated datasets. No fabricated benchmark numbers.

## Roadmap

```text
PHASE 00  Project Engineering      done
PHASE 01  Audio Foundation         done
PHASE 02  Tiny CTC ASR             <- current
PHASE 03  Real Tamil ASR
PHASE 04  Noise Research
PHASE 05  Noise-Robust ASR
PHASE 06  Tamil Diversity
PHASE 07  Streaming ASR
PHASE 08  Tamil Language Core
PHASE 09  Tamil TTS
PHASE 10  Complete Voice Runtime
PHASE 11  Compression & Optimization
PHASE 12  CPU / Edge / Android
PHASE 13  Real-World Testing
PHASE 14  Public Release
```

## Working with an AI agent

`AGENTS.md` holds the permanent engineering rules. `MEMORY.md` holds the current
state and is updated before every commit, so any new agent or new model resumes
with full context instead of starting from zero.

## License

MIT — see [`LICENSE`](LICENSE).

Dataset licenses are separate and are documented per dataset in
`docs/02_data/licensing.md`. The repository license does not relicense any
corpus.