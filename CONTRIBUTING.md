# Contributing

This project is a research and engineering record. Contributions are judged on
measurement and honesty, not on volume of code.

## Read these first

1. `AGENTS.md` — permanent engineering rules. They apply to humans and to AI
   agents equally.
2. `MEMORY.md` — the current state. If you are resuming work, read this first.

## Before you write code

GUIDE section 2: **Do not create all implementation files immediately.** Build one
component, prove it, then build the next. A directory skeleton is fine. A file
full of `pass`, `TODO` stubs, or hardcoded placeholder values is not — a stub that
returns a plausible-looking number without doing the work is worse than no code
at all, because it survives review and poisons a measurement.

## Definition of done

A component is done when:

1. It does the real thing. No placeholder values standing in for results.
2. It has unit tests that fail when the logic is wrong.
3. `pytest`, `ruff check src tests` and `mypy` all pass.
4. Its measured performance is recorded, with the numbers and the setup used.
5. `MEMORY.md` is updated.

## Quality gates

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy
```

All three must pass before a commit. If one cannot run — a missing wheel, a
missing dataset, an unavailable GPU — say which one and why. Reporting a gate as
passed when it did not run is the single worst thing you can do in this repo.

## Commit messages

Imperative, scoped, honest.

```text
good:  add energy-based VAD with hangover merge and unit tests
bad:   update vad stuff
bad:   fix everything
bad:   add VAD (results are great)   <- unmeasured claim
```

State what changed and why. If a change is a hypothesis, say so.

## Data and datasets

- Audio is never committed. Manifests are.
- Every dataset carries its license, source URL, retrieval date and permitted
  use in `docs/02_data/licensing.md` before it is used.
- If a dataset has no clear license, do not use it.
- Never fabricate a dataset, and never present a synthetic sample as a real
  measurement.

## Benchmarks

- A number without the command, config, commit and hardware that produced it is
  not a result.
- Overfitting small data is a diagnostic, not a benchmark.
- Report what you measured, including failures.

## Working with AI agents

`AGENTS.md` is binding on agents. If an agent's output conflicts with it, the file
wins. Update `MEMORY.md` before you stop, so the next session resumes with full
context.