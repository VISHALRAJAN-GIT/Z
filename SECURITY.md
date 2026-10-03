# Security Policy

## Scope

This repository contains source code, configuration and documentation. It
currently contains no dataset, no model weights and no executable release, so
there is no deployed attack surface to report.

That changes as the project grows. The policy below is written for when it does.

## Reporting a vulnerability

Report privately. Do not open a public issue for a security problem.

Include: what the issue is, which files or components are affected, how to
reproduce it, and the impact. A reproduction is worth more than a hypothesis.

Expected response: acknowledgement within a few days, an assessment of severity,
and a fix or a mitigation plan. There is no bounty.

## What matters here

**Dependency and license integrity.** The largest real risk in this project is
taking on a dataset, model weight or vendored asset whose license does not permit
the intended use. This is not primarily a security issue, but it is treated with
the same seriousness. Record every source before use. See
`docs/02_data/licensing.md` and GUIDE section 32.

**Secrets.** This project needs no credentials to run. Do not commit tokens,
cookies, `.env` files, or machine-specific paths. `.gitignore` already excludes
`.env`, `checkpoints/`, `artifacts/` and `data/`.

**Audio is biometric data.** Tamil voice recordings can be identifying, and in
combination with a transcript they can be sensitive. Treat raw audio and
transcripts as personal data: keep them out of version control, out of issue
trackers, and out of any third-party service without a stated legal basis.

**Untrusted input.** When this project starts accepting third-party corpora or
model files, do not execute anything from them. Treat archives as hostile: no
pickles from unknown sources, no code from unvetted repositories, no unpinned
remotes.

**Deserialization.** Prefer safetensors over pickle for weights. Never
`torch.load` a file with `weights_only=False` unless you produced it and trust its
provenance.

## Supported versions

Only `main` is supported. This is pre-1.0 and unreleased.

## Before any public release

The release checklist in GUIDE section 53 applies. Security review is part of it,
not separate from it.