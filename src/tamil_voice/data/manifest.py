"""JSONL manifest writing and reading.

AGENTS.md section 8: manifests are JSONL, one record per line, and are small
enough to commit. A record holds everything downstream code needs and nothing it
does not: relative audio path, transcript, speaker, recording batch, duration and
provenance. Audio is never loaded here — durations come from the file header.

Audio paths are stored relative to the ``data/`` root so the manifest stays valid
if the repository moves. Text is written with ``ensure_ascii=False`` so Tamil is
stored as Tamil, not as ``\\u0bxx`` escapes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from .corpus import CorpusError, Utterance, read_transcript


@dataclass(frozen=True)
class ManifestRecord:
    """One line of a split manifest."""

    utterance_id: str
    audio_path: str
    text: str
    speaker_id: str
    prefix: str
    shipped_split: str
    duration_seconds: float
    sample_rate: int
    dataset_id: str


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _duration_seconds(path: Path) -> tuple[float, int]:
    """Header-only read of ``(duration_seconds, sample_rate)`` for one file."""
    try:
        info = sf.info(str(path))
    except Exception as exc:  # soundfile raises libsndfile's own error type
        raise CorpusError(f"cannot read audio header {path}: {exc}") from exc
    if info.samplerate <= 0:
        raise CorpusError(f"audio reports a non-positive sample rate: {path}")
    return info.frames / info.samplerate, int(info.samplerate)


def build_records(utterances: Iterable[Utterance], *, dataset_id: str, data_root: Path) -> list[ManifestRecord]:
    """Turn discovered utterances into manifest records, sorted by id.

    Every file is opened for its header, so a corrupt or missing file fails here
    rather than in training.
    """
    records: list[ManifestRecord] = []
    for item in utterances:
        duration, sample_rate = _duration_seconds(item.audio_path)
        records.append(
            ManifestRecord(
                utterance_id=item.utterance_id,
                audio_path=_relative_to(item.audio_path, data_root),
                text=read_transcript(item.text_path),
                speaker_id=item.speaker_id,
                prefix=item.prefix,
                shipped_split=item.shipped_split,
                duration_seconds=round(duration, 6),
                sample_rate=sample_rate,
                dataset_id=dataset_id,
            )
        )
    records.sort(key=lambda record: record.utterance_id)
    return records


def write_manifest(path: str | Path, records: Iterable[ManifestRecord]) -> int:
    """Write records as JSONL, overwriting ``path``. Returns the record count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL manifest into a list of dicts. Blank lines are skipped."""
    path = Path(path)
    if not path.is_file():
        raise CorpusError(f"manifest not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise CorpusError(f"invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows
