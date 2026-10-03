"""IISc-MILE Tamil ASR corpus discovery.

The corpus ships as ``train/`` and ``test/`` folders, each holding ``audio_files/``
of 16 kHz mono PCM WAV and ``trans_files/`` of matching UTF-8 transcripts. The
shipped split is **not** speaker-disjoint (the MILE and MICI speakers appear in
both halves), so it is recorded on every utterance but is never used for
splitting; see ``splits.py``.

Filename scheme, verified against the corpus and OpenSLR SLR127::

    <PREFIX>_<SPEAKER>_<UTTERANCE>.wav

* ``PREFIX`` is a recording batch: ``ISTL``, ``MILE`` or ``MICI``.
* ``SPEAKER`` is a 7-digit id, unique across prefixes. The union over all prefixes
  is exactly 531, matching OpenSLR's "531 speakers". A speaker can appear under
  more than one prefix (107 do), because the prefix is a condition, not an
  identity.
* ``UTTERANCE`` is a 7-digit index within the speaker.

Discovery reads directory entries and transcript text only. It never decodes
audio, so it stays fast on the ~89k-file corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Batch/condition prefix followed by 7-digit speaker and utterance ids.
NAME_PATTERN = re.compile(r"^(?P<prefix>[A-Za-z]+)_(?P<speaker>\d+)_(?P<utterance>\d+)$")

#: The two folders the corpus ships with, in a fixed order for determinism.
SHIPPED_SPLITS: tuple[str, ...] = ("train", "test")

#: Directory names inside each shipped split.
AUDIO_DIRNAME = "audio_files"
TRANSCRIPT_DIRNAME = "trans_files"


class CorpusError(Exception):
    """The corpus is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Utterance:
    """One audio file and its transcript, as found on disk."""

    utterance_id: str
    speaker_id: str
    prefix: str
    shipped_split: str
    audio_path: Path
    text_path: Path

    @property
    def filename(self) -> str:
        return self.audio_path.name


def parse_iisc_mile_name(stem: str) -> tuple[str, str, str]:
    """Split a filename stem into ``(prefix, speaker_id, utterance_id)``.

    Raises ``CorpusError`` when the name does not follow the corpus scheme, so a
    renamed or foreign file is reported rather than silently mis-labelled.
    """
    match = NAME_PATTERN.match(stem)
    if match is None:
        raise CorpusError(f"filename does not match <PREFIX>_<SPEAKER>_<UTTERANCE>: {stem!r}")
    return match.group("prefix"), match.group("speaker"), match.group("utterance")


def read_transcript(path: Path) -> str:
    """Read a UTF-8 transcript and strip surrounding whitespace.

    An empty transcript is returned as an empty string; the caller decides whether
    to keep it. Missing or undecodable files raise, because a silently dropped
    utterance is a dataset bug that surfaces much later as an unexplained gap.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CorpusError(f"cannot read transcript {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise CorpusError(f"transcript is not valid UTF-8: {path}: {exc}") from exc
    return text.strip()


def discover_iisc_mile(root: str | Path) -> list[Utterance]:
    """Find every utterance under a corpus root, in a deterministic order.

    ``root`` is the folder that contains ``train/`` and ``test/`` (the extracted
    ``mile_tamil_asr_corpus`` directory). Utterances are returned sorted by
    ``(shipped_split, utterance_id)`` so the same tree always yields the same list.
    """
    root = Path(root)
    if not root.is_dir():
        raise CorpusError(f"corpus root is not a directory: {root}")

    utterances: list[Utterance] = []
    for split in SHIPPED_SPLITS:
        audio_dir = root / split / AUDIO_DIRNAME
        text_dir = root / split / TRANSCRIPT_DIRNAME
        if not audio_dir.is_dir():
            raise CorpusError(f"missing audio directory: {audio_dir}")
        if not text_dir.is_dir():
            raise CorpusError(f"missing transcript directory: {text_dir}")

        wavs = sorted(audio_dir.glob("*.wav"))
        if not wavs:
            raise CorpusError(f"no .wav files under {audio_dir}")
        for wav in wavs:
            prefix, speaker, _ = parse_iisc_mile_name(wav.stem)
            text_path = text_dir / f"{wav.stem}.txt"
            if not text_path.is_file():
                raise CorpusError(f"missing transcript for {wav.name}: {text_path}")
            utterances.append(
                Utterance(
                    utterance_id=wav.stem,
                    speaker_id=speaker,
                    prefix=prefix,
                    shipped_split=split,
                    audio_path=wav,
                    text_path=text_path,
                )
            )
    return utterances
