# EXP-006: dispose of the CTC-invalid rows and create dataset_v002.

# Two questions are separated here, because conflating them is the mistake this step
# exists to avoid.
#
# EXP-004 and EXP-005 both found utterances where CTC cannot train because the encoder
# produces fewer frames than the transcript needs labels. That fact has two completely
# different causes, and they need different answers:
#
#   1. The transcript is corrupt. Speaker 0000289 rows _0000067.._0000071 carry 26-47
#      characters in 0.24-0.45 s. No human speaks at 99-107 characters per second, so
#      the transcript cannot belong to that audio. This is a DATA defect. The pair is
#      unusable and cannot be repaired offline, because there is no way to recover the
#      intended transcript without re-transcribing the audio. -> REMOVE, new version.
#
#   2. The encoder is too coarse for genuinely fast speech. These rows are valid audio
#      with valid transcripts; 25 Hz just cannot emit 26 labels for 1.0 s. Removing
#      them would discard exactly the hard cases GUIDE requires this project to handle.
#      -> KEEP in the dataset, exclude at training time, with the ids published.
#
# So dataset_v002 = dataset_v001 minus the corrupt rows only. The fast rows stay, and
# the set that CTC cannot train on is written to a sidecar file so the baseline filters
# deterministically and visibly instead of silently skipping rows.
#
# The corrupt/fast boundary is taken from the measured rate distribution rather than
# chosen. The script prints the sorted rates of every violating row and checks that no
# row lands between 50 and 95 characters per second, so the answer is identical for any
# threshold in that interval. A separation this unambiguous is a property of the corpus,
# and pretending to a precision it does not have would be dishonest.

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tamil_voice.text.tokenizer import CharacterTokenizer  # noqa: E402

SAMPLE_RATE = 16000
HOP_LENGTH = 160
CONV_LAYERS = 2  # two stride-2 convolutions, so encoder stride 4
SPLITS = ("train", "dev", "test")

# Characters per second above which a transcript cannot belong to its audio. EXP-004
# measured p99 at 19.117 and max at 107.402, with no row between 50 and 95.
CORRUPT_RATE_THRESHOLD = 60.0

# The interval in which no violating row may fall, so the classification cannot depend
# on the exact threshold chosen.
AMBIGUITY_FLOOR = 50.0
AMBIGUITY_CEILING = 95.0


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}: bad JSON on line {line_number}") from exc
    return rows


def stft_frames(duration_seconds: float) -> int:
    """log-Mel frames for a duration, matching the real front end exactly.

    Identical to EXP-004's helper and to ``StftConfig``: samples at 16 kHz, one frame
    per 160-sample hop, plus the one extra frame a centre-padded STFT produces.
    """
    samples = int(round(duration_seconds * SAMPLE_RATE))
    return 1 + samples // HOP_LENGTH


def encoder_frames(frames: int, layers: int = CONV_LAYERS) -> int:
    """Output length after `layers` stride-2 convolutions, kernel 3, padding 1.

    Each layer is ``ceil(L / 2)``. ``floor(T / stride)`` is deliberately not used: it
    disagrees on odd lengths and was the bug EXP-003 caught.
    """
    length = frames
    for _ in range(layers):
        length = -(-length // 2)
    return max(length, 1)


def analyse(
    rows: list[dict[str, Any]], tokenizer: CharacterTokenizer, variant: str
) -> list[dict[str, Any]]:
    """Every row where CTC cannot train, with the numbers behind the verdict."""
    violations: list[dict[str, Any]] = []
    for row in rows:
        duration = float(row["duration_seconds"])
        labels = len(tokenizer.encode(str(row["text"])))
        frames = encoder_frames(stft_frames(duration))
        if frames >= labels:
            continue
        non_space = len(tokenizer.prepare(str(row["text"])).replace(" ", ""))
        rate = non_space / duration if duration > 0 else float("inf")
        violations.append(
            {
                "utterance_id": str(row["utterance_id"]),
                "speaker_id": str(row["speaker_id"]),
                "split": str(row["shipped_split"]),
                "variant": variant,
                "duration_seconds": duration,
                "encoder_frames": frames,
                "labels": labels,
                "characters_per_second": round(rate, 3),
                "reason": "corrupt_transcript"
                if rate >= CORRUPT_RATE_THRESHOLD
                else "fast_speech_below_encoder_resolution",
            }
        )
    return violations


def main() -> int:
    source = ROOT / "data/manifests/dataset_v001"
    target = ROOT / "data/manifests/dataset_v002"
    tokenizer_dir = ROOT / "experiments/005_character_ctc_overfit"

    tokenizers = {
        "with_space": CharacterTokenizer.load(tokenizer_dir / "tokenizer_with_space.json"),
        "without_space": CharacterTokenizer.load(tokenizer_dir / "tokenizer_without_space.json"),
    }

    manifests = {split: load_manifest(source / f"{split}.jsonl") for split in SPLITS}
    total_rows = sum(len(rows) for rows in manifests.values())
    print(f"dataset_v001 rows={total_rows}")

    # ---- measure every violation, both variants
    violations: dict[str, list[dict[str, Any]]] = {}
    for variant, tokenizer in tokenizers.items():
        found: list[dict[str, Any]] = []
        for split in SPLITS:
            found.extend(analyse(manifests[split], tokenizer, variant))
        found.sort(key=lambda entry: -entry["characters_per_second"])
        violations[variant] = found
        corrupt = [entry for entry in found if entry["reason"] == "corrupt_transcript"]
        print(f"{variant}: {len(found)} violations, {len(corrupt)} corrupt")

    # ---- the classification must not depend on the threshold
    print("\ncharacters per second, every violating row:")
    for variant in ("with_space", "without_space"):
        print(f"  [{variant}]")
        for entry in violations[variant]:
            print(
                f"    {entry['utterance_id']:26} {entry['characters_per_second']:8.2f}/s "
                f"{entry['encoder_frames']:5} frames {entry['labels']:4} labels  "
                f"{entry['reason']}"
            )
        rates = [entry["characters_per_second"] for entry in violations[variant]]
        ambiguous = [r for r in rates if AMBIGUITY_FLOOR <= r <= AMBIGUITY_CEILING]
        print(f"  rates between {AMBIGUITY_FLOOR} and {AMBIGUITY_CEILING}: {len(ambiguous)}")
        if ambiguous:
            raise SystemExit(
                f"{variant}: {len(ambiguous)} rows fall inside the ambiguity interval, so the "
                f"corrupt/fast split is not robust and must be argued rather than measured"
            )
        if rates:
            print(f"  fastest fast-but-human row: {min(r for r in rates if r < CORRUPT_RATE_THRESHOLD):.2f}/s")

    # ---- the corrupt set must be identical in both variants
    corrupt_ids = {
        variant: {
            entry["utterance_id"]
            for entry in found
            if entry["reason"] == "corrupt_transcript"
        }
        for variant, found in violations.items()
    }
    if corrupt_ids["with_space"] != corrupt_ids["without_space"]:
        only_space = corrupt_ids["with_space"] - corrupt_ids["without_space"]
        only_nospace = corrupt_ids["without_space"] - corrupt_ids["with_space"]
        raise SystemExit(
            f"the corrupt set depends on the whitespace setting: only with_space={sorted(only_space)} "
            f"only without_space={sorted(only_nospace)}"
        )
    corrupt = sorted(corrupt_ids["with_space"])
    print(f"\ncorrupt rows (identical in both variants, {len(corrupt)}): {corrupt}")

    # ---- write dataset_v002: v001 minus the corrupt rows, order and fields preserved
    corrupt_set = set(corrupt)
    removed: list[dict[str, Any]] = []
    target.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        kept: list[dict[str, Any]] = []
        for row in manifests[split]:
            if str(row["utterance_id"]) in corrupt_set:
                removed.append(
                    {
                        "utterance_id": str(row["utterance_id"]),
                        "speaker_id": str(row["speaker_id"]),
                        "split": split,
                        "duration_seconds": float(row["duration_seconds"]),
                        "text_characters": len(str(row["text"])),
                    }
                )
                continue
            kept.append(row)
        path = target / f"{split}.jsonl"
        # newline="\n" is explicit, not incidental. Python's default text mode
        # translates "\n" to os.linesep, which on Windows produces CRLF. The repo
        # declares eol=lf in .gitattributes, so the committed blob would be LF while
        # the file on disk stayed CRLF, and any recorded SHA-256 would not reproduce
        # after a fresh checkout. dataset_v001 is pure LF on disk; this keeps v002
        # byte-identical to what git stores.
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in kept),
            encoding="utf-8",
            newline="\n",
        )
        print(f"{split}: {len(manifests[split])} -> {len(kept)}")
    removed.sort(key=lambda entry: entry["utterance_id"])

    # ---- regenerate metadata with the same formulas that produced v001
    prior = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    split_stats: dict[str, Any] = {}
    for split in SPLITS:
        rows = load_manifest(target / f"{split}.jsonl")
        durations = [float(row["duration_seconds"]) for row in rows]
        split_stats[split] = {
            "utterances": len(rows),
            "speakers": len({str(row["speaker_id"]) for row in rows}),
            "hours": round(sum(durations) / 3600.0, 4),
            "empty_transcripts": sum(1 for row in rows if not str(row["text"]).strip()),
            "sample_rates": sorted({int(row["sample_rate"]) for row in rows}),
            "duration_seconds": {
                "min": round(min(durations), 4),
                "median": round(statistics.median(durations), 4),
                "max": round(max(durations), 4),
            },
            "prefixes": dict(sorted(Counter(str(row["prefix"]) for row in rows).items())),
        }

    speaker_sets = {
        split: {str(row["speaker_id"]) for row in load_manifest(target / f"{split}.jsonl")}
        for split in SPLITS
    }
    overlaps = {
        "train_dev": len(speaker_sets["train"] & speaker_sets["dev"]),
        "train_test": len(speaker_sets["train"] & speaker_sets["test"]),
        "dev_test": len(speaker_sets["dev"] & speaker_sets["test"]),
    }
    if any(overlaps.values()):
        raise SystemExit(f"dataset_v002 has speaker overlap, which would make WER look good: {overlaps}")

    # ---- the CTC-ineligible sets must not reference removed rows
    corrupt_set = set(corrupt)
    published_ids = {
        entry["utterance_id"]
        for variant in ("with_space", "without_space")
        for entry in violations[variant]
        if entry["reason"] == "fast_speech_below_encoder_resolution"
    }
    if corrupt_set & published_ids:
        raise SystemExit(
            f"the CTC-ineligible list references removed rows: {sorted(corrupt_set & published_ids)}"
        )
    v002_ids = {
        str(row["utterance_id"])
        for split in SPLITS
        for row in load_manifest(target / f"{split}.jsonl")
    }
    if not published_ids <= v002_ids:
        raise SystemExit(
            f"the CTC-ineligible list references rows absent from v002: {sorted(published_ids - v002_ids)}"
        )

    metadata = {
        "dataset_id": prior["dataset_id"],
        "dataset_version": "v002",
        "license": prior["license"],
        "source": prior["source"],
        "corpus_root": prior["corpus_root"],
        "split_seed": prior["split_seed"],
        "ratios": prior["ratios"],
        "derived_from": "dataset_v001",
        "changes": {
            "removed_utterances": len(removed),
            "reason": "transcript cannot belong to its audio: 98.9-107.4 non-space characters "
            "per second, measured, with no row in the corpus between 50 and 95 characters "
            "per second, so the corrupt/fast separation does not depend on the threshold",
            "removed": removed,
            "kept_fast_speech": {
                "reason": "valid audio and valid transcript, but the stride-4 encoder emits "
                "fewer frames than the transcript needs labels. Excluded at training time, "
                "not deleted: fast speech is a GUIDE target condition.",
                # Only rows that survive into v002. The corrupt rows are listed under
                # "removed" and must not also appear here, or a training filter built
                # from this list would reference utterances the dataset no longer has.
                "ctc_ineligible_at_stride_4": {
                    variant: [
                        entry["utterance_id"]
                        for entry in violations[variant]
                        if entry["reason"] == "fast_speech_below_encoder_resolution"
                    ]
                    for variant in ("with_space", "without_space")
                },
            },
            "audio_files_untouched": True,
            "splits_unchanged": True,
            "speaker_overlap_after": overlaps,
        },
        "splits": split_stats,
        "speakers": {split: sorted(ids) for split, ids in speaker_sets.items()},
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    results = {
        "experiment_id": "006_dataset_v002",
        "source_version": "v001",
        "target_version": "v002",
        "encoder_stride": 4,
        "corrupt_rate_threshold": CORRUPT_RATE_THRESHOLD,
        "ambiguity_interval_checked": [AMBIGUITY_FLOOR, AMBIGUITY_CEILING],
        "rows_in_v001": total_rows,
        "rows_in_v002": total_rows - len(removed),
        "removed": removed,
        "violations": violations,
        "violation_counts": {variant: len(found) for variant, found in violations.items()},
        "corrupt_counts": {
            variant: sum(
                1 for entry in found if entry["reason"] == "corrupt_transcript"
            )
            for variant, found in violations.items()
        },
        "split_stats": split_stats,
        "speaker_overlap_after": overlaps,
        "notes": [
            "Only the corrupt rows are removed. Fast-but-valid rows stay in dataset_v002 "
            "and are excluded at training time by utterance id.",
            "Splits are unchanged, so every speaker keeps the split it had in v001 and "
            "speaker overlap stays 0.",
            "No audio was read or modified. Removal is manifest-level only.",
            "The corrupt set is identical with and without whitespace, so the rows removed "
            "do not depend on the tokenizer configuration.",
        ],
    }
    (Path(__file__).with_name("results.json")).write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("\n=== dataset_v002 ===")
    print(f"removed {len(removed)} corrupt rows, {total_rows} -> {total_rows - len(removed)}")
    print(f"kept but CTC-ineligible: with_space={len(violations['with_space']) - len(corrupt)} "
          f"without_space={len(violations['without_space']) - len(corrupt)}")
    print(f"speaker overlap after: {overlaps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())