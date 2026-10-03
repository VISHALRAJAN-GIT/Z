"""EXP-002 verification: re-check the written manifests independently.

``build_manifests.py`` reports its own numbers. This script does not trust them. It
re-reads the JSONL files from disk and re-derives every claim from the bytes:

1. the three files parse as JSONL and their line counts match ``results.json``
2. no speaker appears in two splits
3. no utterance appears twice within a split or across splits
4. no transcript is empty and none contains a Unicode replacement character
5. every audio path is relative, exists on disk, and its header length agrees with
   the recorded duration
6. the recorded durations sum to the reported hours

Exit code 0 means every criterion passed. Run after ``build_manifests.py``:

    python experiments/002_data_split/verify_manifests.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import soundfile as sf  # noqa: E402

from tamil_voice.common.config import get_paths  # noqa: E402

SPLIT_ORDER: tuple[str, ...] = ("train", "dev", "test")
SAMPLE_RATE_TOLERANCE = 1.0 / 16000.0


class Report:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []

    def check(self, name: str, ok: bool, detail: str) -> None:
        (self.passed if ok else self.failed).append(f"{name}: {detail}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON on line {number} of {path}: {exc}") from exc
    return rows


def main() -> int:
    report = Report()
    paths = get_paths()
    manifest_dir = paths.root / "data/manifests/dataset_v001"
    results_path = Path(__file__).with_name("results.json")

    if not results_path.is_file():
        raise SystemExit(f"results.json not found: {results_path}")
    results = json.loads(results_path.read_text(encoding="utf-8"))

    rows: dict[str, list[dict[str, Any]]] = {}
    for name in SPLIT_ORDER:
        path = manifest_dir / f"{name}.jsonl"
        if not path.is_file():
            raise SystemExit(f"manifest not found: {path}")
        rows[name] = load_rows(path)
        reported = results["records_written"][name]
        report.check(
            f"1.{name} line count",
            len(rows[name]) == reported,
            f"{len(rows[name])} rows on disk, results.json reports {reported}",
        )

    speakers: dict[str, set[str]] = {
        name: {row["speaker_id"] for row in rows[name]} for name in SPLIT_ORDER
    }
    overlaps = {}
    for index, first in enumerate(SPLIT_ORDER):
        for second in SPLIT_ORDER[index + 1 :]:
            shared = speakers[first] & speakers[second]
            overlaps[f"{first}&{second}"] = shared
            report.check(
                f"2.{first}&{second} speaker disjoint",
                not shared,
                f"{len(shared)} shared speakers out of {len(speakers[first])} and {len(speakers[second])}",
            )
    total_speakers = len(speakers["train"]) + len(speakers["dev"]) + len(speakers["test"])
    report.check(
        "2.total speakers",
        total_speakers == results["total"]["speakers"],
        f"{total_speakers} distinct speakers, results.json reports {results['total']['speakers']}",
    )

    all_ids: dict[str, str] = {}
    duplicates: list[str] = []
    for name in SPLIT_ORDER:
        for row in rows[name]:
            previous = all_ids.get(row["utterance_id"])
            if previous is not None:
                duplicates.append(f"{row['utterance_id']} in {previous} and {name}")
            else:
                all_ids[row["utterance_id"]] = name
    report.check("3.utterance ids unique", not duplicates, f"{len(duplicates)} duplicates")

    empty = sum(1 for name in SPLIT_ORDER for row in rows[name] if not row["text"].strip())
    replaced = sum(
        1 for name in SPLIT_ORDER for row in rows[name] if "\ufffd" in row["text"]
    )
    report.check("4a.no empty transcripts", empty == 0, f"{empty} empty")
    report.check("4b.no replacement characters", replaced == 0, f"{replaced} rows contain U+FFFD")

    data_root = paths.data
    missing: list[str] = []
    absolute: list[str] = []
    for name in SPLIT_ORDER:
        for row in rows[name]:
            candidate = Path(row["audio_path"])
            if candidate.is_absolute():
                absolute.append(row["audio_path"])
                continue
            if not (data_root / candidate).is_file():
                missing.append(row["audio_path"])
    report.check("5a.audio paths relative", not absolute, f"{len(absolute)} absolute paths")
    report.check("5b.audio files exist", not missing, f"{len(missing)} missing of {len(all_ids)}")

    # Re-measure a deterministic sample rather than all 89k files, and say so.
    step = max(1, len(all_ids) // 2000)
    sampled_ids = sorted(all_ids)[::step][:2000]
    mismatches: list[str] = []
    rates: set[int] = set()
    sampled_seconds = 0.0
    for utterance_id in sampled_ids:
        row = next(
            candidate
            for candidate in rows[all_ids[utterance_id]]
            if candidate["utterance_id"] == utterance_id
        )
        absolute_path = data_root / row["audio_path"]
        if not absolute_path.is_file():
            continue
        info = sf.info(str(absolute_path))
        rates.add(int(info.samplerate))
        actual = info.frames / info.samplerate
        sampled_seconds += actual
        if abs(actual - float(row["duration_seconds"])) > SAMPLE_RATE_TOLERANCE:
            mismatches.append(f"{utterance_id}: manifest {row['duration_seconds']} vs file {actual:.6f}")
    report.check(
        "5c.durations match file headers",
        not mismatches,
        f"{len(sampled_ids)} sampled every {step}th utterance, {len(mismatches)} mismatches, rates={sorted(rates)}",
    )

    recorded_hours = sum(
        float(row["duration_seconds"]) for name in SPLIT_ORDER for row in rows[name]
    ) / 3600.0
    reported_hours = float(results["total"]["hours"])
    report.check(
        "6.total hours",
        abs(recorded_hours - reported_hours) < 0.01,
        f"{recorded_hours:.4f} h from the manifests, results.json reports {reported_hours:.4f} h",
    )

    print("== EXP-002 manifest verification ==")
    for line in report.passed:
        print(f"  PASS  {line}")
    for line in report.failed:
        print(f"  FAIL  {line}")
    print(f"\n{len(report.passed)} pass, {len(report.failed)} fail")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())