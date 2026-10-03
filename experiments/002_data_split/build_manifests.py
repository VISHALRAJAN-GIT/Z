"""EXP-002: build speaker-disjoint JSONL manifests for the IISc-MILE corpus.

Reads ``config.yaml``, discovers the corpus, plans a speaker-disjoint split, writes
``train/dev/test.jsonl`` plus ``metadata.json`` under the configured manifest
directory, and records the measured counts in ``results.json`` next to itself.

Run from anywhere:

    python experiments/002_data_split/build_manifests.py

Audio is never loaded: durations come from file headers. No audio enters version
control; the manifests do.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tamil_voice.data.corpus import discover_iisc_mile  # noqa: E402
from tamil_voice.data.manifest import ManifestRecord, build_records, write_manifest  # noqa: E402
from tamil_voice.data.splits import (  # noqa: E402
    SplitConfig,
    SplitRatios,
    check_speaker_disjoint,
    plan_speaker_split,
)
from tamil_voice.common.config import get_paths, load_yaml  # noqa: E402

SPLIT_ORDER: tuple[str, ...] = ("train", "dev", "test")


def summarise(records: list[ManifestRecord]) -> dict[str, Any]:
    """Measured summary of one split's records."""
    durations = [record.duration_seconds for record in records]
    prefixes: dict[str, int] = {}
    for record in records:
        prefixes[record.prefix] = prefixes.get(record.prefix, 0) + 1
    return {
        "utterances": len(records),
        "speakers": len({record.speaker_id for record in records}),
        "hours": round(sum(durations) / 3600.0, 4),
        "empty_transcripts": sum(1 for record in records if not record.text.strip()),
        "sample_rates": sorted({record.sample_rate for record in records}),
        "duration_seconds": {
            "min": round(min(durations), 4),
            "median": round(statistics.median(durations), 4),
            "max": round(max(durations), 4),
        },
        "prefixes": dict(sorted(prefixes.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build EXP-002 speaker-disjoint manifests.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    paths = get_paths()
    paths.ensure()

    corpus_cfg = config["corpus"]
    split_cfg = config["split"]
    out_cfg = config["output"]

    corpus_root = paths.root / corpus_cfg["root"]
    manifest_dir = paths.root / out_cfg["manifest_dir"]
    dataset_id = corpus_cfg["id"]

    split_config = SplitConfig(
        ratios=SplitRatios(**split_cfg["ratios"]),
        seed=int(split_cfg["seed"]),
    )

    print(f"discovering corpus under {corpus_root}")
    started = time.perf_counter()
    utterances = discover_iisc_mile(corpus_root)
    print(f"  found {len(utterances)} utterances in {time.perf_counter() - started:.1f}s")

    plan = plan_speaker_split(utterances, split_config)
    check_speaker_disjoint(plan)

    print("reading audio headers and transcripts (no audio decoded)")
    started = time.perf_counter()
    records_by_split = {
        name: build_records(plan.by_split[name], dataset_id=dataset_id, data_root=paths.data)
        for name in SPLIT_ORDER
    }
    built = sum(len(records) for records in records_by_split.values())
    print(f"  built {built} records in {time.perf_counter() - started:.1f}s")

    manifest_dir.mkdir(parents=True, exist_ok=True)
    written = {
        name: write_manifest(manifest_dir / f"{name}.jsonl", records_by_split[name]) for name in SPLIT_ORDER
    }

    summary = {name: summarise(records_by_split[name]) for name in SPLIT_ORDER}
    total = summarise([record for name in SPLIT_ORDER for record in records_by_split[name]])

    speaker_sets = {name: set(plan.speakers(name)) for name in SPLIT_ORDER}
    overlap = {
        f"{first}&{second}": sorted(speaker_sets[first] & speaker_sets[second])
        for index, first in enumerate(SPLIT_ORDER)
        for second in SPLIT_ORDER[index + 1 :]
    }

    metadata_path = manifest_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset_version": split_cfg["dataset_version"],
                "license": corpus_cfg["license"],
                "source": corpus_cfg["source"],
                "corpus_root": corpus_cfg["root"],
                "split_seed": split_config.seed,
                "ratios": split_config.ratios.as_dict(),
                "splits": summary,
                "speakers": {name: sorted(speaker_sets[name]) for name in SPLIT_ORDER},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    results: dict[str, Any] = {
        "experiment": "EXP-002",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": (
            "A deterministic, speaker-disjoint split of the IISc-MILE Tamil corpus can be "
            "produced from its filenames and recorded as versioned JSONL manifests."
        ),
        "dataset": {
            "id": dataset_id,
            "version": split_cfg["dataset_version"],
            "license": corpus_cfg["license"],
            "source": corpus_cfg["source"],
            "corpus_root": corpus_cfg["root"],
        },
        "split": {"seed": split_config.seed, "ratios": split_config.ratios.as_dict()},
        "artifacts": {
            **{name: f"{out_cfg['manifest_dir']}/{name}.jsonl" for name in SPLIT_ORDER},
            "metadata": f"{out_cfg['manifest_dir']}/metadata.json",
        },
        "splits": summary,
        "total": total,
        "speaker_overlap": overlap,
        "records_written": written,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    Path(__file__).with_name("results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n== EXP-002 split summary ==")
    for name in SPLIT_ORDER:
        size = summary[name]
        rate = "" if size["sample_rates"] == [16000] else f"  rates={size['sample_rates']}"
        print(
            f"  {name:5s}  utts={size['utterances']:6d}  speakers={size['speakers']:4d}  "
            f"hours={size['hours']:8.3f}{rate}"
        )
    print(f"  total  utts={total['utterances']}  speakers={total['speakers']}  hours={total['hours']}")
    print("  speaker overlap:", {key: len(value) for key, value in overlap.items()})
    print(f"  wrote manifests to {manifest_dir}")

    if any(overlap.values()):
        print("ERROR: splits are not speaker-disjoint", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
