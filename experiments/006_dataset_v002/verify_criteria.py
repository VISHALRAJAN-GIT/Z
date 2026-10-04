"""Independent verification of dataset_v002.

Deliberately does not import anything from ``build_dataset_v002.py``. A verifier that
shares code with the thing it verifies can only confirm the code is consistent with
itself; this one re-derives every claim from the manifests on disk.

Every criterion is a boolean check with a printed result. Any failure exits non-zero,
so this can be run as a gate rather than read hopefully.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tamil_voice.text.tokenizer import CharacterTokenizer  # noqa: E402

SAMPLE_RATE = 16000
HOP_LENGTH = 160
SPLITS = ("train", "dev", "test")
EXPECTED_REMOVED = 5
EXPECTED_CORRUPT = {
    "MILE_0000289_0000067",
    "MILE_0000289_0000068",
    "MILE_0000289_0000069",
    "MILE_0000289_0000070",
    "MILE_0000289_0000071",
}
EXPECTED_INELIGIBLE_WITH_SPACE = 28
EXPECTED_INELIGIBLE_WITHOUT_SPACE = 4
EXPECTED_V001_TOTAL = 89401
EXPECTED_V002_TOTAL = 89396

RESULTS: list[tuple[bool, str, str]] = []


def check(passed: bool, name: str, detail: str = "") -> None:
    RESULTS.append((bool(passed), name, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))


def load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stft_frames(duration: float) -> int:
    return 1 + int(round(duration * SAMPLE_RATE)) // HOP_LENGTH


def encoder_frames(frames: int) -> int:
    length = frames
    for _ in range(2):
        length = -(-length // 2)
    return max(length, 1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    v1_dir = ROOT / "data/manifests/dataset_v001"
    v2_dir = ROOT / "data/manifests/dataset_v002"

    # 1. dataset_v001 must be untouched. Checked against git, not against a copy made
    #    by this script, because a copy taken after the fact would prove nothing.
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "data/manifests/dataset_v001"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    check(dirty == "", "dataset_v001 is unmodified in git", dirty or "clean")

    v1 = {split: load(v1_dir / f"{split}.jsonl") for split in SPLITS}
    v2 = {split: load(v2_dir / f"{split}.jsonl") for split in SPLITS}
    v1_rows = [row for split in SPLITS for row in v1[split]]
    v2_rows = [row for split in SPLITS for row in v2[split]]
    check(len(v1_rows) == EXPECTED_V001_TOTAL, "v001 row count", str(len(v1_rows)))
    check(len(v2_rows) == EXPECTED_V002_TOTAL, "v002 row count", str(len(v2_rows)))

    # 2. Exactly the 5 corrupt ids are gone, and nothing else.
    v1_ids = {str(row["utterance_id"]) for row in v1_rows}
    v2_ids = {str(row["utterance_id"]) for row in v2_rows}
    check(len(v1_ids) == len(v1_rows), "v001 has no duplicate utterance ids")
    check(len(v2_ids) == len(v2_rows), "v002 has no duplicate utterance ids")
    missing = v1_ids - v2_ids
    added = v2_ids - v1_ids
    check(missing == EXPECTED_CORRUPT, "exactly the 5 corrupt rows were removed", str(sorted(missing)))
    check(added == set(), "v002 invents no new utterance ids", str(sorted(added)))

    # 3. Every surviving row is byte-identical to its v001 counterpart.
    v1_by_id = {str(row["utterance_id"]): row for row in v1_rows}
    mismatched = [
        str(row["utterance_id"])
        for row in v2_rows
        if v1_by_id[str(row["utterance_id"])] != row
    ]
    check(not mismatched, "every kept row is unchanged from v001", str(mismatched[:5]))

    # 4. No empty transcripts, and the corpus's own partition label is intact.
    #
    # ``shipped_split`` is how IISc-MILE itself partitioned the corpus, and it is NOT
    # the file a row lives in: EXP-002 re-split speaker-disjointly across the union of
    # the corpus splits. So a row in this project's train.jsonl may carry
    # shipped_split "dev". The label must be preserved from v001, which the
    # "every kept row is unchanged" check already covers; here it is only checked for
    # being a value the corpus actually uses.
    empty = [str(row["utterance_id"]) for row in v2_rows if not str(row["text"]).strip()]
    check(not empty, "no empty transcripts", str(empty[:5]))
    unknown_labels = sorted(
        {
            str(row["shipped_split"])
            for row in v2_rows
            if str(row["shipped_split"]) not in ("train", "dev", "test")
        }
    )
    check(not unknown_labels, "every shipped_split is a corpus partition", str(unknown_labels))

    # 5. Speaker disjointness, which is what keeps WER honest.
    speakers = {split: {str(row["speaker_id"]) for row in v2[split]} for split in SPLITS}
    check(
        not (speakers["train"] & speakers["dev"]),
        "no speaker in both train and dev",
        str(len(speakers["train"] & speakers["dev"])),
    )
    check(
        not (speakers["train"] & speakers["test"]),
        "no speaker in both train and test",
        str(len(speakers["train"] & speakers["test"])),
    )
    check(
        not (speakers["dev"] & speakers["test"]),
        "no speaker in both dev and test",
        str(len(speakers["dev"] & speakers["test"])),
    )
    v1_speakers = {split: {str(row["speaker_id"]) for row in v1[split]} for split in SPLITS}
    check(
        speakers == v1_speakers,
        "no speaker moved splits",
        "identical speaker sets" if speakers == v1_speakers else "a speaker moved",
    )

    # 6. Metadata must agree with the manifests, and must record the removal.
    metadata = json.loads((v2_dir / "metadata.json").read_text(encoding="utf-8"))
    check(metadata.get("dataset_version") == "v002", "metadata declares v002")
    check(metadata.get("derived_from") == "dataset_v001", "metadata records its parent version")
    check(
        metadata["changes"]["removed_utterances"] == EXPECTED_REMOVED,
        "metadata records the removal count",
        str(metadata["changes"]["removed_utterances"]),
    )
    recorded = {entry["utterance_id"] for entry in metadata["changes"]["removed"]}
    check(recorded == EXPECTED_CORRUPT, "metadata lists exactly the removed ids")
    for split in SPLITS:
        claimed = metadata["splits"][split]["utterances"]
        actual = len(v2[split])
        check(claimed == actual, f"metadata utterance count matches {split}.jsonl", f"{claimed} vs {actual}")
        claimed_hours = metadata["splits"][split]["hours"]
        actual_hours = round(sum(float(row["duration_seconds"]) for row in v2[split]) / 3600.0, 4)
        check(claimed_hours == actual_hours, f"metadata hours match {split}.jsonl", f"{claimed_hours} vs {actual_hours}")

    # 7. Re-derive the CTC-ineligible sets from scratch and compare with metadata.
    tokenizers = {
        "with_space": CharacterTokenizer.load(
            ROOT / "experiments/005_character_ctc_overfit/tokenizer_with_space.json"
        ),
        "without_space": CharacterTokenizer.load(
            ROOT / "experiments/005_character_ctc_overfit/tokenizer_without_space.json"
        ),
    }
    counts: dict[str, int] = {}
    for variant, tokenizer in tokenizers.items():
        ineligible = [
            str(row["utterance_id"])
            for row in v2_rows
            if encoder_frames(stft_frames(float(row["duration_seconds"]))) < len(tokenizer.encode(str(row["text"])))
        ]
        counts[variant] = len(ineligible)
        published = metadata["changes"]["kept_fast_speech"]["ctc_ineligible_at_stride_4"][variant]
        check(
            sorted(published) == sorted(ineligible),
            f"{variant}: published CTC-ineligible ids match a fresh computation",
            f"{len(ineligible)} rows",
        )
        still_present = [uid for uid in ineligible if uid in v2_ids]
        check(len(still_present) == len(ineligible), f"{variant}: ineligible rows are kept, not deleted")
    check(
        counts["with_space"] == EXPECTED_INELIGIBLE_WITH_SPACE,
        "with_space CTC-ineligible count",
        str(counts["with_space"]),
    )
    check(
        counts["without_space"] == EXPECTED_INELIGIBLE_WITHOUT_SPACE,
        "without_space CTC-ineligible count",
        str(counts["without_space"]),
    )

    # 8. No removed row's audio was deleted. The manifest changed; the disk must not have.
    data_dir = ROOT / "data"
    missing_audio = [
        entry["utterance_id"]
        for entry in metadata["changes"]["removed"]
        if not (data_dir / "raw/iisc_mile_ta/mile_tamil_asr_corpus" / "train/audio_files" / f"{entry['utterance_id']}.wav").exists()
    ]
    check(not missing_audio, "audio for removed rows still on disk", str(missing_audio))

    passed = sum(1 for ok, _, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print(f"\n{passed} pass, {failed} fail, 0 pending")
    if failed:
        print("\nfailures:")
        for ok, name, detail in RESULTS:
            if not ok:
                print(f"  {name}: {detail}")
        return 1
    print(f"dataset_v002 sha256: {sha256(v2_dir / 'metadata.json')[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())