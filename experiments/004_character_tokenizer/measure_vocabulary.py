"""EXP-004: measure the tokenizer options before choosing one.

This script decides nothing. It produces numbers so that the choice in GUIDE
section 28 can be made from measurement instead of preference:

    "Start with character-level tokens."  "Do not assume BPE is automatically
    better. Benchmark it."

Measured here, all from `dataset_v001`:

1. Unicode normalization audit. Tamil has composed and decomposed spellings of
   the same grapheme; a character tokenizer that counts codepoints is only
   meaningful once it is known whether the corpus is already in one form.
2. Character inventory, split by script, and character-level OOV on dev/test.
3. Target-sequence length cost. Character targets are roughly six times longer
   than word targets, and CTC cannot train where encoder frames < labels. The
   fraction of utterances that violate that, at several encoder strides, is the
   number that decides whether character-level CTC is viable on this corpus.
4. The word-level most-frequent-N OOV curve, for comparison against (2).
5. Output-layer parameter cost of each option, against the project's < 100 MB
   weight target.

Run:  python experiments/004_character_tokenizer/measure_vocabulary.py
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = Path(__file__).parent
SPLITS = ("train", "dev", "test")

HOP_LENGTH = 160
SAMPLE_RATE = 16000
N_MELS = 80

# Word-frequency restriction sizes to measure. Includes the full vocabulary so the
# curve ends at the same point EXP-003 step 1 already measured.
TOP_N = (100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 138047)

# Total encoder time strides to test character-level CTC against, as powers of
# two so each one is a whole number of stride-2 convolutions. EXP-003 used 4
# (two convs), which runs the GRU at 25 Hz over the 100 Hz log-Mel front end.
STRIDES = (4, 8, 16, 32)

# Output-layer width used to price each vocabulary. Matches the trained EXP-003
# model so the comparison is like for like.
HIDDEN = 256


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


def script_of(char: str) -> str:
    codepoint = ord(char)
    if 0x0B80 <= codepoint <= 0x0BFF:
        return "tamil_block"
    if char.isdigit() and codepoint < 128:
        return "ascii_digit"
    if ("a" <= char <= "z") or ("A" <= char <= "Z"):
        return "ascii_latin"
    if codepoint < 128:
        return "ascii_other"
    if char.isspace():
        return "whitespace"
    return "other_non_tamil"


def stft_frames(duration_seconds: float) -> int:
    """log-Mel frames for a duration, matching the EXP-003 front end exactly."""
    samples = int(round(duration_seconds * SAMPLE_RATE))
    return 1 + samples // HOP_LENGTH


def encoder_frames(frames: int, layers: int) -> int:
    """Output length after `layers` stride-2 convolutions.

    Mirrors `TinyCTC.subsampled_length` from EXP-003 step 2 for kernel 3,
    padding 1: each layer is ceil(L / 2). `floor(T / stride)` is deliberately
    not used; it disagrees on odd lengths and was the bug EXP-003 caught.
    """
    length = frames
    for _ in range(layers):
        length = -(-length // 2)
    return max(length, 1)


def main() -> int:
    manifest_dir = ROOT / "data/manifests/dataset_v001"
    rows: dict[str, list[dict[str, Any]]] = {
        split: load_manifest(manifest_dir / f"{split}.jsonl") for split in SPLITS
    }
    print(
        "utterances: "
        + ", ".join(f"{split}={len(rows[split])}" for split in SPLITS)
    )

    # ---- 1. Unicode normalization audit -------------------------------------
    normalization: dict[str, Any] = {}
    for form in ("NFC", "NFD", "NFKC"):
        changed = 0
        total = 0
        for split in SPLITS:
            for row in rows[split]:
                text = str(row["text"])
                total += 1
                if unicodedata.normalize(form, text) != text:
                    changed += 1
        normalization[form] = {"transcripts_changed": changed, "transcripts_total": total}
    raw_distinct = len({c for split in SPLITS for row in rows[split] for c in str(row["text"])})
    nfc_distinct = len(
        {
            c
            for split in SPLITS
            for row in rows[split]
            for c in unicodedata.normalize("NFC", str(row["text"]))
        }
    )
    normalization["distinct_codepoints_raw"] = raw_distinct
    normalization["distinct_codepoints_nfc"] = nfc_distinct
    normalization["already_nfc"] = normalization["NFC"]["transcripts_changed"] == 0
    print(
        "normalization: "
        + f"NFC changed {normalization['NFC']['transcripts_changed']}"
        f"/{normalization['NFC']['transcripts_total']}, "
        f"distinct codepoints raw={raw_distinct} nfc={nfc_distinct}"
    )

    # ---- 2. character inventory and character OOV ---------------------------
    rows_flat = [row for split in SPLITS for row in rows[split]]
    train_texts = [unicodedata.normalize("NFC", str(r["text"])) for r in rows["train"]]
    train_chars = Counter(c for text in train_texts for c in text)
    train_char_set = set(train_chars)

    inventory: dict[str, int] = {}
    for char in train_chars:
        inventory[script_of(char)] = inventory.get(script_of(char), 0) + 1

    character_stats: dict[str, Any] = {}
    for split in SPLITS:
        texts = [unicodedata.normalize("NFC", str(r["text"])) for r in rows[split]]
        chars = [c for text in texts for c in text]
        unseen = [c for c in chars if c not in train_char_set]
        non_space = [c for c in chars if not c.isspace()]
        unseen_non_space = [c for c in non_space if c not in train_char_set]
        character_stats[split] = {
            "utterances": len(texts),
            "characters": len(chars),
            "non_space_characters": len(non_space),
            "distinct_characters": len(set(chars)),
            "unseen_character_tokens": len(unseen),
            "unseen_character_rate": len(unseen) / len(chars) if chars else 0.0,
            "unseen_non_space_tokens": len(unseen_non_space),
            "unseen_non_space_rate": (
                len(unseen_non_space) / len(non_space) if non_space else 0.0
            ),
            "distinct_unseen_characters": sorted(set(unseen)),
        }
    train_inventory = {
        "distinct_characters": len(train_char_set),
        "by_script": dict(sorted(inventory.items())),
        "rareest_20": sorted(train_chars.items(), key=lambda kv: kv[1])[:20],
        "most_common_20": train_chars.most_common(20),
    }
    print(
        f"characters: train distinct={len(train_char_set)} "
        f"scripts={train_inventory['by_script']}"
    )
    for split in SPLITS:
        stats = character_stats[split]
        print(
            f"  {split}: chars={stats['characters']} "
            f"unseen={stats['unseen_character_tokens']} "
            f"rate={stats['unseen_character_rate']:.6f} "
            f"unseen_non_space={stats['unseen_non_space_tokens']} "
            f"rate={stats['unseen_non_space_rate']:.6f}"
        )

    # ---- 3. CTC feasibility: encoder frames versus label count --------------
    def feasibility(unit) -> dict[str, Any]:
        per_stride: dict[str, Any] = {}
        for stride in STRIDES:
            layers = stride.bit_length() - 1
            violations = 0
            tight = 0
            ratios: list[float] = []
            for split in SPLITS:
                for row in rows[split]:
                    labels = unit(str(row["text"]))
                    if not labels:
                        continue
                    frames = encoder_frames(stft_frames(float(row["duration_seconds"])), layers)
                    ratio = frames / len(labels)
                    ratios.append(ratio)
                    if frames < len(labels):
                        violations += 1
                    elif ratio < 2.0:
                        tight += 1
            ratios.sort()
            per_stride[str(stride)] = {
                "encoder_hz": SAMPLE_RATE / HOP_LENGTH / stride,
                "conv_layers_stride2": layers,
                "utterances_violating_frames_lt_labels": violations,
                "utterances_ratio_below_2": tight,
                "ratio_min": round(ratios[0], 3) if ratios else 0.0,
                "ratio_p05": round(ratios[len(ratios) // 20], 3) if ratios else 0.0,
                "ratio_median": round(ratios[len(ratios) // 2], 3) if ratios else 0.0,
            }
        return per_stride

    word_unit = lambda text: text.split()  # noqa: E731
    char_unit = lambda text: [c for c in text if not c.isspace()]  # noqa: E731
    char_with_space = lambda text: list(text)  # noqa: E731

    feasibility_results = {
        "word": feasibility(word_unit),
        "char_no_space": feasibility(char_unit),
        "char_with_space": feasibility(char_with_space),
    }
    print("CTC feasibility (utterances where encoder frames < labels):")
    for name, table in feasibility_results.items():
        line = ", ".join(
            f"stride {s}: {v['utterances_violating_frames_lt_labels']} "
            f"(ratio p05 {v['ratio_p05']})"
            for s, v in table.items()
        )
        print(f"  {name}: {line}")

    # ---- 4. word-level most-frequent-N OOV curve ----------------------------
    train_words = Counter(w for text in train_texts for w in text.split())
    word_order = [w for w, _ in sorted(train_words.items(), key=lambda kv: (-kv[1], kv[0]))]
    curve: list[dict[str, Any]] = []
    for n in TOP_N:
        allowed = set(word_order[:n])
        point: dict[str, Any] = {
            "n": n,
            "fc_parameters_at_hidden_256": HIDDEN * (n + 1) + (n + 1),
            "splits": {},
        }
        for split in SPLITS:
            tokens = [w for text in (unicodedata.normalize("NFC", str(r["text"])) for r in rows[split]) for w in text.split()]
            oov = sum(1 for w in tokens if w not in allowed)
            utterances = rows[split]
            oov_utterances = sum(
                1
                for row in utterances
                if any(w not in allowed for w in str(row["text"]).split())
            )
            point["splits"][split] = {
                "tokens": len(tokens),
                "oov_tokens": oov,
                "oov_rate": oov / len(tokens) if tokens else 0.0,
                "oov_utterance_rate": oov_utterances / len(utterances) if utterances else 0.0,
            }
        curve.append(point)
        dev = point["splits"]["dev"]
        test = point["splits"]["test"]
        print(
            f"  word N={n:>6}: dev OOV {dev['oov_rate']:.4f} "
            f"test OOV {test['oov_rate']:.4f} "
            f"fc_params {point['fc_parameters_at_hidden_256']:,}"
        )

    # ---- 5. output-layer cost of each option --------------------------------
    char_vocab_size = len(train_char_set)
    options = {
        "char_no_space": {
            "vocabulary_size": char_vocab_size,
            "note": "whitespace is not a target symbol; words are re-joined at decode time",
            "fc_parameters_at_hidden_256": HIDDEN * (char_vocab_size + 1) + (char_vocab_size + 1),
        },
        "word_top_1000": {
            "vocabulary_size": 1000,
            "fc_parameters_at_hidden_256": HIDDEN * 1001 + 1001,
        },
        "word_top_10000": {
            "vocabulary_size": 10000,
            "fc_parameters_at_hidden_256": HIDDEN * 10001 + 10001,
        },
        "word_full": {
            "vocabulary_size": len(word_order),
            "fc_parameters_at_hidden_256": HIDDEN * (len(word_order) + 1) + (len(word_order) + 1),
        },
    }
    for name, option in options.items():
        weights_mb = option["fc_parameters_at_hidden_256"] * 4 / 1e6
        option["fc_weights_mb_fp32"] = round(weights_mb, 2)
        option["fits_100mb_weight_target"] = weights_mb < 100
    print("output layer cost:")
    for name, option in options.items():
        print(
            f"  {name}: vocab={option['vocabulary_size']} "
            f"fc={option['fc_parameters_at_hidden_256']:,} "
            f"{option['fc_weights_mb_fp32']} MB "
            f"fits<100MB={option['fits_100mb_weight_target']}"
        )

    # ---- sequence length, the cost side of the trade-off --------------------
    def lengths(unit) -> dict[str, float]:
        total = 0
        count = 0
        per_utt: list[int] = []
        for row in rows["train"]:
            n = len(unit(str(row["text"])))
            total += n
            count += 1
            per_utt.append(n)
        per_utt.sort()
        return {
            "train_labels_total": total,
            "train_utterances": count,
            "mean_labels_per_utterance": total / count if count else 0.0,
            "median_labels_per_utterance": per_utt[len(per_utt) // 2] if per_utt else 0,
            "max_labels_per_utterance": per_utt[-1] if per_utt else 0,
        }

# ---- 6. speaking-rate audit --------------------------------------------
    # The nine character-level CTC violations are not all the same kind of problem.
    # A long utterance that is merely fast is a legitimate hard case; an utterance
    # whose transcript cannot physically fit in its audio is a corrupt row. Only
    # measurement separates them, so measure the rate for every utterance.
    def rate_audit(unit, label: str) -> dict[str, Any]:
        rows_out: list[dict[str, Any]] = []
        per_split: dict[str, Any] = {}
        for split in SPLITS:
            rates: list[float] = []
            speakers: Counter[str] = Counter()
            for row in rows[split]:
                count = len(unit(str(row["text"])))
                duration = float(row["duration_seconds"])
                if count == 0 or duration <= 0:
                    continue
                rate = count / duration
                rates.append(rate)
                rows_out.append(
                    {
                        "split": split,
                        "utterance_id": row["utterance_id"],
                        "speaker_id": row["speaker_id"],
                        "duration_seconds": duration,
                        "labels": count,
                        "labels_per_second": round(rate, 3),
                    }
                )
                speakers[row["speaker_id"]] += 1
            rates.sort()
            per_split[split] = {
                "utterances": len(rates),
                "labels_per_second_p50": round(rates[len(rates) // 2], 3) if rates else 0.0,
                "labels_per_second_p95": round(rates[int(len(rates) * 0.95)], 3) if rates else 0.0,
                "labels_per_second_p99": round(rates[int(len(rates) * 0.99)], 3) if rates else 0.0,
                "labels_per_second_max": round(rates[-1], 3) if rates else 0.0,
                "above_15_per_second": sum(1 for r in rates if r > 15),
                "above_20_per_second": sum(1 for r in rates if r > 20),
                "above_25_per_second": sum(1 for r in rates if r > 25),
                "above_30_per_second": sum(1 for r in rates if r > 30),
                "speakers": len(speakers),
            }
        rows_out.sort(key=lambda item: -item["labels_per_second"])
        worst = rows_out[:20]
        worst_speakers = Counter(item["speaker_id"] for item in rows_out[:200])
        return {
            "unit": label,
            "per_split": per_split,
            "worst_20": worst,
            "speaker_concentration_in_worst_200": worst_speakers.most_common(10),
        }

    rate_results = {
        "char_no_space": rate_audit(char_unit, "non-space characters"),
        "word": rate_audit(word_unit, "whitespace words"),
    }
    print("speaking rate (non-space characters per second):")
    for split in SPLITS:
        stats = rate_results["char_no_space"]["per_split"][split]
        print(
            f"  {split}: p50={stats['labels_per_second_p50']} "
            f"p99={stats['labels_per_second_p99']} "
            f"max={stats['labels_per_second_max']} "
            f">20/s={stats['above_20_per_second']} >30/s={stats['above_30_per_second']}"
        )
    print(
        "  worst speakers (of the 200 fastest): "
        f"{rate_results['char_no_space']['speaker_concentration_in_worst_200'][:5]}"
    )

    # Which of the stride-4 CTC violations are corrupt rather than merely fast?
    violation_ids = {item["utterance_id"] for item in rate_results["char_no_space"]["worst_20"]}
    violations_detail = []
    for stride in (4,):
        layers = stride.bit_length() - 1
        for row in rows_flat:
            labels = char_unit(str(row["text"]))
            if not labels:
                continue
            frames = encoder_frames(stft_frames(float(row["duration_seconds"])), layers)
            if frames < len(labels):
                duration = float(row["duration_seconds"])
                rate = len(labels) / duration if duration else 0.0
                violations_detail.append(
                    {
                        "utterance_id": row["utterance_id"],
                        "speaker_id": row["speaker_id"],
                        "shipped_split": row.get("shipped_split"),
                        "duration_seconds": duration,
                        "characters": len(labels),
                        "encoder_frames_at_stride_4": frames,
                        "characters_per_second": round(rate, 3),
                        "in_fastest_200": row["utterance_id"] in violation_ids,
                    }
                )
    violations_detail.sort(key=lambda item: -item["characters_per_second"])
    print(
        f"  stride-4 character CTC violations: {len(violations_detail)}; "
        f"of the fastest 200: {sum(1 for v in violations_detail if v['in_fastest_200'])}"
    )

    results = {
        "experiment_id": "004_character_tokenizer",
        "purpose": "measure tokenizer options so the choice follows GUIDE section 28 from data",
        "manifest_dir": str(manifest_dir.relative_to(ROOT)).replace("\\", "/"),
        "dataset_version": "dataset_v001",
        "normalization": normalization,
        "character_inventory_train": train_inventory,
        "character_stats": character_stats,
        "ctc_feasibility": feasibility_results,
        "word_top_n_curve": curve,
        "vocabulary_options": options,
        "rate_audit": rate_results,
        "ctc_violations_stride4_char": violations_detail,
        "sequence_lengths": {
            "word": lengths(word_unit),
            "char_no_space": lengths(char_unit),
            "char_with_space": lengths(char_with_space),
        },
        "notes": [
            "Encoder frames are computed with ceil applied once per stride-2 conv, which is "
            "the arithmetic verified against nn.Conv1d in EXP-003 step 2. floor(T/stride) "
            "would understate them and is not used.",
            "Word-level OOV rates here are recomputed from the manifests rather than copied "
            "from EXP-003, so this experiment can be checked on its own.",
        ],
    }
    (OUTPUT_DIR / "vocabulary_measurements.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote vocabulary_measurements.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())