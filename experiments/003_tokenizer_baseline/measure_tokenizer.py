"""EXP-003: tokenizer measurements (word-level) and tiny CTC baseline plan.

This script does word-level tokenisation on the dataset_v001 manifests, measures
vocabulary, token frequency, and OOV between train/dev/test splits. It also selects
a small 10-30 minute subset from train to use for the overfit test later.

Gates: 220 tests pass. This is the first EXP-003 step.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST_DIR = ROOT / "data/manifests/dataset_v001"
OUTPUT_DIR = Path(__file__).parent
SPLITS = ["train", "dev", "test"]
TARGET_MINUTES = 15.0  # aim for ~15 min subset for overfit check


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                raise RuntimeError(f"Bad JSON line {i} in {path}")
    return rows


def tokenize_wordlevel(text: str) -> list[str]:
    # Very simple whitespace tokenization. If text contains punctuation, keep it as token or split; for Tamil often just whitespace.
    text = text.strip()
    if not text:
        return []
    # Split on whitespace
    tokens = text.split()
    return tokens


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data: dict[str, list[dict[str, Any]]] = {}
    for s in SPLITS:
        data[s] = load_manifest(MANIFEST_DIR / f"{s}.jsonl")

    # Build vocab from train
    train_counter: Counter[str] = Counter()
    for r in data["train"]:
        train_counter.update(tokenize_wordlevel(r["text"]))
    train_vocab = set(train_counter.keys())

    stats: dict[str, Any] = {}
    for s in SPLITS:
        c: Counter[str] = Counter()
        oov = 0
        total = 0
        for r in data[s]:
            toks = tokenize_wordlevel(r["text"])
            total += len(toks)
            for t in toks:
                c[t] += 1
                if t not in train_vocab:
                    oov += 1
        stats[s] = {
            "utterances": len(data[s]),
            "tokens": total,
            "vocab_size": len(c),
            "oov_tokens": oov,
            "oov_rate": (oov / total) if total else 0.0,
            "top10": c.most_common(10),
        }

    # Build train vocab list sorted (by freq desc then token)
    train_vocab_sorted = [w for w, _ in train_counter.most_common()]

    # Build id maps
    token2id = {w: i + 1 for i, w in enumerate(train_vocab_sorted)}  # 0 for blank
    id2token = {i + 1: w for i, w in enumerate(train_vocab_sorted)}
    unk_id = len(token2id) + 1
    token2id_with_unk = dict(token2id)
    token2id_with_unk["<UNK>"] = unk_id

    # Select subset
    subset: list[dict[str, Any]] = []
    total_dur = 0.0
    for r in sorted(data["train"], key=lambda x: x.get("duration_seconds", 0.0)):
        subset.append(r)
        total_dur += float(r.get("duration_seconds", 0.0))
        if total_dur >= TARGET_MINUTES * 60.0:
            break
    subset_hours = total_dur / 3600.0
    subset_minutes = total_dur / 60.0

    out = {
        "target_minutes": TARGET_MINUTES,
        "train_vocab_size": len(train_vocab_sorted),
        "total_train_tokens": sum(train_counter.values()),
        "splits": stats,
        "unk_id": unk_id,
        "token2id_has_unk": True,
        "subset": {
            "utterances": len(subset),
            "minutes": round(subset_minutes, 2),
            "hours": round(subset_hours, 4),
            "duration_seconds": round(total_dur, 2),
            "max_duration": round(max(float(r.get("duration_seconds", 0.0)) for r in subset), 2)
            if subset else 0,
            "min_duration": round(min(float(r.get("duration_seconds", 0.0)) for r in subset), 2)
            if subset else 0,
        },
        "subset_sample": subset[:3] if subset else [],
    }
    (OUTPUT_DIR / "tokenizer_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
    (OUTPUT_DIR / "word_vocab.txt").write_text("\n".join(train_vocab_sorted) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "token2id.json").write_text(json.dumps(token2id_with_unk, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")

    print("EXP-003 tokenizer measurements")
    print(f"train_vocab_size={out['train_vocab_size']} total_tokens={out['total_train_tokens']}")
    for s in SPLITS:
        ss = out["splits"][s]
        print(
            f"{s}: utt={ss['utterances']} toks={ss['tokens']} oov={ss['oov_tokens']} oov_rate={ss['oov_rate']:.4f} vocab={ss['vocab_size']}"
        )
    print(f"subset: utt={out['subset']['utterances']} min={out['subset']['minutes']} h={out['subset']['hours']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
