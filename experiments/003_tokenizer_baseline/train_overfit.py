"""EXP-003 step 2: train a tiny CTC model until it overfits the 15-minute subset.

The purpose of this script is not accuracy. It is to prove the pipeline can drive
its own loss to zero on data it memorises. If a model cannot overfit 661
utterances, the fault is in the data, the features, the padding, the CTC output
lengths, the blank id or the decoder, and no amount of extra data will reveal it.

Three choices are deliberate and were each made after measuring something:

1. The vocabulary is rebuilt from the subset (1360 words, 0.99 % of the 138047-word
   train vocabulary). The full-vocab output layer measures 70819137 parameters,
   which is not a tiny model and would make this a vocabulary test instead of a
   pipeline test. The full-vocab figure is still computed and recorded below.

2. There is no BatchNorm, and the GRU is unidirectional. Both are about padding:
   BatchNorm averages the zero-padded tail into the statistics of the real
   frames, and a bidirectional GRU reads that tail backwards into the last valid
   frames. With a unidirectional stack and true CTC input lengths, padded frames
   cannot influence any valid output.

3. Encoder lengths are computed with exact convolution arithmetic and then
   checked against what ``nn.Conv1d`` actually produces. ``floor(T / 4)`` is
   wrong for 497 of the 661 subset rows; the helper below is verified, not
   trusted.

CTC runs with ``zero_infinity=False`` on purpose. That flag would hide exactly
the input-shorter-than-target bug this step exists to catch.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tamil_voice.audio.features import StftConfig, log_mel_spectrogram  # noqa: E402
from tamil_voice.audio.io import ValidationLimits, load_audio  # noqa: E402
from tamil_voice.audio.resampling import resample_to_canonical  # noqa: E402
from tamil_voice.common.config import get_paths, load_yaml  # noqa: E402
from tamil_voice.common.logging import get_logger  # noqa: E402
from tamil_voice.common.seed import seed_everything  # noqa: E402

LOGGER = get_logger(__name__)


class OverfitCheckFailed(RuntimeError):
    """The run finished but did not overfit. Reported, never smoothed over."""


@dataclass(frozen=True)
class Example:
    utterance_id: str
    speaker_id: str
    text: str
    tokens: tuple[str, ...]
    duration_seconds: float


# --------------------------------------------------------------------------- data


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


def select_subset(
    rows: list[dict[str, Any]], target_minutes: float
) -> tuple[list[dict[str, Any]], float]:
    """Shortest utterances first until the duration budget is met.

    This must stay identical to ``measure_tokenizer.py``; otherwise step 1 and
    step 2 would be measuring different subsets and neither number would mean
    anything next to the other.
    """
    target_seconds = target_minutes * 60.0
    ordered = sorted(rows, key=lambda row: float(row.get("duration_seconds", 0.0)))
    subset: list[dict[str, Any]] = []
    total = 0.0
    for row in ordered:
        subset.append(row)
        total += float(row.get("duration_seconds", 0.0))
        if total >= target_seconds:
            break
    return subset, total


def build_vocabulary(examples: list[Example]) -> tuple[dict[str, int], dict[int, str]]:
    """Frequency-ordered, ties broken lexicographically so the ids are stable."""
    counter: Counter[str] = Counter()
    for example in examples:
        counter.update(example.tokens)
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    token2id = {token: index + 1 for index, (token, _) in enumerate(ordered)}
    id2token = {index: token for token, index in token2id.items()}
    return token2id, id2token


def extract_features(
    rows: list[dict[str, Any]], feature_config: StftConfig, paths: Any
) -> tuple[list[np.ndarray], float]:
    """Compute log-Mel once for every row and keep it in memory.

    900 s of audio is ~29 MB of features, so caching costs nothing and saves
    re-running the STFT on every epoch.
    """
    limits = ValidationLimits()
    features: list[np.ndarray] = []
    started = time.time()
    for position, row in enumerate(rows, 1):
        audio_path = (paths.data / row["audio_path"]).resolve()
        audio = load_audio(audio_path, limits=limits)
        audio = resample_to_canonical(audio)
        mel = log_mel_spectrogram(audio.waveform, feature_config)
        features.append(np.asarray(mel, dtype=np.float32))
        if position % 100 == 0:
            LOGGER.info("features", extra={"done": position, "total": len(rows)})
    return features, time.time() - started


class SubsetDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int, int, int]]):
    def __init__(
        self,
        examples: list[Example],
        features: list[np.ndarray],
        token2id: dict[str, int],
        unk_id: int,
    ) -> None:
        self.examples = examples
        self.features = features
        self.token2id = token2id
        self.unk_id = unk_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        example = self.examples[index]
        feature = torch.from_numpy(self.features[index])
        labels = torch.tensor(
            [self.token2id.get(token, self.unk_id) for token in example.tokens],
            dtype=torch.long,
        )
        return feature, labels, index, feature.shape[0], labels.shape[0]


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features, labels, indices, feature_lengths, label_lengths = zip(*batch)
    max_frames = max(feature_lengths)
    max_labels = max(label_lengths)
    padded_features = torch.zeros(len(batch), max_frames, features[0].shape[1])
    padded_labels = torch.zeros(len(batch), max_labels, dtype=torch.long)
    for row, feature in enumerate(features):
        padded_features[row, : feature.shape[0]] = feature
    for row, label in enumerate(labels):
        padded_labels[row, : label.shape[0]] = label
    return (
        padded_features,
        padded_labels,
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(feature_lengths, dtype=torch.long),
        torch.tensor(label_lengths, dtype=torch.long),
    )


# --------------------------------------------------------------------------- model


class TinyCTC(nn.Module):
    def __init__(
        self,
        n_mels: int,
        num_classes: int,
        conv_channels: list[int],
        conv_kernel: int,
        conv_strides: list[int],
        conv_padding: list[int],
        gru_hidden: int,
        gru_layers: int,
        gru_bidirectional: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        convs: list[nn.Module] = []
        in_channels = n_mels
        for out_channels, stride, padding in zip(conv_channels, conv_strides, conv_padding):
            convs.append(nn.Conv1d(in_channels, out_channels, conv_kernel, stride, padding))
            convs.append(nn.ReLU())
            in_channels = out_channels
        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=gru_bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(gru_hidden * (2 if gru_bidirectional else 1), num_classes)
        self.conv_kernel = conv_kernel
        self.conv_strides = list(conv_strides)
        self.conv_padding = list(conv_padding)
        self.n_mels = n_mels

    def subsampled_length(self, frames: int) -> int:
        """Exact output length of the conv stack, one layer at a time."""
        length = frames
        for stride, padding in zip(self.conv_strides, self.conv_padding):
            length = (length + 2 * padding - self.conv_kernel) // stride + 1
        return max(length, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.convs(features.transpose(1, 2)).transpose(1, 2)
        hidden = self.dropout(hidden)
        hidden, _ = self.gru(hidden)
        return self.fc(hidden)


def verify_length_math(model: TinyCTC, lengths: list[int]) -> None:
    """Check the arithmetic against ``nn.Conv1d`` itself, on this machine.

    A length helper that is merely plausible is the exact class of bug an
    overfit test is supposed to catch, so it is measured rather than reviewed.
    """
    unique = sorted(set(lengths))
    probe = unique if len(unique) <= 12 else unique[:6] + unique[-6:]
    device = next(model.parameters()).device
    with torch.no_grad():
        batched = torch.zeros(len(probe), max(probe), model.n_mels, device=device)
        actual_batch = model.subsampled_length(max(probe))
        predicted_batch = model(batched).shape[1]
        if actual_batch != predicted_batch:
            raise OverfitCheckFailed(
                f"encoder length mismatch on a padded batch: predicted {predicted_batch}, "
                f"conv stack produced {actual_batch}"
            )
        for frames in probe:
            single = torch.zeros(1, frames, model.n_mels, device=device)
            predicted = model.subsampled_length(frames)
            actual = model(single).shape[1]
            if predicted != actual:
                raise OverfitCheckFailed(
                    f"encoder length mismatch at {frames} frames: predicted {predicted}, "
                    f"conv stack produced {actual}"
                )
    LOGGER.info("encoder length math verified", extra={"probes": len(probe)})


# -------------------------------------------------------------------------- decode


def greedy_decode(logits: torch.Tensor, lengths: torch.Tensor, blank_id: int) -> list[list[int]]:
    """Best-path decode: argmax, collapse repeats, drop blanks."""
    best = logits.argmax(dim=-1).cpu()
    decoded: list[list[int]] = []
    for row, length in enumerate(lengths.tolist()):
        sequence: list[int] = []
        previous = -1
        for step in range(length):
            token = int(best[row, step])
            if token != previous and token != blank_id:
                sequence.append(token)
            previous = token
        decoded.append(sequence)
    return decoded


def edit_distance(reference: list[int], hypothesis: list[int]) -> tuple[int, int, int]:
    """Token-level Levenshtein. Returns (errors, ref_len, hyp_len)."""
    if not reference:
        return len(hypothesis), 0, len(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, 1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1], len(reference), len(hypothesis)


# --------------------------------------------------------------------------- train


@torch.no_grad()
def evaluate(
    model: TinyCTC,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    examples: list[Example],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    errors = 0
    reference_tokens = 0
    hypothesis_tokens = 0
    exact = 0
    for features, labels, indices, feature_lengths, label_lengths in loader:
        logits = model(features.to(device))
        input_lengths = torch.tensor(
            [model.subsampled_length(int(x)) for x in feature_lengths], dtype=torch.long
        )
        decoded = greedy_decode(logits, input_lengths, blank_id=0)
        for row, index in enumerate(indices.tolist()):
            reference = labels[row, : label_lengths[row]].tolist()
            row_errors, ref_len, hyp_len = edit_distance(reference, decoded[row])
            errors += row_errors
            reference_tokens += ref_len
            hypothesis_tokens += hyp_len
            if row_errors == 0:
                exact += 1
    return {
        "token_error_rate": errors / reference_tokens if reference_tokens else 1.0,
        "exact_match_rate": exact / len(examples) if examples else 0.0,
        "reference_tokens": float(reference_tokens),
        "hypothesis_tokens": float(hypothesis_tokens),
    }


def build_model(cfg: dict[str, Any], num_classes: int, n_mels: int) -> TinyCTC:
    model_cfg = cfg["model"]
    return TinyCTC(
        n_mels=n_mels,
        num_classes=num_classes,
        conv_channels=[int(c) for c in model_cfg["conv_channels"]],
        conv_kernel=int(model_cfg["conv_kernel"]),
        conv_strides=[int(s) for s in model_cfg["conv_strides"]],
        conv_padding=[int(p) for p in model_cfg["conv_padding"]],
        gru_hidden=int(model_cfg["gru_hidden"]),
        gru_layers=int(model_cfg["gru_layers"]),
        gru_bidirectional=bool(model_cfg["gru_bidirectional"]),
        dropout=float(model_cfg["dropout"]),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise OverfitCheckFailed("config asked for cuda but torch reports it unavailable")
    return torch.device(requested)


def main(argv: list[str] | None = None) -> int:
    config_path = Path(__file__).with_name("config.yaml")
    cfg = load_yaml(config_path)
    paths = get_paths()
    paths.ensure()

    seed_report = seed_everything(int(cfg["training"]["seed"]))
    LOGGER.info("seed", extra={"report": seed_report.__dict__})

    device = resolve_device(str(cfg["training"]["device"]))
    print(f"device={device}")

    # ---- data: train split only, never dev or test.
    manifest_dir = paths.root / cfg["data"]["manifest_dir"]
    rows: list[dict[str, Any]] = []
    for split in cfg["data"]["splits"]:
        if split != "train":
            raise OverfitCheckFailed(
                f"this step must not read {split!r}: the overfit check is train-only by definition"
            )
        rows.extend(load_manifest(manifest_dir / f"{split}.jsonl"))
    subset, subset_seconds = select_subset(rows, float(cfg["data"]["subset_minutes"]))
    print(f"subset utterances={len(subset)} seconds={subset_seconds:.1f}")

    blank_id = int(cfg["tokenizer"]["blank_id"])
    examples = [
        Example(
            utterance_id=str(row["utterance_id"]),
            speaker_id=str(row["speaker_id"]),
            text=str(row["text"]),
            tokens=tuple(str(row["text"]).split()),
            duration_seconds=float(row["duration_seconds"]),
        )
        for row in subset
    ]
    empty = [e.utterance_id for e in examples if not e.tokens]
    if empty:
        raise OverfitCheckFailed(f"{len(empty)} subset rows have no tokens: {empty[:5]}")

    token2id, id2token = build_vocabulary(examples)
    unk_id = max(token2id.values()) + 1
    num_classes = unk_id + 1  # includes blank and unk
    print(f"subset vocabulary={len(token2id)} num_classes={num_classes} blank={blank_id}")

    # ---- features
    feature_cfg = StftConfig(
        n_fft=int(cfg["features"]["n_fft"]),
        win_length=int(cfg["features"]["win_length"]),
        hop_length=int(cfg["features"]["hop_length"]),
        n_mels=int(cfg["features"]["n_mels"]),
    )
    features, feature_seconds = extract_features(subset, feature_cfg, paths)
    print(f"features: {len(features)} utterances in {feature_seconds:.1f}s")

    dataset = SubsetDataset(examples, features, token2id, unk_id)
    collated = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=int(cfg["training"]["num_workers"]),
    )
    generator = torch.Generator().manual_seed(int(cfg["training"]["seed"]))
    train_loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=int(cfg["training"]["num_workers"]),
        generator=generator,
    )

    # ---- model, and the length arithmetic that decides whether CTC is even valid
    model = build_model(cfg, num_classes, feature_cfg.n_mels).to(device)
    parameter_count = count_parameters(model)
    full_vocab_classes = 138049  # measured by measure_tokenizer.py, hardcoded as a recorded fact
    print(f"parameters={parameter_count}")

    feature_lengths_all = [int(f.shape[0]) for f in features]
    verify_length_math(model, feature_lengths_all)

    output_lengths = [model.subsampled_length(n) for n in feature_lengths_all]
    label_lengths_all = [len(e.tokens) for e in examples]
    starved = [
        (e.utterance_id, out, len(e.tokens))
        for e, out in zip(examples, output_lengths)
        if out < len(e.tokens)
    ]
    if starved:
        raise OverfitCheckFailed(
            f"{len(starved)} rows have fewer encoder frames than labels, which CTC cannot "
            f"train on: {starved[:5]}"
        )
    print(
        f"frames: min={min(feature_lengths_all)} max={max(feature_lengths_all)} "
        f"encoder min={min(output_lengths)} max={max(output_lengths)} "
        f"labels max={max(label_lengths_all)}"
    )

    criterion = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    grad_clip = float(cfg["training"]["grad_clip"])

    checkpoint_dir = paths.checkpoints / "exp003_overfit"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        for step, (feats, labels, _, feature_lengths, label_lengths) in enumerate(train_loader, 1):
            logits = model(feats.to(device))
            input_lengths = torch.tensor(
                [model.subsampled_length(int(x)) for x in feature_lengths],
                dtype=torch.long,
                device=device,
            )
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            # CTCLoss wants the labels concatenated with no padding between them,
            # so the target tensor must have exactly sum(target_lengths) entries.
            # Flattening the padded (B, S) block is wrong: CTCLoss counts blank as a
            # target symbol, so the padding would be trained on as if it were text.
            label_lengths = label_lengths.to(device)
            keep = torch.arange(labels.shape[1], device=device)[None, :] < label_lengths[:, None]
            targets = labels.to(device)[keep]
            loss = criterion(log_probs, targets, input_lengths, label_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += float(loss.detach())
            steps += 1
            if step % int(cfg["training"]["log_interval"]) == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={float(loss.detach()):.4f}")
        mean_loss = running_loss / steps if steps else float("inf")
        record: dict[str, Any] = {"epoch": epoch, "train_loss": mean_loss}
        if epoch % int(cfg["training"]["eval_every"]) == 0:
            record.update(evaluate(model, collated, examples, device))
            print(
                f"epoch {epoch} loss={mean_loss:.4f} "
                f"token_error_rate={record['token_error_rate']:.4f} "
                f"exact_match={record['exact_match_rate']:.4f}"
            )
        history.append(record)
        if epoch % int(cfg["training"]["save_every"]) == 0:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "config": cfg},
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
            )
    elapsed = time.time() - started

    final = history[-1]
    results = {
        "experiment_id": cfg["experiment_id"],
        "step": cfg["step"],
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": seed_report.__dict__,
        "subset": {
            "utterances": len(subset),
            "duration_seconds": round(subset_seconds, 2),
            "minutes": round(subset_seconds / 60.0, 2),
            "selection_rule": "shortest utterances first until subset_minutes is met",
            "speakers": len({e.speaker_id for e in examples}),
            "tokens": sum(len(e.tokens) for e in examples),
            "min_duration_seconds": round(min(e.duration_seconds for e in examples), 3),
            "max_duration_seconds": round(max(e.duration_seconds for e in examples), 3),
        },
        "vocabulary": {
            "source": cfg["data"]["vocab_source"],
            "size": len(token2id),
            "num_classes": num_classes,
            "blank_id": blank_id,
            "unk_id": unk_id,
            "unk_hits": 0,
            "as_fraction_of_train_vocab": round(len(token2id) / 138047, 6),
        },
        "features": {
            "n_fft": feature_cfg.n_fft,
            "win_length": feature_cfg.win_length,
            "hop_length": feature_cfg.hop_length,
            "n_mels": feature_cfg.n_mels,
            "extract_seconds": round(feature_seconds, 2),
            "frames_min": min(feature_lengths_all),
            "frames_max": max(feature_lengths_all),
            "encoder_stride": int(np.prod(cfg["model"]["conv_strides"])),
            "encoder_frames_min": min(output_lengths),
            "encoder_frames_max": max(output_lengths),
            "length_math_verified_against_conv1d": True,
            "rows_where_floor_over_stride_would_be_wrong": sum(
                1
                for n, out in zip(feature_lengths_all, output_lengths)
                if n // int(np.prod(cfg["model"]["conv_strides"])) != out
            ),
        },
        "model": {
            "architecture": "Conv1d x2 (no BatchNorm) -> GRU -> Linear",
            "gru_bidirectional": bool(cfg["model"]["gru_bidirectional"]),
            "parameters": parameter_count,
            "parameters_millions": round(parameter_count / 1e6, 4),
            "full_train_vocab_would_be_classes": full_vocab_classes,
            "full_train_vocab_fc_parameters": 256 * full_vocab_classes + full_vocab_classes,
            "parameter_note": (
                "with the measured 138047-word train vocabulary the output layer alone is "
                f"{256 * full_vocab_classes + full_vocab_classes} parameters, so the full-vocab "
                "model cannot be called tiny and does not fit the <100 MB weight target"
            ),
        },
        "training": {
            "epochs": int(cfg["training"]["epochs"]),
            "batch_size": int(cfg["training"]["batch_size"]),
            "lr": float(cfg["training"]["lr"]),
            "optimizer": "AdamW",
            "grad_clip": grad_clip,
            "elapsed_seconds": round(elapsed, 1),
            "seconds_per_epoch": round(elapsed / max(len(history), 1), 2),
        },
        "final": {
            "train_loss": final["train_loss"],
            "token_error_rate": final.get("token_error_rate"),
            "exact_match_rate": final.get("exact_match_rate"),
        },
        "history": history,
        "notes": [
            "Metrics are on the training subset. That is the point: an overfit test "
            "measures memorisation, not generalisation. No WER is claimed.",
            "dev and test were never opened by this script.",
        ],
    }

    (Path(__file__).with_name("subset_vocab.json")).write_text(
        json.dumps(
            {
                "source": "train subset only",
                "selection_rule": "frequency desc, ties lexicographic",
                "size": len(token2id),
                "blank_id": blank_id,
                "unk_id": unk_id,
                "tokens": sorted(token2id, key=lambda t: token2id[t]),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (Path(__file__).with_name("results.json")).write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save({"model": model.state_dict(), "config": cfg}, checkpoint_dir / "final.pt")

    thresholds = cfg["training"]
    ter = final.get("token_error_rate")
    ter_ok = ter is not None and ter <= float(thresholds["accept_max_token_error_rate"])
    loss_ok = final["train_loss"] <= float(thresholds["accept_max_loss"])
    results["acceptance"] = {
        "accept_max_token_error_rate": float(thresholds["accept_max_token_error_rate"]),
        "accept_max_loss": float(thresholds["accept_max_loss"]),
        "token_error_rate_pass": ter_ok,
        "loss_pass": loss_ok,
        "passed": bool(ter_ok and loss_ok),
    }
    (Path(__file__).with_name("results.json")).write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"done in {elapsed / 60:.1f} min | final loss={final['train_loss']:.4f} "
        f"token_error_rate={ter if ter is None else round(ter, 4)} "
        f"exact_match={final.get('exact_match_rate')}"
    )
    if not (ter_ok and loss_ok):
        print("ACCEPTANCE FAILED: see acceptance block in results.json")
        return 1
    print("ACCEPTANCE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())