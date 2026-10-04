"""EXP-005: train a tiny CTC model on character targets until it overfits.

EXP-003 proved the pipeline could drive its loss to zero on word targets. This step
repeats that with the targets EXP-004 chose and the tokenizer now living in
``src/tamil_voice/text/tokenizer.py``, for two reasons:

1. The tokenizer has never been inside a training loop. Unit tests prove it encodes
   and decodes; only this proves the ids it produces are usable as CTC targets.
2. The whitespace variant was chosen by CTC arithmetic (9 invalid utterances against
   33) but implemented the other way round, because a target sequence with no space
   symbol decodes with no word boundaries. Both are run here on identical data with
   identical hyperparameters, so the choice is measured instead of argued.

Everything except the targets is held identical to EXP-003 step 2: same
architecture, same features, same subset rule, same hyperparameters. Anything else
moving would make the two experiments incomparable, which defeats the purpose.

The model class below is a deliberate copy of the one in EXP-003. The research
record must stay runnable on its own, so it is not edited to import from another
experiment. The move of this model into ``src/tamil_voice/asr/`` belongs to the
baseline step, which is where a model is first needed by something other than a
test.

CTC runs with ``zero_infinity=False`` on purpose. That flag would hide exactly the
input-shorter-than-target failure this step is most likely to hit, because character
targets make it much more likely than word targets did.
"""

from __future__ import annotations

import json
import sys
import time
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
from tamil_voice.text.tokenizer import (  # noqa: E402
    CharacterTokenizer,
    TokenizerConfig,
    build_from_jsonl,
)

LOGGER = get_logger(__name__)


class OverfitCheckFailed(RuntimeError):
    """The run finished but did not overfit. Reported, never smoothed over."""


@dataclass(frozen=True)
class Example:
    utterance_id: str
    speaker_id: str
    text: str
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

    Must stay identical to EXP-003's ``select_subset``. If the two disagree about
    what "the subset" is, neither experiment's number means anything beside the
    other's.
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


def extract_features(
    rows: list[dict[str, Any]], feature_config: StftConfig, paths: Any
) -> tuple[list[np.ndarray], float]:
    """Compute log-Mel once and keep it in memory.

    Shared by both variants on purpose: the two runs must see identical features,
    and recomputing them would be the only way they could differ.
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


class CharacterDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int, int, int]]):
    """Holds pre-encoded ids so the tokenizer is not re-run per epoch."""

    def __init__(
        self,
        examples: list[Example],
        features: list[np.ndarray],
        target_ids: list[tuple[int, ...]],
    ) -> None:
        self.examples = examples
        self.features = features
        self.target_ids = target_ids

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        feature = torch.from_numpy(self.features[index])
        labels = torch.tensor(self.target_ids[index], dtype=torch.long)
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
    """Copy of the EXP-003 model. Unidirectional, no BatchNorm, both on purpose.

    BatchNorm averages the zero-padded tail into the statistics of the real frames,
    and a bidirectional GRU reads that tail backwards into the last valid frames.
    With a unidirectional stack and true CTC input lengths, padded frames cannot
    influence any valid output.
    """

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

    A length helper that is merely plausible is the exact class of bug this
    experiment exists to catch, so it is measured rather than reviewed.
    """
    unique = sorted(set(lengths))
    probe = unique if len(unique) <= 12 else unique[:6] + unique[-6:]
    device = next(model.parameters()).device
    with torch.no_grad():
        batched = torch.zeros(len(probe), max(probe), model.n_mels, device=device)
        if model.subsampled_length(max(probe)) != model(batched).shape[1]:
            raise OverfitCheckFailed("encoder length mismatch on a padded batch")
        for frames in probe:
            single = torch.zeros(1, frames, model.n_mels, device=device)
            if model.subsampled_length(frames) != model(single).shape[1]:
                raise OverfitCheckFailed(f"encoder length mismatch at {frames} frames")
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


def edit_distance(reference: list[Any], hypothesis: list[Any]) -> tuple[int, int, int]:
    """Levenshtein over any hashable token. Returns (errors, ref_len, hyp_len)."""
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


@torch.no_grad()
def evaluate(
    model: TinyCTC,
    loader: DataLoader[Any],
    tokenizer: CharacterTokenizer,
    device: torch.device,
) -> dict[str, float | None]:
    """Character token error rate always; word error rate only when meaningful.

    WER is reported only when whitespace is a target symbol. Without it there are no
    word boundaries to recover, and a "WER" computed by guessing boundaries would be
    a made-up number.
    """
    model.eval()
    errors = reference_tokens = hypothesis_tokens = 0
    word_errors = reference_words = hypothesis_words = 0
    exact = 0
    rows = 0
    for features, labels, indices, feature_lengths, label_lengths in loader:
        logits = model(features.to(device))
        input_lengths = torch.tensor(
            [model.subsampled_length(int(x)) for x in feature_lengths], dtype=torch.long
        )
        decoded = greedy_decode(logits, input_lengths, blank_id=tokenizer.blank_id)
        for row, index in enumerate(indices.tolist()):
            reference = labels[row, : label_lengths[row]].tolist()
            row_errors, ref_len, hyp_len = edit_distance(reference, decoded[row])
            errors += row_errors
            reference_tokens += ref_len
            hypothesis_tokens += hyp_len
            rows += 1
            if row_errors == 0:
                exact += 1
            if tokenizer.config.include_space:
                reference_words_list = tokenizer.decode(reference).split(" ")
                hypothesis_words_list = tokenizer.decode(decoded[row]).split(" ")
                w_err, w_ref, w_hyp = edit_distance(reference_words_list, hypothesis_words_list)
                word_errors += w_err
                reference_words += w_ref
                hypothesis_words += w_hyp
    return {
        "token_error_rate": errors / reference_tokens if reference_tokens else 1.0,
        "exact_match_rate": exact / rows if rows else 0.0,
        "reference_tokens": float(reference_tokens),
        "hypothesis_tokens": float(hypothesis_tokens),
        "word_error_rate": (
            word_errors / reference_words if reference_words else None
        ),
        "reference_words": float(reference_words),
    }


# --------------------------------------------------------------------------- train


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


def run_variant(
    name: str,
    tokenizer: CharacterTokenizer,
    examples: list[Example],
    features: list[np.ndarray],
    cfg: dict[str, Any],
    feature_cfg: StftConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Train one variant end to end and return its measured record."""
    target_ids = [tuple(tokenizer.encode(example.text)) for example in examples]
    empty = [e.utterance_id for e, ids in zip(examples, target_ids) if not ids]
    if empty:
        raise OverfitCheckFailed(f"{len(empty)} subset rows encode to nothing: {empty[:5]}")

    model = build_model(cfg, len(tokenizer.symbols), feature_cfg.n_mels).to(device)
    parameter_count = count_parameters(model)

    frame_lengths = [int(f.shape[0]) for f in features]
    verify_length_math(model, frame_lengths)
    encoder_lengths = [model.subsampled_length(n) for n in frame_lengths]

    # Character targets make this failure far more likely than word targets did, so
    # it is measured and reported rather than assumed away. Rows that cannot satisfy
    # frames >= labels are excluded from this variant and listed by id. That is a
    # property of the overfit subset, not a dataset change: dataset_v001 is
    # untouched and no manifest is rewritten.
    invalid = [
        {"utterance_id": e.utterance_id, "encoder_frames": out, "labels": len(ids)}
        for e, out, ids in zip(examples, encoder_lengths, target_ids)
        if out < len(ids)
    ]
    keep = [i for i, out in enumerate(encoder_lengths) if out >= len(target_ids[i])]
    if not keep:
        raise OverfitCheckFailed(f"variant {name!r}: every subset row violates frames >= labels")
    kept_examples = [examples[i] for i in keep]
    kept_features = [features[i] for i in keep]
    kept_ids = [target_ids[i] for i in keep]
    print(
        f"[{name}] subset rows: {len(examples)} total, {len(invalid)} violate frames>=labels, "
        f"{len(keep)} trainable"
    )
    for entry in invalid:
        print(
            f"[{name}] excluded {entry['utterance_id']}: "
            f"{entry['encoder_frames']} frames < {entry['labels']} labels"
        )

    dataset = CharacterDataset(kept_examples, kept_features, kept_ids)
    collate_args = {"collate_fn": collate, "num_workers": int(cfg["training"]["num_workers"])}
    eval_loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), **collate_args)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
        **collate_args,
    )

    criterion = nn.CTCLoss(blank=tokenizer.blank_id, reduction="mean", zero_infinity=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    grad_clip = float(cfg["training"]["grad_clip"])

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
            # CTCLoss counts blank as a target symbol, so the target tensor must hold
            # exactly sum(target_lengths) entries. Flattening the padded (B, S) block
            # would train on the padding as if it were text.
            label_lengths = label_lengths.to(device)
            keep_mask = torch.arange(labels.shape[1], device=device)[None, :] < label_lengths[:, None]
            targets = labels.to(device)[keep_mask]
            loss = criterion(log_probs, targets, input_lengths, label_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += float(loss.detach())
            steps += 1
            if step % int(cfg["training"]["log_interval"]) == 0:
                print(f"[{name}] epoch {epoch} step {step}/{len(train_loader)} loss={float(loss.detach()):.4f}")
        mean_loss = running_loss / steps if steps else float("inf")
        record: dict[str, Any] = {"epoch": epoch, "train_loss": mean_loss}
        if epoch % int(cfg["training"]["eval_every"]) == 0:
            record.update(evaluate(model, eval_loader, tokenizer, device))
            print(
                f"[{name}] epoch {epoch} loss={mean_loss:.4f} "
                f"token_error_rate={record['token_error_rate']:.4f} "
                f"exact_match={record['exact_match_rate']:.4f} "
                f"word_error_rate={record['word_error_rate']}"
            )
        history.append(record)
    elapsed = time.time() - started

    checkpoint_dir = get_paths().checkpoints / "exp005_character_overfit"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "variant": name, "config": cfg},
        checkpoint_dir / f"{name}_final.pt",
    )

    final = history[-1]
    thresholds = cfg["training"]
    ter = final.get("token_error_rate")
    ter_ok = ter is not None and ter <= float(thresholds["accept_max_token_error_rate"])
    loss_ok = final["train_loss"] <= float(thresholds["accept_max_loss"])
    wer = final.get("word_error_rate")
    return {
        "name": name,
        "tokenizer": {
            "include_space": tokenizer.config.include_space,
            "form": tokenizer.config.form,
            "real_symbols": len(tokenizer.real_symbols),
            "num_classes": len(tokenizer.symbols),
            "blank_id": tokenizer.blank_id,
            "unk_id": tokenizer.unk_id,
        },
        "subset": {
            "rows_before_filtering": len(examples),
            "rows_violating_frames_ge_labels": len(invalid),
            "rows_trained": len(keep),
            "violations": invalid,
            "total_labels": sum(len(ids) for ids in kept_ids),
            "max_labels": max(len(ids) for ids in kept_ids),
        },
        "model": {
            "architecture": "Conv1d x2 (no BatchNorm) -> GRU -> Linear",
            "parameters": parameter_count,
            "parameters_millions": round(parameter_count / 1e6, 4),
            "encoder_stride": int(np.prod(cfg["model"]["conv_strides"])),
        },
        "training": {
            "epochs": int(cfg["training"]["epochs"]),
            "elapsed_seconds": round(elapsed, 1),
            "seconds_per_epoch": round(elapsed / max(len(history), 1), 2),
        },
        "final": {
            "train_loss": final["train_loss"],
            "token_error_rate": ter,
            "exact_match_rate": final.get("exact_match_rate"),
            "word_error_rate": wer,
        },
        "acceptance": {
            "accept_max_token_error_rate": float(thresholds["accept_max_token_error_rate"]),
            "accept_max_loss": float(thresholds["accept_max_loss"]),
            "token_error_rate_pass": ter_ok,
            "loss_pass": loss_ok,
            "passed": bool(ter_ok and loss_ok),
        },
        "history": history,
    }


def main(argv: list[str] | None = None) -> int:
    # The config path is overridable so a short smoke run can be done without
    # overwriting this experiment's real results. Outputs are written beside the
    # config that produced them, never beside the script.
    arguments = list(sys.argv[1:] if argv is None else argv)
    config_path = Path(arguments[0]) if arguments else Path(__file__).with_name("config.yaml")
    out_dir = config_path.parent
    cfg = load_yaml(config_path)
    paths = get_paths()
    paths.ensure()

    seed_report = seed_everything(int(cfg["training"]["seed"]))
    LOGGER.info("seed", extra={"report": seed_report.__dict__})
    device = resolve_device(str(cfg["training"]["device"]))
    print(f"device={device}")

    manifest_dir = paths.root / cfg["data"]["manifest_dir"]
    train_manifest = manifest_dir / "train.jsonl"
    for split in cfg["data"]["splits"]:
        if split != "train":
            raise OverfitCheckFailed(
                f"this step must not read {split!r}: the overfit check is train-only by definition"
            )
    rows = load_manifest(train_manifest)
    subset, subset_seconds = select_subset(rows, float(cfg["data"]["subset_minutes"]))
    print(f"train rows={len(rows)} subset utterances={len(subset)} seconds={subset_seconds:.1f}")

    examples = [
        Example(
            utterance_id=str(row["utterance_id"]),
            speaker_id=str(row["speaker_id"]),
            text=str(row["text"]),
            duration_seconds=float(row["duration_seconds"]),
        )
        for row in subset
    ]

    # Built from the FULL train manifest, not the subset. See config.yaml.
    variants_cfg = cfg["tokenizer"]["variants"]
    tokenizers: dict[str, CharacterTokenizer] = {}
    for variant in variants_cfg:
        settings = TokenizerConfig(
            form=str(cfg["tokenizer"]["form"]),
            include_space=bool(variant["include_space"]),
            min_frequency=int(cfg["tokenizer"]["min_frequency"]),
            max_symbols=cfg["tokenizer"]["max_symbols"],
        )
        tokenizer = build_from_jsonl(train_manifest, config=settings)
        tokenizers[str(variant["name"])] = tokenizer
        tokenizer.save(out_dir / f"tokenizer_{variant['name']}.json")
        print(
            f"tokenizer {variant['name']}: real_symbols={len(tokenizer.real_symbols)} "
            f"num_classes={len(tokenizer.symbols)}"
        )

    feature_cfg = StftConfig(
        n_fft=int(cfg["features"]["n_fft"]),
        win_length=int(cfg["features"]["win_length"]),
        hop_length=int(cfg["features"]["hop_length"]),
        n_mels=int(cfg["features"]["n_mels"]),
    )
    features, feature_seconds = extract_features(subset, feature_cfg, paths)
    print(f"features: {len(features)} utterances in {feature_seconds:.1f}s")

    records = [
        run_variant(
            str(variant["name"]),
            tokenizers[str(variant["name"])],
            examples,
            features,
            cfg,
            feature_cfg,
            device,
            int(cfg["training"]["seed"]),
        )
        for variant in variants_cfg
    ]

    results = {
        "experiment_id": cfg["experiment_id"],
        "step": cfg["step"],
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": seed_report.__dict__,
        "dataset_version": "dataset_v001",
        "subset": {
            "utterances": len(subset),
            "duration_seconds": round(subset_seconds, 2),
            "minutes": round(subset_seconds / 60.0, 2),
            "selection_rule": cfg["data"]["subset_rule"],
            "identical_to_exp003_subset": True,
            "speakers": len({e.speaker_id for e in examples}),
            "min_duration_seconds": round(min(e.duration_seconds for e in examples), 3),
            "max_duration_seconds": round(max(e.duration_seconds for e in examples), 3),
        },
        "features": {
            "n_fft": feature_cfg.n_fft,
            "win_length": feature_cfg.win_length,
            "hop_length": feature_cfg.hop_length,
            "n_mels": feature_cfg.n_mels,
            "extract_seconds": round(feature_seconds, 2),
            "length_math_verified_against_conv1d": True,
        },
        "variants": records,
        "notes": [
            "Both variants share one feature extraction and one seed, so any "
            "difference between them is the targets and not the audio or the order.",
            "Metrics are on the training subset, which is memorisation, not "
            "generalisation. No WER is claimed as a generalisation result.",
            "dev and test were never opened by this script.",
            "Rows violating frames >= labels are excluded per variant and listed by "
            "utterance id. dataset_v001 is not modified and no manifest is rewritten; "
            "the permanent disposition of those rows is still an open decision.",
        ],
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== summary ===")
    all_passed = True
    for record in records:
        all_passed = all_passed and bool(record["acceptance"]["passed"])
        print(
            f"{record['name']}: loss={record['final']['train_loss']:.4f} "
            f"ter={record['final']['token_error_rate']} "
            f"exact={record['final']['exact_match_rate']} "
            f"wer={record['final']['word_error_rate']} "
            f"params={record['model']['parameters']} "
            f"passed={record['acceptance']['passed']}"
        )
    if not all_passed:
        print("ACCEPTANCE FAILED: see acceptance blocks in results.json")
        return 1
    print("ACCEPTANCE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())