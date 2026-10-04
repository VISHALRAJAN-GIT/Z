"""The tiny CTC acoustic model, and the length arithmetic it depends on.

This is the first model in ``src``. It was previously written twice, as a copy in
EXP-003 and a copy in EXP-005, because a committed experiment has to stay runnable
on its own. The architecture never actually diverged between those copies — a
line-by-line comparison found differences only in a docstring, in a diagnostic
message, and in a type hint — so there was no behaviour to reconcile, only duplication
to remove.

The experiments keep their own copies on purpose. Deleting them would rewrite the
research record and make EXP-003 and EXP-005 unrunnable. What this module removes is
the need for the *third* copy.

**The architecture is unidirectional and has no BatchNorm, both deliberately.**

Over a padded ``(B, C, T)`` batch, BatchNorm averages the zero-padded tail into the
statistics of the real frames, and a bidirectional GRU reads that tail backwards into
the last valid frames. Both corrupt valid output using padding alone. A unidirectional
stack fed true CTC input lengths cannot do this, because padded frames only ever
influence frames after them. Phase 07 needs streaming, which wants unidirectional
anyway, so the constraint and the requirement agree.

**The length arithmetic is part of the model, not a convenience.** CTC requires
``encoder_frames >= target_labels`` for every utterance, and a stride that is too
coarse silently invalidates real speech — 39 % of the corpus at stride 8. So
``subsampled_length`` is a method on the model rather than a free function, and
:func:`verify_length_math` checks it against ``nn.Conv1d`` itself on the running
machine. A length helper that is merely plausible is exactly the bug an overfit test
exists to catch, so it is measured rather than reviewed.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

#: Encoder subsampling factor, the product of the conv strides. With ``[2, 2]`` the
#: GRU runs at 25 Hz on a 16 kHz signal with a 10 ms hop.
DEFAULT_CONV_STRIDES = (2, 2)


class LengthMathError(RuntimeError):
    """The conv stack's real output length disagrees with ``subsampled_length``.

    A subclass of ``RuntimeError`` so that experiment code catching either name still
    works, because the two committed copies raise their own ``OverfitCheckFailed``.
    """


class TinyCTC(nn.Module):
    """Conv subsampling front end, GRU, linear classifier. Emits per-frame logits.

    There is no softmax: CTC scores the sequence directly against the frame
    distribution, and applying one here would force a training-time recomputation of
    log-probabilities that are cheaper to keep as logits.
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
        for out_channels, stride, padding in zip(
            conv_channels, conv_strides, conv_padding, strict=True
        ):
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

    @property
    def subsample_factor(self) -> int:
        """Product of the conv strides: input frames per output frame."""
        return math.prod(self.conv_strides)

    def subsampled_length(self, frames: int) -> int:
        """Exact output length of the conv stack, one layer at a time.

        This is ``nn.Conv1d``'s own output formula, applied to the length instead of
        to the data. It must stay exactly equal to what the conv stack produces, and
        :func:`verify_length_math` is what keeps it that way.
        """
        length = frames
        for stride, padding in zip(self.conv_strides, self.conv_padding, strict=True):
            length = (length + 2 * padding - self.conv_kernel) // stride + 1
        return max(length, 1)

    def output_lengths(self, input_lengths: Tensor) -> Tensor:
        """``subsampled_length`` over a batch of lengths, for use as CTC ``input_lengths``.

        Int64 on purpose: ``nn.CTCLoss`` requires it, and a float tensor here fails
        deep inside the loss rather than at the call site.
        """
        if input_lengths.dtype != torch.int64:
            raise TypeError(f"input_lengths must be torch.int64, got {input_lengths.dtype}")
        return torch.tensor(
            [self.subsampled_length(int(length)) for length in input_lengths],
            dtype=torch.int64,
            device=input_lengths.device,
        )

    def forward(self, features: Tensor) -> Tensor:
        """Logits of shape ``(batch, encoder_frames, num_classes)``."""
        hidden = self.convs(features.transpose(1, 2)).transpose(1, 2)
        hidden = self.dropout(hidden)
        hidden, _ = self.gru(hidden)
        return self.fc(hidden)


def verify_length_math(model: TinyCTC, lengths: list[int], probe_limit: int = 12) -> None:
    """Check ``subsampled_length`` against ``nn.Conv1d`` itself, on this machine.

    Both a padded batch and each length in isolation, because the batch path and the
    single path can disagree — a padded ``(B, C, T)`` tensor has ``T`` columns for
    every row while only ``lengths[i]`` of them are real. ``model.subsampled_length``
    is what CTC is told the lengths are, so if the two disagree then either the loss
    is being fed a wrong length or the model is emitting frames that do not exist.

    Raises:
        LengthMathError: with both the predicted and the observed length, so the
            failure says which direction the arithmetic is wrong. A bare
            "mismatch" here is close to useless when the whole point is the numbers.
    """
    unique = sorted(set(lengths))
    probe = unique if len(unique) <= probe_limit else unique[: probe_limit // 2] + unique[-(probe_limit // 2) :]
    if not probe:
        raise LengthMathError("no lengths to probe; got an empty sequence")
    device = next(model.parameters()).device
    batched = torch.zeros(len(probe), max(probe), model.n_mels, device=device)
    predicted_batch = model.subsampled_length(max(probe))
    observed_batch = model(batched).shape[1]
    if predicted_batch != observed_batch:
        raise LengthMathError(
            f"encoder length mismatch on a padded batch: subsampled_length says "
            f"{predicted_batch}, the conv stack produced {observed_batch}"
        )
    for frames in probe:
        single = torch.zeros(1, frames, model.n_mels, device=device)
        predicted = model.subsampled_length(frames)
        observed = model(single).shape[1]
        if predicted != observed:
            raise LengthMathError(
                f"encoder length mismatch at {frames} frames: subsampled_length says "
                f"{predicted}, the conv stack produced {observed}"
            )
