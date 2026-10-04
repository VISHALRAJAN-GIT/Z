from __future__ import annotations

import pytest
import torch
from torch import nn

from tamil_voice.asr.model import (
    DEFAULT_CONV_STRIDES,
    LengthMathError,
    TinyCTC,
    verify_length_math,
)

# The EXP-005 overfit configuration, which is the one the project measured against.
# Reproducing its parameter count is a regression test that moving the model did not
# quietly change it.
EXPERIMENTAL = {
    "n_mels": 80,
    "num_classes": 50,
    "conv_channels": [32, 64],
    "conv_kernel": 3,
    "conv_strides": [2, 2],
    "conv_padding": [1, 1],
    "gru_hidden": 256,
    "gru_layers": 2,
    "gru_bidirectional": False,
    "dropout": 0.0,
}


def _model(**overrides) -> TinyCTC:
    kwargs = {**EXPERIMENTAL, **overrides}
    return TinyCTC(**kwargs)  # type: ignore[arg-type]


def test_default_strides_subsample_by_four() -> None:
    assert DEFAULT_CONV_STRIDES == (2, 2)
    assert _model().subsample_factor == 4


def test_subsampled_length_matches_the_conv_stack() -> None:
    """The core invariant: the helper is the conv stack's own output formula."""
    model = _model()
    for frames in range(1, 200):
        observed = model(torch.zeros(1, frames, model.n_mels)).shape[1]
        assert model.subsampled_length(frames) == observed, frames


def test_subsampled_length_uses_ceil_not_floor() -> None:
    """Odd lengths are where floor(T/4) silently over-counts valid utterances."""
    model = _model()
    for frames in (1, 3, 5, 7, 9, 101):
        expected = -(-frames // 2)
        expected = -(-expected // 2)
        assert model.subsampled_length(frames) == expected, frames


def test_subsampled_length_never_returns_zero() -> None:
    assert _model().subsampled_length(0) == 1


def test_verify_length_math_passes_on_a_correct_model() -> None:
    verify_length_math(_model(), [13, 40, 100, 401, 1000])


def test_verify_length_math_catches_a_wrong_helper() -> None:
    """If the helper drifts from the conv stack, the failure must name both numbers."""
    model = _model()
    model.conv_kernel = 5  # the helper reads this, the built conv layers do not
    with pytest.raises(LengthMathError, match=r"subsampled_length says .* conv stack produced"):
        verify_length_math(model, [40, 100])


def test_verify_length_math_rejects_an_empty_probe() -> None:
    with pytest.raises(LengthMathError, match="empty"):
        verify_length_math(_model(), [])


def test_verify_length_math_samples_long_length_lists() -> None:
    verify_length_math(_model(), list(range(10, 400, 7)))


def test_output_lengths_are_int64_and_track_the_batch() -> None:
    model = _model()
    lengths = torch.tensor([37, 100, 256], dtype=torch.int64)
    out = model.output_lengths(lengths)
    assert out.dtype == torch.int64
    assert out.tolist() == [model.subsampled_length(n) for n in (37, 100, 256)]


def test_output_lengths_rejects_a_float_tensor() -> None:
    """nn.CTCLoss demands int64, and failing inside the loss hides the call site."""
    with pytest.raises(TypeError, match="torch.int64"):
        _model().output_lengths(torch.tensor([37.0, 100.0]))


def test_forward_emits_one_row_per_subsampled_frame() -> None:
    model = _model().eval()
    features = torch.randn(3, 150, model.n_mels)
    with torch.no_grad():
        logits = model(features)
    assert logits.shape == (3, model.subsampled_length(150), EXPERIMENTAL["num_classes"])
    assert torch.isfinite(logits).all()


def test_num_classes_out_matches_the_classifier() -> None:
    assert _model().fc.out_features == 50


def test_parameter_count_matches_the_measured_experiment() -> None:
    """EXP-005 measured 668818 parameters for this configuration on whitespace-kept targets."""
    assert sum(p.numel() for p in _model().parameters()) == 668818


def _receptive_end(model: TinyCTC, out_index: int) -> int:
    """Highest input frame index that output frame ``out_index`` depends on.

    Walks the conv stack backwards. One layer with stride ``s``, padding ``p`` and
    kernel ``k`` maps output ``j`` to inputs ``[j*s - p, j*s - p + k - 1]``, so a
    stack's receptive field is the union over layers. For the ``[2,2]`` stride-4 stack
    this works out to ``4*out_index + 3``.

    Deriving the bound instead of assuming a frame ratio is what keeps this test from
    asserting on frames that legitimately read the padding.
    """
    lo = hi = out_index
    layers = list(
        zip(
            model.conv_strides,
            model.conv_padding,
            [model.conv_kernel] * len(model.conv_strides),
            strict=True,
        )
    )
    for stride, padding, kernel in reversed(layers):
        lo = stride * lo - padding
        hi = stride * hi - padding + kernel - 1
    return hi


def test_padded_frames_cannot_influence_valid_frames() -> None:
    """The reason this model is unidirectional and has no BatchNorm.

    Overwriting the padded tail with noise must not change the logits at any frame
    whose receptive field ends before the padding starts. BatchNorm would fail this by
    folding the tail into its statistics, and a bidirectional GRU would fail it by
    reading the tail backwards into the last valid frames. The claim is in the module
    docstring, so it gets tested rather than asserted in prose.

    Only the frames that genuinely precede the padding are compared. Comparing the
    whole output would be wrong: the stack emits fewer frames than it consumes, so most
    of the output is *derived from* the padding this test perturbs, and those frames
    are supposed to change.
    """
    model = _model().eval()
    valid = 60
    padded = 150
    base = torch.randn(1, padded, model.n_mels)
    perturbed = base.clone()
    perturbed[:, valid:, :] = torch.randn(1, padded - valid, model.n_mels) * 10.0
    safe = [
        m for m in range(model.subsampled_length(padded)) if _receptive_end(model, m) < valid
    ]
    assert len(safe) >= 10, "the probe must leave enough clean frames to be meaningful"
    assert not any(_receptive_end(model, m) >= valid for m in safe)
    with torch.no_grad():
        a = model(base)[:, safe, :]
        b = model(perturbed)[:, safe, :]
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max().item()


def test_receptive_end_matches_the_stride_four_stack() -> None:
    model = _model()
    assert [_receptive_end(model, m) for m in range(5)] == [3, 7, 11, 15, 19]


def test_dropout_is_inactive_in_eval_mode() -> None:
    model = _model(dropout=0.5).eval()
    features = torch.randn(2, 80, model.n_mels)
    with torch.no_grad():
        assert torch.equal(model(features), model(features))


def test_batch_size_does_not_change_per_row_logits() -> None:
    """Row independence: a batched forward must equal per-row forwards."""
    model = _model().eval()
    features = torch.randn(4, 90, model.n_mels)
    with torch.no_grad():
        batched = model(features)
        singles = torch.cat([model(features[i : i + 1]) for i in range(4)], dim=0)
    assert torch.allclose(batched, singles, atol=1e-5)


def test_no_batchnorm_anywhere_in_the_model() -> None:
    """Named explicitly so a future convenience refactor cannot add one silently."""
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in _model().modules())


def test_bidirectional_configuration_doubles_the_classifier() -> None:
    unidirectional = _model()
    bidirectional = _model(gru_bidirectional=True)
    assert bidirectional.fc.in_features == 2 * EXPERIMENTAL["gru_hidden"]
    assert unidirectional.fc.in_features == EXPERIMENTAL["gru_hidden"]
