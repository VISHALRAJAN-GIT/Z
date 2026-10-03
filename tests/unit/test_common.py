from __future__ import annotations

import io
import json
import logging
import os
import random

import numpy as np
import pytest

from tamil_voice.common.logging import LOGGER_NAME, get_logger, setup_logging
from tamil_voice.common.seed import DEFAULT_SEED, SeedReport, seed_everything, torch_available

# --------------------------------------------------------------------------- seed


def test_seed_everything_makes_python_reproducible() -> None:
    seed_everything(42)
    first = [random.random() for _ in range(5)]
    seed_everything(42)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_seed_everything_makes_numpy_reproducible() -> None:
    seed_everything(7)
    first = np.random.rand(5).tolist()
    seed_everything(7)
    second = np.random.rand(5).tolist()
    assert first == second


def test_different_seeds_differ() -> None:
    seed_everything(1)
    first = [random.random() for _ in range(5)]
    seed_everything(2)
    second = [random.random() for _ in range(5)]
    assert first != second


def test_seed_everything_sets_pythonhashseed() -> None:
    previous = os.environ.get("PYTHONHASHSEED")
    try:
        report = seed_everything(123)
        assert os.environ["PYTHONHASHSEED"] == "123"
        assert report.python_hash_seed is True
    finally:
        if previous is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = previous


def test_seed_everything_reports_numpy_was_seeded() -> None:
    report = seed_everything(0)
    assert report.numpy is True
    assert report.seed == 0


def test_seed_everything_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_seed_report_does_not_claim_torch_when_absent() -> None:
    report = seed_everything(5)
    assert isinstance(report, SeedReport)
    assert report.torch is torch_available()
    if not torch_available():
        assert report.fully_deterministic is False


def test_default_seed_is_stable() -> None:
    assert DEFAULT_SEED == 1337


# ------------------------------------------------------------------------- logging


def test_setup_logging_emits_json_lines() -> None:
    stream = io.StringIO()
    logger = setup_logging("INFO", stream=stream)
    logger.info("hello", extra={"stage": "phase00"})
    payload = json.loads(stream.getvalue().strip())
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["logger"] == LOGGER_NAME
    assert payload["stage"] == "phase00"
    assert "ts" in payload


def test_json_formatter_keeps_non_ascii_readable() -> None:
    stream = io.StringIO()
    logger = setup_logging("INFO", stream=stream)
    logger.info("நாளைக்கு காலை")
    assert "நாளைக்கு காலை" in stream.getvalue()


def test_setup_logging_is_idempotent() -> None:
    first = setup_logging("INFO", stream=io.StringIO())
    second = setup_logging("INFO", stream=io.StringIO())
    assert len(first.handlers) == 1
    assert len(second.handlers) == 1


def test_log_level_is_respected() -> None:
    stream = io.StringIO()
    logger = setup_logging("WARNING", stream=stream)
    logger.info("should be dropped")
    logger.warning("should appear")
    assert "should be dropped" not in stream.getvalue()
    assert "should appear" in stream.getvalue()
    setup_logging("INFO", stream=io.StringIO())


def test_invalid_level_falls_back_to_info() -> None:
    logger = setup_logging("NOT_A_LEVEL", stream=io.StringIO())
    assert logger.level == logging.INFO


def test_get_logger_returns_child() -> None:
    child = get_logger("audio")
    assert child.name == f"{LOGGER_NAME}.audio"
    assert get_logger().name == LOGGER_NAME
