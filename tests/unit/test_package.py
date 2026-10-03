from __future__ import annotations

import importlib

import pytest

import tamil_voice

SUBPACKAGES = (
    "audio",
    "vad",
    "enhancement",
    "asr",
    "text",
    "language",
    "tts",
    "optimization",
    "runtime",
    "common",
)


def test_version_is_exposed() -> None:
    assert isinstance(tamil_voice.__version__, str)
    assert tamil_voice.__version__.count(".") == 2


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(f"tamil_voice.{name}")
    assert module.__name__ == f"tamil_voice.{name}"


def test_public_surface_is_declared() -> None:
    assert set(tamil_voice.__all__) == {"__version__"}


def test_guide_says_modules_must_not_pretend_to_work() -> None:
    """A stub that raises NotImplementedError is fine; a stub that fakes success is not.

    Guards against the exact failure mode this project already hit once: a
    module that returns plausible values without doing the work.
    """
    offenders: list[str] = []
    for name in SUBPACKAGES:
        module = importlib.import_module(f"tamil_voice.{name}")
        for attr in dir(module):
            if attr.startswith("_"):
                continue
            value = getattr(module, attr)
            if callable(value) and getattr(value, "__module__", "") == module.__name__:
                offenders.append(f"tamil_voice.{name}.{attr}")
    assert offenders == [], f"unimplemented callables should not be exported yet: {offenders}"
