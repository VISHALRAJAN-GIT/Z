from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from tamil_voice.common.config import (
    CANONICAL_AUDIO,
    CANONICAL_CHANNELS,
    CANONICAL_DTYPE,
    CANONICAL_SAMPLE_RATE,
    AudioSpec,
    get_paths,
    load_yaml,
    project_root,
    resolve_config_path,
)


def test_canonical_audio_matches_guide() -> None:
    assert CANONICAL_SAMPLE_RATE == 16_000
    assert CANONICAL_CHANNELS == 1
    assert CANONICAL_DTYPE == "float32"
    assert CANONICAL_AUDIO.sample_rate == 16_000


def test_audio_spec_is_frozen() -> None:
    spec = AudioSpec()
    with pytest.raises(FrozenInstanceError):
        spec.sample_rate = 8_000  # type: ignore[misc]


def test_audio_spec_ndim() -> None:
    assert AudioSpec(channels=1).ndim == 1
    assert AudioSpec(channels=2).ndim == 2


def test_audio_spec_frame_length() -> None:
    window, hop = CANONICAL_AUDIO.frame_length(0.025, 0.010)
    assert window == 400
    assert hop == 160


@pytest.mark.parametrize("window,hop", [(0, 0.01), (0.025, 0), (-1, 0.01)])
def test_audio_spec_rejects_non_positive_frames(window: float, hop: float) -> None:
    with pytest.raises(ValueError):
        CANONICAL_AUDIO.frame_length(window, hop)


def test_project_root_contains_pyproject() -> None:
    root = project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "tamil_voice").is_dir()


def test_get_paths_finds_every_root() -> None:
    paths = get_paths()
    assert paths.root == project_root()
    for path in (paths.data, paths.configs, paths.checkpoints, paths.artifacts):
        assert isinstance(path, Path)
    assert paths.manifests == paths.data / "manifests"
    assert paths.raw == paths.data / "raw"
    assert paths.processed == paths.data / "processed"


def test_paths_relative_rendering() -> None:
    paths = get_paths()
    assert paths.relative(paths.manifests / "asr_train.jsonl") == "data/manifests/asr_train.jsonl"


def test_paths_relative_handles_foreign_path(tmp_path: Path) -> None:
    paths = get_paths()
    assert paths.relative(tmp_path) == tmp_path.as_posix()


def test_load_yaml_reads_mapping(tmp_path: Path) -> None:
    target = tmp_path / "cfg.yaml"
    target.write_text("sample_rate: 16000\nmel_bins: 80\n", encoding="utf-8")
    loaded = load_yaml(target)
    assert loaded == {"sample_rate": 16000, "mel_bins": 80}


def test_load_yaml_empty_file_is_empty_mapping(tmp_path: Path) -> None:
    target = tmp_path / "empty.yaml"
    target.write_text("", encoding="utf-8")
    assert load_yaml(target) == {}


def test_load_yaml_missing_file_raises_with_absolute_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_yaml(missing)
    assert str(missing) in str(excinfo.value)


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    target = tmp_path / "list.yaml"
    target.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(TypeError):
        load_yaml(target)


def test_resolve_config_path_prefers_existing_file() -> None:
    configs = get_paths().configs
    configs.mkdir(parents=True, exist_ok=True)
    target = configs / "asr" / "tiny.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("d_model: 192\n", encoding="utf-8")
    try:
        assert resolve_config_path("asr/tiny") == target
        assert resolve_config_path("asr/tiny.yaml") == target
    finally:
        target.unlink(missing_ok=True)


def test_resolve_config_path_adds_yaml_suffix_for_missing_file() -> None:
    resolved = resolve_config_path("audio/vad")
    assert resolved.suffix == ".yaml"
    assert resolved.parent.name == "audio"


def test_env_override_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    get_paths.cache_clear()
    monkeypatch.setenv("TVF_DATA_ROOT", "custom_data")
    try:
        assert get_paths().data == project_root() / "custom_data"
    finally:
        monkeypatch.delenv("TVF_DATA_ROOT", raising=False)
        get_paths.cache_clear()


def test_shipped_configs_are_valid_yaml() -> None:
    configs = get_paths().configs
    files = sorted(configs.rglob("*.yaml"))
    for path in files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{path} must contain a mapping"
        assert os.path.basename(path) == path.name
