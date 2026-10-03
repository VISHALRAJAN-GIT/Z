from __future__ import annotations

from pathlib import Path

import pytest

from tamil_voice.data.corpus import Utterance
from tamil_voice.data.splits import (
    SpeakerSplitPlan,
    SplitConfig,
    SplitError,
    SplitRatios,
    check_speaker_disjoint,
    plan_speaker_split,
)


def _utt(speaker: str, index: int, prefix: str = "MILE") -> Utterance:
    stem = f"{prefix}_{speaker}_{index:07d}"
    return Utterance(
        utterance_id=stem,
        speaker_id=speaker,
        prefix=prefix,
        shipped_split="train",
        audio_path=Path(f"{stem}.wav"),
        text_path=Path(f"{stem}.txt"),
    )


def _corpus(n_speakers: int, utts_per_speaker: int) -> list[Utterance]:
    return [_utt(f"{s:07d}", i) for s in range(n_speakers) for i in range(utts_per_speaker)]


def test_every_speaker_assigned_to_exactly_one_split() -> None:
    plan = plan_speaker_split(_corpus(50, 4), SplitConfig(seed=1))
    assert len(plan.assignment) == 50
    assert set(plan.assignment.values()) <= {"train", "dev", "test"}
    check_speaker_disjoint(plan)
    assert sum(len(items) for items in plan.by_split.values()) == 200


def test_a_speakers_utterances_stay_together() -> None:
    plan = plan_speaker_split(_corpus(30, 5), SplitConfig(seed=2))
    for name, items in plan.by_split.items():
        for item in items:
            assert plan.assignment[item.speaker_id] == name
    counts = plan.speaker_counts()
    assert sum(counts.values()) == 30


def test_split_is_deterministic_for_a_seed() -> None:
    first = plan_speaker_split(_corpus(80, 3), SplitConfig(seed=7)).assignment
    second = plan_speaker_split(_corpus(80, 3), SplitConfig(seed=7)).assignment
    third = plan_speaker_split(_corpus(80, 3), SplitConfig(seed=8)).assignment
    assert first == second
    assert first != third


def test_ratios_are_approximated_by_utterance_share() -> None:
    plan = plan_speaker_split(_corpus(100, 10), SplitConfig(ratios=SplitRatios(0.9, 0.05, 0.05), seed=3))
    counts = plan.utterance_counts()
    total = sum(counts.values())
    assert total == 1000
    assert 0.86 <= counts["train"] / total <= 0.94
    assert 0 < counts["dev"] / total < 0.10
    assert 0 < counts["test"] / total < 0.10


def test_empty_corpus_raises() -> None:
    with pytest.raises(SplitError):
        plan_speaker_split([], SplitConfig())


@pytest.mark.parametrize("ratios", [SplitRatios(0.9, 0.05, 0.10), SplitRatios(0.9, 0.0, 0.10)])
def test_invalid_ratios_raise(ratios: SplitRatios) -> None:
    with pytest.raises(SplitError):
        ratios.validate()


def test_check_speaker_disjoint_detects_leak() -> None:
    speaker = "0000001"
    plan = SpeakerSplitPlan(
        assignment={speaker: "train"},
        by_split={"train": [_utt(speaker, 0)], "dev": [_utt(speaker, 1)], "test": []},
        config=SplitConfig(),
    )
    with pytest.raises(SplitError):
        check_speaker_disjoint(plan)
