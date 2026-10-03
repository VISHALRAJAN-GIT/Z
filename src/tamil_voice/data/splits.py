"""Speaker-disjoint dataset splitting.

AGENTS.md section 8: split by **speaker**, never by individual recording. If the
same speaker appears in more than one split, the model recognises the voice rather
than the words, and the error rate looks better than it is. The IISc-MILE shipped
split violates this (MILE and MICI speakers appear in both halves), so this module
ignores it and re-splits by speaker.

The algorithm is deliberately simple and fully deterministic:

1. group utterances by speaker;
2. shuffle the speakers with a seeded RNG;
3. assign each speaker, in that order, to whichever split is furthest below its
   target utterance count (ties go to the first split in declaration order).

Step 3 spreads the small splits across the whole speaker list instead of dumping a
contiguous block, and tracks the requested utterance proportions even though
speakers contribute very different numbers of utterances (11 to 262 in this
corpus). Determinism matters: the same corpus, ratios and seed must always produce
the same split, or a benchmark is not reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from .corpus import Utterance

#: Split names, in tie-break precedence order (train wins an exact tie).
SPLIT_NAMES: tuple[str, ...] = ("train", "dev", "test")

_RATIO_TOLERANCE = 1e-6


class SplitError(Exception):
    """The requested split is impossible or the configuration is invalid."""


@dataclass(frozen=True)
class SplitRatios:
    """Target utterance fractions. Must be non-zero and sum to 1."""

    train: float = 0.90
    dev: float = 0.05
    test: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "dev": self.dev, "test": self.test}

    def validate(self) -> None:
        values = self.as_dict()
        for name, value in values.items():
            if value <= 0:
                raise SplitError(f"ratio for {name!r} must be positive, got {value}")
        total = sum(values.values())
        if abs(total - 1.0) > _RATIO_TOLERANCE:
            raise SplitError(f"ratios must sum to 1.0, got {total}")


@dataclass(frozen=True)
class SplitConfig:
    """Everything that determines a split, so it can be recorded and replayed."""

    ratios: SplitRatios = SplitRatios()
    seed: int = 1337


@dataclass(frozen=True)
class SpeakerSplitPlan:
    """The result of a speaker-disjoint split.

    ``assignment`` maps every speaker to exactly one split. ``by_split`` holds the
    utterances grouped by the split their speaker was assigned to; together they
    guarantee no speaker spans two splits.
    """

    assignment: dict[str, str]
    by_split: dict[str, list[Utterance]]
    config: SplitConfig

    def speakers(self, split: str) -> set[str]:
        return {s for s, name in self.assignment.items() if name == split}

    def utterance_counts(self) -> dict[str, int]:
        return {name: len(items) for name, items in self.by_split.items()}

    def speaker_counts(self) -> dict[str, int]:
        return {name: len(self.speakers(name)) for name in self.by_split}


def plan_speaker_split(utterances: Iterable[Utterance], config: SplitConfig | None = None) -> SpeakerSplitPlan:
    """Assign whole speakers to splits so no speaker spans two splits."""
    config = config or SplitConfig()
    config.ratios.validate()

    items = list(utterances)
    if not items:
        raise SplitError("cannot split an empty utterance list")

    by_speaker: dict[str, list[Utterance]] = {}
    for item in items:
        by_speaker.setdefault(item.speaker_id, []).append(item)

    ratios = config.ratios.as_dict()
    names = list(SPLIT_NAMES)
    target = {name: ratios[name] * len(items) for name in names}
    filled = {name: 0 for name in names}

    speakers = sorted(by_speaker)
    rng = random.Random(config.seed)
    rng.shuffle(speakers)

    assignment: dict[str, str] = {}
    for speaker in speakers:
        # Furthest below target; ties broken by SPLIT_NAMES order via -index.
        chosen = max(names, key=lambda name: (target[name] - filled[name], -names.index(name)))
        assignment[speaker] = chosen
        filled[chosen] += len(by_speaker[speaker])

    by_split: dict[str, list[Utterance]] = {name: [] for name in names}
    for item in items:
        by_split[assignment[item.speaker_id]].append(item)

    return SpeakerSplitPlan(assignment=assignment, by_split=by_split, config=config)


def check_speaker_disjoint(plan: SpeakerSplitPlan) -> None:
    """Raise ``SplitError`` if any speaker appears in more than one split.

    Guaranteed by construction, but the check is cheap and turns a silent split bug
    into a loud failure before any training runs.
    """
    seen: dict[str, str] = {}
    for name, items in plan.by_split.items():
        for item in items:
            previous = seen.setdefault(item.speaker_id, name)
            if previous != name:
                raise SplitError(f"speaker {item.speaker_id} appears in both {previous} and {name}")
