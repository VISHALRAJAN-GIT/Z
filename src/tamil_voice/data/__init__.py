"""Dataset lifecycle: discovery, speaker-disjoint splitting, manifests.

Pipeline position (AGENTS.md section 8)::

    raw -> interim -> processed -> manifest -> training

Nothing in this package trains or reads audio samples. It turns a downloaded
corpus into versioned JSONL manifests with a speaker-disjoint split, which is the
only split that yields a meaningful error rate later.
"""

from __future__ import annotations

__all__ = ["corpus", "manifest", "splits"]
