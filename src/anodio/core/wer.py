"""Word error rate between two transcripts."""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> list[str]:
    """Lower-case words without punctuation; elisions such as "l'oiseau" stay one word."""
    text = unicodedata.normalize("NFC", text).lower().replace("’", "'")
    return re.sub(r"[^\w\s']", " ", text).split()


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """(substitutions + deletions + insertions) / reference words; None for an empty reference."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return None
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word))
        previous = current
    return previous[-1] / len(ref)
