"""Anonymization of an Original into an Anonymized recording."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from anodio.core.audio import Audio
from anodio.core.params import Params
from anodio.core.stages import stage_named


class Anonymizer(Protocol):
    """An interchangeable Anonymization method. The Chain is the only one in v1 (see ADR 0001)."""

    def __call__(self, original: Audio, params: Params) -> Audio: ...


def chain(original: Audio, params: Params) -> Audio:
    """Apply the enabled Stages in order. Pure: same Original and Params, same output."""
    if not params.stages:
        return Audio(original.samples.copy(), original.sample_rate)
    samples = original.samples.astype(np.float64)
    for name in params.stages:
        samples = stage_named(name).apply(samples, original.sample_rate, params.values, params.seed)
    return Audio(_match_peak(samples, original.samples), original.sample_rate)


anonymize: Anonymizer = chain


def _match_peak(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples.astype(np.float32)
    return (samples * (np.max(np.abs(reference)) / peak)).astype(np.float32)
