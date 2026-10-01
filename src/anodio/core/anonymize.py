"""Anonymization of an Original into an Anonymized recording."""

from __future__ import annotations

import numpy as np

from anodio.core.audio import Audio
from anodio.core.params import Params
from anodio.core.stages import mcadams


def anonymize(original: Audio, params: Params) -> Audio:
    """Pure function: same Original and Params always give the same Anonymized."""
    samples = mcadams(original.samples, original.sample_rate, params.mcadams_alpha)
    return Audio(_match_peak(samples, original.samples), original.sample_rate)


def _match_peak(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples.astype(np.float32)
    return (samples * (np.max(np.abs(reference)) / peak)).astype(np.float32)
