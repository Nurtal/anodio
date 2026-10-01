"""Anonymization of an Original into an Anonymized recording."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from anodio.core.audio import Audio
from anodio.core.params import Params
from anodio.core.stages import mcadams, psola, vtln


class Anonymizer(Protocol):
    """An interchangeable Anonymization method. The Chain is the only one in v1 (see ADR 0001)."""

    def __call__(self, original: Audio, params: Params) -> Audio: ...


def chain(original: Audio, params: Params) -> Audio:
    """Apply the enabled Stages in order. Pure: same Original and Params, same output."""
    samples = original.samples.astype(np.float64)
    sr = original.sample_rate
    for stage in params.stages:
        if stage == "mcadams":
            samples = mcadams(samples, sr, params.mcadams_alpha)
        elif stage == "psola":
            rng = np.random.default_rng([params.seed, 1])
            samples = psola(samples, sr, params.pitch_semitones, params.formant_ratio, rng)
        elif stage == "vtln":
            samples = vtln(samples, sr, params.vtln_alpha)
    if not params.stages:
        return Audio(original.samples.copy(), sr)
    return Audio(_match_peak(samples, original.samples), sr)


anonymize: Anonymizer = chain


def _match_peak(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples.astype(np.float32)
    return (samples * (np.max(np.abs(reference)) / peak)).astype(np.float32)
