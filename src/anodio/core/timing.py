"""Pause detection from the energy envelope."""

from __future__ import annotations

import numpy as np

FRAME_SECONDS = 0.02
HOP_SECONDS = 0.01
SILENCE_DB = -35.0
MIN_PAUSE_SECONDS = 0.15


def frame_rms(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = int(FRAME_SECONDS * sample_rate)
    hop = int(HOP_SECONDS * sample_rate)
    if len(samples) < frame:
        return np.array([np.sqrt(np.mean(samples**2))])
    starts = range(0, len(samples) - frame + 1, hop)
    return np.array([np.sqrt(np.mean(samples[s : s + frame] ** 2)) for s in starts])


def pauses(samples: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
    """Internal pauses (silences between two voiced stretches) as (start, end) in seconds."""
    rms = frame_rms(samples, sample_rate)
    peak = rms.max()
    if peak == 0:
        return []
    silent = 20 * np.log10(np.maximum(rms, 1e-12) / peak) < SILENCE_DB
    voiced = np.flatnonzero(~silent)
    result = []
    run_start = None
    for i in range(voiced[0], voiced[-1] + 1):
        if silent[i] and run_start is None:
            run_start = i
        elif not silent[i] and run_start is not None:
            start, end = run_start * HOP_SECONDS, i * HOP_SECONDS
            if end - start >= MIN_PAUSE_SECONDS:
                result.append((start, end))
            run_start = None
    return result
