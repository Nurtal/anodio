"""Synthetic speech-like signals with known properties, used as test fixtures."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from anodio.core import Audio

FORMANTS_HZ = ((700, 80), (1220, 90), (2600, 120))


def _resonator(freq: float, bandwidth: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.exp(-np.pi * bandwidth / sample_rate)
    theta = 2 * np.pi * freq / sample_rate
    a = np.array([1.0, -2 * r * np.cos(theta), r * r])
    return np.array([a.sum()]), a


def voiced_segment(
    duration: float,
    sample_rate: int,
    f0: float = 120.0,
    jitter: float = 0.0,
    shimmer: float = 0.0,
    vibrato_semitones: float = 1.0,
    seed: int = 0,
    formants=FORMANTS_HZ,
) -> np.ndarray:
    """Glottal pulse train through formant resonators.

    `jitter` / `shimmer` are the relative standard deviations of consecutive
    period lengths / pulse amplitudes.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    source = np.zeros(n)
    t = 0.0
    while t < duration:
        idx = int(t * sample_rate)
        if idx >= n:
            break
        contour = f0 * 2 ** (vibrato_semitones / 12 * np.sin(2 * np.pi * 1.5 * t))
        source[idx] = 1.0 + shimmer * rng.standard_normal()
        t += (1.0 / contour) * (1.0 + jitter * rng.standard_normal())
    # Smooth the impulses into a crude glottal pulse.
    source = lfilter([1.0], [1.0, -0.95], source)
    signal = source
    for freq, bw in formants:
        b, a = _resonator(freq, bw, sample_rate)
        signal = lfilter(b, a, signal)
    ramp = min(int(0.02 * sample_rate), n // 2)
    envelope = np.ones(n)
    envelope[:ramp] = np.linspace(0, 1, ramp)
    envelope[n - ramp :] = np.linspace(1, 0, ramp)
    signal = signal * envelope
    return signal / np.max(np.abs(signal))


# (start, end) of voiced segments in `utterance`, in seconds.
UTTERANCE_SEGMENTS = ((0.3, 1.1), (1.5, 2.1))
UTTERANCE_DURATION = 2.4


def utterance(sample_rate: int = 16000, f0: float = 120.0, seed: int = 0, **kwargs) -> Audio:
    """Voiced / silent / voiced, framed by silence."""
    samples = np.zeros(int(UTTERANCE_DURATION * sample_rate))
    for i, (start, end) in enumerate(UTTERANCE_SEGMENTS):
        seg = voiced_segment(end - start, sample_rate, f0=f0, seed=seed + i, **kwargs)
        s = int(start * sample_rate)
        samples[s : s + len(seg)] = 0.5 * seg
    return Audio(samples.astype(np.float32), sample_rate)
