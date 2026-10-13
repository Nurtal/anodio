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
    syllable_rate: float = 0.0,
) -> np.ndarray:
    """Glottal pulse train through formant resonators.

    `jitter` / `shimmer` are the relative standard deviations of consecutive
    period lengths / pulse amplitudes. `syllable_rate` (Hz) dips the amplitude
    between syllable nuclei, as in connected speech.
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
    if syllable_rate:
        t = np.arange(n) / sample_rate
        envelope *= 1 - 0.8 * (0.5 + 0.5 * np.cos(2 * np.pi * syllable_rate * t))
    signal = signal * envelope
    return signal / np.max(np.abs(signal))


# (start, end) of voiced segments in `utterance`, in seconds.
UTTERANCE_SEGMENTS = ((0.3, 1.1), (1.5, 2.1))
UTTERANCE_DURATION = 2.4


def utterance(
    sample_rate: int = 16000,
    f0: float = 120.0,
    seed: int = 0,
    segments=UTTERANCE_SEGMENTS,
    duration: float = UTTERANCE_DURATION,
    **kwargs,
) -> Audio:
    """Voiced segments separated and framed by silence (by default voiced / silent / voiced)."""
    samples = np.zeros(int(duration * sample_rate))
    for i, (start, end) in enumerate(segments):
        seg = voiced_segment(end - start, sample_rate, f0=f0, seed=seed + i, **kwargs)
        s = int(start * sample_rate)
        samples[s : s + len(seg)] = 0.5 * seg
    return Audio(samples.astype(np.float32), sample_rate)


def long_utterance(**kwargs) -> Audio:
    """About 10 s: eight voiced stretches of 1 s separated by 0.25 s pauses."""
    segments = tuple((0.3 + 1.25 * i, 1.3 + 1.25 * i) for i in range(8))
    return utterance(segments=segments, duration=segments[-1][1] + 0.3, **kwargs)


def band_limited_utterance(sample_rate: int, vibrato: float = 0.06, shift: float = 0.0) -> Audio:
    """The same continuous signal whatever the sample rate: harmonics below 6 kHz, defined analytically.

    `utterance` places its pulses on the sample grid, so its content depends on the sample rate.
    `vibrato` is the relative F0 excursion; `shift` delays the first voiced segment (seconds).
    """
    t = np.arange(int(UTTERANCE_DURATION * sample_rate)) / sample_rate
    f0 = 130.0
    # Slow vibrato plus a fast, small frequency modulation that gives the voice some jitter.
    phase = 2 * np.pi * f0 * (
        t - vibrato / (2 * np.pi * 1.5) * np.cos(2 * np.pi * 1.5 * t) - 0.004 / (2 * np.pi * 31) * np.cos(2 * np.pi * 31 * t)
    )
    signal = np.zeros_like(t)
    for harmonic in range(1, 40):
        weight = 0.02 + sum(np.exp(-0.5 * ((harmonic * f0 - freq) / (2 * bw)) ** 2) for freq, bw in FORMANTS_HZ)
        signal += weight * np.sin(harmonic * phase)
    envelope = np.zeros_like(t)
    for start, end in ((UTTERANCE_SEGMENTS[0][0] + shift, UTTERANCE_SEGMENTS[0][1] + shift), UTTERANCE_SEGMENTS[1]):
        envelope += np.clip((t - start) / 0.03, 0, 1) * np.clip((end - t) / 0.03, 0, 1)
    envelope *= 1 - 0.6 * (0.5 + 0.5 * np.cos(2 * np.pi * 4 * t))  # syllables
    envelope *= 1 + 0.04 * np.sin(2 * np.pi * 23 * t)  # shimmer
    signal *= envelope
    return Audio((0.5 * signal / np.max(np.abs(signal))).astype(np.float32), sample_rate)
