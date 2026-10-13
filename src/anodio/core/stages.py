"""Stages of the anonymization Chain. Each one is a pure function of (samples, sample_rate, param).

`REGISTRY`, at the end of this module, is the only place that lists the Stages: the Params,
the Chain, the CLI and the GUI are all derived from it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import get_window, lfilter

LPC_ORDER = 20
FRAME_SECONDS = 0.02


def _lpc(frame: np.ndarray, order: int) -> np.ndarray | None:
    """LPC coefficients [1, a1..ap] by the autocorrelation method, or None for a silent frame."""
    autocorr = np.correlate(frame, frame, mode="full")[len(frame) - 1 : len(frame) + order]
    if autocorr[0] <= 1e-10:
        return None
    autocorr = autocorr.copy()
    autocorr[0] *= 1.0 + 1e-9  # white-noise correction for numerical stability
    coeffs = solve_toeplitz(autocorr[:-1], -autocorr[1:])
    return np.concatenate([[1.0], coeffs])


def mcadams(samples: np.ndarray, sample_rate: int, alpha: float) -> np.ndarray:
    """Shift formants by raising the angle of every LPC pole to the power `alpha`.

    As in the VoicePrivacy Challenge B2 baseline.
    """
    return _warp_lpc_poles(samples, sample_rate, lambda angle: angle**alpha)


def vtln(samples: np.ndarray, sample_rate: int, alpha: float) -> np.ndarray:
    """Vocal tract length normalisation: bilinear (all-pass) warping of the LPC pole angles.

    alpha > 0 moves the low and mid frequencies up, alpha < 0 moves them down.
    """
    return _warp_lpc_poles(
        samples, sample_rate, lambda w: w + 2 * np.arctan(alpha * np.sin(w) / (1 - alpha * np.cos(w)))
    )


def _warp_lpc_poles(samples: np.ndarray, sample_rate: int, warp) -> np.ndarray:
    """Move the spectral envelope by remapping the angle (in ]0, pi[) of every LPC pole.

    Frame-wise LPC analysis / resynthesis with 50 % overlap-add. Each resynthesised
    frame is rescaled to the energy of its input frame: the gain of the warped filter
    differs from frame to frame, which would otherwise modulate the cycle amplitudes
    (raising the shimmer). The residual (excitation) is kept, so the pulse timing of
    the voice source is untouched.
    """
    frame_len = int(FRAME_SECONDS * sample_rate) // 2 * 2
    hop = frame_len // 2
    # sqrt-Hann at analysis and synthesis: their product (Hann) sums to 1 at 50 % overlap.
    window = np.sqrt(get_window("hann", frame_len))
    padded = np.concatenate([np.zeros(frame_len), samples.astype(np.float64), np.zeros(frame_len)])
    output = np.zeros_like(padded)
    for start in range(0, len(padded) - frame_len + 1, hop):
        frame = padded[start : start + frame_len] * window
        a = _lpc(frame, LPC_ORDER)
        if a is None:
            continue
        residual = lfilter(a, [1.0], frame)
        synthesized = lfilter([1.0], _warp_polynomial(a, warp), residual)
        energy = np.sum(synthesized**2)
        if energy > 0:
            synthesized *= np.sqrt(np.sum(frame**2) / energy)
        output[start : start + frame_len] += synthesized * window
    return output[frame_len : frame_len + len(samples)]


def _warp_polynomial(a: np.ndarray, warp) -> np.ndarray:
    poles = np.roots(a)
    angles = np.angle(poles)
    complex_ = np.abs(np.imag(poles)) > 1e-12
    new_poles = poles.copy()
    # Warp the upper half-plane poles; their conjugates mirror them.
    upper = complex_ & (angles > 0)
    lower = complex_ & (angles < 0)
    new_poles[upper] = np.abs(poles[upper]) * np.exp(1j * np.clip(warp(angles[upper]), 1e-3, np.pi - 1e-3))
    new_poles[lower] = np.abs(poles[lower]) * np.exp(-1j * np.clip(warp(-angles[lower]), 1e-3, np.pi - 1e-3))
    return np.real(np.poly(new_poles))


UNVOICED_HOP_SECONDS = 0.01
MAX_VOICED_GAP_SECONDS = 0.02
# Larger cycle-to-cycle deviations are pulse-detection errors (Praat's period factor is 1.3).
MAX_PERIOD_DEVIATION = 0.15
MAX_AMPLITUDE_DEVIATION = 0.5


def _glottal_pulses(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    import parselmouth
    from parselmouth.praat import call

    from anodio.core.features import pitch_range

    sound = parselmouth.Sound(samples.astype(np.float64), sample_rate)
    point_process = call(sound, "To PointProcess (periodic, cc)", *pitch_range(sound))
    n = call(point_process, "Get number of points")
    return np.array([call(point_process, "Get time from index", i) for i in range(1, n + 1)])


def _voiced_runs(pulses: np.ndarray) -> list[np.ndarray]:
    if len(pulses) == 0:
        return []
    breaks = np.flatnonzero(np.diff(pulses) > MAX_VOICED_GAP_SECONDS) + 1
    return [run for run in np.split(pulses, breaks) if len(run) >= 4]


def _relative_deviations(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a per-period series into a smooth contour and relative cycle-to-cycle deviations."""
    from scipy.ndimage import median_filter

    smooth = median_filter(values, size=5, mode="nearest")
    return smooth, values / smooth - 1.0


def psola(
    samples: np.ndarray,
    sample_rate: int,
    pitch_semitones: float,
    formant_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pitch-synchronous overlap-add that shifts F0 and formants while keeping jitter and shimmer.

    New pulse marks follow the original smoothed period contour divided by the pitch ratio,
    and each new period / pulse amplitude gets a relative deviation drawn from the original
    ones. Interpolating a pitch contour (as Praat does) would smooth those cycle-to-cycle
    perturbations away. Grains are time-scaled by `formant_ratio` to move the formants.
    Unvoiced stretches go through the same overlap-add at a fixed hop, with unchanged timing.
    """
    pitch_ratio = 2 ** (pitch_semitones / 12)
    n = len(samples)
    x = samples.astype(np.float64)
    # Each grain: (output centre, source centre, left half-width, right half-width, gain), in seconds.
    grains: list[tuple[float, float, float, float, float]] = []
    voiced_spans = []
    for run in _voiced_runs(_glottal_pulses(samples, sample_rate)):
        periods = np.diff(run)
        smooth_period, period_dev = _relative_deviations(periods)
        amplitudes = np.array([_peak(x, t, p, sample_rate) for t, p in zip(run[:-1], periods)])
        smooth_amp, amp_dev = _relative_deviations(np.maximum(amplitudes, 1e-9))
        period_pool = _plausible(period_dev, MAX_PERIOD_DEVIATION)
        amp_pool = _plausible(amp_dev, MAX_AMPLITUDE_DEVIATION)
        start, end = run[0], run[-1]
        voiced_spans.append((start, end))
        t = start
        while t <= end:
            j = min(int(np.searchsorted(run, t)), len(periods) - 1)
            j = j if j == 0 or abs(run[j] - t) <= abs(run[j - 1] - t) else j - 1
            j = min(j, len(periods) - 1)
            left = periods[j - 1] if j > 0 else periods[j]
            gain = (1 + rng.choice(amp_pool)) / (1 + amp_dev[j])
            grains.append((t, run[j], left, periods[j], gain))
            t += smooth_period[j] / pitch_ratio * (1 + rng.choice(period_pool))
    hop = UNVOICED_HOP_SECONDS
    margin = MAX_VOICED_GAP_SECONDS / 2
    for t in np.arange(0.0, n / sample_rate + hop, hop):
        if not any(s - margin <= t <= e + margin for s, e in voiced_spans):
            grains.append((t, t, hop, hop, 1.0))

    output = np.zeros(n)
    for out_t, src_t, left, right, gain in grains:
        grain = _windowed_grain(x, src_t, left, right, sample_rate)
        if formant_ratio != 1.0:
            grain = _time_scale(grain, 1 / formant_ratio)
            left /= formant_ratio
        begin = int(round((out_t - left) * sample_rate))
        lo, hi = max(begin, 0), min(begin + len(grain), n)
        if hi > lo:
            output[lo:hi] += gain * grain[lo - begin : hi - begin]
    return _match_envelope(output, x, sample_rate)


def _plausible(deviations: np.ndarray, limit: float) -> np.ndarray:
    """Deviations to draw from, without pulse-detection errors (which Praat's jitter also ignores)."""
    kept = deviations[np.abs(deviations) < limit]
    return kept if len(kept) else np.zeros(1)


def _peak(x: np.ndarray, t: float, period: float, sample_rate: int) -> float:
    lo = max(int((t - period / 2) * sample_rate), 0)
    hi = int((t + period / 2) * sample_rate) + 1
    return float(np.max(np.abs(x[lo:hi]))) if hi > lo else 0.0


def _windowed_grain(x: np.ndarray, t: float, left: float, right: float, sample_rate: int) -> np.ndarray:
    """Samples from t-left to t+right under an asymmetric Hann window (zero-padded at the edges)."""
    n_left, n_right = int(round(left * sample_rate)), int(round(right * sample_rate))
    centre = int(round(t * sample_rate))
    idx = np.arange(centre - n_left, centre + n_right)
    valid = (idx >= 0) & (idx < len(x))
    segment = np.zeros(len(idx))
    segment[valid] = x[idx[valid]]
    window = np.concatenate([np.hanning(2 * n_left + 1)[:n_left], np.hanning(2 * n_right + 1)[n_right:-1]])
    return segment * window


def _time_scale(grain: np.ndarray, factor: float) -> np.ndarray:
    """Stretch a grain in time by `factor` (linear interpolation)."""
    new_len = max(int(round(len(grain) * factor)), 2)
    return np.interp(np.linspace(0, len(grain) - 1, new_len), np.arange(len(grain)), grain)


ENVELOPE_SECONDS = 0.05


def _match_envelope(output: np.ndarray, reference: np.ndarray, sample_rate: int) -> np.ndarray:
    """Give `output` the slow energy envelope of `reference` (PSOLA changes grain density)."""
    from scipy.ndimage import uniform_filter1d

    size = int(ENVELOPE_SECONDS * sample_rate)
    ref = np.sqrt(np.maximum(uniform_filter1d(reference**2, size), 0.0))
    out = np.sqrt(np.maximum(uniform_filter1d(output**2, size), 0.0))
    gain = np.where(out > 1e-8, ref / np.maximum(out, 1e-8), 0.0)
    return output * np.minimum(gain, 10.0)


@dataclass(frozen=True)
class ParamSpec:
    """A Param of a Stage."""

    name: str
    label: str
    # Intervals in which the Param is drawn from the Seed. The gaps around the identity
    # exclude values too weak to anonymize.
    safe_ranges: tuple[tuple[float, float], ...]
    # Value for which the Stage leaves the signal unchanged.
    identity: float
    # Widest values an Override may take.
    bounds: tuple[float, float]

    def is_safe(self, value: float) -> bool:
        return any(low <= value <= high for low, high in self.safe_ranges)


@dataclass(frozen=True)
class Stage:
    name: str
    label: str
    # Rank in the Chain, which need not be the rank in the registry.
    position: int
    params: tuple[ParamSpec, ...]
    # (samples, sample rate, Param values by name, Seed) -> samples
    apply: Callable[[np.ndarray, int, Mapping[str, float], int], np.ndarray]


def _apply_mcadams(samples: np.ndarray, sample_rate: int, values: Mapping[str, float], seed: int) -> np.ndarray:
    return mcadams(samples, sample_rate, values["mcadams_alpha"])


def _apply_psola(samples: np.ndarray, sample_rate: int, values: Mapping[str, float], seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, 1])
    return psola(samples, sample_rate, values["pitch_semitones"], values["formant_ratio"], rng)


def _apply_vtln(samples: np.ndarray, sample_rate: int, values: Mapping[str, float], seed: int) -> np.ndarray:
    return vtln(samples, sample_rate, values["vtln_alpha"])


# Params are drawn from the Seed in the order of this registry: append a new Stage at the
# end, whatever its `position` in the Chain, so that every Seed keeps its Pseudo-voice.
REGISTRY: tuple[Stage, ...] = (
    Stage(
        name="mcadams",
        label="McAdams",
        position=20,
        params=(
            ParamSpec(
                "mcadams_alpha", "Coefficient McAdams α", safe_ranges=((0.5, 0.9),), identity=1.0, bounds=(0.4, 1.0)
            ),
        ),
        apply=_apply_mcadams,
    ),
    Stage(
        name="psola",
        label="Pitch / formants (PSOLA)",
        position=10,
        params=(
            ParamSpec(
                "pitch_semitones",
                "Décalage F0 (demi-tons)",
                safe_ranges=((-5.0, -2.0), (2.0, 5.0)),
                identity=0.0,
                bounds=(-8.0, 8.0),
            ),
            ParamSpec(
                "formant_ratio",
                "Ratio de formants",
                safe_ranges=((0.85, 0.95), (1.05, 1.15)),
                identity=1.0,
                bounds=(0.75, 1.25),
            ),
        ),
        apply=_apply_psola,
    ),
    Stage(
        name="vtln",
        label="VTLN",
        position=30,
        params=(
            ParamSpec(
                "vtln_alpha",
                "Warping VTLN",
                safe_ranges=((-0.15, -0.05), (0.05, 0.15)),
                identity=0.0,
                bounds=(-0.25, 0.25),
            ),
        ),
        apply=_apply_vtln,
    ),
)


def chain_stages() -> tuple[Stage, ...]:
    """The registered Stages, in the order the Chain applies them."""
    return tuple(sorted(REGISTRY, key=lambda stage: stage.position))


def stage_named(name: str) -> Stage:
    for stage in REGISTRY:
        if stage.name == name:
            return stage
    raise ValueError(f"Stage inconnu : {name}")
