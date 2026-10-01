"""Stages of the anonymization Chain. Each one is a pure function of (samples, sample_rate, param)."""

from __future__ import annotations

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

    Frame-wise LPC analysis / resynthesis with 50 % overlap-add,
    as in the VoicePrivacy Challenge B2 baseline. The residual (excitation) is kept,
    so the pulse timing of the voice source is untouched.
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
        poles = np.roots(a)
        angles = np.angle(poles)
        # Only warp complex poles in the upper half-plane; conjugates follow.
        upper = (angles > 0) & (angles < np.pi) & (np.abs(np.imag(poles)) > 1e-12)
        new_poles = poles.copy()
        new_poles[upper] = np.abs(poles[upper]) * np.exp(1j * angles[upper] ** alpha)
        lower = (angles < 0) & (angles > -np.pi) & (np.abs(np.imag(poles)) > 1e-12)
        new_poles[lower] = np.abs(poles[lower]) * np.exp(-1j * (-angles[lower]) ** alpha)
        new_a = np.real(np.poly(new_poles))
        output[start : start + frame_len] += lfilter([1.0], new_a, residual) * window
    return output[frame_len : frame_len + len(samples)]
