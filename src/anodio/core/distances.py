"""Distances between two feature distributions, for the Informational Axis."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

HISTOGRAM_BINS = 30
EPSILON = 1e-10


def _histograms(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = min(a.min(), b.min()), max(a.max(), b.max())
    if high <= low:
        high = low + 1.0
    p, _ = np.histogram(a, bins=HISTOGRAM_BINS, range=(low, high))
    q, _ = np.histogram(b, bins=HISTOGRAM_BINS, range=(low, high))
    p = (p + EPSILON) / (p + EPSILON).sum()
    q = (q + EPSILON) / (q + EPSILON).sum()
    return p, q


def kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """KL(P_a || P_b) in bits, on shared histograms."""
    p, q = _histograms(a, b)
    return float(np.sum(p * np.log2(p / q)))


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, between 0 and 1."""
    p, q = _histograms(a, b)
    m = (p + q) / 2
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def distances(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    """KL, JS and Wasserstein between two samples; columns of 2-D samples are averaged."""
    a2, b2 = a.reshape(len(a), -1), b.reshape(len(b), -1)
    if len(a2) == 0 or len(b2) == 0:
        return {"kl": None, "js": None, "wasserstein": None}
    columns = range(a2.shape[1])
    return {
        "kl": float(np.mean([kl_divergence(a2[:, i], b2[:, i]) for i in columns])),
        "js": float(np.mean([js_divergence(a2[:, i], b2[:, i]) for i in columns])),
        "wasserstein": float(np.mean([wasserstein_distance(a2[:, i], b2[:, i]) for i in columns])),
    }
