"""Evaluation of an Original / Anonymized pair along each Axis."""

from __future__ import annotations

import math

import numpy as np

from anodio.core.audio import Audio
from anodio.core.models import SpeakerEmbedder
from anodio.core.features import clinical_features, energy_db, energy_dtw_db, f0_contour, f0_correlation
from anodio.core.params import Params
from anodio.core.report import AxisResult, Measure, Report
from anodio.core.timing import pauses

DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    # Indicative ECAPA decision threshold: recalibrate it with a Corpus evaluation.
    "privacy": {"max_cosine_similarity": 0.25},
    "prosody": {"max_pause_boundary_error_ms": 20.0, "min_f0_correlation": 0.9},
    # Per-feature limits override the default "max_relative_deviation". The HNR is in dB and
    # partly depends on the vocal tract (which the Anonymization moves on purpose), so it is
    # judged by an absolute difference. The local shimmer is very sensitive to the waveform shape.
    "clinical": {
        "max_relative_deviation": 0.10,
        "max_relative_deviation:shimmer_local_pct": 0.25,
        "max_absolute_deviation:hnr_db": 3.0,
    },
}


def evaluate(
    original: Audio,
    anonymized: Audio,
    params: Params | None = None,
    thresholds: dict[str, dict[str, float]] | None = None,
    embedder: SpeakerEmbedder | None = None,
) -> Report:
    """Measure each Axis. Without an `embedder`, the Privacy Axis is left out."""
    merged = {axis: {**values, **(thresholds or {}).get(axis, {})} for axis, values in DEFAULT_THRESHOLDS.items()}
    contour_o, contour_a = f0_contour(original), f0_contour(anonymized)
    axes = {}
    if embedder is not None:
        axes["privacy"] = _privacy(original, anonymized, embedder, merged["privacy"])
    axes["prosody"] = _prosody(original, anonymized, contour_o, contour_a, merged["prosody"])
    axes["clinical"] = _clinical(original, anonymized, contour_o, contour_a, merged["clinical"])
    return Report(axes=axes, params=params)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _privacy(original, anonymized, embedder: SpeakerEmbedder, thresholds: dict[str, float]) -> AxisResult:
    similarity = cosine_similarity(embedder.embed(original), embedder.embed(anonymized))
    limit = thresholds["max_cosine_similarity"]
    failures = []
    if similarity >= limit:
        failures.append(
            f"L'ASV attacker relie encore la voix au Speaker : similarité cosinus de {similarity:.2f} "
            f"(seuil : {limit:.2f})."
        )
    measures: dict[str, Measure] = {"cosine_similarity": similarity, "embedder": embedder.name}
    return AxisResult(measures, dict(thresholds), verdict=not failures, failures=failures)


def _prosody(original, anonymized, contour_o, contour_a, thresholds: dict[str, float]) -> AxisResult:
    pauses_o = pauses(original.samples, original.sample_rate)
    pauses_a = pauses(anonymized.samples, anonymized.sample_rate)
    boundary_error_ms = None
    if len(pauses_o) == len(pauses_a):
        diffs = [abs(a - b) for po, pa in zip(pauses_o, pauses_a) for a, b in zip(po, pa)]
        boundary_error_ms = round(1000 * float(np.mean(diffs)), 3) if diffs else 0.0
    correlation = f0_correlation(contour_o, contour_a)

    measures: dict[str, Measure] = {
        "duration_original_s": original.duration,
        "duration_anonymized_s": anonymized.duration,
        "pause_count_original": len(pauses_o),
        "pause_count_anonymized": len(pauses_a),
        "total_pause_original_s": round(sum(e - s for s, e in pauses_o), 3),
        "total_pause_anonymized_s": round(sum(e - s for s, e in pauses_a), 3),
        "pause_boundary_error_ms": boundary_error_ms,
        "f0_correlation": correlation,
        "energy_dtw_db": energy_dtw_db(energy_db(original), energy_db(anonymized)),
    }
    failures = []
    if abs(original.duration - anonymized.duration) > 1 / original.sample_rate:
        failures.append("La durée de la version anonymisée diffère de celle de l'original.")
    if boundary_error_ms is None:
        failures.append(f"Nombre de pauses différent : {len(pauses_o)} contre {len(pauses_a)}.")
    elif boundary_error_ms > thresholds["max_pause_boundary_error_ms"]:
        failures.append(
            f"Décalage moyen des pauses de {boundary_error_ms:.0f} ms "
            f"(seuil : {thresholds['max_pause_boundary_error_ms']:.0f} ms)."
        )
    if correlation is None:
        failures.append("Pas assez de trames voisées communes pour comparer l'intonation.")
    elif correlation < thresholds["min_f0_correlation"]:
        failures.append(
            f"Corrélation des contours de F0 de {correlation:.2f} (seuil : {thresholds['min_f0_correlation']:.2f})."
        )
    return AxisResult(measures, dict(thresholds), verdict=not failures, failures=failures)


def _clinical(original, anonymized, contour_o, contour_a, thresholds: dict[str, float]) -> AxisResult:
    features_o = clinical_features(original, contour_o)
    features_a = clinical_features(anonymized, contour_a)
    measures: dict[str, Measure] = {}
    failures = []
    for name, value_o in features_o.items():
        value_a = features_a[name]
        measures[f"{name}.original"] = _finite_or_none(value_o)
        measures[f"{name}.anonymized"] = _finite_or_none(value_a)
        if f"max_absolute_deviation:{name}" in thresholds:
            limit = thresholds[f"max_absolute_deviation:{name}"]
            deviation = _absolute_deviation(value_o, value_a)
            measures[f"{name}.absolute_deviation"] = deviation
            description = f"écart de {deviation:.1f} (seuil : {limit:g})" if deviation is not None else ""
        else:
            limit = thresholds.get(f"max_relative_deviation:{name}", thresholds["max_relative_deviation"])
            deviation = _relative_deviation(value_o, value_a)
            measures[f"{name}.relative_deviation"] = deviation
            description = (
                f"écart relatif de {100 * deviation:.0f} % (seuil : {100 * limit:.0f} %)" if deviation is not None else ""
            )
        if deviation is None:
            failures.append(f"{name} : mesure impossible.")
        elif deviation > limit:
            failures.append(f"{name} : {description}.")
    return AxisResult(measures, dict(thresholds), verdict=not failures, failures=failures)


def _absolute_deviation(original: float, anonymized: float) -> float | None:
    if not (math.isfinite(original) and math.isfinite(anonymized)):
        return None
    return abs(anonymized - original)


def _relative_deviation(original: float, anonymized: float) -> float | None:
    if not (math.isfinite(original) and math.isfinite(anonymized)):
        return None
    if original == anonymized:
        return 0.0
    return abs(anonymized - original) / max(abs(original), 1e-9)


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None
