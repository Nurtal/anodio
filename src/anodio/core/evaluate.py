"""Evaluation of an Original / Anonymized pair along each Axis."""

from __future__ import annotations

import numpy as np

from anodio.core.audio import Audio
from anodio.core.params import Params
from anodio.core.report import AxisResult, Measure, Report
from anodio.core.timing import pauses

DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "prosody": {"max_pause_boundary_error_ms": 20.0},
}


def evaluate(
    original: Audio,
    anonymized: Audio,
    params: Params | None = None,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> Report:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    return Report(axes={"prosody": _prosody(original, anonymized, thresholds["prosody"])}, params=params)


def _prosody(original: Audio, anonymized: Audio, thresholds: dict[str, float]) -> AxisResult:
    pauses_o = pauses(original.samples, original.sample_rate)
    pauses_a = pauses(anonymized.samples, anonymized.sample_rate)
    boundary_error_ms = None
    if len(pauses_o) == len(pauses_a):
        diffs = [abs(a - b) for po, pa in zip(pauses_o, pauses_a) for a, b in zip(po, pa)]
        boundary_error_ms = round(1000 * float(np.mean(diffs)), 3) if diffs else 0.0

    measures: dict[str, Measure] = {
        "duration_original_s": original.duration,
        "duration_anonymized_s": anonymized.duration,
        "pause_count_original": len(pauses_o),
        "pause_count_anonymized": len(pauses_a),
        "total_pause_original_s": round(sum(e - s for s, e in pauses_o), 3),
        "total_pause_anonymized_s": round(sum(e - s for s, e in pauses_a), 3),
        "pause_boundary_error_ms": boundary_error_ms,
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
    return AxisResult(measures, dict(thresholds), verdict=not failures, failures=failures)
